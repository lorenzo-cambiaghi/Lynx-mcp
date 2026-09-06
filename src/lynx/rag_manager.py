"""
RAG Manager - 100% LOCAL.

No data ever leaves the host. No calls to external services.
- Embeddings: HuggingFace local model (default BAAI/bge-small-en-v1.5,
  ~130 MB, downloaded once and cached)
- Vector store: ChromaDB persistent on local disk
- LLM: Disabled (None) - we only do vector retrieval, no generation
- Telemetry: Disabled
"""

import fnmatch
import hashlib
import logging
import os
import sys
import threading
import time

# ============================================================================
# BLOCK ANY EXTERNAL COMMUNICATION AND SILENCE NOISY LOGS
# ============================================================================
# Disable LlamaIndex / HuggingFace anonymous telemetry.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Drop any OpenAI key so LlamaIndex cannot accidentally call OpenAI.
os.environ.pop("OPENAI_API_KEY", None)
# Silence HuggingFace progress bars and warnings: their output on stdout
# would corrupt the JSON-RPC stream MCP uses.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# OFFLINE MODE is decided by `config.configure_hf_offline()`, which every
# entry point calls BEFORE importing this module: offline when the needed
# models are already cached (privacy guarantee), online for the one run
# that has to download them. Hard-coding HF_HUB_OFFLINE=1 here used to
# brick the first `lynx serve` on a clean machine.

# Silence the recurring "llama-index-readers-file package not found" warning:
# the default FlatReader handles plain-text files (code, markdown, JSON, etc.)
# correctly. The optional package adds PDF/Word/EPUB parsers we don't need.
logging.getLogger("llama_index.core.readers.file.base").setLevel(logging.ERROR)


def log(msg):
    """Write logs to stderr so we don't corrupt the JSON-RPC stream on stdout."""
    print(msg, file=sys.stderr)


import re
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from .embeddings import OnnxEmbedding
from rank_bm25 import BM25Okapi
from pathlib import Path
import json
from datetime import datetime

from .chunking import chunk_file, CHUNKER_VERSION
from . import fs_scan


# ----------------------------------------------------------------------------
# Code-aware tokenizer for BM25.
# ----------------------------------------------------------------------------
# Goal: make queries like "IDamageable", "AStarPathFinder", "calculateSpacing"
# match identifiers in the code even when the casing differs slightly.
#
# Strategy: keep the FULL identifier as a token AND its constituent parts
# (CamelCase / snake_case / dot.notation split). Lowercase everything.
# This roughly doubles index size but dramatically improves recall on
# identifier-heavy queries.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD = re.compile(r"[^A-Za-z0-9]+")


def _tokenize_code(text: str) -> list:
    """Split text into BM25 tokens, preserving both whole identifiers and parts."""
    if not text:
        return []
    tokens = []
    for raw in _NON_WORD.split(text):
        if not raw:
            continue
        lower_raw = raw.lower()
        tokens.append(lower_raw)
        # Add CamelCase / Pascal pieces as separate tokens for partial matching.
        if any(c.isupper() for c in raw):
            for piece in _CAMEL_BOUNDARY.split(raw):
                piece_lower = piece.lower()
                if piece_lower and piece_lower != lower_raw:
                    tokens.append(piece_lower)
    return tokens


def _normalize_extension(ext: str) -> str:
    """Make sure an extension has a leading dot and is lowercased."""
    ext = ext.lower().strip()
    return ext if ext.startswith(".") else f".{ext}"


# Below this body length we never dedup: short identical bodies (a one-line
# getter, `pass`, `return None`, `=> base.GetHashCode();`) are common across
# genuinely DISTINCT symbols, so collapsing them would hide real results. Long
# identical bodies, on the other hand, are real duplicates — a vendored or
# `build/`/`dist/` copy of the same file.
_DEDUP_MIN_CHARS = 120

# Extra candidates fetched beyond top_k so content-dedup can drop duplicates
# without leaving the final list short of top_k.
_DEDUP_HEADROOM = 5


def _dedup_by_content(results: list) -> list:
    """Drop later results whose (substantial) body is byte-identical to an
    earlier one — vendored/build copies of the same code.

    Order-preserving, so the highest-ranked occurrence survives. Bodies shorter
    than `_DEDUP_MIN_CHARS` are never collapsed (see the note above), so common
    boilerplate shared by distinct symbols is always kept.
    """
    seen: set = set()
    out: list = []
    for r in results:
        body = (r.get("content") or "").strip()
        if len(body) < _DEDUP_MIN_CHARS:
            out.append(r)  # too short to be a confident duplicate — keep it
            continue
        if body in seen:
            continue
        seen.add(body)
        out.append(r)
    return out


def _matches_filters(item: dict, file_glob, extensions, path_contains, paths=None) -> bool:
    """Apply optional post-retrieval filters to a single result dict.

    All provided filters are AND-ed. None / empty filters are ignored.
    Both `file` (basename) and `file_path` (full path) are checked, so a
    glob like '*.cs' works regardless of which field carries the path.

    `paths` is an explicit allowlist of relative paths (typically from
    `git diff --name-only`). When set, the item passes only if its
    `file_path` ends with one of them OR contains one of them as a
    substring. Used by `search_diff` to restrict results to files
    modified in the current branch.
    """
    file_name = item.get("file") or ""
    file_path = item.get("file_path") or ""

    if extensions:
        norm_exts = {_normalize_extension(e) for e in extensions}
        # Use file_name when available; fall back to file_path basename.
        candidate = file_name or os.path.basename(file_path)
        suffix = os.path.splitext(candidate)[1].lower()
        if suffix not in norm_exts:
            return False

    if path_contains:
        needle = path_contains
        if needle not in file_path and needle not in file_name:
            return False

    if file_glob:
        # fnmatch handles Unix-shell glob (*, ?, [seq]). Allow it to match
        # against both the basename and the full path so users can write
        # either '*.py' or '**/Bullet*/**'.
        if not (
            fnmatch.fnmatch(file_name, file_glob)
            or fnmatch.fnmatch(file_path, file_glob)
            # Path normalization: file_path uses OS separators; allow forward-
            # slash globs to still match by trying a normalized variant.
            or fnmatch.fnmatch(file_path.replace(os.sep, "/"), file_glob)
        ):
            return False

    if paths:
        # Normalize separators on both sides so a posix `subdir/foo.py`
        # from `git diff` matches Windows `subdir\foo.py` in chunks.
        norm_file_path = file_path.replace("\\", "/")
        if not any(
            norm_file_path.endswith(p.replace("\\", "/"))
            or p.replace("\\", "/") in norm_file_path
            for p in paths
        ):
            return False

    return True


# Severity levels for config drift detection. See _build_config_snapshot
# and check_config_drift for what triggers each.
DRIFT_CRITICAL = "critical"
DRIFT_WARNING = "warning"

# Schema version for the per-source file_hashes.json. Bump if the file shape
# changes incompatibly (the loader auto-discards mismatched files and a full
# rebuild kicks in, so existing installs upgrade transparently).
FILE_HASHES_SCHEMA_VERSION = 1

# Read buffer for SHA-256 hashing. 1 MiB is enough to keep IO efficient on
# big files (.shader / .compute can be hundreds of KB) while staying small
# in memory.
_HASH_READ_CHUNK_BYTES = 1 << 20

# Mapping of impacting field -> severity. Fields not in this mapping are
# considered runtime-only and do not trigger drift detection.
_DRIFT_SEVERITY = {
    "embedding_model_name": DRIFT_CRITICAL,  # vectors become incomparable
    "codebase_path": DRIFT_CRITICAL,         # entirely different code indexed
    "chunker_version": DRIFT_CRITICAL,       # chunk boundaries / metadata change
    "supported_extensions": DRIFT_WARNING,   # missing files / orphan vectors
    "collection_name": DRIFT_WARNING,        # points at a different collection
}


def _configure_local_only(embedding_model_name: str):
    """Configure LlamaIndex to run 100% offline.

    Must be called BEFORE any operation on the index.
    """
    # Local embeddings: a small open model run on CPU through ONNX Runtime
    # (no PyTorch in the install). One runtime is shared by every source
    # that uses the same model.
    Settings.embed_model = OnnxEmbedding(model_name=embedding_model_name)

    # LLM = None: we don't want LlamaIndex to ever call a remote LLM.
    # We only use the vector retriever (similarity search); the AI client
    # connected via MCP is responsible for any generation.
    Settings.llm = None


def _norm(path) -> str:
    """Normalize a path the same way LlamaIndex stores it: absolute, OS-native separators."""
    return os.path.normpath(os.path.abspath(str(path)))


class CodebaseRAG:
    """Manages indexing and search over a codebase.

    Everything runs locally: ChromaDB on disk, HuggingFace embeddings on CPU.
    """

    def __init__(
        self,
        codebase_path,
        rag_storage_path: str,
        supported_extensions,
        embedding_model_name: str,
        collection_name: str,
        search_mode: str = "hybrid",
        rrf_k: int = 60,
        candidate_pool_size: int = 30,
        reranker_config=None,  # RerankerConfig | None — optional cross-encoder rerank
        ignored_path_fragments=(),
        follower: bool = False,
    ):
        # A follower shares a store another Lynx process owns: it reads and
        # never writes, and it re-reads what the owner wrote (see
        # `refresh_if_stale`). See lynx/ownership.py for who owns what.
        self.follower = bool(follower)
        self._watermark = None
        self._last_staleness_check = 0.0

        self.codebase_path = Path(codebase_path)
        self.storage_path = Path(rag_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Frozenset of supported file extensions (lowercase, with leading dot).
        self.supported_extensions = frozenset(
            ext.lower() for ext in supported_extensions
        )

        # Path fragments to exclude from indexing (e.g. node_modules, dist).
        # Tracked as instance state because it now affects the indexed file set
        # (so it's part of the drift config snapshot) and because the shared
        # fs_scan walk needs it on every (re)build.
        self.ignored_path_fragments = tuple(ignored_path_fragments or ())

        # Stored as instance state because they are part of the config snapshot
        # used for drift detection.
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name

        # Search behavior: dense / sparse / hybrid (RRF-fused).
        if search_mode not in ("hybrid", "dense", "sparse"):
            raise ValueError(
                f"search_mode must be one of 'hybrid' | 'dense' | 'sparse', got {search_mode!r}"
            )
        self.search_mode = search_mode
        self.rrf_k = int(rrf_k)
        self.candidate_pool_size = int(candidate_pool_size)

        # Optional cross-encoder reranker. We hold the config so search()
        # knows top_n_before_rerank; the actual Reranker object is
        # lazy-constructed on the first search call that needs it (saves
        # ~80MB RAM when enabled but never queried).
        self.reranker_config = reranker_config
        self._reranker = None

        # Tri-state cache for batched-query embedding (search_batch):
        #   None  = not yet checked, True = batched path verified to match the
        #   single-query embedding, False = it diverged → use per-query fallback.

        # BM25 sparse index, lazily built from chunks living in ChromaDB.
        # _bm25_docs maps chunk_id -> tokenized content; the BM25Okapi object
        # is rebuilt on demand from this dict whenever it goes stale.
        self._bm25_docs: dict = {}
        self._bm25_meta: dict = {}  # chunk_id -> {"file": ..., "content": ...}
        self._bm25_okapi = None
        self._bm25_doc_ids: list = []

        # Serialize index mutations between the watcher, explicit updates,
        # a follower's refresh, and the search path. Re-entrant so a path that
        # already holds it can call a helper that takes it again: with several
        # sessions on one store these paths interleave far more than they used
        # to, and a plain Lock turns that into a deadlock instead of a wait.
        self._write_lock = threading.RLock()

        # Configure LlamaIndex for fully offline operation.
        _configure_local_only(embedding_model_name)

        # Initialize ChromaDB (local, persistent vector database).
        self.vector_store = ChromaVectorStore(
            chroma_collection=self._get_or_create_collection()
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # Metadata file tracks the last indexed git commit, update time,
        # and the config snapshot used to build the index.
        self.metadata_file = self.storage_path / "metadata.json"
        self.metadata = self._load_metadata()

        # Per-file SHA-256 cache for incremental rebuilds. The loader returns
        # an empty dict when the file is missing or its snapshot disagrees
        # with the live config — both cases force a full rebuild downstream.
        self.file_hashes_file = self.storage_path / "file_hashes.json"
        self._file_hashes: dict = self._load_file_hashes()

        # Where the store stood BEFORE we opened it, which is the safe order.
        # Opening applies the write-ahead log up to that moment; a watermark
        # read afterwards could already include a write the owner made in
        # between, and we would never go back for it. Reading early can only
        # cost one redundant refresh.
        self._watermark = self._store_watermark()

        self.index = self._load_or_build_index()

        if self._watermark is None:
            # First build: there was no store to read a moment ago, there is
            # one now. Start following from here.
            self._watermark = self._store_watermark()

        # After index is loaded, check whether the live config matches the
        # config that built it. Surfaces silent drift (e.g. embedding model
        # changed but old vectors still in store).
        self._check_and_log_drift()

    def _collection_count(self):
        """Chunks currently in the collection, or None when the store can't
        answer. Used to verify that writes actually landed — never to decide
        emptiness (that's `_load_or_build_index`, which must not mask a read
        failure as 0)."""
        try:
            return self.vector_store._collection.count()
        except Exception as e:
            log(f"[rag] could not read collection count: {e}")
            return None

    def _get_or_create_collection(self):
        """Open or create the persistent ChromaDB collection on local disk."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self._chroma_client = chromadb.PersistentClient(
            path=str(self.storage_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        return self._chroma_client.get_or_create_collection(self.collection_name)

    # ------------------------------------------------------------------
    # Follower mode: reading a store another Lynx process owns
    # ------------------------------------------------------------------
    # Several processes can read one ChromaDB store at the same time; only one
    # may write it (see lynx/ownership.py). A follower therefore searches
    # normally but has to be told when the owner has written something, because
    # retrieval answers from state loaded at open time: the HNSW segment applies
    # the write-ahead log when the store is opened, and the BM25 corpus is
    # cached in memory. `count()` alone stays live, which is exactly why this
    # went unnoticed — the numbers move while the results do not.

    _STALENESS_INTERVAL_SEC = 2.0
    # Above this many changed files, reloading the whole BM25 corpus is
    # cheaper than patching it file by file.
    _MAX_DELTA_FILES = 50

    def _store_watermark(self):
        """Monotonic "something was written" counter for this store.

        Read with plain sqlite rather than through a Chroma client: it runs
        before searches, so it has to cost nothing and must not contend with
        the owner. None when there is no store yet.
        """
        import sqlite3

        path = self.storage_path / "chroma.sqlite3"
        if not path.exists():
            return None
        try:
            db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        except sqlite3.Error:
            return None
        try:
            row = db.execute("SELECT max(seq_id) FROM embeddings_queue").fetchone()
            return (row[0] if row else 0) or 0
        except sqlite3.Error:
            return None
        finally:
            db.close()

    def _files_written_since(self, watermark):
        """``(files, exact)`` for the writes logged above `watermark`.

        `exact` is False when the log no longer covers the range we missed.
        Chroma purges it on its own schedule and `heal_wal` rewrites it, so a
        partial list would silently leave stale chunks in the BM25 cache; the
        caller reloads everything instead.
        """
        import sqlite3

        path = self.storage_path / "chroma.sqlite3"
        if watermark is None or not path.exists():
            return [], False
        try:
            db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        except sqlite3.Error:
            return [], False
        try:
            low, high = db.execute(
                "SELECT min(seq_id), max(seq_id) FROM embeddings_queue").fetchone()
            if low is None or low > watermark + 1 or high < watermark:
                return [], False
            files = set()
            for (meta,) in db.execute(
                    "SELECT metadata FROM embeddings_queue WHERE seq_id > ?",
                    (watermark,)):
                try:
                    m = json.loads(meta) if meta else {}
                except ValueError:
                    continue
                fp = m.get("file_path") or (m.get("metadata") or {}).get("file_path")
                if fp:
                    files.add(fp)
            return sorted(files), True
        except sqlite3.Error:
            return [], False
        finally:
            db.close()

    def _reopen_store(self):
        """Re-open ChromaDB so this process sees what another one wrote.

        Building a new client is what makes the segment replay the log;
        Chroma's process-level cache has to be dropped first or the same
        system is handed back. Collections other sources already hold keep
        working (verified), and this costs 0.13s on a 270MB store.
        """
        try:
            from chromadb.api.shared_system_client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception as e:
            log(f"[rag] could not drop the Chroma client cache: {e}")
        self._chroma_client = None
        self.vector_store = ChromaVectorStore(
            chroma_collection=self._get_or_create_collection()
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

    def _bm25_reload_file(self, abs_path: str) -> None:
        """Re-read one file's chunks from the store into the BM25 cache.

        The watcher path patches the cache from the nodes it just wrote; a
        follower never sees those nodes, so it reads them back instead.
        """
        self._bm25_drop_file_entries(abs_path)
        try:
            data = self.vector_store._collection.get(
                where={"file_path": abs_path},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            log(f"[bm25] could not re-read {abs_path}: {e}")
            return
        for cid, content, meta in zip(data.get("ids") or [],
                                      data.get("documents") or [],
                                      data.get("metadatas") or []):
            meta = meta or {}
            self._bm25_docs[cid] = _tokenize_code(content or "")
            self._bm25_meta[cid] = {
                "file": meta.get("file_name", "unknown"),
                "file_path": meta.get("file_path", ""),
                "symbol_name": meta.get("symbol_name", ""),
                "symbol_kind": meta.get("symbol_kind", ""),
                "language": meta.get("language", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "content": content or "",
            }

    def refresh_if_stale(self) -> bool:
        """Pick up the owner's writes. No-op for an owner; throttled otherwise.

        Returns True when something was actually reloaded.
        """
        if not self.follower:
            return False
        now = time.monotonic()
        if now - self._last_staleness_check < self._STALENESS_INTERVAL_SEC:
            return False
        self._last_staleness_check = now
        mark = self._store_watermark()
        if mark is None or mark == self._watermark:
            return False

        files, exact = self._files_written_since(self._watermark)
        with self._write_lock:
            self._reopen_store()
            patchable = (exact and files and len(files) <= self._MAX_DELTA_FILES
                         and bool(self._bm25_docs))
            if patchable:
                for abs_path in files:
                    self._bm25_reload_file(abs_path)
                self._bm25_okapi = None
            else:
                self._invalidate_bm25()
            self._watermark = mark
        what = (f"picked up the owner's changes ({len(files)} file(s))"
                if patchable else "reloaded after the owner wrote")
        log(f"[rag] {self.collection_name}: {what}")
        return True

    def _load_metadata(self):
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return {"last_commit": None, "last_update": None, "config_snapshot": None}

    def _save_metadata(self):
        self.metadata_file.write_text(json.dumps(self.metadata, indent=2))

    # ------------------------------------------------------------------
    # Per-file SHA-256 cache (incremental rebuild support)
    # ------------------------------------------------------------------

    def _compute_file_hash(self, abs_path: str) -> str:
        """SHA-256 of a file's bytes. Returns empty string on read error so
        callers can decide whether to treat that as 'always re-index'."""
        h = hashlib.sha256()
        try:
            with open(abs_path, "rb") as f:
                while True:
                    block = f.read(_HASH_READ_CHUNK_BYTES)
                    if not block:
                        break
                    h.update(block)
            return h.hexdigest()
        except OSError as e:
            log(f"[rag] hash failed for {abs_path}: {e}")
            return ""

    def _file_hashes_snapshot(self) -> dict:
        """Snapshot of the config fields the hash file must be consistent with.

        Mirrors `_build_config_snapshot` BUT excludes `codebase_path`: the
        hashes are keyed by absolute path, so a path change naturally
        invalidates them (files at the new path simply aren't in the cache).

        `chunker_version` is included so any change to the chunking logic
        (tree-sitter rules, fallback behavior, metadata shape) discards the
        cache and triggers a full rebuild — old chunks no longer align with
        new ones, and skipping unchanged files would leave stale boundaries.
        """
        return {
            "embedding_model_name": self.embedding_model_name,
            "collection_name": self.collection_name,
            "supported_extensions": sorted(self.supported_extensions),
            "chunker_version": CHUNKER_VERSION,
        }

    def _load_file_hashes(self) -> dict:
        """Load the per-file SHA cache, or empty dict on any inconsistency.

        Returns {} (forcing a full rebuild) when:
          - the file is missing (first run, or wiped)
          - schema_version mismatches
          - the stored snapshot disagrees with the live config (any tracked
            change makes the stored vectors / chunk identities unreliable)
        """
        if not self.file_hashes_file.exists():
            return {}
        try:
            raw = json.loads(self.file_hashes_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log(f"[rag] file_hashes.json unreadable ({e}); ignoring and rebuilding.")
            return {}

        if raw.get("schema_version") != FILE_HASHES_SCHEMA_VERSION:
            log(
                f"[rag] file_hashes.json schema_version mismatch "
                f"(got {raw.get('schema_version')!r}, expected "
                f"{FILE_HASHES_SCHEMA_VERSION}); discarding cache."
            )
            return {}

        stored_snapshot = raw.get("config_snapshot") or {}
        live_snapshot = self._file_hashes_snapshot()
        if stored_snapshot != live_snapshot:
            log(
                "[rag] file_hashes.json config snapshot disagrees with live "
                "config; cache is invalid, will rebuild from scratch."
            )
            return {}

        return raw.get("files") or {}

    def _save_file_hashes(self):
        """Persist the current in-memory hash cache + the matching snapshot."""
        payload = {
            "schema_version": FILE_HASHES_SCHEMA_VERSION,
            "config_snapshot": self._file_hashes_snapshot(),
            "files": self._file_hashes,
        }
        try:
            self.file_hashes_file.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except OSError as e:
            log(f"[rag] could not persist file_hashes.json: {e}")

    def _record_file_hash(self, abs_path: str, sha: str, *, chunks: int | None = None):
        """Update one entry in the in-memory cache. Caller persists via
        _save_file_hashes when convenient (batched at end of a rebuild,
        or per-event in the watcher path).

        `chunks` is how many chunks of this file are in the index right now — 0
        is a legitimate value (nothing chunkable in the file). It exists so a
        coverage check can tell "absent because empty" from "absent because the
        insert failed"; entries written before this field existed carry None,
        which reads as "expected to be present". An entry must only be written
        AFTER the chunks are in the index (see _build_index / update_file):
        this cache is what `_partition_files` trusts to skip work."""
        self._file_hashes[abs_path] = {
            "sha256": sha,
            "last_indexed_at": datetime.now().isoformat(),
            "chunks": chunks,
        }

    def _forget_file_hash(self, abs_path: str):
        """Drop one entry. No-op if absent."""
        self._file_hashes.pop(abs_path, None)

    # ------------------------------------------------------------------
    # AST-aware chunking pipeline (M3)
    # ------------------------------------------------------------------

    def _build_nodes_for_file(self, abs_path: str) -> list:
        """Read `abs_path`, run it through the chunker, return TextNode list.

        Returns [] on any read/decode error (the file is logged and skipped —
        a single unreadable file shouldn't abort a build of thousands).

        Every node carries the metadata fields that downstream code expects:
          - file_path        (used by _delete_file_chunks for the where-clause)
          - file_name        (used by search-result formatters and BM25 cache)
          - symbol_name      (AST chunker only; "<chunk1>" for fallback)
          - symbol_kind      ("method_declaration", ..., "text_window")
          - language         ("c_sharp", "python", ..., "text")
          - chunker          ("tree_sitter" | "sentence_splitter")
          - start_line, end_line  (1-based inclusive)
        """
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            log(f"[rag] read failed for {abs_path}: {e}")
            return []

        chunks = chunk_file(abs_path, content)
        nodes = []
        for c in chunks:
            metadata = {
                "file_path": c["file_path"],
                "file_name": c["file_name"],
                "symbol_name": c.get("symbol_name", ""),
                "symbol_kind": c.get("symbol_kind", ""),
                "language": c.get("language", "text"),
                "chunker": c.get("chunker", "sentence_splitter"),
                "start_line": c.get("start_line", 1),
                "end_line": c.get("end_line", 1),
            }
            nodes.append(TextNode(text=c["text"], metadata=metadata))
        return nodes

    def _reset_collection(self):
        """Drop the ChromaDB collection so a rebuild does not duplicate data."""
        try:
            self._chroma_client.delete_collection(self.collection_name)
        except Exception as e:
            log(f"warning: delete_collection failed (probably did not exist): {e}")
        collection = self._chroma_client.get_or_create_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def reset_index(self):
        """Empty the collection and forget the SHA cache so the next build is a
        full destructive rebuild.

        Uses ChromaDB's delete API (no file removal), so it is safe to call
        while this process holds the store open — unlike deleting the storage
        directory, which Windows blocks while the open HNSW segment handle is
        live (WinError 32). Used by the `reset` flow to recover/clean a source
        in place."""
        if self.follower:
            from .errors import StoreNotOwnedError
            raise StoreNotOwnedError(self.collection_name, str(self.storage_path))
        with self._write_lock:
            self._reset_collection()
            self._file_hashes = {}
            self._save_file_hashes()
            # The collection is now empty; drop the BM25 cache so a stale warm
            # corpus can't serve results until the next build repopulates it.
            self._invalidate_bm25()

    def _list_candidate_files(self) -> list:
        """Return abs paths of files we would index — the canonical "what should
        be in the index" set used by the SHA partitioner.

        Delegates to the shared `fs_scan.list_candidate_files` so the vector
        index and the graph layer walk the codebase identically (same dot-dir
        exclusion, same extension filter, same `ignored_path_fragments`).
        """
        return fs_scan.list_candidate_files(
            self.codebase_path,
            self.supported_extensions,
            self.ignored_path_fragments,
        )

    def _partition_files(self, candidate_files: list) -> tuple:
        """Diff disk-state against the SHA cache.

        Returns (unchanged, changed, added, removed) as four disjoint sets of
        absolute paths:
          - unchanged: in cache, SHA still matches → preserve existing chunks
          - changed:   in cache, SHA differs       → delete chunks + re-index
          - added:     not in cache                → index
          - removed:   in cache but not on disk    → delete chunks + drop entry

        Computes a SHA for every file that exists in BOTH sets (the common
        case). For 1.5k files at ~50KB average this is ~1s on a modern SSD —
        cheap compared to the embedding cost it lets us skip.
        """
        candidate_set = set(candidate_files)
        cached_set = set(self._file_hashes.keys())

        added = candidate_set - cached_set
        removed = cached_set - candidate_set

        unchanged = set()
        changed = set()
        for abs_path in candidate_set & cached_set:
            cached_sha = self._file_hashes[abs_path].get("sha256")
            live_sha = self._compute_file_hash(abs_path)
            if live_sha and live_sha == cached_sha:
                unchanged.add(abs_path)
            else:
                changed.add(abs_path)

        return unchanged, changed, added, removed

    def _build_index(self):
        """Build or incrementally update the index.

        Two regimes:
          - No usable SHA cache (first build, snapshot mismatch, or a wiped
            file_hashes.json): full destructive rebuild — wipe any leftover
            chunks (they may be bound to a stale embedding space) and index
            everything.
          - Valid SHA cache: delta rebuild — skip files whose SHA still
            matches, delete + re-index files whose SHA changed, drop
            files removed from disk, index files newly added.
        """
        candidate_files = self._list_candidate_files()

        if not self._file_hashes:
            log(f"[rag] Full rebuild from {self.codebase_path} ({len(candidate_files)} files)")
            # Stale chunks (from a previous embedding model or extension set)
            # would corrupt search results; wipe before inserting.
            try:
                existing_count = self.vector_store._collection.count()
            except Exception:
                existing_count = 0
            if existing_count > 0:
                self._reset_collection()
            unchanged, changed, added, removed = set(), set(), set(candidate_files), set()
        else:
            unchanged, changed, added, removed = self._partition_files(candidate_files)
            log(
                f"[rag] Delta rebuild from {self.codebase_path}: "
                f"{len(unchanged)} unchanged, {len(changed)} changed, "
                f"{len(added)} added, {len(removed)} removed"
            )

        # Short-circuit when there's nothing to do (delta only — full rebuild
        # always falls through to the indexing step).
        if not changed and not added and not removed:
            log("[rag] Index already in sync with disk; no work to do.")
            return VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        # Drop chunks for files that need re-indexing or removal.
        for abs_path in changed:
            self._delete_file_chunks(abs_path)
        for abs_path in removed:
            self._delete_file_chunks(abs_path)
            self._forget_file_hash(abs_path)

        # Index added + changed via the AST-aware chunker (chunking.chunk_file).
        # The chunker yields TextNode objects with full metadata; we pass them
        # to insert_nodes() so LlamaIndex does NOT re-run its own SentenceSplitter
        # on top of our chunks.
        files_to_index = sorted(changed | added)
        index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        if files_to_index:
            chunks_before = self._collection_count()
            total_nodes = 0
            indexed = {}   # abs_path -> chunk count actually inserted
            failed = []    # insert raised: nothing of this file is in the index
            for abs_path in files_to_index:
                nodes = self._build_nodes_for_file(abs_path)
                if not nodes:
                    # Legitimately empty (no chunks produced): record it below
                    # with chunks=0 so we don't re-chunk it on every rebuild,
                    # and so a coverage check knows the absence is expected.
                    indexed[abs_path] = 0
                    continue
                try:
                    index.insert_nodes(nodes)
                    total_nodes += len(nodes)
                    indexed[abs_path] = len(nodes)
                except Exception as e:
                    log(f"[rag] insert failed for {abs_path}: {e}")
                    failed.append(abs_path)

            # Prove the writes landed before trusting them. An insert can fail
            # WITHOUT raising — stale rows in chroma's `acquire_write` (left by
            # a killed process) make every write a silent no-op, which is how an
            # index stayed frozen for 25 days while each pass logged success.
            # So the cache is only updated when the collection grew by exactly
            # the number of chunks we sent; on any mismatch nothing is recorded
            # and the next pass retries the whole batch (safe: re-indexing a
            # file deletes its chunks first, so a partial success can't
            # duplicate).
            chunks_after = self._collection_count()
            grew = (chunks_after - chunks_before) if None not in (
                chunks_before, chunks_after) else None
            if total_nodes and grew is not None and grew != total_nodes:
                log(f"[rag] REFUSING to record {len(indexed)} file(s) as indexed: "
                    f"sent {total_nodes} chunk(s) but the collection grew by "
                    f"{grew}. The store is not accepting writes — check "
                    f"`lynx manager doctor` (stale write locks are the usual "
                    f"cause). Nothing is lost: the files stay queued for the "
                    f"next pass.")
            else:
                # Record SHAs ONLY for the files whose chunks actually landed in
                # the index. Recording a failed one would mark it "unchanged"
                # forever: _partition_files trusts this cache, so the file would
                # never be retried and would stay invisible to search — silently,
                # since the failure was only ever a log line. (Real incident
                # 2026-07-03: 202 files of the `framework` source cached as
                # indexed, absent from the index, never retried for 25 days.)
                for abs_path, chunk_count in indexed.items():
                    sha = self._compute_file_hash(abs_path)
                    if sha:
                        self._record_file_hash(abs_path, sha, chunks=chunk_count)
                # A file that failed must not keep a stale entry either —
                # dropping it is what makes the next pass treat it as `added`.
                for abs_path in failed:
                    self._forget_file_hash(abs_path)

                if failed:
                    log(f"[rag] {len(failed)} file(s) failed to insert and will "
                        f"be retried on the next update (not recorded).")
                log(f"[rag] Indexed {len(indexed)} file(s) -> {total_nodes} chunk(s).")

        self._save_file_hashes()
        # Collection contents changed: drop the BM25 cache so the next search
        # rebuilds it from the new ChromaDB state.
        self._invalidate_bm25()
        return index

    def _load_or_build_index(self):
        """Load existing index or build a new one.

        Source of truth is the ChromaDB collection: if it already contains
        documents we reuse the index instead of rebuilding it on every start.

        On a true first-run build we also persist metadata immediately, so
        subsequent code paths (drift detection, git-update detection) have a
        valid baseline without needing an extra explicit update() call.
        """
        try:
            existing_count = self.vector_store._collection.count()
        except Exception as e:
            # A count() failure here is NOT "empty" — a fresh collection
            # returns 0 without raising. An exception means ChromaDB cannot
            # read the store (incompatible version, truncated segment). Don't
            # mask it as 0 (which silently reuses a broken index and breaks the
            # dashboard); surface it so the caller can mark the source corrupt
            # and offer a reset.
            from .errors import CorruptIndexError
            raise CorruptIndexError(
                self.collection_name, f"{type(e).__name__}: {e}",
                str(self.storage_path),
            ) from e

        if existing_count > 0:
            log(f"[rag] ChromaDB already populated ({existing_count} chunks), reusing.")
            return VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        # count == 0: before rebuilding (a destructive operation through
        # _reset_collection), distinguish "true first run" from "race / lock
        # contention with another process". If metadata records a previous
        # build, do not re-index: reuse the store as-is and let an explicit
        # update() call force a rebuild if the user really wants one.
        if self.metadata.get("last_commit"):
            log(
                "[rag] Empty collection but metadata indicates a previous build: "
                "skipping rebuild (likely contention with another process). "
                "Reusing existing store."
            )
            return VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        # Genuine first run: build, then persist metadata in the same step
        # so the next constructor call sees a complete state and so a
        # follow-up update(force=True) does not redundantly rebuild.
        index = self._build_index()
        self._record_build_metadata()
        return index

    def needs_update(self):
        """Return True if a new git commit has been made since the last indexing.

        Returns False (without raising) when the codebase is not a git repo
        or git integration is not desired.
        """
        try:
            from git import Repo
            repo = Repo(self.codebase_path)
            current_commit = repo.head.commit.hexsha
            last_commit = self.metadata.get("last_commit")
            if current_commit != last_commit:
                return True
        except Exception as e:
            # Not a git repo, or git not available: that's fine, just skip.
            log(f"[rag] git check skipped: {e}")
        return False

    def _record_build_metadata(self):
        """Persist the post-build metadata: git head, timestamp, config snapshot.

        Called from every code path that completes a fresh _build_index().
        Centralized here so that '_load_or_build_index' (first-run case) and
        'update()' (explicit rebuild case) cannot drift out of sync.
        """
        try:
            from git import Repo
            repo = Repo(self.codebase_path)
            self.metadata["last_commit"] = repo.head.commit.hexsha
        except Exception as e:
            log(f"[rag] could not read git HEAD ({e}), saving last_commit='unknown'")
            self.metadata["last_commit"] = "unknown"
        self.metadata["last_update"] = datetime.now().isoformat()
        # A full rebuild aligns the index with the current config; refresh
        # the snapshot so subsequent drift checks compare against the right
        # baseline.
        self.metadata["config_snapshot"] = self._build_config_snapshot()
        self._save_metadata()

    def update(self, force=False):
        """Rebuild the index if there are git changes (or if force=True)."""
        if self.follower:
            from .errors import StoreNotOwnedError
            raise StoreNotOwnedError(self.collection_name, str(self.storage_path))
        if not force and not self.needs_update():
            log("[rag] Index already up to date.")
            return

        log("[rag] Updating codebase index...")
        self.index = self._build_index()
        self._record_build_metadata()
        log("[rag] Index updated successfully.")

    def search(
        self,
        query: str,
        top_k=5,
        *,
        file_glob=None,
        extensions=None,
        path_contains=None,
        paths=None,
    ):
        """Hybrid (default) / dense / sparse search over the codebase.

        Mode is set at construction time via `search_mode`. In hybrid mode we
        fetch `candidate_pool_size` candidates from each retriever (dense and
        BM25) then fuse rankings via Reciprocal Rank Fusion.

        Optional filters (all AND-ed together) narrow the result set:
          - file_glob: fnmatch pattern matched against basename and full path.
          - extensions: iterable of extensions ('.py' or 'py'), case-insensitive.
          - path_contains: substring required in file path or filename.
          - paths: explicit allowlist of relative paths (used by search_diff
            to restrict results to files modified vs a base branch).

        When any filter is present the underlying retrieval over-fetches a
        wider pool (5x top_k) so post-filtering still leaves enough results.
        """
        # No auto-update here: the watcher keeps the index live and any
        # git-based catch-up is handled on demand by update_codebase_index.
        return self._search_once(
            query,
            top_k=top_k,
            mode=self.search_mode,
            file_glob=file_glob,
            extensions=extensions,
            path_contains=path_contains,
            paths=paths,
        )

    def _search_once(
        self,
        query: str,
        top_k: int,
        mode: str,
        *,
        file_glob=None,
        extensions=None,
        path_contains=None,
        paths=None,
    ):
        """Single-pass retrieval helper shared by search() and deep_search().

        `mode` is explicit (rather than reading self.search_mode) so deep_search
        can override it per call without mutating instance state.
        """
        return self._retrieve_one(
            query, top_k, mode,
            file_glob=file_glob, extensions=extensions,
            path_contains=path_contains, paths=paths,
        )

    def _retrieve_one(
        self,
        query: str,
        top_k: int,
        mode: str,
        *,
        file_glob=None,
        extensions=None,
        path_contains=None,
        paths=None,
        embedding=None,
    ):
        """The single shared retrieval core for ONE query: dense/sparse/hybrid
        + post-filters + optional cross-encoder rerank, returning the final
        top_k hits.

        `search()`/`deep_search()` (single) and `search_batch()` BOTH route
        through here, so the two paths can never drift (mode dispatch, over-fetch,
        filters, rerank all live in one place). `embedding` lets the batch path
        pass a precomputed query vector to skip re-embedding; when None the dense
        retriever embeds the query itself.
        """
        # A follower answers from what it loaded at open time, so ask the store
        # whether the owner has written since. Throttled and, when nothing
        # moved, one read-only sqlite query.
        self.refresh_if_stale()

        has_filters = bool(file_glob or extensions or path_contains or paths)
        rerank_enabled = bool(
            self.reranker_config is not None and self.reranker_config.enabled
        )
        # Over-fetch when filtering OR reranking: filtered-out candidates leave
        # a smaller pool, and the reranker needs a bigger pool to actually
        # reorder things (no point reranking exactly top_k results). A small
        # constant headroom is always added so content-dedup (which can drop
        # duplicate chunks) doesn't leave the dense/sparse path short of top_k.
        fetch_n = top_k + _DEDUP_HEADROOM
        if has_filters:
            fetch_n = top_k * 5
        if rerank_enabled:
            fetch_n = max(fetch_n, self.reranker_config.top_n_before_rerank)

        if mode == "dense":
            results = self._dense_lookup(query, fetch_n, embedding=embedding)
        elif mode == "sparse":
            results = self._bm25_lookup(query, fetch_n)
        elif mode == "hybrid":
            pool = max(fetch_n, self.candidate_pool_size)
            dense_hits = self._dense_lookup(query, pool, embedding=embedding)
            sparse_hits = self._bm25_lookup(query, pool)
            results = self._rrf_fuse([dense_hits, sparse_hits], k=self.rrf_k, top_k=pool)
        else:
            raise ValueError(
                f"mode must be one of 'hybrid' | 'dense' | 'sparse', got {mode!r}"
            )

        if has_filters:
            results = [
                r for r in results
                if _matches_filters(r, file_glob, extensions, path_contains, paths=paths)
            ]

        # Collapse chunks with byte-identical bodies: vendored copies, `build/`
        # / `dist/` mirrors of source, or a file checked in twice produce
        # different ids (so RRF keeps them all) but read as duplicates. Keep the
        # highest-ranked occurrence. Done before rerank + truncation so a dup
        # can't crowd a distinct result out of top_k.
        results = _dedup_by_content(results)

        # Cross-encoder rerank (opt-in, after filters so we don't waste
        # cycles on chunks the user filtered out anyway). Lazy-construct
        # the Reranker on the first call that needs it — saves ~80MB
        # RAM for sources where the flag is on but never queried.
        if rerank_enabled and len(results) > 1:
            if self._reranker is None:
                from .reranker import Reranker
                self._reranker = Reranker(
                    model_name=self.reranker_config.model_name,
                )
            pool = results[: self.reranker_config.top_n_before_rerank]
            return self._reranker.rerank(query, pool, top_k=top_k)  # already top_k

        return results[:top_k]

    def deep_search(
        self,
        queries,
        top_k: int = 5,
        *,
        mode=None,
        file_glob=None,
        extensions=None,
        path_contains=None,
        min_score=None,
        min_results=None,
        return_all_variants: bool = False,
        score_thresholds=None,
    ):
        """Multi-query fallback search.

        Tries each query in `queries` in order. The first variant whose result
        set passes the weakness threshold wins and is returned. If all variants
        fail the threshold, returns the *strongest weak set* with a marker so
        the caller can decide what to do.

        Args:
            queries: list[str] of query variants in priority order.
            top_k: results to return.
            mode: per-call override of retrieval mode ("dense"/"sparse"/"hybrid").
                  None = use self.search_mode.
            file_glob, extensions, path_contains: same semantics as search().
                  Applied to every variant.
            min_score: per-call override of the score threshold. None = use the
                  mode-specific default from score_thresholds.
            min_results: per-call override of the min-results threshold.
                  None = 2.
            return_all_variants: when True, the response dict includes the
                  per-variant result lists for debugging.
            score_thresholds: dict {mode: threshold} from config. Optional;
                  callers without config can omit it and rely on min_score.

        Returns:
            dict with shape:
              {
                "results": [...],          # the final ranked results
                "winning_variant_index": int or None,  # 1-based, None if all weak
                "variants_tried": int,
                "all_weak": bool,          # True if no variant crossed threshold
                "per_variant": [...],      # only when return_all_variants
              }
        """
        if not isinstance(queries, (list, tuple)):
            raise TypeError(f"queries must be a list of strings, got {type(queries).__name__}")
        if len(queries) == 0:
            raise ValueError("queries must contain at least one string")
        if not all(isinstance(q, str) and q.strip() for q in queries):
            raise ValueError("every query must be a non-empty string")

        effective_mode = mode or self.search_mode
        if effective_mode not in ("hybrid", "dense", "sparse"):
            raise ValueError(
                f"mode must be one of 'hybrid' | 'dense' | 'sparse', got {effective_mode!r}"
            )

        # Resolve the threshold for this call. Precedence:
        #   1. explicit min_score arg
        #   2. score_thresholds dict from config (mode-specific entry)
        #   3. built-in fallback defaults
        builtin_defaults = {"dense": 0.45, "hybrid": 0.012, "sparse": 3.0}
        if min_score is not None:
            threshold = float(min_score)
        elif score_thresholds and effective_mode in score_thresholds:
            threshold = float(score_thresholds[effective_mode])
        else:
            threshold = builtin_defaults[effective_mode]

        effective_min_results = int(min_results) if min_results is not None else 2

        per_variant = []
        best_results = []
        best_top_score = float("-inf")
        winning_idx = None

        for idx, variant in enumerate(queries):
            results = self._search_once(
                variant,
                top_k=top_k,
                mode=effective_mode,
                file_glob=file_glob,
                extensions=extensions,
                path_contains=path_contains,
            )
            top_score = results[0]["score"] if results else float("-inf")
            passes = (
                len(results) >= effective_min_results
                and top_score >= threshold
            )

            if return_all_variants:
                per_variant.append({
                    "query": variant,
                    "results": list(results),
                    "top_score": top_score if results else None,
                    "passed_threshold": passes,
                })

            # Track the strongest weak set in case all variants fail.
            if top_score > best_top_score:
                best_top_score = top_score
                best_results = results

            if passes:
                response = {
                    "results": results,
                    "winning_variant_index": idx + 1,
                    "variants_tried": idx + 1,
                    "all_weak": False,
                }
                if return_all_variants:
                    response["per_variant"] = per_variant
                return response

        # All variants failed the threshold.
        response = {
            "results": best_results,
            "winning_variant_index": None,
            "variants_tried": len(queries),
            "all_weak": True,
        }
        if return_all_variants:
            response["per_variant"] = per_variant
        return response

    # ------------------------------------------------------------------
    # Retrieval primitives: dense (vectors), sparse (BM25), fusion (RRF)
    # ------------------------------------------------------------------

    def search_batch(
        self,
        queries,
        top_k: int = 5,
        *,
        mode=None,
        file_glob=None,
        extensions=None,
        path_contains=None,
        paths=None,
    ) -> list:
        """Search many queries at once, batching the query EMBEDDING into a
        single model call (the dominant per-query cost — ~13ms each on CPU).

        Returns a list aligned to `queries`: one result list per query. Each
        query runs through the SAME `_retrieve_one` core as `search()` (dense/
        sparse/hybrid + filters + rerank), so results are identical to calling
        `search()` once per query — just faster for N > 1, because the embedding
        is batched (BM25 / RRF / rerank still run per query).
        """
        if not isinstance(queries, (list, tuple)) or len(queries) == 0:
            raise ValueError("queries must be a non-empty list of strings")
        if not all(isinstance(q, str) and q.strip() for q in queries):
            raise ValueError("every query must be a non-empty string")

        mode = mode or self.search_mode
        if mode not in ("hybrid", "dense", "sparse"):
            raise ValueError(
                f"mode must be one of 'hybrid' | 'dense' | 'sparse', got {mode!r}"
            )

        # The win: one batched embedding call for every query instead of N.
        embeddings = None
        if mode in ("hybrid", "dense"):
            embeddings = self._embed_queries_batch(list(queries))

        return [
            self._retrieve_one(
                q, top_k, mode,
                file_glob=file_glob, extensions=extensions,
                path_contains=path_contains, paths=paths,
                embedding=(embeddings[i] if embeddings is not None else None),
            )
            for i, q in enumerate(queries)
        ]

    def _embed_queries_batch(self, queries: list) -> list:
        """Embed N queries in one model call, producing the SAME vectors the
        single-query path (`get_query_embedding`) would, so batch results match
        `search()`.

        BGE-style models prepend a *query* instruction ("Represent this
        question for searching…") that texts don't get (cosine ~0.92 apart).
        `OnnxEmbedding.embed_queries` is that path in batch form: same
        formatting, same graph, one run. Any other embedding model falls back
        to one call per query.
        """
        embed_model = Settings.embed_model
        batch = getattr(embed_model, "embed_queries", None)
        if callable(batch):
            return batch(list(queries))
        return [embed_model.get_query_embedding(q) for q in queries]

    def _dense_lookup(self, query: str, top_n: int, embedding=None) -> list:
        """Pure semantic retrieval via the LlamaIndex / ChromaDB pipeline.

        Propagates the AST-aware chunker metadata (symbol_name, symbol_kind,
        language, start_line, end_line) so the formatter can cite results
        precisely (e.g. "MyClass.handleClick at L42-58").

        When `embedding` is provided (a precomputed query vector), the retriever
        skips re-embedding and uses it directly — this is what `search_batch`
        relies on to embed N queries in a single model call.
        """
        retriever = self.index.as_retriever(similarity_top_k=top_n)
        if embedding is not None:
            from llama_index.core import QueryBundle
            results = retriever.retrieve(
                QueryBundle(query_str=query, embedding=list(embedding))
            )
        else:
            results = retriever.retrieve(query)
        return [
            {
                "id": getattr(getattr(r, "node", None), "id_", None) or r.metadata.get("file_path", ""),
                "file": r.metadata.get("file_name", "unknown"),
                "file_path": r.metadata.get("file_path", ""),
                "symbol_name": r.metadata.get("symbol_name", ""),
                "symbol_kind": r.metadata.get("symbol_kind", ""),
                "language": r.metadata.get("language", ""),
                "start_line": r.metadata.get("start_line", 0),
                "end_line": r.metadata.get("end_line", 0),
                "content": r.get_content(),
                "score": r.score,
            }
            for r in results
        ]

    def _ensure_bm25(self):
        """Ensure the BM25Okapi index is current.

        Two regimes, both lazy (only runs when a sparse/hybrid search needs it):
          - **warm cache**: `_bm25_docs` is already populated (kept in sync by
            the per-file watcher path). Just rebuild the BM25Okapi from the
            cached, already-tokenized corpus — NO ChromaDB read, NO re-tokenize.
          - **cold cache**: `_bm25_docs` is empty (first search, or after a bulk
            (re)build / reset cleared it). Pull every chunk from ChromaDB once,
            tokenize, and populate the cache.

        The cost this avoids: previously every single watcher save invalidated
        the whole BM25 index, so the next search re-read the entire collection
        from ChromaDB and re-ran the (regex-heavy) code tokenizer over every
        chunk. Now an edit only re-tokenizes the chunks of the edited file.
        """
        if self._bm25_okapi is not None:
            return
        if not self._bm25_docs:
            self._bm25_load_from_store()
        self._bm25_build_okapi()

    def _bm25_load_from_store(self):
        """Cold-load the tokenized corpus from ChromaDB into the cache."""
        try:
            data = self.vector_store._collection.get(include=["documents", "metadatas"])
        except Exception as e:
            log(f"[bm25] could not read ChromaDB collection: {e}")
            self._bm25_docs = {}
            self._bm25_meta = {}
            return
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        # Built aside and swapped in at the end: a search on another thread
        # then sees either the old corpus or the new one, never one being
        # filled in.
        fresh_docs: dict = {}
        fresh_meta: dict = {}
        for cid, content, meta in zip(ids, docs, metas):
            fresh_docs[cid] = _tokenize_code(content or "")
            fresh_meta[cid] = {
                "file": (meta or {}).get("file_name", "unknown"),
                "file_path": (meta or {}).get("file_path", ""),
                "symbol_name": (meta or {}).get("symbol_name", ""),
                "symbol_kind": (meta or {}).get("symbol_kind", ""),
                "language": (meta or {}).get("language", ""),
                "start_line": (meta or {}).get("start_line", 0),
                "end_line": (meta or {}).get("end_line", 0),
                "content": content or "",
            }
        self._bm25_docs = fresh_docs
        self._bm25_meta = fresh_meta

    def _bm25_build_okapi(self):
        """(Re)build the BM25Okapi object from the current cached corpus.

        The snapshot is taken under the write lock: the watcher and, on a
        follower, the refresh both edit this cache from another thread, and
        iterating a dict while it is being written raises. Only the copy is
        locked, not the BM25 construction, so searches are not serialised
        behind it.
        """
        with self._write_lock:
            doc_ids = list(self._bm25_docs.keys())
            corpus = [self._bm25_docs[cid] for cid in doc_ids]
        self._bm25_doc_ids = doc_ids
        self._bm25_okapi = BM25Okapi(corpus) if doc_ids else None

    def _invalidate_bm25(self):
        """Drop the BM25 cache entirely so the next search cold-loads from
        ChromaDB. Used after bulk (re)builds and resets, where so much changed
        that a full reload is the honest, simplest choice."""
        self._bm25_okapi = None
        self._bm25_docs = {}
        self._bm25_meta = {}
        self._bm25_doc_ids = []

    def _bm25_drop_file_entries(self, abs_path: str):
        """Remove every cached BM25 entry whose chunk belongs to `abs_path`."""
        drop = [
            cid for cid, m in self._bm25_meta.items()
            if m.get("file_path") == abs_path
        ]
        for cid in drop:
            self._bm25_docs.pop(cid, None)
            self._bm25_meta.pop(cid, None)

    def _bm25_apply_file_update(self, abs_path: str, nodes: list):
        """Incrementally refresh the BM25 cache for one re-indexed file.

        Only maintains the cache when it is already WARM (`_bm25_docs`
        populated). A cold cache is left cold: the next search cold-loads from
        ChromaDB, which already reflects the just-inserted nodes — so doing
        nothing here is both correct and cheaper than a redundant load.
        """
        if not self._bm25_docs:
            return
        self._bm25_drop_file_entries(abs_path)
        for n in nodes:
            cid = getattr(n, "id_", None)
            if not cid:
                continue
            meta = getattr(n, "metadata", None) or {}
            content = getattr(n, "text", "") or ""
            self._bm25_docs[cid] = _tokenize_code(content)
            self._bm25_meta[cid] = {
                "file": meta.get("file_name", "unknown"),
                "file_path": meta.get("file_path", ""),
                "symbol_name": meta.get("symbol_name", ""),
                "symbol_kind": meta.get("symbol_kind", ""),
                "language": meta.get("language", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "content": content,
            }
        # Rebuild the Okapi from the cache on the next search (no store read).
        self._bm25_okapi = None

    def _bm25_apply_file_removal(self, abs_path: str):
        """Incrementally drop a removed file's chunks from a warm BM25 cache."""
        if not self._bm25_docs:
            return
        self._bm25_drop_file_entries(abs_path)
        self._bm25_okapi = None

    def _bm25_lookup(self, query: str, top_n: int) -> list:
        """BM25 sparse retrieval over the in-memory tokenized corpus."""
        self._ensure_bm25()
        if self._bm25_okapi is None or not self._bm25_doc_ids:
            return []
        query_tokens = _tokenize_code(query)
        if not query_tokens:
            return []
        scores = self._bm25_okapi.get_scores(query_tokens)
        # Pair scores with doc ids, sort desc, take top_n.
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        out = []
        for i in ranked_idx:
            if scores[i] <= 0:
                # rank-bm25 returns 0 for documents without any matching token.
                # Skip them to keep the candidate list meaningful.
                continue
            cid = self._bm25_doc_ids[i]
            meta = self._bm25_meta.get(cid, {})
            out.append({
                "id": cid,
                "file": meta.get("file", "unknown"),
                "file_path": meta.get("file_path", ""),
                "symbol_name": meta.get("symbol_name", ""),
                "symbol_kind": meta.get("symbol_kind", ""),
                "language": meta.get("language", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "content": meta.get("content", ""),
                "score": float(scores[i]),
            })
        return out

    def _rrf_fuse(self, rankings: list, k: int, top_k: int) -> list:
        """Reciprocal Rank Fusion: combine multiple rankings into one.

        Each document gets score = sum over rankings of 1 / (k + rank+1).
        """
        fused_scores: dict = {}
        item_by_id: dict = {}
        for ranking in rankings:
            for rank, item in enumerate(ranking):
                cid = item.get("id")
                if not cid:
                    continue
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                # Keep the first metadata seen for this id; sources agree on file/content.
                item_by_id.setdefault(cid, item)
        sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        out = []
        for cid in sorted_ids:
            item = dict(item_by_id[cid])
            item["score"] = fused_scores[cid]  # fused score replaces raw per-source score
            out.append(item)
        return out

    # ------------------------------------------------------------------
    # Incremental, per-file update (driven by the file watcher)
    # ------------------------------------------------------------------

    def is_supported(self, filepath) -> bool:
        """True if the file has a supported extension and lives inside the codebase."""
        p = Path(filepath)
        if p.suffix.lower() not in self.supported_extensions:
            return False
        try:
            p.resolve().relative_to(self.codebase_path.resolve())
        except (ValueError, OSError):
            return False
        return True

    def _delete_file_chunks(self, abs_path: str) -> int:
        """Delete all chunks belonging to a file from the collection.

        Returns the number of chunks removed.
        """
        try:
            col = self.vector_store._collection
            existing = col.get(where={"file_path": abs_path})
            ids = existing.get("ids") or []
            if ids:
                col.delete(ids=ids)
            return len(ids)
        except Exception as e:
            log(f"[rag] delete chunks failed for {abs_path}: {e}")
            return 0

    def update_file(self, filepath) -> bool:
        """Re-index a single file: drop the old chunks, insert the new ones.

        Skips the re-index entirely when the file's SHA-256 matches the cached
        one — this filters out phantom saves (editor "save" that didn't change
        any bytes, whitespace-only diffs the editor normalized back, etc.) so
        the watcher path stays cheap in noisy IDE environments.
        """
        if self.follower:
            # Last line of defence. The watcher is not started on a follower
            # and stop_watcher cancels the timers it armed, but a write that
            # slipped through either would be the second writer on one store.
            return False
        abs_path = _norm(filepath)
        if not self.is_supported(abs_path):
            return False
        if not os.path.isfile(abs_path):
            # Modify followed by delete before flush: treat as remove.
            return self.remove_file(abs_path)

        # Cheap pre-check: if content hash hasn't changed, do nothing.
        new_sha = self._compute_file_hash(abs_path)
        if new_sha:
            cached_sha = self._file_hashes.get(abs_path, {}).get("sha256")
            if cached_sha == new_sha:
                # Genuine no-op save.
                return False

        with self._write_lock:
            removed = self._delete_file_chunks(abs_path)
            nodes = self._build_nodes_for_file(abs_path)
            if not nodes:
                log(f"[rag] no chunks produced from {abs_path}")
                return False

            before = self._collection_count()
            try:
                self.index.insert_nodes(nodes)
            except Exception as e:
                log(f"[rag] insert failed for {abs_path}: {e}")
                return False

            # Same silent-no-op guard as the batch path: a store with stale
            # write locks accepts the call and writes nothing. Recording the SHA
            # then would retire this file from every future pass.
            after = self._collection_count()
            if None not in (before, after) and after - before != len(nodes):
                log(f"[rag] insert of {Path(abs_path).name} did not land "
                    f"({after - before} of {len(nodes)} chunk(s) stored); not "
                    f"recording it as indexed. Run `lynx manager doctor`.")
                return False

            # Chunks changed: refresh only this file's BM25 entries (no full
            # collection re-read / re-tokenize). Rebuilt lazily on next search.
            self._bm25_apply_file_update(abs_path, nodes)

            # Persist the new SHA. If the hash function failed earlier we
            # leave the cache entry stale; the next change will recompute.
            # Reached only after insert_nodes succeeded (the except above
            # returns), so the entry never claims more than the index holds.
            if new_sha:
                self._record_file_hash(abs_path, new_sha, chunks=len(nodes))
                self._save_file_hashes()

        log(f"[rag] Re-indexed {Path(abs_path).name} ({removed} old chunks dropped, {len(nodes)} new)")
        return True

    def remove_file(self, filepath) -> bool:
        """Remove all chunks of a file from the index (on delete or move)."""
        if self.follower:
            return False
        abs_path = _norm(filepath)
        if Path(abs_path).suffix.lower() not in self.supported_extensions:
            return False
        with self._write_lock:
            removed = self._delete_file_chunks(abs_path)
            if removed:
                self._bm25_apply_file_removal(abs_path)
            # Drop the SHA entry regardless of whether chunks existed: the
            # source of truth is that this file is gone.
            had_hash = abs_path in self._file_hashes
            self._forget_file_hash(abs_path)
            if had_hash:
                self._save_file_hashes()
        if removed:
            log(f"[rag] Removed {removed} chunks for {Path(abs_path).name}")
        return removed > 0

    # ------------------------------------------------------------------
    # Config drift detection
    # ------------------------------------------------------------------

    def _build_config_snapshot(self) -> dict:
        """Snapshot of the config fields that, if changed, invalidate the index.

        See _DRIFT_SEVERITY for which fields are tracked. Fields like
        watcher.debounce_seconds, search.default_top_k, loading_timeout_seconds
        are intentionally omitted: changing them has no effect on the stored
        vectors.

        ignored_path_fragments is deliberately NOT in the snapshot. It now
        affects the indexed file set (via fs_scan), but the SHA partitioner
        self-heals on the next build — files that became ignored fall out of
        the candidate set and their chunks are dropped as "removed". Adding it
        here would make every pre-upgrade index report a false WARNING drift
        (the stored snapshot lacks the key, so `None != []` for the common
        no-fragments case) — pure noise for the majority of users.
        """
        return {
            "codebase_path": str(self.codebase_path.resolve()),
            "supported_extensions": sorted(self.supported_extensions),
            "embedding_model_name": self.embedding_model_name,
            "collection_name": self.collection_name,
            "chunker_version": CHUNKER_VERSION,
        }

    def check_config_drift(self):
        """Compare the live config against the snapshot in metadata.json.

        Returns a dict shaped like:
            {
              "severity": "critical" | "warning",
              "changes": {field: {"old": ..., "new": ...}, ...}
            }
        or None when no drift is detected (or when there is no stored
        snapshot yet, in which case we silently record the current one).
        """
        current = self._build_config_snapshot()
        stored = self.metadata.get("config_snapshot")

        if stored is None:
            # No baseline yet: record the current snapshot silently.
            # Happens on first run after upgrading from a version without
            # drift detection.
            self.metadata["config_snapshot"] = current
            self._save_metadata()
            return None

        if stored == current:
            return None

        changes = {}
        worst = DRIFT_WARNING
        for field, new_value in current.items():
            old_value = stored.get(field)
            if old_value != new_value:
                changes[field] = {"old": old_value, "new": new_value}
                if _DRIFT_SEVERITY.get(field) == DRIFT_CRITICAL:
                    worst = DRIFT_CRITICAL

        if not changes:
            return None

        return {"severity": worst, "changes": changes}

    def _check_and_log_drift(self):
        """Run drift check and emit a warning to stderr if anything changed."""
        drift = self.check_config_drift()
        if drift is None:
            return
        sev = drift["severity"].upper()
        log(f"[drift] {sev}: config has changed since the index was built.")
        for field, delta in drift["changes"].items():
            log(f"[drift]   {field}: {delta['old']!r} -> {delta['new']!r}")
        if drift["severity"] == DRIFT_CRITICAL:
            log("[drift] Search results may be wrong. Run a full rebuild "
                "(update_codebase_index with force=True) to clear the warning.")
        else:
            log("[drift] The index may be stale. Consider a rebuild.")

    def drift_status_text(self) -> str:
        """Human-readable drift summary for inclusion in get_rag_status()."""
        drift = self.check_config_drift()
        if drift is None:
            return "No config drift detected."
        sev = drift["severity"].upper()
        lines = [f"Config drift: {sev}"]
        for field, delta in drift["changes"].items():
            lines.append(f"  - {field}: {delta['old']!r} -> {delta['new']!r}")
        if drift["severity"] == DRIFT_CRITICAL:
            lines.append(
                "  Search results may be wrong. Run update_codebase_index(force=True) to rebuild."
            )
        else:
            lines.append("  The index may be stale. Consider a rebuild.")
        return "\n".join(lines)
