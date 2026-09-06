"""
Configuration loader for the lynx server (schema v2).

Reads `config.json` and exposes a typed `Config` dataclass used by
`server.py`, `cli.py`, and `source_manager.py`.

Resolution order (first match wins):
  1. Explicit `config_path` argument to `load_config()`.
  2. `RAG_CONFIG_PATH` environment variable.
  3. `./config.json` in the current working directory.

The CLI's `--config` flag plumbs straight to (1). The env var is the
recommended escape hatch when an MCP client launches the server from an
unpredictable working directory.

Schema v2 — top-level shape:

    {
      "config_version": 2,
      "storage_path": "./rag_storage",
      "embedding": {...},
      "search": {...},
      "sources": {
        "<source-name>": {
          "type": "codebase" | "webdoc" | "pdf" | ...,
          ... type-specific fields ...
        },
        ...
      }
    }

A v1 config (recognized by the presence of a top-level `codebase_path` field
and the absence of `sources`) is rejected with a pointer to
`lynx migrate-config`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


CURRENT_CONFIG_VERSION = 2
SUPPORTED_SOURCE_TYPES = ("codebase", "webdoc", "pdf")

# Source names are passed as the `source` argument of the MCP tools and used
# as ChromaDB collection names. Restrict to a conservative subset so they are
# always valid identifiers and easy to type. See README for the full rule.
SOURCE_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,39}$")


# ---------------------------------------------------------------------------
# Shared (server-wide) sub-configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class DeepSearchConfig:
    """Tunables for the `deep_search` fallback tool.

    See the docs section on hybrid retrieval and deep_search for the
    rationale on mode-specific thresholds.
    """
    min_results: int = 2
    score_thresholds: dict = field(default_factory=lambda: {
        "dense": 0.45,
        "hybrid": 0.012,
        "sparse": 3.0,
    })


@dataclass(frozen=True)
class RerankerConfig:
    """Cross-encoder reranker that runs after hybrid RRF fusion.

    Default disabled — utenti esistenti non vedono cambi. When enabled,
    after the standard pipeline produces N candidates we feed the top
    `top_n_before_rerank` to a small cross-encoder model, get content-
    aware relevance scores, and return the actually best `top_k`.

    See README "Reranking" section for cost/benefit.
    """
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many results to feed the reranker. Bigger = better recall but
    # slower (cross-encoder is O(n) in candidates). 30 is the sweet spot
    # for ms-marco-MiniLM-L-6-v2 on CPU.
    top_n_before_rerank: int = 30


@dataclass(frozen=True)
class SearchConfig:
    default_top_k: int = 5
    mode: str = "hybrid"            # "hybrid" | "dense" | "sparse"
    rrf_k: int = 60                 # standard RRF constant
    candidate_pool_size: int = 30   # per-retriever candidate pool size
    deep: DeepSearchConfig = field(default_factory=DeepSearchConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)


# ---------------------------------------------------------------------------
# Top-level Config object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    config_version: int
    storage_path: Path
    loading_timeout_seconds: int
    embedding: EmbeddingConfig
    search: SearchConfig
    # `sources` maps source-name -> raw dict from JSON (already validated /
    # path-resolved). Per-source typed dataclasses are intentionally NOT used
    # here because each source type has its own shape; the SourceManager and
    # backends know how to read their own fields.
    sources: Dict[str, dict] = field(default_factory=dict)
    # Where `lynx graph export` writes graph view files. None => the consumer
    # defaults to `<storage_path>/reports`. Settable in config.json
    # ("reports_path") and, later, from the LynxManager UI.
    reports_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Path / utility helpers
# ---------------------------------------------------------------------------


def _default_config_path() -> Path:
    return Path.cwd() / "config.json"


def _normalize_ignored_fragments(fragments: List[str]) -> tuple:
    """Convert forward-slash fragments to OS-native separators so they match
    actual filesystem paths on Windows."""
    sep = os.sep
    return tuple(frag.replace("/", sep) for frag in fragments)


def _resolve_path(value: str, base: Path) -> Path:
    """Resolve a path from the config file. Relative paths are anchored to
    the directory containing the config file, not the current working
    directory (which is unpredictable when launched by an MCP client)."""
    p = Path(value)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


# ---------------------------------------------------------------------------
# Per-type source validation
# ---------------------------------------------------------------------------


def _validate_codebase_source(name: str, raw: dict, base_dir: Path) -> dict:
    """Validate and normalize a `type=codebase` source entry."""
    path_raw = raw.get("path")
    if not path_raw:
        _config_error(f"source {name!r}: 'path' is required for type=codebase")
    path = _resolve_path(path_raw, base_dir)
    if not path.is_dir():
        _config_error(
            f"source {name!r}: path does not exist or is not a directory: {path}"
        )

    extensions = raw.get("supported_extensions") or [".py", ".md", ".txt"]
    extensions = frozenset(ext.lower() for ext in extensions)

    ignored = _normalize_ignored_fragments(raw.get("ignored_path_fragments") or [])

    watcher_raw = raw.get("watcher") or {}
    watcher = {
        "enabled": bool(watcher_raw.get("enabled", True)),
        "debounce_seconds": float(watcher_raw.get("debounce_seconds", 2.0)),
    }

    git_raw = raw.get("git_integration") or {}
    git_integration = {"enabled": bool(git_raw.get("enabled", True))}

    # Opt-in graph layer (call graph + import graph + community analysis).
    # Default disabled — existing configs keep working without changes; the
    # `graph_query` MCP tool is only registered when at least one source
    # sets this to true.
    graph_raw = raw.get("graph")
    if graph_raw is None:
        graph_raw = {}
    elif not isinstance(graph_raw, dict):
        _config_error(
            f"source {name!r}: 'graph' must be an object like "
            f"{{ \"enabled\": true }}, got {type(graph_raw).__name__} {graph_raw!r}"
        )
    graph = {"enabled": bool(graph_raw.get("enabled", False))}

    return {
        "type": "codebase",
        "path": path,
        "supported_extensions": extensions,
        "ignored_path_fragments": ignored,
        "watcher": watcher,
        "git_integration": git_integration,
        "graph": graph,
    }


def _validate_webdoc_source(name: str, raw: dict, base_dir: Path) -> dict:
    """Validate and normalize a `type=webdoc` source entry.

    Required: `url` (the crawl starting point).
    Everything else has a sensible default; see WebdocBackend.__init__ for
    the runtime semantics. We validate ranges here so misconfiguration
    surfaces at config-load time, not at fetch time.
    """
    url = raw.get("url")
    if not url or not isinstance(url, str):
        _config_error(f"source {name!r}: 'url' is required for type=webdoc")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        _config_error(
            f"source {name!r}: url must be http or https, got scheme {parsed.scheme!r}"
        )
    if not parsed.netloc:
        _config_error(f"source {name!r}: url is missing a host component: {url!r}")

    def _opt_int(key, default, min_val=0, max_val=None):
        val = int(raw.get(key, default))
        if val < min_val:
            _config_error(f"source {name!r}: {key} must be >= {min_val}, got {val}")
        if max_val is not None and val > max_val:
            _config_error(f"source {name!r}: {key} must be <= {max_val}, got {val}")
        return val

    max_depth = _opt_int("max_depth", 3, min_val=0, max_val=20)
    max_pages = _opt_int("max_pages", 500, min_val=1, max_val=50_000)

    delay = float(raw.get("request_delay_seconds", 0.5))
    if delay < 0:
        _config_error(f"source {name!r}: request_delay_seconds cannot be negative")

    include_patterns = list(raw.get("include_url_patterns") or [])
    exclude_patterns = list(raw.get("exclude_url_patterns") or [])
    if not all(isinstance(p, str) for p in include_patterns + exclude_patterns):
        _config_error(
            f"source {name!r}: include/exclude_url_patterns must be lists of strings"
        )

    # Opt-in JS rendering (headless Chromium via Playwright) for SPA /
    # client-side-rendered docs sites. Off by default: it needs the
    # `webdoc-js` extra (`lynx manager install webdoc-js`) and is an order
    # of magnitude slower per page than the plain HTTP fetch.
    render_js = bool(raw.get("render_js", False))
    render_wait_until = str(raw.get("render_wait_until", "networkidle")).lower()
    if render_wait_until not in ("load", "domcontentloaded", "networkidle"):
        _config_error(
            f"source {name!r}: render_wait_until must be one of "
            f"'load' | 'domcontentloaded' | 'networkidle', got {render_wait_until!r}"
        )
    render_timeout = float(raw.get("render_timeout_seconds", 30.0))
    if render_timeout <= 0:
        _config_error(
            f"source {name!r}: render_timeout_seconds must be positive, got {render_timeout}"
        )

    return {
        "type": "webdoc",
        "url": url,
        "max_depth": max_depth,
        "max_pages": max_pages,
        "same_origin_only": bool(raw.get("same_origin_only", True)),
        "include_url_patterns": include_patterns,
        "exclude_url_patterns": exclude_patterns,
        "request_delay_seconds": delay,
        "user_agent": raw.get("user_agent"),  # None → backend uses default
        "render_js": render_js,
        "render_wait_until": render_wait_until,
        "render_timeout_seconds": render_timeout,
    }


def _validate_pdf_source(name: str, raw: dict, base_dir: Path) -> dict:
    """Validate and normalize a `type=pdf` source entry.

    Required: `path` — directory containing the .pdf files to index.
    Optional sub-block `extractor` (defaults applied per-field below).
    Optional sub-block `watcher` — DISABLED by default for PDF sources
    because re-extraction is expensive (10-30s per medium PDF) and PDFs
    typically change much less often than source code.
    """
    path_raw = raw.get("path")
    if not path_raw:
        _config_error(f"source {name!r}: 'path' is required for type=pdf")
    path = _resolve_path(path_raw, base_dir)
    if not path.is_dir():
        _config_error(
            f"source {name!r}: path does not exist or is not a directory: {path}"
        )

    recursive = bool(raw.get("recursive", True))
    file_glob = str(raw.get("file_glob", "**/*.pdf"))

    extractor_raw = raw.get("extractor")
    if extractor_raw is None:
        extractor_raw = {}
    elif not isinstance(extractor_raw, dict):
        _config_error(
            f"source {name!r}: 'extractor' must be an object like "
            f"{{ \"backend\": \"auto\", \"max_file_mb\": 100, ... }}, got "
            f"{type(extractor_raw).__name__} {extractor_raw!r}"
        )

    backend = str(extractor_raw.get("backend", "auto"))
    if backend not in ("auto", "pypdf", "pymupdf"):
        _config_error(
            f"source {name!r}: extractor.backend must be one of "
            f"'auto' | 'pypdf' | 'pymupdf', got {backend!r}"
        )

    def _opt_int(key, default, min_val, max_val):
        val = int(extractor_raw.get(key, default))
        if val < min_val:
            _config_error(f"source {name!r}: extractor.{key} must be >= {min_val}, got {val}")
        if val > max_val:
            _config_error(f"source {name!r}: extractor.{key} must be <= {max_val}, got {val}")
        return val

    max_file_mb = _opt_int("max_file_mb", 100, 1, 2000)
    max_pages_per_file = _opt_int("max_pages_per_file", 5000, 1, 100_000)

    watcher_raw = raw.get("watcher") or {}
    if not isinstance(watcher_raw, dict):
        _config_error(
            f"source {name!r}: 'watcher' must be an object like "
            f"{{ \"enabled\": true }}, got {type(watcher_raw).__name__}"
        )
    watcher = {
        "enabled": bool(watcher_raw.get("enabled", False)),  # OFF by default for PDFs
        "debounce_seconds": float(watcher_raw.get("debounce_seconds", 5.0)),
    }

    # The graph layer is meaningful only for codebase sources (it needs
    # AST). If the user enables it on a PDF source we don't error out
    # (so they can flip type back and forth without re-editing the
    # block) but we warn at boot and silently ignore the flag here.
    graph_raw = raw.get("graph") or {}
    if isinstance(graph_raw, dict) and graph_raw.get("enabled"):
        print(
            f"[config] warning: source {name!r} sets graph.enabled=true but "
            f"graph layer is only supported for type=codebase. Ignoring.",
            file=sys.stderr,
        )

    return {
        "type": "pdf",
        "path": path,
        "recursive": recursive,
        "file_glob": file_glob,
        "extractor": {
            "backend": backend,
            "max_file_mb": max_file_mb,
            "max_pages_per_file": max_pages_per_file,
            "skip_password_protected": bool(extractor_raw.get("skip_password_protected", True)),
            "skip_if_text_empty": bool(extractor_raw.get("skip_if_text_empty", True)),
        },
        "watcher": watcher,
    }


_TYPE_VALIDATORS = {
    "codebase": _validate_codebase_source,
    "webdoc": _validate_webdoc_source,
    "pdf": _validate_pdf_source,
}


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def _config_error(msg: str) -> None:
    """Print a friendly error to stderr and exit. Used by the loader."""
    print(f"\n[config] ERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def _looks_like_v1(raw: dict) -> bool:
    """Heuristic: v1 had `codebase_path` at the top level and no `sources`."""
    return "codebase_path" in raw and "sources" not in raw


def resolve_config_path(explicit: "Path | str | None" = None) -> Path:
    """Apply the standard resolution chain and return the chosen path.

    Order: explicit argument > `RAG_CONFIG_PATH` env var > `./config.json`
    in the current working directory. The returned path is NOT guaranteed
    to exist — callers should `.is_file()` and react accordingly. Kept
    as a single source of truth so `lynx serve`, `lynx manager ui`,
    `lynx manager doctor`, and `lynx manager install --model` never
    disagree on which config they're looking at.
    """
    if explicit is not None:
        return Path(explicit)
    env_override = os.environ.get("RAG_CONFIG_PATH")
    if env_override:
        return Path(env_override)
    return _default_config_path()


def reports_dir(config: "Config") -> Path:
    """Where `export_graph` writes its self-contained graph views.

    `reports_path` when the config sets one, `<storage_path>/reports`
    otherwise. Single source of truth because three entry points now write
    there — the MCP tool, `lynx graph export`, and the web UI — and a view
    exported from one has to be findable by the others.
    """
    rp = getattr(config, "reports_path", None)
    return Path(rp) if rp else Path(config.storage_path) / "reports"


def load_config(config_path: Path | None = None) -> Config:
    """Load and validate the JSON config file.

    Exits with a clear error message if the config is missing, malformed,
    or in an unsupported schema version.
    """
    config_path = resolve_config_path(config_path)

    if not config_path.is_file():
        _config_error(
            f"config file not found at {config_path}\n"
            f"[config] Copy 'config.example.json' to '{config_path.name}' and edit "
            f"'sources' to point at your code / docs."
        )

    try:
        # utf-8-sig, not utf-8: PowerShell's `Out-File -Encoding utf8` and
        # several Windows editors prepend a BOM, which plain utf-8 decodes
        # into the string and json.loads then rejects with "Unexpected UTF-8
        # BOM" — a config that looks perfectly fine in an editor failing to
        # load with an error that points at column 1. Identical to utf-8 for
        # files without a BOM.
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        _config_error(f"invalid JSON in {config_path}: {e}")

    base_dir = config_path.resolve().parent

    # --- version handling --------------------------------------------------
    version = raw.get("config_version")
    if version is None:
        if _looks_like_v1(raw):
            _config_error(
                f"{config_path} looks like a v1 config (has 'codebase_path' at the "
                f"top level). Schema v2 reorganizes the file under a 'sources' "
                f"block.\n[config] Run:  lynx migrate-config "
                f"--input \"{config_path}\"\n[config] (or rewrite manually — see "
                f"config.example.json for the v2 template.)"
            )
        _config_error(
            f"'config_version' field is missing from {config_path}. "
            f"Expected: 'config_version': {CURRENT_CONFIG_VERSION}."
        )
    if not isinstance(version, int):
        _config_error(f"'config_version' must be an integer, got {version!r}")
    if version < CURRENT_CONFIG_VERSION:
        _config_error(
            f"'config_version': {version} is older than this package's schema "
            f"(v{CURRENT_CONFIG_VERSION}). Run "
            f"'lynx migrate-config --input \"{config_path}\"' "
            f"to upgrade."
        )
    if version > CURRENT_CONFIG_VERSION:
        _config_error(
            f"'config_version': {version} is newer than this package's schema "
            f"(v{CURRENT_CONFIG_VERSION}). Upgrade the package, or downgrade the "
            f"config to match."
        )

    # --- shared / top-level fields ----------------------------------------
    storage_path = _resolve_path(raw.get("storage_path", "./rag_storage"), base_dir)
    loading_timeout_seconds = int(raw.get("loading_timeout_seconds", 600))

    embedding_raw = raw.get("embedding") or {}
    embedding = EmbeddingConfig(
        model_name=str(embedding_raw.get("model_name", "BAAI/bge-small-en-v1.5")),
    )

    search_raw = raw.get("search") or {}
    mode = str(search_raw.get("mode", "hybrid")).lower()
    if mode not in ("hybrid", "dense", "sparse"):
        _config_error(
            f"search.mode must be one of 'hybrid' | 'dense' | 'sparse', "
            f"got {mode!r}"
        )

    deep_raw = search_raw.get("deep") or {}
    deep_thresholds_raw = deep_raw.get("score_thresholds") or {}
    deep_thresholds = {
        "dense": float(deep_thresholds_raw.get("dense", 0.45)),
        "hybrid": float(deep_thresholds_raw.get("hybrid", 0.012)),
        "sparse": float(deep_thresholds_raw.get("sparse", 3.0)),
    }
    deep = DeepSearchConfig(
        min_results=int(deep_raw.get("min_results", 2)),
        score_thresholds=deep_thresholds,
    )

    # Optional reranker block — disabled by default for backward compat.
    reranker_raw = search_raw.get("reranker")
    if reranker_raw is None:
        reranker_raw = {}
    elif not isinstance(reranker_raw, dict):
        _config_error(
            f"search.reranker must be an object like "
            f"{{ \"enabled\": true, \"model_name\": \"...\" }}, got "
            f"{type(reranker_raw).__name__} {reranker_raw!r}"
        )
    rerank_top_n = int(reranker_raw.get("top_n_before_rerank", 30))
    if rerank_top_n < 1 or rerank_top_n > 200:
        _config_error(
            f"search.reranker.top_n_before_rerank must be between 1 and 200, "
            f"got {rerank_top_n}"
        )
    reranker = RerankerConfig(
        enabled=bool(reranker_raw.get("enabled", False)),
        model_name=str(reranker_raw.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        top_n_before_rerank=rerank_top_n,
    )
    search = SearchConfig(
        default_top_k=int(search_raw.get("default_top_k", 5)),
        mode=mode,
        rrf_k=int(search_raw.get("rrf_k", 60)),
        candidate_pool_size=int(search_raw.get("candidate_pool_size", 30)),
        deep=deep,
        reranker=reranker,
    )

    # --- sources -----------------------------------------------------------
    # An empty `sources: {}` is allowed: that's the state right after
    # `lynx manager init`, before the user has added any source via the UI.
    # `lynx serve` will run with zero MCP tools registered, and the manager
    # UI will surface an empty-state prompt to add one.
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, dict):
        _config_error(
            "'sources' must be an object mapping source name -> "
            "source config. See config.example.json for the shape."
        )

    sources: Dict[str, dict] = {}
    for name, entry in sources_raw.items():
        if not SOURCE_NAME_RE.match(name):
            _config_error(
                f"source name {name!r} is invalid. Allowed pattern: "
                f"letter followed by letters / digits / underscore "
                f"(max 40 chars). Example valid names: 'myproject', "
                f"'unityDoc', 'avalonia_docs'."
            )
        if not isinstance(entry, dict):
            _config_error(f"source {name!r} must be a JSON object")
        type_name = entry.get("type")
        if type_name not in SUPPORTED_SOURCE_TYPES:
            _config_error(
                f"source {name!r}: type {type_name!r} is not supported. "
                f"Supported: {list(SUPPORTED_SOURCE_TYPES)}"
            )
        validator = _TYPE_VALIDATORS[type_name]
        sources[name] = validator(name, entry, base_dir)

    raw_reports = raw.get("reports_path")
    reports_path = _resolve_path(raw_reports, base_dir) if raw_reports else None

    return Config(
        config_version=version,
        storage_path=storage_path,
        loading_timeout_seconds=loading_timeout_seconds,
        embedding=embedding,
        search=search,
        sources=sources,
        reports_path=reports_path,
    )


# ---------------------------------------------------------------------------
# HuggingFace offline-mode decision
# ---------------------------------------------------------------------------


def _hf_cache_dir() -> Path:
    """Resolve the HuggingFace hub cache directory (mirrors hf defaults)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


# Where a model's ONNX graph may sit inside its folder, in order of preference.
# Kept here (stdlib only) so `model_files.py`, the doctor and this probe agree.
ONNX_GRAPH_CANDIDATES = ("onnx/model.onnx", "model.onnx")


def snapshot_has_runtime_files(model_dir: Path) -> bool:
    """True when `model_dir` holds what the ONNX runtime loads: a graph and
    `tokenizer.json`.

    A snapshot fetched by an install older than 1.9 has the PyTorch weights
    only. It reports False here, so the next run stays online long enough to
    fetch the graph into the same snapshot (see `configure_hf_offline`)."""
    try:
        if not (model_dir / "tokenizer.json").is_file():
            return False
        return any((model_dir / rel).is_file() for rel in ONNX_GRAPH_CANDIDATES)
    except OSError:
        return False


def _hf_model_cached(model_name: str) -> bool:
    """True if a usable snapshot of `model_name` exists: either a local
    folder, or an entry in the HF cache with the ONNX graph and tokenizer.

    Pure-stdlib directory probe (no huggingface_hub import — see
    `configure_hf_offline` for why importing it here would defeat the
    purpose)."""
    local = Path(model_name)
    if local.is_dir():
        return snapshot_has_runtime_files(local)
    safe = model_name.replace("/", "--")
    snapshots = _hf_cache_dir() / f"models--{safe}" / "snapshots"
    try:
        if not snapshots.is_dir():
            return False
        return any(
            entry.is_dir() and snapshot_has_runtime_files(entry)
            for entry in snapshots.iterdir()
        )
    except OSError:
        return False


def sanitize_tls_keylog_env() -> None:
    """Remove ``SSLKEYLOGFILE`` before anything imports ``ssl``.

    Setting ``SSLKEYLOGFILE`` makes Python's ``ssl`` open that path through
    OpenSSL's file BIO the first time a default SSL context is built. On the
    uv / python-build-standalone interpreter we ship with — which does NOT
    provide ``OPENSSL_Applink`` — that file-BIO call ABORTS the whole process::

        OPENSSL_Uplink(...): no OPENSSL_Applink

    no traceback, mid-startup. **Any** value triggers it: a normal writable path
    crashes exactly like the ``\\\\.\\aswMonFltProxy\\...`` device path that
    HTTPS-inspecting antivirus (Avast/AVG) injects into every process. So a
    "keep legit-looking paths" heuristic is unsafe — verified: a normal path
    aborts the interpreter too.

    Lynx never needs a TLS key-log: it runs offline and makes no outbound TLS in
    normal operation (the only HTTPS is the first-run model download, which works
    fine without key logging). So we unconditionally drop the variable for this
    process. A developer who genuinely wants TLS key logging on an
    applink-capable interpreter can keep it with ``LYNX_KEEP_SSLKEYLOGFILE=1``.

    Stdlib-only; must run before the first ``import ssl`` (transitively via
    requests / huggingface_hub / chromadb)."""
    value = os.environ.get("SSLKEYLOGFILE")
    if not value:
        return
    if os.environ.get("LYNX_KEEP_SSLKEYLOGFILE"):
        return  # explicit opt-in — caller accepts the applink crash risk
    os.environ.pop("SSLKEYLOGFILE", None)
    print(
        f"[config] removed SSLKEYLOGFILE={value!r} from this process: it crashes "
        f"the bundled interpreter on first TLS use (no OPENSSL_Applink) and Lynx "
        f"never needs TLS key logging. Set LYNX_KEEP_SSLKEYLOGFILE=1 to keep it.",
        file=sys.stderr,
    )


def configure_hf_offline(config) -> None:
    """Set the HF offline env flags only when every required model is cached.

    Lynx guarantees "no data egress during search" by running HuggingFace
    in offline mode. But hard-coding HF_HUB_OFFLINE=1 broke the FIRST run:
    with an empty cache the embedding model can never be downloaded and
    `lynx serve` dies before answering a single query. So the rule is:

      - the user already set HF_HUB_OFFLINE → always respected, no override;
      - every model the config needs is in the local cache → go offline;
      - something is missing → stay online for THIS run so it can be
        fetched once; every later run flips back to offline.

    MUST be called before the first import of huggingface_hub/transformers
    (both freeze the offline flags at import time) — i.e. right after
    `load_config()` and before importing `source_manager`. That's why this
    lives in config.py, which is stdlib-only.
    """
    if os.environ.get("HF_HUB_OFFLINE") is not None:
        return  # explicit user choice wins

    models = [config.embedding.model_name]
    if config.search.reranker.enabled:
        models.append(config.search.reranker.model_name)

    missing = [m for m in models if not _hf_model_cached(m)]
    if not missing:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        print(
            f"[config] model(s) not in local cache yet: {', '.join(missing)} — "
            f"allowing network for this run to download them once. "
            f"Subsequent runs are fully offline.",
            file=sys.stderr,
        )
