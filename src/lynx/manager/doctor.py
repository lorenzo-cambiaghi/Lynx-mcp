"""Diagnostic checks for Lynx — `lynx manager doctor`.

Goal: one command tells you "is this install healthy?". We check the
things that silently break searches:

  - Python version (Lynx needs ≥ 3.10)
  - HuggingFace model cache (the embedding model downloads automatically
    on the first run; offline mode kicks in once it's cached)
  - Config file (exists, valid JSON, validates with the loader)
  - Per-source state (path exists, ChromaDB readable, drift status,
    git_integration optionally validated)
  - Optional extras (which ones are installed vs available)
  - Disk space in storage_path

Each check returns a `CheckResult` with a status flag and optional
details. The CLI entry point aggregates them, prints a colored summary,
and exits with code 0/1/2 so this can be wired into CI.

Pure functions — no global state, no side effects beyond reading the
filesystem and the active Python process metadata.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .ansi import success, warn, error, bold, dim, heading, bullet


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


# Status levels. Order matters: ERROR > WARN > OK. The CLI exit code
# is derived from the worst status across all checks.
STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_ERROR = "error"


@dataclass
class CheckResult:
    name: str
    status: str  # one of STATUS_OK / STATUS_WARN / STATUS_ERROR
    summary: str
    details: list = field(default_factory=list)  # list[str], bullet lines

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "details": list(self.details),
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version() -> CheckResult:
    """Lynx requires Python 3.10+ for `str | None` syntax in many places."""
    v = sys.version_info
    if v >= (3, 10):
        return CheckResult(
            name="Python version",
            status=STATUS_OK,
            summary=f"Python {v.major}.{v.minor}.{v.micro}",
        )
    return CheckResult(
        name="Python version",
        status=STATUS_ERROR,
        summary=f"Python {v.major}.{v.minor}.{v.micro} — Lynx requires >= 3.10",
        details=["Install Python 3.10 or newer (3.12 recommended)."],
    )


def _hf_cache_dir_for(model_name: str) -> Path:
    """Return the expected HuggingFace cache directory for a model.

    HF Hub stores models under
    `<hub-cache>/models--{owner}--{repo}/snapshots/<sha>/` where
    `{owner}--{repo}` comes from replacing `/` with `--` in the model
    name. The hub cache root honors `HF_HUB_CACHE` / `HF_HOME` (resolved
    via `config._hf_cache_dir`) so we probe the SAME directory the runtime
    uses, not a hardcoded `~/.cache` that ignores those overrides.
    """
    from ..config import _hf_cache_dir
    safe = model_name.replace("/", "--")
    return _hf_cache_dir() / f"models--{safe}"


def check_hf_model_cache(model_name: str, label: str = "embedding model") -> CheckResult:
    """Verify that a HuggingFace model is present in the local cache.

    We don't try to download here — that's `lynx manager install --model`'s
    job. We just report whether the cache is populated so the user
    knows whether `lynx serve` will hit network on first query.
    """
    cache_dir = _hf_cache_dir_for(model_name)
    if not cache_dir.exists():
        return CheckResult(
            name=f"HF cache: {label}",
            status=STATUS_WARN,
            summary=f"{model_name} not in HF cache",
            details=[
                f"Expected at {cache_dir}",
                "Run `lynx manager install --model` to fetch it now, or "
                "let the first `lynx serve` download it automatically "
                "(requires network at that moment; later runs are offline).",
                "Behind a firewall / can't reach huggingface.co? Set "
                "HF_ENDPOINT to a reachable mirror, or import a shared model "
                "archive with `lynx manager install --from-archive <path|url>`.",
            ],
        )
    # Look for at least one snapshot directory with the key files.
    snapshots = cache_dir / "snapshots"
    if not snapshots.exists() or not any(snapshots.iterdir()):
        return CheckResult(
            name=f"HF cache: {label}",
            status=STATUS_WARN,
            summary=f"{model_name} cache directory exists but is empty",
            details=[
                f"At {cache_dir}",
                "Re-download via `lynx manager install --model`.",
            ],
        )
    # Any snapshot with the files the ONNX runtime opens makes the model
    # usable. Otherwise report what the first one lacks: an install older
    # than 1.9 left only the PyTorch weights here, which are no longer read.
    from ..config import ONNX_GRAPH_CANDIDATES, snapshot_has_runtime_files
    snapshot_dirs = [s for s in snapshots.iterdir() if s.is_dir()]
    if not any(snapshot_has_runtime_files(s) for s in snapshot_dirs):
        first_snapshot = snapshot_dirs[0] if snapshot_dirs else snapshots
        required = ("config.json", "tokenizer.json")
        missing = [f for f in required if not (first_snapshot / f).exists()]
        if not any((first_snapshot / rel).is_file() for rel in ONNX_GRAPH_CANDIDATES):
            missing.append("onnx/model.onnx")
        return CheckResult(
            name=f"HF cache: {label}",
            status=STATUS_WARN,
            summary=f"{model_name} cache incomplete (missing: {', '.join(missing)})",
            details=[
                "Re-fetch with `lynx manager install --model`: it adds the "
                "missing files to this snapshot. Lynx loads models through "
                "ONNX Runtime; installs before 1.9 downloaded only the "
                "PyTorch weights.",
            ],
        )
    return CheckResult(
        name=f"HF cache: {label}",
        status=STATUS_OK,
        summary=f"{model_name} present in HF cache",
    )


def check_hf_endpoint() -> CheckResult:
    """Informational: report the HF cache location and whether a mirror
    (HF_ENDPOINT) is configured. Always OK — this is context, not a problem.

    Useful on restricted networks where huggingface.co is blocked: the user
    can point HF_ENDPOINT at a reachable mirror and HF_HOME/HF_HUB_CACHE at a
    shared cache, and this check confirms what the runtime will actually use.
    """
    from ..config import _hf_cache_dir
    endpoint = os.environ.get("HF_ENDPOINT")
    details = [f"Hub cache: {_hf_cache_dir()}"]
    if endpoint:
        details.append(f"HF_ENDPOINT: {endpoint} (downloads use this mirror)")
        summary = f"using mirror {endpoint}"
    else:
        details.append(
            "HF_ENDPOINT not set — downloads go to huggingface.co. On a "
            "restricted network, set it to a reachable mirror, or import a "
            "shared archive via `lynx manager install --from-archive`."
        )
        summary = "huggingface.co (default; reachability not checked)"
    return CheckResult(
        name="HF endpoint",
        status=STATUS_OK,
        summary=summary,
        details=details,
    )


def check_config_file(config_path: Optional[Path]) -> "tuple[CheckResult, Optional[object]]":
    """Validate the config file. Returns the check result AND the loaded
    config object (or None on failure) so callers can chain into per-
    source checks without re-parsing."""
    if config_path is None:
        # Fall back to the default lookup order used by `load_config`.
        for candidate in (Path("config.json"), Path("./config.json")):
            if candidate.exists():
                config_path = candidate
                break
    if config_path is None or not config_path.exists():
        return (
            CheckResult(
                name="Config file",
                status=STATUS_ERROR,
                summary="config.json not found",
                details=[
                    "Pass --config PATH, set RAG_CONFIG_PATH, or run "
                    "`lynx manager init` to scaffold one.",
                ],
            ),
            None,
        )
    try:
        from ..config import load_config
        cfg = load_config(config_path)
    except SystemExit as e:
        # load_config calls sys.exit on validation errors — capture it.
        return (
            CheckResult(
                name="Config file",
                status=STATUS_ERROR,
                summary=f"{config_path} failed validation",
                details=[f"Exit code: {e.code}. Re-run the failing command "
                         f"for the full error message."],
            ),
            None,
        )
    except Exception as e:
        return (
            CheckResult(
                name="Config file",
                status=STATUS_ERROR,
                summary=f"{config_path} could not be loaded",
                details=[f"{type(e).__name__}: {e}"],
            ),
            None,
        )
    return (
        CheckResult(
            name="Config file",
            status=STATUS_OK,
            summary=f"{config_path}: valid, {len(cfg.sources)} source(s)",
        ),
        cfg,
    )


def check_source(name: str, src_cfg: dict, storage_path: Path) -> CheckResult:
    """Check one source: path exists, ChromaDB readable, optional drift.

    Drift detection requires bootstrapping the source backend which
    loads the HF model — too expensive for `doctor`. We skip drift here
    and surface only the path / Chroma-presence problems. The deeper
    `get_rag_status` MCP tool can be used for drift inspection at
    runtime.
    """
    src_type = src_cfg.get("type", "?")
    details: list = []

    # 1. Source path / URL existence
    if src_type in ("codebase", "pdf"):
        path = Path(src_cfg.get("path", ""))
        if not path.exists():
            return CheckResult(
                name=f"Source {name!r}",
                status=STATUS_ERROR,
                summary=f"path {path} does not exist",
            )
        if not path.is_dir():
            return CheckResult(
                name=f"Source {name!r}",
                status=STATUS_ERROR,
                summary=f"path {path} is not a directory",
            )
        details.append(f"Source path: {path}")
    elif src_type == "webdoc":
        url = src_cfg.get("url", "")
        if not url:
            return CheckResult(
                name=f"Source {name!r}",
                status=STATUS_ERROR,
                summary="webdoc has no URL",
            )
        details.append(f"URL: {url}")

    # 2. ChromaDB directory presence
    source_storage = storage_path / name
    chroma_db = source_storage / "chroma.sqlite3"
    if source_storage.exists():
        if chroma_db.exists():
            size_mb = chroma_db.stat().st_size / (1024 * 1024)
            details.append(f"ChromaDB: {size_mb:.1f} MB at {chroma_db}")

            # 2b. Integrity probe + WAL verdict. The WAL fingerprint (queued
            # rows older than 10 min with a segment behind them) is NOT proof
            # of the chromadb deadlock: a clean build whose size is not a
            # multiple of the HNSW sync threshold ends in exactly that state
            # and sits there while idle — this check called two healthy,
            # searchable indexes "wedged" for 16 days. What separates the two
            # is the out-of-process probe: a healthy store answers it in
            # about a second, a wedged one never does. So the probe decides
            # and the fingerprint only explains a timeout. The probe opens
            # chroma in a child (no model load) and classifies a store held
            # by another process as `in_use` up front, so it is cheap here.
            from ..integrity import inspect_wal, check_index
            try:
                wal = inspect_wal(source_storage)
            except Exception:
                wal = None
            probe = check_index(source_storage, name, timeout=20)
            verdict = probe.get("status")
            if verdict == "in_use":
                details.append(
                    "Index: open in another running Lynx process (usually "
                    "`lynx serve`) — healthy; verdicts that need the store "
                    "are skipped"
                )
                if wal and wal["pending_ops"]:
                    details.append(
                        f"WAL: {wal['pending_ops']} queued write(s), index in "
                        f"use, skipping the wedge check"
                    )
            elif verdict == "corrupt":
                return CheckResult(
                    name=f"Source {name!r}",
                    status=STATUS_ERROR,
                    summary="index is unreadable or crashes on open",
                    details=details + [
                        probe.get("detail", ""),
                        f"Fix: `lynx reset --source {name}` wipes it and "
                        f"rebuilds from the files on disk.",
                    ],
                )
            elif verdict == "unverified":
                if wal and wal["wedged"]:
                    mins = int(wal["oldest_pending_s"] // 60)
                    return CheckResult(
                        name=f"Source {name!r}",
                        status=STATUS_ERROR,
                        summary=(f"index WAL is wedged — {wal['unapplied_ops']} "
                                 f"write(s) stuck for {mins} min and the index "
                                 f"never answered the probe"),
                        details=details + [
                            "A process was likely killed mid-write; ChromaDB "
                            "deadlocks on this state (every open hangs forever, "
                            "surfacing as probe timeouts / 'index not verified').",
                            f"Files awaiting re-index: {len(wal['affected_files'])}",
                            f"Fix: stop every Lynx process, then run "
                            f"`lynx manager doctor --heal-wal {name}` — it purges "
                            f"the stuck writes and queues the affected files for "
                            f"re-indexing. `lynx reset --source {name}` also "
                            f"works but rebuilds the whole index.",
                        ],
                    )
                return CheckResult(
                    name=f"Source {name!r}",
                    status=STATUS_WARN,
                    summary="index could not be verified (probe timed out)",
                    details=details + [
                        probe.get("detail", ""),
                        "Close other Lynx processes and rerun; a large index on "
                        "a slow or antivirus-scanned disk can also take longer "
                        "than the probe budget.",
                    ],
                )
            else:
                if verdict == "ok" and probe.get("count") is not None:
                    details.append(
                        f"Index: {probe['count']} chunk(s), answered the probe")
                elif verdict == "empty":
                    details.append("Index: empty (0 chunks)")
                if wal and wal["pending_ops"]:
                    details.append(
                        f"WAL: {wal['pending_ops']} queued write(s), "
                        f"{wal['unapplied_ops']} awaiting replay by a segment "
                        f"(normal — the index answered the probe)"
                    )
            if wal and wal["discarding_writes"] and verdict != "in_use":
                return CheckResult(
                    name=f"Source {name!r}",
                    status=STATUS_ERROR,
                    summary="the index silently discards every write",
                    details=details + [
                        f"The write queue numbers below the segments' watermark "
                        f"({wal['watermark']}). Queue ids restart at 1 "
                        f"(INTEGER PRIMARY KEY, no AUTOINCREMENT), so every new "
                        f"write is numbered below what the segments already "
                        f"consumed and is dropped as a duplicate.",
                        "Nothing raises: indexing logs success, count() stays "
                        "healthy, and the index never moves again.",
                        f"Fix: stop every Lynx process, then run "
                        f"`lynx manager doctor --heal-wal {name}` — it anchors "
                        f"the queue above the watermark so writes land again.",
                    ],
                )

            # 2c. Coverage drift. The SHA cache is what decides which files get
            # work: an entry written for a file whose chunks never landed makes
            # that file permanently invisible to search — silently, and with a
            # perfectly healthy count(). Cheap read-only comparison.
            try:
                from ..integrity import inspect_coverage
                coverage = inspect_coverage(source_storage)
            except Exception:
                coverage = None
            if coverage and coverage["drifted"]:
                phantom = len(coverage["phantom_files"])
                unvectorized = coverage["unvectorized"]
                parts = []
                if phantom:
                    parts.append(f"{phantom} file(s) cached as indexed are "
                                 f"missing from the index")
                if unvectorized:
                    parts.append(f"{unvectorized} chunk(s) have no vector")
                findings = details + [
                    f"Indexed files: {coverage['indexed_files']} of "
                    f"{coverage['cached_files']} in the hash cache.",
                ]
                if phantom:
                    findings.append(
                        "Those files are skipped as 'unchanged' on every pass, so "
                        "they will never be retried: searches simply cannot find "
                        "them.")
                if unvectorized:
                    findings.append(
                        f"The metadata hands out {unvectorized} id(s) the vector "
                        f"index can't resolve (from "
                        f"{len(coverage['unvectorized_files'])} file(s)): every "
                        f"search fails with 'Error querying knn' / 'Error finding "
                        f"id' even though count() looks healthy.")
                findings.append(
                    f"Fix: stop every Lynx process, then run "
                    f"`lynx manager doctor --heal-coverage {name}` — it drops "
                    f"just those entries so the next update re-indexes them.")
                return CheckResult(
                    name=f"Source {name!r}",
                    status=STATUS_ERROR,
                    summary=" and ".join(parts),
                    details=findings,
                )
        else:
            details.append(f"ChromaDB: not yet built (run `lynx build --source {name}`)")
    else:
        details.append("Storage dir not created yet — run `lynx build` to bootstrap.")

    # 3. Watcher status (for codebase + pdf)
    if src_type in ("codebase", "pdf"):
        watcher_enabled = (src_cfg.get("watcher") or {}).get("enabled")
        if watcher_enabled is False:
            details.append("Watcher: disabled (manual refresh only)")
        elif watcher_enabled:
            details.append("Watcher: enabled (live updates on file changes)")

    return CheckResult(
        name=f"Source {name!r}",
        status=STATUS_OK,
        summary=f"type={src_type}, ok",
        details=details,
    )


def check_optional_extras() -> CheckResult:
    """Report which optional extras are installed.

    We probe each known optional extra by trying to import its key
    package. Result is informational (always OK) — missing extras are
    not errors, just options the user might want.
    """
    extras_status: list = []
    # (extra_name, package_to_probe, what_it_enables)
    known = [
        ("pdf-fast", "pymupdf", "faster PDF extraction on multi-column layouts"),
    ]
    for extra, package, what_for in known:
        try:
            __import__(package)
            extras_status.append(f"[installed] {extra} — {what_for}")
        except ImportError:
            extras_status.append(
                f"[not installed] {extra} — install with "
                f"`lynx manager install {extra}` ({what_for})"
            )
    return CheckResult(
        name="Optional extras",
        status=STATUS_OK,
        summary=f"{len(known)} known extras",
        details=extras_status,
    )


def check_disk_space(storage_path: Path) -> CheckResult:
    """Warn when the storage path's filesystem is running low."""
    target = storage_path if storage_path.exists() else storage_path.parent
    if not target.exists():
        target = Path(".").resolve()
    try:
        usage = shutil.disk_usage(target)
    except OSError as e:
        return CheckResult(
            name="Disk space",
            status=STATUS_WARN,
            summary=f"Couldn't determine free space at {target}: {e}",
        )
    free_mb = usage.free / (1024 * 1024)
    if free_mb < 100:
        return CheckResult(
            name="Disk space",
            status=STATUS_ERROR,
            summary=f"Only {free_mb:.0f} MB free at {target} (< 100 MB minimum)",
            details=["Free up disk space or move `storage_path` to a larger volume."],
        )
    if free_mb < 1000:
        return CheckResult(
            name="Disk space",
            status=STATUS_WARN,
            summary=f"{free_mb:.0f} MB free at {target} (under 1 GB — tight for indexing)",
        )
    free_gb = free_mb / 1024
    return CheckResult(
        name="Disk space",
        status=STATUS_OK,
        summary=f"{free_gb:.1f} GB free at {target}",
    )


# ---------------------------------------------------------------------------
# Aggregation + CLI entry point
# ---------------------------------------------------------------------------


def run_all_checks(config_path: Optional[Path]) -> list:
    """Run every diagnostic and return the list of CheckResult."""
    results: list = []

    # Python first — if this fails everything else likely fails too,
    # but we keep going so the user sees the full picture.
    results.append(check_python_version())

    # Config — short-circuit per-source checks if invalid.
    cfg_result, cfg = check_config_file(config_path)
    results.append(cfg_result)

    # HF model cache — use config value when available, fall back to default.
    if cfg is not None:
        embedding_model = cfg.embedding.model_name
    else:
        embedding_model = "BAAI/bge-small-en-v1.5"
    results.append(check_hf_model_cache(embedding_model, label="embedding"))

    # Where models come from / go (mirror + cache location).
    results.append(check_hf_endpoint())

    # Reranker model only if config says it's enabled.
    if cfg is not None and cfg.search.reranker.enabled:
        results.append(
            check_hf_model_cache(cfg.search.reranker.model_name, label="reranker")
        )

    # Per-source.
    if cfg is not None:
        storage_path = Path(cfg.storage_path)
        for name, src_cfg in cfg.sources.items():
            results.append(check_source(name, src_cfg, storage_path))
        results.append(check_disk_space(storage_path))

    # Extras (always — informational).
    results.append(check_optional_extras())

    return results


def _worst_status(results: list) -> str:
    """Aggregate the worst status across all checks."""
    has_error = any(r.status == STATUS_ERROR for r in results)
    if has_error:
        return STATUS_ERROR
    has_warn = any(r.status == STATUS_WARN for r in results)
    if has_warn:
        return STATUS_WARN
    return STATUS_OK


def _format_human(results: list) -> str:
    """Render the results as colored, human-readable text."""
    out: list = [heading("Lynx diagnostic"), ""]
    for r in results:
        if r.status == STATUS_OK:
            line = success(f"{bold(r.name)}: {r.summary}")
        elif r.status == STATUS_WARN:
            line = warn(f"{bold(r.name)}: {r.summary}")
        else:
            line = error(f"{bold(r.name)}: {r.summary}")
        out.append(line)
        for detail in r.details:
            out.append(bullet(detail))
    worst = _worst_status(results)
    out.append("")
    if worst == STATUS_OK:
        out.append(success(bold("All checks passed.")))
    elif worst == STATUS_WARN:
        out.append(warn(bold("Some checks have warnings — review above.")))
    else:
        out.append(error(bold("Some checks failed — fix the errors above.")))
    return "\n".join(out)


def run_doctor(args) -> int:
    """CLI entry point. Returns the exit code (0 ok, 1 warn, 2 error)."""
    # Same resolution chain as `lynx serve` so `lynx manager doctor`
    # without --config picks up ./config.json or $RAG_CONFIG_PATH.
    from ..config import resolve_config_path
    resolved = resolve_config_path(getattr(args, "config", None))
    config_path = resolved if resolved.is_file() else None

    if getattr(args, "heal_wal", None):
        return _run_heal_wal(args.heal_wal, config_path)

    if getattr(args, "heal_coverage", None):
        return _run_heal_coverage(args.heal_coverage, config_path)

    results = run_all_checks(config_path)

    if getattr(args, "json", False):
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print(_format_human(results))

    worst = _worst_status(results)
    return {STATUS_OK: 0, STATUS_WARN: 1, STATUS_ERROR: 2}[worst]


def _run_heal_wal(name: str, config_path: Optional[Path]) -> int:
    """`lynx manager doctor --heal-wal SOURCE` — surgically unwedge a store
    whose WAL was left half-written by a killed process (the state the
    per-source doctor check flags as 'index WAL is wedged'). Purges the
    stuck writes, queues the affected files for re-indexing, then proves
    the result with the out-of-process integrity probe."""
    if config_path is None:
        print("[doctor] no config file found — pass --config PATH", file=sys.stderr)
        return 2
    from ..config import load_config
    cfg = load_config(config_path)
    if name not in cfg.sources:
        print(f"[doctor] unknown source {name!r}. "
              f"Available: {list(cfg.sources)}", file=sys.stderr)
        return 2

    storage_dir = Path(cfg.storage_path) / name
    from ..integrity import check_index, heal_wal, inspect_wal

    wal = inspect_wal(storage_dir)
    if wal is None:
        print(f"[doctor] source {name!r} has no index yet — nothing to heal.")
        return 0
    if not wal["wedged"] and not wal["discarding_writes"]:
        queued = (f" ({wal['pending_ops']} queued write(s) are the tail a "
                  f"segment replays on its next open — normal)"
                  if wal["pending_ops"] else "")
        print(f"[doctor] source {name!r}: WAL is clean{queued} — nothing to heal.")
        return 0
    if wal["discarding_writes"]:
        print(f"[doctor] segments parked at seq {wal['watermark']} with an "
              f"empty queue: every write is being discarded as already-applied.")
    else:
        # The fingerprint alone is not a wedge (see inspect_wal): a healthy
        # store answers the probe in about a second, a wedged one never
        # does. Purging the queue of a healthy store would throw away the
        # vectors it was about to replay, so confirm before cutting.
        print("[doctor] fingerprint matches a wedge; confirming with the "
              "out-of-process probe...")
        probe = check_index(storage_dir, name, timeout=20)
        if probe["status"] in ("ok", "empty"):
            count = probe.get("count")
            print(f"[doctor] source {name!r}: the index answers the probe"
                  + (f" ({count} chunks)" if count else "")
                  + f"; the {wal['unapplied_ops']} queued write(s) are the tail "
                  f"the vector segment replays on open, not a wedge — nothing "
                  f"to heal.")
            return 0
        if probe["status"] == "in_use":
            print(f"[doctor] can't heal {name!r}: {probe['detail']}",
                  file=sys.stderr)
            return 2

    try:
        outcome = heal_wal(storage_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[doctor] can't heal {name!r}: {e}", file=sys.stderr)
        return 2

    print(f"[doctor] purged {outcome['purged_ops']} stuck write(s) and "
          f"{outcome['purged_locks']} stale write lock(s).")
    if outcome.get("forwarded_segments"):
        print(f"[doctor] fast-forwarded {outcome['forwarded_segments']} lagging "
              f"segment(s) that were waiting on purged rows; the chunks whose "
              f"vectors were lost are re-queued by --heal-coverage.")
    if outcome.get("anchored_seq"):
        print(f"[doctor] anchored the write queue at seq "
              f"{outcome['anchored_seq']} so new writes land above the "
              f"segments' watermark and are accepted again.")
    if outcome["reindex_files"]:
        print(f"[doctor] {len(outcome['reindex_files'])} file(s) queued for "
              f"re-indexing on the next build / watcher pass:")
        for f in outcome["reindex_files"]:
            print(f"  - {f}")

    print("[doctor] verifying the healed index (out-of-process probe)...")
    result = check_index(storage_dir, name)
    if result["status"] in ("ok", "empty"):
        count = result.get("count")
        print(f"[doctor] index verified healthy"
              + (f" ({count} chunks)." if count else "."))
        return 0
    print(f"[doctor] index still {result['status']} after the heal "
          f"({result.get('detail', '')}). Fall back to "
          f"`lynx reset --source {name}`.", file=sys.stderr)
    return 2


def _run_heal_coverage(name: str, config_path: Optional[Path]) -> int:
    """`lynx manager doctor --heal-coverage SOURCE` — drop the SHA-cache entries
    of files that are cached as indexed but absent from the index, so the next
    build re-indexes exactly those. The alternative for this state used to be a
    full rebuild, because nothing detected it in the first place."""
    if config_path is None:
        print("[doctor] no config file found — pass --config PATH", file=sys.stderr)
        return 2
    from ..config import load_config
    cfg = load_config(config_path)
    if name not in cfg.sources:
        print(f"[doctor] unknown source {name!r}. "
              f"Available: {list(cfg.sources)}", file=sys.stderr)
        return 2

    storage_dir = Path(cfg.storage_path) / name
    from ..integrity import heal_coverage, inspect_coverage

    report = inspect_coverage(storage_dir)
    if report is None:
        print(f"[doctor] source {name!r} has no readable index yet — "
              f"nothing to heal.")
        return 0
    if not report["drifted"]:
        print(f"[doctor] source {name!r}: hash cache agrees with the index "
              f"({report['indexed_files']} file(s) indexed) — nothing to heal.")
        return 0

    try:
        outcome = heal_coverage(storage_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[doctor] can't heal {name!r}: {e}", file=sys.stderr)
        return 2

    if outcome["phantom"]:
        print(f"[doctor] {outcome['phantom']} file(s) were cached as indexed "
              f"but absent from the index.")
    if outcome["unvectorized"]:
        print(f"[doctor] {outcome['unvectorized']} chunk(s) across "
              f"{outcome['unvectorized_files']} file(s) had no vector — the "
              f"state that makes every search fail with 'Error querying knn'.")
    print(f"[doctor] dropped {len(outcome['reindex_files'])} cache entry(ies).")
    print(f"[doctor] they will be re-indexed on the next build / watcher pass "
          f"(`lynx build --source {name}`).")
    return 0
