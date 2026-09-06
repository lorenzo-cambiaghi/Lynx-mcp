"""
MCP server for the lynx project (multi-source).

The bulk of the work happens inside `run_server(config_path)`:
  - load config (v2 schema, validates `sources` block)
  - construct `SourceManager` in a background thread (heavy: loads embedding
    model and builds backends)
  - register a FIXED set of MCP tools that take a `source` parameter
  - start per-source watchers once backends are ready
  - call mcp.run() (blocking)

Tool surface design: the tool count is CONSTANT in the number of sources.
Earlier versions registered ~17 tools per source (search_<name>,
get_callers_<name>, ...), which blew past client tool limits and bloated
the model context as soon as a user added a second or third source. Now
every tool takes `source` (optional where it can be defaulted) and the
tool descriptions embed the live source catalog so the client can route
without an extra discovery call.

The stdout-redirect dance at module top must run BEFORE any heavy import,
so it is intentionally a side effect of importing this module — see the
comment by `_REAL_STDOUT_FD` for why.
"""

import os
import sys
import warnings
from typing import Annotated

# Silence EVERYTHING before any library writes to stdout/stderr.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# CRITICAL for MCP: some libraries write to stdout during import/init
# (llama_index prints "LLM is explicitly disabled. Using MockLLM.";
# onnxruntime and huggingface_hub log the model load; etc.). In MCP stdio mode,
# stdout is the JSON-RPC channel and any spurious byte breaks the protocol
# and freezes tool calls. We save the real fd 1 and point fd 1 at fd 2 for the
# whole life of the process: the indexes open in a background thread while the
# client is already talking to us, so there is no moment after which stray
# prints would be safe. The transport alone writes to the saved descriptor
# (see `_run_stdio`).
_REAL_STDOUT_FD = os.dup(1)
os.dup2(2, 1)

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from .config import load_config


# ----------------------------------------------------------------------
# Tool annotations — hints clients use for permission UIs (a read-only
# tool can be auto-approved) and parallelization decisions.
# ----------------------------------------------------------------------

# Retrieval tools: pure reads over the local index, no network.
_ANN_READ = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False,
    idempotentHint=True, openWorldHint=False,
)
# update_source_index rewrites the index; for webdoc sources the rebuild
# crawls the configured site, so the open-world hint is honest.
_ANN_REBUILD = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False,
    idempotentHint=True, openWorldHint=True,
)
# feedback appends to a local log file; nothing ever leaves the machine.
_ANN_FEEDBACK = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False,
    idempotentHint=False, openWorldHint=False,
)
# export_graph WRITES a file to disk (not read-only), but deterministically and
# non-destructively (it only creates/overwrites its own report file).
_ANN_WRITE_FILE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False,
    idempotentHint=True, openWorldHint=False,
)


# ----------------------------------------------------------------------
# Reusable per-parameter descriptions. MCP clients (and directory quality
# scorers like Glama) read the `description` of each input-schema property;
# the prose tool description alone doesn't populate those. FastMCP only picks
# them up from `Annotated[..., Field(description=...)]`, not from docstrings.
# The texts live in `tool_docs.py`, next to the tool descriptions, so the
# whole per-session context cost of the tool list is readable in one file.
# ----------------------------------------------------------------------

from .tool_docs import desc as _tool_desc, param as _param
from .tool_profiles import (
    DEFAULT_PROFILE, PROFILES, ToolProfileError, available_tools, select_tools,
    validate_profile,
)

_SourceArg = Annotated[str | None, Field(description=_param("*.source"))]
# `search` / `deep_search` accept ONE name, a LIST of names, or omit (all):
# a query is scoped to a subset of sources at request time, no restart.
_SourcesArg = Annotated[str | list[str] | None, Field(description=_param("*.sources"))]
_FileGlobArg = Annotated[str | None, Field(description=_param("*.file_glob"))]
_ExtensionsArg = Annotated[list[str] | None, Field(description=_param("*.extensions"))]
_PathContainsArg = Annotated[str | None, Field(description=_param("*.path_contains"))]


def _run_stdio(mcp) -> None:
    """Serve JSON-RPC on the real stdout while fd 1 stays pointed at stderr.

    The SDK's `stdio_server` writes to `sys.stdout` by default; we hand it a
    text stream on the descriptor saved at import instead, so a library that
    prints during the background load cannot corrupt the channel."""
    import io
    import anyio
    from mcp.server.stdio import stdio_server

    raw = os.fdopen(_REAL_STDOUT_FD, "wb", buffering=0, closefd=False)
    real_stdout = io.TextIOWrapper(raw, encoding="utf-8", write_through=True)

    async def _main():
        async with stdio_server(stdout=anyio.wrap_file(real_stdout)) as (read, write):
            await mcp._mcp_server.run(
                read, write, mcp._mcp_server.create_initialization_options(),
            )

    anyio.run(_main)


# ----------------------------------------------------------------------
# Output formatting helpers (shared by per-source and global tools).
# Implementations live in `_format.py`; re-exported here so existing
# `lynx.server._format_*` references (including tests) keep resolving.
# ----------------------------------------------------------------------

from ._format import (
    _format_one_result,
    _format_one_outline,
    _format_outline_results,
    _format_search_results,
    _format_deep_response,
    _build_filter_suffix,
    _format_node_brief,
    _format_edge_lines,
    _format_definition_results,
    _format_usage_results,
    _format_test_results,
    _describe_loc,
    _format_describe_symbol,
    _format_impact,
    _format_module_summary,
    _format_repo_overview,
    _format_similar_results,
    _format_search_diff,
)


# ----------------------------------------------------------------------
# Tool registration: fixed tool set with a `source` parameter
# ----------------------------------------------------------------------
#
# NOTE on descriptions: we pass them via @mcp.tool(description=...) rather
# than as Python docstrings. An f-string as the first statement of a
# function is just an evaluated expression — it never becomes __doc__, so
# FastMCP would see an empty description and the AI client would have no
# idea when to call each tool.


def _source_catalog(manager) -> str:
    """One-line catalog of configured sources, embedded in tool
    descriptions so the client can route without a discovery call."""
    parts = []
    for name, backend in manager.backends.items():
        entry = f"{name} (type={backend.type_name}"
        path = backend.source_config.get("path") or backend.source_config.get("url", "")
        if path:
            entry += f", {path}"
        entry += ")"
        parts.append(entry)
    return "; ".join(parts)


def _capabilities(manager) -> dict:
    """What the configured sources support; decides which tools exist at all."""
    backends = list(manager.backends.values())
    return {
        "has_codebase": any(_is_codebase(b) for b in backends),
        "has_graph": any(getattr(b, "graph", None) is not None for b in backends),
        "has_git": any(
            _is_codebase(b)
            and b.source_config.get("git_integration", {}).get("enabled")
            for b in backends
        ),
    }


def _build_instructions(manager, tools=None, profile: str | None = None) -> str:
    """Handshake instructions sent to the client in the MCP `initialize`
    response. Every client gets this automatically, no rules file needed.

    This is the one place the source catalog and the usage ladder are
    spelled out, so the tool descriptions don't have to repeat them 17
    times. Kept compact: it rides along in the client's context on every
    turn. The full playbook is the `lynx://guide` resource. `tools` is the
    set actually registered; anything outside it is not mentioned, and the
    profile that hid it is named so the model can ask for it."""
    caps = _capabilities(manager)
    has = (lambda name: True) if tools is None else (lambda name: name in set(tools))

    codebase = [n for n, b in manager.backends.items() if _is_codebase(b)]
    graph = [n for n, b in manager.backends.items() if getattr(b, "graph", None) is not None]
    git = [
        n for n, b in manager.backends.items()
        if _is_codebase(b) and b.source_config.get("git_integration", {}).get("enabled")
    ]

    parts = [
        "Lynx: semantic + lexical search over locally indexed sources (code, "
        f"library docs, PDFs). Sources: {_source_catalog(manager)}. ",
    ]
    scoped = []
    if codebase:
        scoped.append(f"codebase: {', '.join(codebase)}")
    if graph:
        scoped.append(f"graph-enabled: {', '.join(graph)}")
    if git:
        scoped.append(f"git-enabled: {', '.join(git)}")
    if scoped:
        parts.append(
            f"Per-source tools ({'; '.join(scoped)}) take `source`; omit it when "
            "only one source qualifies. "
        )
    if has("search"):
        parts.append(
            "Use `search` FIRST for any question about this code or these docs: "
            "describe what the code DOES in plain words ('method that clamps "
            "camera zoom'), not identifier names. Omit `source` to search "
            "everything. Hybrid scores are small by construction: ~0.03 is a "
            "STRONG match. For broad queries or a large top_k use outline=true "
            "(signatures only), then read just the one body you need. "
        )
    lookups = [n for n in ("find_definition", "find_usages", "describe_symbol") if has(n)]
    if lookups:
        parts.append(f"For an identifier you already know: {' / '.join(lookups)}. ")
    if has("deep_search"):
        parts.append("Escalate to `deep_search` only when `search` returns weak or empty results. ")
    structural = [n for n in ("graph_query", "impact", "describe_symbol") if has(n)]
    if caps["has_graph"] and structural:
        parts.append(
            "For structural questions ('who calls X?', 'what breaks if I change "
            f"X?') use {' / '.join(structural)}: they read the code graph, which "
            "textual search cannot see. "
        )
    if tools is not None and profile is not None:
        every = available_tools(**caps)
        hidden = [n for n in every if n not in set(tools)]
        if hidden:
            parts.append(
                f"Tool profile '{profile}' ({len(tools)} of {len(every)} tools); "
                f"not loaded: {', '.join(hidden)}. They come with "
                "`lynx serve --profile full` or `tools.include` in config.json. "
            )
    parts.append(
        "Indexes open in the background right after this handshake; a call that "
        "arrives earlier answers with the loading state, so retry it. "
    )
    parts.append("Read the `lynx://guide` resource for the full playbook. ")
    if has("feedback"):
        parts.append("If you cannot find what you need, call `feedback` before giving up.")
    return "".join(parts).rstrip()


def _build_guide(manager, tools=None, profile: str | None = None) -> str:
    """Full usage playbook, exposed as the `lynx://guide` MCP resource.

    Reuses the same generator that powers the downloadable rules files in
    the manager UI, so there is one source of truth for 'how to use Lynx
    well'. Tools outside the active profile are left out of it, and a
    closing note says which profile is running."""
    from .manager.ui.integrations import render_rules_for_sources
    caps = _capabilities(manager)
    text = render_rules_for_sources(
        list(manager.backends), has_graph=caps["has_graph"], has_git=caps["has_git"],
        tools=tools,
    )
    if tools is not None and profile is not None:
        every = available_tools(**caps)
        hidden = [n for n in every if n not in set(tools)]
        if hidden:
            text += (
                f"\n## Tool profile\n\nThis server runs the '{profile}' profile: "
                f"{len(tools)} of {len(every)} tools. Not loaded: "
                f"{', '.join(hidden)}. The full set comes with "
                "`lynx serve --profile full`, or with `tools.include` in config.json.\n"
            )
    return text


def _resolve_source(manager, source, *, predicate=None, kind: str = "source"):
    """Resolve an optional `source` argument to a concrete source name.

    - explicit name → validated (and checked against `predicate` if given);
    - None → the only matching source if unambiguous, otherwise raises
      ValueError listing the candidates so the client can retry.
    """
    candidates = [
        name
        for name, backend in manager.backends.items()
        if predicate is None or predicate(backend)
    ]
    if not candidates:
        raise ValueError(f"no configured {kind} supports this operation")
    if source is not None:
        if source not in manager.backends:
            raise ValueError(
                f"unknown source {source!r}. Available: {list(manager.backends)}"
            )
        if source not in candidates:
            raise ValueError(
                f"source {source!r} does not support this operation. "
                f"Eligible sources: {candidates}"
            )
        return source
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"multiple eligible sources — pass `source` explicitly. "
        f"Candidates: {candidates}"
    )


def _normalize_sources(manager, source):
    """Normalize the `source` arg of search / deep_search to a validated list of names,
    or None for "all sources". Accepts a single name, a comma-separated string, or a list —
    so a query can be scoped to a subset of sources at request time. Raises ValueError naming
    any unknown source so the client can retry against `list_sources`."""
    if source is None:
        return None
    if isinstance(source, str):
        names = [s.strip() for s in source.split(",") if s.strip()]
    else:
        names = [str(s).strip() for s in source if str(s).strip()]
    if not names:
        return None
    unknown = [n for n in names if n not in manager.backends]
    if unknown:
        raise ValueError(
            f"unknown source(s) {unknown}. Available: {list(manager.backends)}"
        )
    return list(dict.fromkeys(names))  # de-dup, preserve order


def _register_search_tools(mcp, manager):
    """Register `search` and `deep_search` (fixed names, `source` param)."""
    _desc_search = _tool_desc("search")

    @mcp.tool(name="search", description=_desc_search, annotations=_ANN_READ)
    def _search(
        query: Annotated[str, Field(description=_param("search.query"))],
        source: _SourcesArg = None,
        top_k: Annotated[int | None, Field(description=_param("*.top_k"))] = None,
        outline: Annotated[bool, Field(description=_param("search.outline"))] = False,
        file_glob: _FileGlobArg = None,
        extensions: _ExtensionsArg = None,
        path_contains: _PathContainsArg = None,
    ) -> str:
        try:
            effective_top_k = top_k if top_k is not None else manager.config.search.default_top_k
            filters = dict(
                file_glob=file_glob, extensions=extensions, path_contains=path_contains
            )
            names = _normalize_sources(manager, source)
            if not names:
                results = manager.search_all(query, top_k=effective_top_k, **filters)
                label = "all sources"
            elif len(names) == 1:
                results = manager.search(names[0], query, top_k=effective_top_k, **filters)
                label = f"source {names[0]!r}"
            else:
                results = manager.search_all(query, top_k=effective_top_k, only=names, **filters)
                label = f"sources {names}"
            filter_suffix = _build_filter_suffix(file_glob, extensions, path_contains)
            fmt = _format_outline_results if outline else _format_search_results
            return fmt(query, results, label, filter_suffix)
        except Exception as e:
            return f"Error during search: {str(e)}"

    _desc_deep = _tool_desc("deep_search")

    @mcp.tool(name="deep_search", description=_desc_deep, annotations=_ANN_READ)
    def _deep_search(
        queries: Annotated[list[str], Field(description=_param("deep_search.queries"))],
        source: _SourcesArg = None,
        top_k: Annotated[int | None, Field(description=_param("*.top_k"))] = None,
        mode: Annotated[str | None, Field(description=_param("deep_search.mode"))] = None,
        file_glob: _FileGlobArg = None,
        extensions: _ExtensionsArg = None,
        path_contains: _PathContainsArg = None,
        min_score: Annotated[float | None, Field(description=_param("deep_search.min_score"))] = None,
        min_results: Annotated[int | None, Field(description=_param("deep_search.min_results"))] = None,
        return_all_variants: Annotated[bool, Field(description=_param("deep_search.return_all_variants"))] = False,
    ) -> str:
        try:
            effective_top_k = top_k if top_k is not None else manager.config.search.default_top_k
            filters = dict(
                file_glob=file_glob, extensions=extensions, path_contains=path_contains
            )
            names = _normalize_sources(manager, source)
            # mode / return_all_variants are single-source only; a subset (or all) fuses.
            single = names[0] if names and len(names) == 1 else None
            if single is None:
                response = manager.deep_search_all(
                    queries=queries,
                    top_k=effective_top_k,
                    min_score=min_score,
                    min_results=min_results,
                    only=names,  # None = every source; a subset restricts the fusion
                    **filters,
                )
                label = "all sources" if not names else f"sources {names}"
            else:
                manager.get(single)
                response = manager.deep_search(
                    single,
                    queries=queries,
                    top_k=effective_top_k,
                    mode=mode,
                    min_score=min_score,
                    min_results=min_results,
                    return_all_variants=return_all_variants,
                    **filters,
                )
                label = f"source {single!r}"
            meta_parts = []
            if mode and single is not None:
                meta_parts.append(f"mode={mode!r}")
            if file_glob:
                meta_parts.append(f"file_glob={file_glob!r}")
            if extensions:
                meta_parts.append(f"extensions={list(extensions)!r}")
            if path_contains:
                meta_parts.append(f"path_contains={path_contains!r}")
            meta_suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""
            return _format_deep_response(response, queries, label, meta_suffix)
        except Exception as e:
            return f"Error during deep search: {str(e)}"


def _register_global_tools(mcp, manager):
    """Register cross-source / management tools that don't depend on a
    specific source name."""

    @mcp.tool(name="list_sources", description=_tool_desc("list_sources"), annotations=_ANN_READ)
    def list_sources() -> str:
        # Like every other tool, answer with text on failure: while the
        # indexes are still opening, that text is the loading state.
        try:
            statuses = manager.list_sources()
        except Exception as e:
            return f"Error: {e}"
        lines = [f"Sources ({len(manager.backends)}):"]
        for status in statuses:
            line = (
                f"  - {status['name']} (type: {status['type']}, "
                f"chunks: {status.get('chunk_count', 'n/a')})"
            )
            if status.get("path"):
                line += f"\n      path: {status['path']}"
            if status.get("drift_severity"):
                line += f"\n      drift: {status['drift_severity'].upper()}"
            lines.append(line)
        return "\n".join(lines)

    @mcp.tool(name="update_source_index", description=_tool_desc("update_source_index"),
              annotations=_ANN_REBUILD)
    def update_source_index(
        source: Annotated[str, Field(description=_param("update_source_index.source"))],
        force: Annotated[bool, Field(description=_param("update_source_index.force"))] = False,
    ) -> str:
        try:
            manager.update(source, force=force)
            return f"Source {source!r} rebuilt successfully."
        except KeyError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error rebuilding source {source!r}: {str(e)}"

    @mcp.tool(name="get_rag_status", description=_tool_desc("get_rag_status"), annotations=_ANN_READ)
    def get_rag_status(
        source: Annotated[str | None, Field(description=_param("get_rag_status.source"))] = None,
    ) -> str:
        try:
            statuses = (
                [manager.get(source).status()]
                if source is not None
                else [b.status() for b in manager.backends.values()]
            )
            lines = []
            for s in statuses:
                name = s["name"]
                drift_text = manager.get(name).drift_status_text()
                needs = (
                    manager.get(name).needs_update()
                    if hasattr(manager.get(name), "needs_update")
                    else False
                )
                lines.append(f"=== Source: {name} (type: {s['type']}) ===")
                lines.append(f"Status:       {'Needs update' if needs else 'Up to date'}")
                if s.get("path"):
                    lines.append(f"Path:         {s['path']}")
                lines.append(f"Chunks:       {s.get('chunk_count', 'n/a')}")
                if s.get("last_commit"):
                    lines.append(f"Last commit:  {s['last_commit']}")
                lines.append(f"Last update:  {s.get('last_update', 'Never')}")
                lines.append("")
                lines.append(drift_text)
                lines.append("")
            return "\n".join(lines).rstrip()
        except KeyError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading status: {str(e)}"

    _desc_feedback = _tool_desc("feedback")

    @mcp.tool(name="feedback", description=_desc_feedback, annotations=_ANN_FEEDBACK)
    def _feedback(
        trying_to_do: Annotated[str, Field(description=_param("feedback.trying_to_do"))],
        tried: Annotated[str, Field(description=_param("feedback.tried"))],
        stuck: Annotated[str, Field(description=_param("feedback.stuck"))],
    ) -> str:
        try:
            import json as _json
            from datetime import datetime as _dt
            from pathlib import Path as _Path

            feedback_dir = _Path(manager.config.storage_path) / "_feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "at": _dt.now().isoformat(timespec="seconds"),
                "trying_to_do": trying_to_do,
                "tried": tried,
                "stuck": stuck,
                "sources": list(manager.backends),
            }
            with open(feedback_dir / "feedback.jsonl", "a", encoding="utf-8") as f:
                f.write(_json.dumps(record, ensure_ascii=False) + "\n")
            return (
                "Feedback recorded locally (rag_storage/_feedback/feedback.jsonl). "
                "Tell the user their Lynx index didn't cover this, so they can "
                "review the report and adjust sources or filters."
            )
        except Exception as e:
            return f"Error recording feedback: {e}"


# ----------------------------------------------------------------------
# Graph tool (registered when at least one source has graph.enabled=true)
# ----------------------------------------------------------------------


def _register_graph_tools(mcp, manager):
    """Register the single `graph_query` tool covering every graph operation.

    One tool with an `operation` selector instead of 10 tools per source:
    the per-source variants made the tool list explode quadratically
    (sources x operations) and blew client tool limits.
    """
    _desc = _tool_desc("graph_query")

    @mcp.tool(name="graph_query", description=_desc, annotations=_ANN_READ)
    def _graph_query(
        operation: Annotated[str, Field(description=_param("graph_query.operation"))],
        source: _SourceArg = None,
        symbol: Annotated[str | None, Field(description=_param("graph_query.symbol"))] = None,
        target: Annotated[str | None, Field(description=_param("graph_query.target"))] = None,
        relation_filter: Annotated[str | None, Field(description=_param("graph_query.relation_filter"))] = None,
        depth: Annotated[int, Field(description=_param("graph_query.depth"))] = 1,
        limit: Annotated[int, Field(description=_param("graph_query.limit"))] = 50,
        max_hops: Annotated[int, Field(description=_param("graph_query.max_hops"))] = 8,
        top_n: Annotated[int, Field(description=_param("graph_query.top_n"))] = 10,
        min_community_size: Annotated[int, Field(description=_param("graph_query.min_community_size"))] = 3,
    ) -> str:
        try:
            src = _resolve_source(
                manager, source,
                predicate=lambda b: getattr(b, "graph", None) is not None,
                kind="graph-enabled source",
            )
            # Rendering lives in graph/dispatch.py so `lynx graph query`
            # returns exactly what the model sees here.
            from .graph.dispatch import run_graph_query
            return run_graph_query(
                manager, src, operation,
                symbol=symbol, target=target, relation_filter=relation_filter,
                depth=depth, limit=limit, max_hops=max_hops, top_n=top_n,
                min_community_size=min_community_size,
            )
        except Exception as e:
            return f"Error: {e}"


# ----------------------------------------------------------------------
# Main server entry point
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Combined tools (find_definition / find_usages / find_tests_for /
# find_similar / search_diff) — registered when at least one codebase
# source exists. Use the graph layer when present, fall back to search.
# ----------------------------------------------------------------------


def _report_dir(manager):
    """Directory where graph view files are written. The rule lives in
    `config.reports_dir` — the UI and the CLI write there too, and a view
    exported from one has to be findable by the others."""
    from .config import reports_dir
    return reports_dir(manager.config)


def _is_codebase(backend) -> bool:
    return backend.type_name == "codebase"


def _register_combined_tools(mcp, manager, *, has_graph: bool = False):
    """Register find_definition / find_usages / find_tests_for / find_similar /
    describe_symbol / impact / repo_overview (+ search_diff when git is on) for
    codebase sources. `module_summary` and `export_graph` are graph-only — they
    return nothing useful without the call graph — so they're registered solely
    when `has_graph` (consistent with how graph_query is gated).
    """
    _desc_find_def = _tool_desc("find_definition")

    @mcp.tool(name="find_definition", description=_desc_find_def, annotations=_ANN_READ)
    def _find_def(
        symbol: Annotated[str, Field(description=_param("find_definition.symbol"))],
        source: _SourceArg = None,
        limit: Annotated[int, Field(description=_param("*.limit"))] = 10,
    ) -> str:
        try:
            src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
            results = manager.find_definition(src, symbol, limit=limit)
            return _format_definition_results(symbol, results)
        except Exception as e:
            return f"Error: {e}"

    _desc_find_usages = _tool_desc("find_usages")

    @mcp.tool(name="find_usages", description=_desc_find_usages, annotations=_ANN_READ)
    def _find_usages(
        symbol: Annotated[str, Field(description=_param("find_usages.symbol"))],
        source: _SourceArg = None,
        limit: Annotated[int, Field(description=_param("*.limit"))] = 50,
    ) -> str:
        try:
            src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
            results = manager.find_usages(src, symbol, limit=limit)
            return _format_usage_results(symbol, results)
        except Exception as e:
            return f"Error: {e}"

    _desc_find_tests = _tool_desc("find_tests_for")

    @mcp.tool(name="find_tests_for", description=_desc_find_tests, annotations=_ANN_READ)
    def _find_tests(
        symbol: Annotated[str, Field(description=_param("find_tests_for.symbol"))],
        source: _SourceArg = None,
        limit: Annotated[int, Field(description=_param("*.limit"))] = 20,
        test_path_pattern: Annotated[str | None, Field(description=_param("find_tests_for.test_path_pattern"))] = None,
    ) -> str:
        try:
            src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
            results = manager.find_tests_for(
                src, symbol, limit=limit,
                test_path_pattern=test_path_pattern,
            )
            return _format_test_results(symbol, results)
        except Exception as e:
            return f"Error: {e}"

    _desc_find_similar = _tool_desc("find_similar")

    @mcp.tool(name="find_similar", description=_desc_find_similar, annotations=_ANN_READ)
    def _find_similar(
        snippet: Annotated[str, Field(description=_param("find_similar.snippet"))],
        source: _SourceArg = None,
        top_k: Annotated[int, Field(description=_param("*.limit"))] = 10,
    ) -> str:
        try:
            src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
            results = manager.find_similar(src, snippet, top_k=top_k)
            return _format_similar_results(results)
        except Exception as e:
            return f"Error: {e}"

    _desc_describe = _tool_desc("describe_symbol")

    @mcp.tool(name="describe_symbol", description=_desc_describe, annotations=_ANN_READ)
    def _describe_symbol(
        symbol: Annotated[str, Field(description=_param("describe_symbol.symbol"))],
        source: _SourceArg = None,
        callers_limit: Annotated[int, Field(description=_param("describe_symbol.callers_limit"))] = 10,
        callees_limit: Annotated[int, Field(description=_param("describe_symbol.callees_limit"))] = 10,
        tests_limit: Annotated[int, Field(description=_param("describe_symbol.tests_limit"))] = 5,
    ) -> str:
        try:
            src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
            d = manager.describe_symbol(
                src, symbol,
                callers_limit=callers_limit,
                callees_limit=callees_limit,
                tests_limit=tests_limit,
            )
            return _format_describe_symbol(symbol, d)
        except Exception as e:
            return f"Error: {e}"

    _desc_impact = _tool_desc("impact")

    @mcp.tool(name="impact", description=_desc_impact, annotations=_ANN_READ)
    def _impact(
        symbol: Annotated[str, Field(description=_param("impact.symbol"))],
        source: _SourceArg = None,
        max_depth: Annotated[int, Field(description=_param("impact.max_depth"))] = 3,
        tests_limit: Annotated[int, Field(description=_param("impact.tests_limit"))] = 10,
    ) -> str:
        try:
            src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
            d = manager.impact_of(src, symbol, max_depth=max_depth, tests_limit=tests_limit)
            return _format_impact(symbol, d)
        except Exception as e:
            return f"Error: {e}"

    _desc_overview = _tool_desc("repo_overview")

    @mcp.tool(name="repo_overview", description=_desc_overview, annotations=_ANN_READ)
    def _repo_overview(
        source: _SourceArg = None,
    ) -> str:
        try:
            src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
            d = manager.repo_overview(src)
            return _format_repo_overview(d)
        except Exception as e:
            return f"Error: {e}"

    # module_summary and export_graph produce nothing useful without the call
    # graph, so register them only when it's available — consistent with how
    # graph_query is gated, and keeps a non-graph codebase source uncluttered.
    if has_graph:
        _desc_module = _tool_desc("module_summary")

        @mcp.tool(name="module_summary", description=_desc_module, annotations=_ANN_READ)
        def _module_summary(
            file: Annotated[str, Field(description=_param("module_summary.file"))],
            source: _SourceArg = None,
            limit: Annotated[int, Field(description=_param("module_summary.limit"))] = 200,
        ) -> str:
            try:
                src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
                d = manager.module_summary(src, file, limit=limit)
                return _format_module_summary(file, d)
            except Exception as e:
                return f"Error: {e}"

        _desc_export_graph = _tool_desc("export_graph")

        @mcp.tool(name="export_graph", description=_desc_export_graph, annotations=_ANN_WRITE_FILE)
        def _export_graph(
            target: Annotated[str, Field(description=_param("export_graph.target"))],
            mode: Annotated[str, Field(description=_param("export_graph.mode"))] = "symbol",
            source: _SourceArg = None,
            depth: Annotated[int, Field(description=_param("export_graph.depth"))] = 2,
            out: Annotated[str | None, Field(description=_param("export_graph.out"))] = None,
        ) -> str:
            try:
                from pathlib import Path
                src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
                res = manager.export_graph(src, mode, target, depth=depth)
                if res.get("empty"):
                    return f"Nothing to export: {res.get('reason')}"
                out_path = Path(out) if out else _report_dir(manager) / res["suggested_name"]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(res["content"], encoding="utf-8")
                return f"Wrote self-contained graph view to {out_path} — open it in a browser."
            except Exception as e:
                return f"Error: {e}"

    # search_diff only when at least one codebase source has git_integration
    # enabled — without it the diff command can't run.
    def _has_git(backend):
        return (
            _is_codebase(backend)
            and backend.source_config.get("git_integration", {}).get("enabled")
        )

    git_sources = [name for name, b in manager.backends.items() if _has_git(b)]
    if git_sources:
        _desc_search_diff = _tool_desc("search_diff")

        @mcp.tool(name="search_diff", description=_desc_search_diff, annotations=_ANN_READ)
        def _search_diff(
            query: Annotated[str, Field(description=_param("search_diff.query"))],
            source: _SourceArg = None,
            base: Annotated[str | None, Field(description=_param("search_diff.base"))] = None,
            top_k: Annotated[int, Field(description=_param("search_diff.top_k"))] = 8,
        ) -> str:
            try:
                src = _resolve_source(
                    manager, source, predicate=_has_git, kind="git-enabled codebase source"
                )
                out = manager.search_diff(src, query, base=base, top_k=top_k)
            except Exception as e:
                return f"Error: {e}"
            return _format_search_diff(src, out)


class LynxServer(FastMCP):
    """FastMCP plus two things the SDK leaves to the server: a call to a tool
    the profile hid answers with the profile and the fix instead of
    "Unknown tool", and `serverInfo.version` is this package's version, not
    the SDK's (FastMCP falls back to the `mcp` package version when the
    server passes none)."""

    hidden_tools: dict = {}

    def __init__(self, name: str, *, instructions: str | None = None) -> None:
        super().__init__(name, instructions=instructions)
        self.hidden_tools = {}
        self._mcp_server.version = package_version()

    async def call_tool(self, name: str, arguments: dict):
        message = self.hidden_tools.get(name)
        if message is not None:
            raise ToolError(message)
        return await super().call_tool(name, arguments)


def package_version() -> str:
    """The installed lynx-mcp version, or "unknown" outside an install."""
    try:
        from importlib.metadata import version
        return version("lynx-mcp")
    except Exception:
        return "unknown"


def hidden_tool_message(name: str, profile: str) -> str:
    return (
        f"Tool {name!r} is not loaded in tool profile {profile!r}. Start the server "
        f"with `lynx serve --profile full`, set LYNX_TOOL_PROFILE=full, or add "
        f"\"{name}\" to tools.include in config.json."
    )


def resolve_profile(config, override: str | None = None) -> str:
    """Which tool profile this process runs: `lynx serve --profile` wins, then
    the LYNX_TOOL_PROFILE environment variable, then `tools.profile` in the
    config file (default 'standard')."""
    name = override or os.environ.get("LYNX_TOOL_PROFILE") or config.tools.profile
    return validate_profile(name)




def apply_tool_profile(mcp, profile: str, include=(), exclude=()):
    """Prune the registered tools down to a profile. Returns (kept, dropped).

    Registration stays capability-driven (graph tools only with a graph, ...);
    the profile is applied afterwards, so the registrars and the tests that
    enumerate them keep seeing the full surface."""
    registered = list(mcp._tool_manager._tools)
    kept, dropped = select_tools(registered, profile, include, exclude)
    for name in dropped:
        remove = getattr(mcp, "remove_tool", None)
        if callable(remove):
            remove(name)
        else:  # pragma: no cover - older mcp releases
            mcp._tool_manager._tools.pop(name, None)
    _slim_tool_schemas(mcp)
    return kept, dropped


def _slim_tool_schemas(mcp) -> None:
    """Drop from every registered tool what no client reads: pydantic's
    automatic `title` on the argument model and on each property, and the
    output schema FastMCP derives from a `-> str` return type (every Lynx tool
    returns plain text, so the schema only ever said "a string"). Both ride in
    `tools/list` on every session; measured at about 3,000 characters on the
    full profile. With no output schema the SDK also stops wrapping each result
    in a `{"result": ...}` structured copy, which was sent alongside the text."""
    for tool in mcp._tool_manager._tools.values():
        params = tool.parameters
        if isinstance(params, dict):
            params.pop("title", None)
            for prop in (params.get("properties") or {}).values():
                if isinstance(prop, dict):
                    prop.pop("title", None)
        fm = getattr(tool, "fn_metadata", None)
        if fm is not None and getattr(fm, "output_schema", None) is not None:
            fm.output_schema = None
            fm.wrap_output = False


def run_server(config_path=None, profile: str | None = None):
    """Boot the MCP server. Blocks until the client disconnects.

    The handshake does not wait for the indexes. Everything it needs (which
    sources exist, their type, whether the graph or git is on) is in the
    config, so the tools are registered against a `ManagerHandle` built from
    the config alone, the transport starts within a second, and the real
    `SourceManager` is constructed in a background thread. A tool call that
    arrives before the indexes are open waits a bounded time, then answers
    with the loading state and asks to be retried (see `startup.py`)."""
    config = load_config(config_path=config_path)
    try:
        profile_name = resolve_profile(config, profile)
    except ToolProfileError as e:
        print(f"[server] FATAL: {e}", file=sys.stderr)
        sys.exit(2)

    from .startup import ManagerHandle, start_loader
    manager = ManagerHandle(config)
    start_loader(config, manager)

    # The tool set is decided before FastMCP exists, because the handshake
    # `instructions` are passed to its constructor and must name only the
    # tools the client will actually see. `available_tools` mirrors the
    # capability gating below; `tests/test_tool_profiles.py` keeps them equal.
    caps = _capabilities(manager)
    selected, _ = select_tools(
        available_tools(**caps), profile_name,
        config.tools.include, config.tools.exclude,
    )

    # FastMCP is constructed only now so the handshake `instructions` can
    # embed the live source catalog: every client gets the usage playbook
    # automatically, without installing a rules file.
    mcp = LynxServer("lynx", instructions=_build_instructions(manager, selected, profile_name))

    # Full playbook as an MCP resource the client can read on demand.
    guide_text = _build_guide(manager, selected, profile_name)

    @mcp.resource(
        "lynx://guide",
        name="guide",
        description="How to use Lynx well: search phrasing, score "
                    "interpretation, escalation ladder, structural queries.",
        mime_type="text/markdown",
    )
    def _guide() -> str:
        return guide_text

    # Fixed tool set — the tool count does not grow with the number of
    # sources. Conditional tools (graph_query, find_*, search_diff) are
    # registered only when at least one source supports them.
    _register_search_tools(mcp, manager)
    _register_global_tools(mcp, manager)

    # graph_query — only when at least one source has graph.enabled=true.
    # We probe `backend.graph` rather than `backend.type_name == "codebase"`
    # so future source types (e.g. a pdf backend with a graph) can opt in too.
    if any(getattr(b, "graph", None) is not None for b in manager.backends.values()):
        _register_graph_tools(mcp, manager)

    # Combined tools (find_definition / find_usages / find_tests_for /
    # find_similar / search_diff) — only when there is a codebase source.
    if any(b.type_name == "codebase" for b in manager.backends.values()):
        has_graph = any(
            getattr(b, "graph", None) is not None for b in manager.backends.values()
        )
        _register_combined_tools(mcp, manager, has_graph=has_graph)

    # Then the profile: everything above registered what the sources support,
    # this drops what the profile does not want the client to pay for.
    kept, dropped = apply_tool_profile(
        mcp, profile_name, config.tools.include, config.tools.exclude,
    )
    mcp.hidden_tools = {name: hidden_tool_message(name, profile_name) for name in dropped}
    print(
        f"[server] tool profile {profile_name!r}: {len(kept)} tools"
        + (f", not loaded: {', '.join(dropped)}" if dropped else ""),
        file=sys.stderr,
    )

    # Talk to the client now; the loader thread fills in the indexes.
    print("[server] answering the MCP handshake; indexes open in the background",
          file=sys.stderr)
    _run_stdio(mcp)


if __name__ == "__main__":
    # Allow `python -m lynx.server` for ad-hoc invocation.
    # Normal entry point goes through cli:main (`lynx serve`).
    run_server()
