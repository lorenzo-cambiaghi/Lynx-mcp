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
import threading
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
# and freezes tool calls. We save the real fd 1 and redirect it to fd 2 for
# the entire import/loading phase, then restore it just before mcp.run().
_REAL_STDOUT_FD = os.dup(1)
os.dup2(2, 1)

from mcp.server.fastmcp import FastMCP
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
# ----------------------------------------------------------------------

_SourceArg = Annotated[
    str | None,
    Field(description="Source name from `list_sources`. Omit to use the default: "
                      "all sources for `search`/`deep_search` (RRF-fused), or the "
                      "single applicable source for the others."),
]
# `search` / `deep_search` accept ONE name, a LIST of names, or omit (all). This is how you
# scope a query to a subset of sources at request time — no server restart, no config change.
_SourcesArg = Annotated[
    str | list[str] | None,
    Field(description="Which source(s) from `list_sources` to search. Omit = ALL sources "
                      "(RRF-fused). One name = that source. A LIST of names = just those "
                      "sources, fused — the way to contextualize a query to a subset "
                      "(e.g. [\"skelforge\"] or [\"skelforge\", \"framework\"])."),
]
_FileGlobArg = Annotated[
    str | None,
    Field(description="fnmatch glob to restrict results by path/filename, "
                      "e.g. `*.cs` or `**/Editor/*`."),
]
_ExtensionsArg = Annotated[
    list[str] | None,
    Field(description="Restrict results to these file extensions, "
                      "e.g. `['.cs', '.shader']`."),
]
_PathContainsArg = Annotated[
    str | None,
    Field(description="Keep only results whose file path contains this substring."),
]


def _restore_real_stdout():
    """Restore fd 1 to the real stdout, draining the Python buffer first."""
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os.dup2(_REAL_STDOUT_FD, 1)


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


def _build_instructions(manager) -> str:
    """Handshake instructions sent to the client in the MCP `initialize`
    response. Every client gets this automatically — no rules file needed.
    Kept compact (it rides along in the client's context); the full
    playbook lives in the `lynx://guide` resource."""
    has_graph = any(
        getattr(b, "graph", None) is not None for b in manager.backends.values()
    )
    parts = [
        "Lynx provides semantic + lexical search over locally indexed sources "
        "(code, library docs, PDFs). Indexed sources: "
        f"{_source_catalog(manager)}. ",
        "Use `search(query, source?)` FIRST for any question about this code or "
        "these docs — describe what the code DOES in natural language "
        "('method that clamps camera zoom'), not identifier names (for those "
        "use find_definition / find_usages). Omit `source` to search everything. ",
        "Hybrid scores are small by construction: ~0.03 is a STRONG match, "
        "not a weak one. ",
        "Escalate to `deep_search` only when `search` returns weak or empty "
        "results. ",
        "For broad/exploratory queries or a large top_k, prefer "
        "`search(query, outline=true)`: it returns signatures only (cheap to "
        "scan) — triage them, then pull just the one body you need with "
        "`find_definition` or its file:line. Use the default full search when "
        "you'll work with the code right away. ",
    ]
    if has_graph:
        parts.append(
            "For structural questions ('who calls X?', 'what breaks if I "
            "change X?') use `graph_query` / `find_usages` — they read the "
            "code graph, which textual search cannot see. "
        )
    parts.append(
        "Read the `lynx://guide` resource for the full playbook. "
        "If you cannot find what you need, call `feedback` before giving up."
    )
    return "".join(parts)


def _build_guide(manager) -> str:
    """Full usage playbook, exposed as the `lynx://guide` MCP resource.

    Reuses the same generator that powers the downloadable rules files in
    the manager UI, so there is one source of truth for 'how to use Lynx
    well'."""
    from .manager.ui.integrations import render_rules_for_sources
    has_graph = any(
        getattr(b, "graph", None) is not None for b in manager.backends.values()
    )
    has_git = any(
        b.type_name == "codebase"
        and b.source_config.get("git_integration", {}).get("enabled")
        for b in manager.backends.values()
    )
    return render_rules_for_sources(
        list(manager.backends), has_graph=has_graph, has_git=has_git
    )


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
    catalog = _source_catalog(manager)

    _desc_search = (
        f"Semantic + lexical (hybrid) search over an indexed source. "
        f"This is your PRIMARY search tool — use it FIRST for any question about the indexed "
        f"code or docs. Omit `source` to search ALL sources at once (rankings fused via RRF, "
        f"each hit tagged with its source); pass ONE name to target a single source, or a "
        f"LIST of names to fuse just those — the way to scope a query to a subset of sources "
        f"(e.g. source=[\"skelforge\"] or [\"skelforge\", \"framework\"]) at request time. "
        f"Configured sources: {catalog}. "
        f"Best practices: use natural-language descriptions of what the code does, not exact "
        f"identifiers (use grep for those). Good: 'method that handles player damage calculation'. "
        f"Bad: 'CalculateDamage'. Args: query (natural language); source (optional name); "
        f"top_k (default from config); file_glob, extensions, path_contains (optional filters). "
        f"For broad/exploratory queries, or a large top_k, set outline=true to get "
        f"signatures-only results (much cheaper to read): scan them, then pull just the one body "
        f"you need with find_definition or its file:line. Use the default (full bodies) when "
        f"you'll work with the code right away."
    )

    @mcp.tool(name="search", description=_desc_search, annotations=_ANN_READ)
    def _search(
        query: Annotated[str, Field(description="Natural-language description of the behavior to find (e.g. 'method that handles player damage calculation'), NOT an identifier — use grep for exact names.")],
        source: _SourcesArg = None,
        top_k: Annotated[int | None, Field(description="Maximum number of results to return. Defaults to the server's configured value.")] = None,
        outline: Annotated[bool, Field(description="If true, return each hit's signature + first doc line instead of its full body — cheap triage for broad queries or a large top_k. Scan the signatures, then read the one body you need (find_definition, or its file:line). Default false = full bodies, for when you'll use the code right away.")] = False,
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

    _desc_deep = (
        f"Multi-query fallback search. ESCALATION TOOL — use only when `search` returned weak "
        f"or empty results, or when the user explicitly asks for a more thorough search. "
        f"Slower than `search` because it runs multiple retrievals. "
        f"How it works: tries each query variant in order; stops at the first whose results pass "
        f"the weakness threshold. If all fail, returns the strongest weak set with a warning. "
        f"Omit `source` to run across ALL sources (RRF-fused); pass one name for a single "
        f"source, or a LIST of names to fuse just that subset. "
        f"Configured sources: {catalog}. "
        f"Best practices: provide 2-4 GENUINELY DIFFERENT phrasings (different angles, not "
        f"paraphrases). Good: ['player health system', 'damage and healing logic', 'HP component "
        f"lifecycle']. Bad: ['player health', 'health of the player'] (too similar). "
        f"Args: queries (ordered list of variants); source (optional name); top_k; "
        f"mode ('dense'|'sparse'|'hybrid', single-source only); file_glob, extensions, "
        f"path_contains; min_score, min_results (threshold overrides); "
        f"return_all_variants (per-variant diagnostics, single-source only)."
    )

    @mcp.tool(name="deep_search", description=_desc_deep, annotations=_ANN_READ)
    def _deep_search(
        queries: Annotated[list[str], Field(description="2-4 genuinely different phrasings of the same need (different angles, not paraphrases), tried in priority order.")],
        source: _SourcesArg = None,
        top_k: Annotated[int | None, Field(description="Maximum number of results to return. Defaults to the configured value.")] = None,
        mode: Annotated[str | None, Field(description="Retrieval mode override (single-source only): 'dense', 'sparse', or 'hybrid'. Defaults to the server's configured mode.")] = None,
        file_glob: _FileGlobArg = None,
        extensions: _ExtensionsArg = None,
        path_contains: _PathContainsArg = None,
        min_score: Annotated[float | None, Field(description="Override the weakness threshold: a variant's results must beat this score to count as strong.")] = None,
        min_results: Annotated[int | None, Field(description="Override the minimum number of results a variant must return to be considered strong.")] = None,
        return_all_variants: Annotated[bool, Field(description="If true, include per-variant diagnostics in the response (single-source only).")] = False,
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

    @mcp.tool(annotations=_ANN_READ)
    def list_sources() -> str:
        """List all configured sources with their type, location, chunk count, and drift status.

        Call this first when you don't know which sources are available.
        Pass a name from this list as the `source` argument of the other
        tools (`search`, `deep_search`, `graph_query`, `find_*`, ...).
        """
        lines = [f"Sources ({len(manager.backends)}):"]
        for status in manager.list_sources():
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

    @mcp.tool(annotations=_ANN_REBUILD)
    def update_source_index(
        source: Annotated[str, Field(description="Name of the source to rebuild (see `list_sources`).")],
        force: Annotated[bool, Field(description="If true, rebuild even when no new git commits are detected.")] = False,
    ) -> str:
        """Force a full rebuild of a specific source's index.

        Day-to-day the watcher keeps the index in sync automatically; you rarely need this.
        Use it after a complex merge, a bulk rename, or when drift detection flags a
        critical change (e.g. embedding model swap).

        Do NOT call this routinely — it is expensive and blocks until complete.
        Call `get_rag_status` first to check if a rebuild is actually needed.

        Args:
            source: Name of the source to rebuild. Use `list_sources` to discover names.
            force: If True, rebuild even when no new git commits are detected.
        """
        try:
            manager.update(source, force=force)
            return f"Source {source!r} rebuilt successfully."
        except KeyError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error rebuilding source {source!r}: {str(e)}"

    @mcp.tool(annotations=_ANN_READ)
    def get_rag_status(
        source: Annotated[str | None, Field(description="Source name to inspect (see `list_sources`). Omit to report the status of every configured source.")] = None,
    ) -> str:
        """Report state of the RAG index for one source or all sources.

        Use this to check if the index is up to date before deciding whether to
        call `update_source_index`. Also useful for debugging when search results
        seem stale or incomplete.

        Args:
            source: Specific source name to inspect. If None, returns the
                status of every configured source.
        """
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

    _desc_feedback = (
        "Report that you could not find what you needed. Call this BEFORE giving up "
        "after exhausting search / deep_search / graph_query. The report is appended "
        "to a LOCAL log file on this machine (rag_storage/_feedback/feedback.jsonl) — "
        "it is never uploaded anywhere — and helps the index owner tune sources, "
        "filters, and chunking. "
        "Args: trying_to_do (what you were trying to find or answer); "
        "tried (which tools/queries you already tried); "
        "stuck (where exactly you got blocked or what was missing)."
    )

    @mcp.tool(name="feedback", description=_desc_feedback, annotations=_ANN_FEEDBACK)
    def _feedback(
        trying_to_do: Annotated[str, Field(description="What you were trying to find or answer.")],
        tried: Annotated[str, Field(description="Which tools and queries you already tried.")],
        stuck: Annotated[str, Field(description="Where exactly you got blocked, or what was missing.")],
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
    graph_sources = [
        name for name, b in manager.backends.items()
        if getattr(b, "graph", None) is not None
    ]

    _desc = (
        f"Query the code knowledge graph (call graph + inheritance + imports) of a source. "
        f"Graph-enabled sources: {', '.join(graph_sources)}. `source` may be omitted when only "
        f"one source has the graph layer. Symbol matching is fuzzy (case-insensitive substring) — "
        f"pass an identifier, not a description. Operations: "
        f"'callers' (who calls `symbol`? what breaks if I change it?); "
        f"'callees' (what does `symbol` call / depend on?); "
        f"'subclasses' (who inherits from / implements `symbol`?); "
        f"'superclasses' (what does `symbol` extend / implement?); "
        f"'imports' (import edges of a file — pass a file path substring or a symbol in it); "
        f"'neighbors' (everything around `symbol`, args: relation_filter "
        f"'calls'|'imports'|'imports_from'|'contains', depth 1-6); "
        f"'shortest_path' (call chain from `symbol` to `target`, arg: max_hops); "
        f"'overview' (architectural snapshot: most-connected hubs + communities, "
        f"args: top_n, min_community_size — call once at the start of an unfamiliar session); "
        f"'surprising_connections' (bridge edges by betweenness centrality, arg: top_n); "
        f"'status' (node/edge counts and freshness — use to debug empty results). "
        f"Results include file+line so you can cite them."
    )

    @mcp.tool(name="graph_query", description=_desc, annotations=_ANN_READ)
    def _graph_query(
        operation: Annotated[str, Field(description="Graph operation to run, e.g. callers, callees, subclasses, superclasses, imports, neighbors, shortest_path, overview, surprising_connections, status. See the tool description for the full list and which require `symbol`.")],
        source: _SourceArg = None,
        symbol: Annotated[str | None, Field(description="The symbol the operation acts on (required for callers/callees/subclasses/superclasses/imports/neighbors/shortest_path).")] = None,
        target: Annotated[str | None, Field(description="Destination symbol for `shortest_path` (the path runs from `symbol` to `target`).")] = None,
        relation_filter: Annotated[str | None, Field(description="For `neighbors`: restrict to one edge relation, e.g. 'calls', 'inherits', 'imports'.")] = None,
        depth: Annotated[int, Field(description="For `neighbors`: how many hops out to traverse.")] = 1,
        limit: Annotated[int, Field(description="Maximum number of edges/results to return.")] = 50,
        max_hops: Annotated[int, Field(description="For `shortest_path`: maximum path length to search.")] = 8,
        top_n: Annotated[int, Field(description="For `overview` / `surprising_connections`: how many top items to return.")] = 10,
        min_community_size: Annotated[int, Field(description="For `overview`: minimum size of a detected community/cluster.")] = 3,
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
    codebase_sources = [
        name for name, b in manager.backends.items() if _is_codebase(b)
    ]
    src_hint = (
        f"Codebase sources: {', '.join(codebase_sources)}. `source` may be "
        f"omitted when only one codebase source is configured."
    )

    _desc_find_def = (
        f"Find where a symbol is DEFINED. Uses the graph layer when enabled "
        f"(precise file+line from the AST), falls back to BM25 search otherwise. "
        f"Each result carries `source` ('graph' or 'search_bm25') so you can "
        f"communicate confidence. Use for 'where is X declared?', 'show me the "
        f"implementation of X'. {src_hint} "
        f"Args: symbol (identifier name); source; limit (max results, default 10)."
    )

    @mcp.tool(name="find_definition", description=_desc_find_def, annotations=_ANN_READ)
    def _find_def(
        symbol: Annotated[str, Field(description="Identifier to locate the definition of, e.g. `MyClass` or `MyClass.handleClick`.")],
        source: _SourceArg = None,
        limit: Annotated[int, Field(description="Maximum number of results to return.")] = 10,
    ) -> str:
        try:
            src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
            results = manager.find_definition(src, symbol, limit=limit)
            return _format_definition_results(symbol, results)
        except Exception as e:
            return f"Error: {e}"

    _desc_find_usages = (
        f"Find every USE of a symbol: calls (from graph if enabled) AND non-call "
        f"references (typeof, generics, decorators, imports, doc mentions) via "
        f"textual search. Excludes the definition itself. Deduped by (file, line). "
        f"Use for 'who uses X?', 'what breaks if I change X?'. {src_hint} "
        f"Args: symbol (identifier name); source; limit (max results, default 50)."
    )

    @mcp.tool(name="find_usages", description=_desc_find_usages, annotations=_ANN_READ)
    def _find_usages(
        symbol: Annotated[str, Field(description="Identifier to find all uses of (calls plus textual references).")],
        source: _SourceArg = None,
        limit: Annotated[int, Field(description="Maximum number of results to return.")] = 50,
    ) -> str:
        try:
            src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
            results = manager.find_usages(src, symbol, limit=limit)
            return _format_usage_results(symbol, results)
        except Exception as e:
            return f"Error: {e}"

    _desc_find_tests = (
        f"Find tests that mention a symbol. Default pattern matches conventional "
        f"test paths: `/tests/`, `/test/`, `/spec/`, `/__tests__/`, `_test.py`, "
        f"`_test.go`, `.test.{{js,ts}}`, `.spec.{{js,ts}}`, `*Test.cs`, `*Tests.cs`. "
        f"Pass a custom regex via `test_path_pattern` for non-standard layouts. "
        f"Use for 'are there tests for X?', 'how is X tested?'. {src_hint} "
        f"Args: symbol; source; limit (default 20); test_path_pattern (optional regex)."
    )

    @mcp.tool(name="find_tests_for", description=_desc_find_tests, annotations=_ANN_READ)
    def _find_tests(
        symbol: Annotated[str, Field(description="Identifier to find tests for.")],
        source: _SourceArg = None,
        limit: Annotated[int, Field(description="Maximum number of results to return.")] = 20,
        test_path_pattern: Annotated[str | None, Field(description="Custom regex for test file paths, overriding the default conventional patterns.")] = None,
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

    _desc_find_similar = (
        f"Find code structurally / semantically similar to a given snippet. "
        f"Pure dense (semantic) search — BM25 would just bring back chunks that "
        f"share identifiers, not the same. Truncates snippets longer than 2000 "
        f"chars. Excludes chunks that are byte-identical to the input. "
        f"Use for 'before I write this function, does something similar exist?'. "
        f"{src_hint} Args: snippet (the code block); source; top_k (max results, default 10)."
    )

    @mcp.tool(name="find_similar", description=_desc_find_similar, annotations=_ANN_READ)
    def _find_similar(
        snippet: Annotated[str, Field(description="Code block to find structurally / semantically similar code to (truncated above 2000 chars).")],
        source: _SourceArg = None,
        top_k: Annotated[int, Field(description="Maximum number of results to return.")] = 10,
    ) -> str:
        try:
            src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
            results = manager.find_similar(src, snippet, top_k=top_k)
            return _format_similar_results(results)
        except Exception as e:
            return f"Error: {e}"

    _desc_describe = (
        f"One-shot context for a symbol — DEFINITION + CALLED BY + CALLS + TESTS "
        f"in a single call, instead of running find_definition, graph callers/callees, "
        f"and find_tests_for separately. The fastest way to understand an unfamiliar "
        f"function/class before changing it: where it lives, who depends on it (blast "
        f"radius), what it depends on, and how it's tested. Call data (CALLED BY / CALLS) "
        f"comes from the graph layer and is present only when it's enabled for the source; "
        f"definition + tests always work. {src_hint} "
        f"Args: symbol (identifier name); source; callers_limit (default 10); "
        f"callees_limit (default 10); tests_limit (default 5)."
    )

    @mcp.tool(name="describe_symbol", description=_desc_describe, annotations=_ANN_READ)
    def _describe_symbol(
        symbol: Annotated[str, Field(description="Identifier to describe, e.g. `MyClass` or `MyClass.handleClick`.")],
        source: _SourceArg = None,
        callers_limit: Annotated[int, Field(description="Max 'called by' (caller) edges to include.")] = 10,
        callees_limit: Annotated[int, Field(description="Max 'calls' (callee) edges to include.")] = 10,
        tests_limit: Annotated[int, Field(description="Max test references to include.")] = 5,
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

    _desc_impact = (
        f"Blast radius of changing a symbol: every function that reaches it "
        f"TRANSITIVELY through the call graph (with hop distance), plus the tests "
        f"that exercise it. Answers 'if I change X, what could break and what should "
        f"I re-run?' — broader than find_usages (direct refs only) and symbol-scoped "
        f"unlike search_diff (branch-scoped). Transitive callers need the graph layer; "
        f"tests resolve via search regardless. {src_hint} "
        f"Args: symbol; source; max_depth (call-graph hops, default 3, max 6); "
        f"tests_limit (default 10)."
    )

    @mcp.tool(name="impact", description=_desc_impact, annotations=_ANN_READ)
    def _impact(
        symbol: Annotated[str, Field(description="Identifier whose change-impact (transitive callers + tests) to compute.")],
        source: _SourceArg = None,
        max_depth: Annotated[int, Field(description="How many call-graph hops to walk outward (1-6).")] = 3,
        tests_limit: Annotated[int, Field(description="Max test references to include.")] = 10,
    ) -> str:
        try:
            src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
            d = manager.impact_of(src, symbol, max_depth=max_depth, tests_limit=tests_limit)
            return _format_impact(symbol, d)
        except Exception as e:
            return f"Error: {e}"

    _desc_overview = (
        f"Orientation map for a codebase — the 'what is this and where do I start' answer "
        f"when you land in an unfamiliar repo: detected languages (by file count), "
        f"frameworks, manifest files, likely entry points (main/CLI/server), and suggested "
        f"build/test/run commands. Pure filesystem scan; no graph or index needed. "
        f"Call it ONCE at the start of an unfamiliar session. {src_hint} Args: source."
    )

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
        _desc_module = (
            f"High-level summary of a FILE: the public symbols it defines, what it imports, "
            f"and which other files depend on it (via the call graph). Lets you grasp a unit "
            f"without reading the whole thing — ideal before editing it. {src_hint} "
            f"Args: file (path or path fragment, e.g. 'VoxelWorld.cs'); source; limit "
            f"(max symbols, default 200)."
        )

        @mcp.tool(name="module_summary", description=_desc_module, annotations=_ANN_READ)
        def _module_summary(
            file: Annotated[str, Field(description="File path or path fragment to summarize, e.g. `src/foo.py` or `VoxelWorld.cs`.")],
            source: _SourceArg = None,
            limit: Annotated[int, Field(description="Max defined symbols to list.")] = 200,
        ) -> str:
            try:
                src = _resolve_source(manager, source, predicate=_is_codebase, kind="codebase source")
                d = manager.module_summary(src, file, limit=limit)
                return _format_module_summary(file, d)
            except Exception as e:
                return f"Error: {e}"

        _desc_export_graph = (
            f"Render a SHAREABLE, self-contained graph view as a single offline file "
            f"(no server, no internet) — for a human to look at, attach to a PR, or archive. "
            f"mode='symbol' draws a symbol's blast radius (callers above, callees below); "
            f"mode='module' draws a file as a hub (imports + dependents). WRITES the file to "
            f"the configured reports dir unless `out` is given, and returns the path. {src_hint} "
            f"Args: target (symbol name for mode=symbol, file path/fragment for mode=module); "
            f"mode ('symbol'|'module', default symbol); source; depth (hops, symbol mode, "
            f"default 2); out (output file path, optional)."
        )

        @mcp.tool(name="export_graph", description=_desc_export_graph, annotations=_ANN_WRITE_FILE)
        def _export_graph(
            target: Annotated[str, Field(description="Symbol name (mode=symbol) or file path/fragment (mode=module) to render.")],
            mode: Annotated[str, Field(description="'symbol' (blast radius) or 'module' (file hub).")] = "symbol",
            source: _SourceArg = None,
            depth: Annotated[int, Field(description="Call-graph hops for symbol mode (1-6).")] = 2,
            out: Annotated[str | None, Field(description="Output file path; defaults to the configured reports dir.")] = None,
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
        _desc_search_diff = (
            f"Search restricted to files added/modified vs a base branch. "
            f"Default base is auto-detected (`main`, then `master`, then `develop`); "
            f"pass `base=` explicitly to override. "
            f"Killer for code review: 'I changed the discount logic; what else uses "
            f"the same formula?' — only chunks from files YOU edited in this branch. "
            f"Returns base + modified_files list + hits. Excludes deleted files. "
            f"Git-enabled sources: {', '.join(git_sources)}. `source` may be omitted "
            f"when only one qualifies. Args: query (natural language); source; "
            f"base (optional branch name); top_k (max hits, default 8)."
        )

        @mcp.tool(name="search_diff", description=_desc_search_diff, annotations=_ANN_READ)
        def _search_diff(
            query: Annotated[str, Field(description="Natural-language description of the behavior to find, restricted to files changed vs the base branch.")],
            source: _SourceArg = None,
            base: Annotated[str | None, Field(description="Base branch to diff against. Defaults to the auto-detected `main` / `master` / `develop`.")] = None,
            top_k: Annotated[int, Field(description="Maximum number of hits to return.")] = 8,
        ) -> str:
            try:
                src = _resolve_source(
                    manager, source, predicate=_has_git, kind="git-enabled codebase source"
                )
                out = manager.search_diff(src, query, base=base, top_k=top_k)
            except Exception as e:
                return f"Error: {e}"
            return _format_search_diff(src, out)


def run_server(config_path=None):
    """Boot the MCP server. Blocks on mcp.run() until the client disconnects."""
    config = load_config(config_path=config_path)
    state = {
        "manager": None,
        "ready": threading.Event(),
        "error": None,
    }

    def _load_background():
        try:
            # Decide HF offline mode BEFORE the heavy imports freeze the
            # env flags (see configure_hf_offline for the full rationale).
            from .config import configure_hf_offline
            configure_hf_offline(config)
            from .source_manager import SourceManager
            mgr = SourceManager(config)
            state["manager"] = mgr
        except Exception as e:
            state["error"] = str(e)
            state["ready"].set()
            return

        try:
            mgr.start_watchers()
        except Exception as e:
            print(f"[server] failed to start watchers: {e}", file=sys.stderr)
        state["ready"].set()

    threading.Thread(target=_load_background, daemon=True).start()

    # Wait for the manager to load before opening the JSON-RPC transport.
    # Any spurious print() during model load (llama_index's MockLLM warning,
    # onnxruntime / huggingface_hub logs) lands on stderr because fd 1 is redirected.
    state["ready"].wait(timeout=config.loading_timeout_seconds)

    if state["error"] is not None:
        print(f"[server] FATAL: source manager failed to load: {state['error']}", file=sys.stderr)
        sys.exit(1)
    if state["manager"] is None:
        print(
            "[server] FATAL: source manager did not load within "
            f"{config.loading_timeout_seconds}s (loading_timeout_seconds in config).",
            file=sys.stderr,
        )
        sys.exit(1)

    manager = state["manager"]

    # FastMCP is constructed only now so the handshake `instructions` can
    # embed the live source catalog — every client gets the usage playbook
    # automatically, without installing a rules file.
    mcp = FastMCP("lynx", instructions=_build_instructions(manager))

    # Full playbook as an MCP resource the client can read on demand.
    guide_text = _build_guide(manager)

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

    # Loading phase done: restore fd 1 and start the transport.
    _restore_real_stdout()
    mcp.run()


if __name__ == "__main__":
    # Allow `python -m lynx.server` for ad-hoc invocation.
    # Normal entry point goes through cli:main (`lynx serve`).
    run_server()
