"""Argument surface for the `lynx` command — argparse construction only.

Split out of `cli.py`, which had grown past 1600 lines with two thirds of
it being flag declarations. Nothing here executes a command: the parser is
data about what the CLI accepts, the handlers live next door, and keeping
them apart means a change to one is readable without scrolling past the
other.

Deliberately free of heavy imports. `_build_parser` runs on EVERY
invocation, `lynx serve` included, so this module must stay as cheap as
`argparse` itself.
"""
from __future__ import annotations

import argparse

from . import __version__


# Graph operations `lynx graph query --op` accepts. Duplicated from
# `lynx.graph.dispatch.OPERATIONS` on purpose: the parser is built on EVERY
# invocation (including `lynx serve`), and importing the graph package to
# read one tuple costs 340ms warm (1.6s cold) measured with -X importtime,
# against 84ms for all of lynx.cli. The bulk is networkx, pulled in by
# graph/builder.py; tree-sitter, via graph/extractor.py, is the rest. Drift
# is caught by test_cli_graph_query.py, which asserts the two lists match.
_GRAPH_OPERATIONS = (
    "callers", "callees", "subclasses", "superclasses", "imports",
    "neighbors", "shortest_path", "overview", "surprising_connections",
    "status",
)

# Source types `lynx source add --type` accepts, mirroring the v2 config
# schema's per-type validators in config.py.
_SOURCE_TYPES = ("codebase", "webdoc", "pdf")


def _add_query_parsers(sub) -> None:
    """Register the retrieval / navigation subcommands.

    Names map 1:1 onto the MCP tools (underscores become hyphens) so the
    tool table in the docs doubles as the CLI reference — nobody should
    have to translate `find_definition` into some other spelling. The
    implementations live in `query_cli.py`; only the argument surface is
    here, next to every other subcommand.
    """
    def _common(p, *, source_help: str, with_json: bool = True):
        p.add_argument("--config", "-c", metavar="PATH")
        p.add_argument("--source", "-s", metavar="NAME", help=source_help)
        if with_json:
            p.add_argument(
                "--json", action="store_true", dest="as_json",
                help="Emit the raw result as JSON instead of text.",
            )
        return p

    _CODE_SRC = ("Codebase source to use. Optional when only one is "
                 "configured.")

    sp_deep = sub.add_parser(
        "deep-search",
        help="Multi-query escalation search: tries each phrasing until one "
             "returns strong results. Use when `search` came back weak.",
    )
    sp_deep.add_argument(
        "queries", nargs="+", metavar="QUERY",
        help="2-4 genuinely different phrasings (different angles, not "
             "paraphrases), tried in order.",
    )
    sp_deep.add_argument("--config", "-c", metavar="PATH")
    sp_deep.add_argument(
        "--source", "-s", metavar="NAME", action="append",
        help="Source to query. Repeat to fuse a subset. Omitted: every source.",
    )
    sp_deep.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit the raw result as JSON instead of text.",
    )
    sp_deep.add_argument("--top-k", "-k", type=int, default=None)
    sp_deep.add_argument(
        "--mode", choices=["hybrid", "dense", "sparse"], default=None,
        help="Retrieval mode override (single-source only).",
    )
    sp_deep.add_argument("--ext", action="append", metavar="EXT")
    sp_deep.add_argument("--glob", metavar="PATTERN")
    sp_deep.add_argument("--path-contains", metavar="SUBSTRING")
    sp_deep.add_argument(
        "--min-score", type=float, default=None,
        help="Override the score a variant must beat to count as strong.",
    )
    sp_deep.add_argument(
        "--min-results", type=int, default=None,
        help="Override how many results a variant needs to count as strong.",
    )
    sp_deep.add_argument(
        "--return-all-variants", action="store_true",
        help="Include per-variant diagnostics (single-source only).",
    )

    sp_def = sub.add_parser(
        "find-definition",
        help="Where is a symbol defined? AST-precise with the graph layer, "
             "BM25 fallback otherwise.",
    )
    sp_def.add_argument("symbol", help="Identifier to locate.")
    _common(sp_def, source_help=_CODE_SRC)
    sp_def.add_argument("--limit", type=int, default=10)

    sp_use = sub.add_parser(
        "find-usages", help="Where is a symbol referenced?",
    )
    sp_use.add_argument("symbol", help="Identifier to look for.")
    _common(sp_use, source_help=_CODE_SRC)
    sp_use.add_argument("--limit", type=int, default=50)

    sp_tests = sub.add_parser(
        "find-tests-for", help="Which tests exercise a symbol?",
    )
    sp_tests.add_argument("symbol", help="Identifier under test.")
    _common(sp_tests, source_help=_CODE_SRC)
    sp_tests.add_argument("--limit", type=int, default=20)
    sp_tests.add_argument(
        "--test-path-pattern", metavar="SUBSTRING",
        help="Restrict to test files whose path contains this (e.g. `/spec/`).",
    )

    sp_sim = sub.add_parser(
        "find-similar",
        help="Find code that resembles a snippet — duplicates, parallel "
             "implementations, the other place that needs the same fix.",
    )
    sp_sim.add_argument(
        "snippet", nargs="?",
        help="Code to match. Omit and use --file for anything multi-line.",
    )
    _common(sp_sim, source_help=_CODE_SRC)
    sp_sim.add_argument(
        "--file", "-f", metavar="PATH",
        help="Read the snippet from a file instead of the command line.",
    )
    sp_sim.add_argument("--top-k", "-k", type=int, default=10)

    sp_desc = sub.add_parser(
        "describe-symbol",
        help="One-shot context: definition + called by + calls + tests.",
    )
    sp_desc.add_argument("symbol", help="Identifier to describe.")
    _common(sp_desc, source_help=_CODE_SRC)
    sp_desc.add_argument("--callers-limit", type=int, default=10)
    sp_desc.add_argument("--callees-limit", type=int, default=10)
    sp_desc.add_argument("--tests-limit", type=int, default=5)

    sp_imp = sub.add_parser(
        "impact",
        help="Blast radius: everything that transitively reaches a symbol, "
             "plus the tests to re-run.",
    )
    sp_imp.add_argument("symbol", help="Identifier whose impact to compute.")
    _common(sp_imp, source_help=_CODE_SRC)
    sp_imp.add_argument("--max-depth", type=int, default=3,
                        help="Call-graph hops to walk outward (1-6).")
    sp_imp.add_argument("--tests-limit", type=int, default=10)

    sp_ovw = sub.add_parser(
        "repo-overview",
        help="Orientation map: languages, frameworks, entry points.",
    )
    _common(sp_ovw, source_help=_CODE_SRC)

    sp_mod = sub.add_parser(
        "module-summary",
        help="What a file contains and how it connects: symbols, imports, "
             "dependents.",
    )
    sp_mod.add_argument("file", metavar="FILE",
                        help="File path (or a symbol inside it).")
    _common(sp_mod, source_help=_CODE_SRC)
    sp_mod.add_argument("--limit", type=int, default=40)

    sp_diff = sub.add_parser(
        "search-diff",
        help="Search only the files you changed vs a base branch.",
    )
    sp_diff.add_argument("query", help="Natural-language search query.")
    _common(sp_diff, source_help="Git-enabled codebase source.")
    sp_diff.add_argument(
        "--base", metavar="REF",
        help="Base branch. Default: auto-detected main / master / develop.",
    )
    sp_diff.add_argument("--top-k", "-k", type=int, default=8)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lynx",
        description="Lynx — self-hosted MCP server with semantic + lexical "
                    "search over your local code and documentation. Multi-source, "
                    "100 percent local, no data egress.",
    )
    parser.add_argument(
        "--version", action="version", version=f"lynx {__version__}"
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sp_serve = sub.add_parser("serve", help="Run the MCP server (default)")
    sp_serve.add_argument("--config", "-c", metavar="PATH")
    sp_serve.add_argument(
        "--profile", choices=["core", "standard", "full"],
        help="Which tools to expose: core (5), standard (10, default) or full "
             "(17). Overrides LYNX_TOOL_PROFILE and tools.profile in config.json.",
    )

    sp_build = sub.add_parser("build", help="Force a full rebuild of a source's index")
    sp_build.add_argument("--config", "-c", metavar="PATH")
    sp_build.add_argument(
        "--source", "-s", metavar="NAME",
        help="Source to rebuild. Optional when only one source is configured.",
    )

    sp_reset = sub.add_parser(
        "reset",
        help="Wipe a source's index and rebuild from scratch (fixes a corrupt index)",
    )
    sp_reset.add_argument("--config", "-c", metavar="PATH")
    sp_reset.add_argument(
        "--source", "-s", metavar="NAME",
        help="Source to reset. Optional when only one source is configured.",
    )
    sp_reset.add_argument(
        "--all", action="store_true", help="Reset every configured source.",
    )
    sp_reset.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt.",
    )
    sp_reset.add_argument(
        "--no-rebuild", action="store_true",
        help="Only wipe the index; don't rebuild (a later build/launch will).",
    )

    sp_search = sub.add_parser("search", help="Run an ad-hoc search query")
    sp_search.add_argument("query", help="Natural-language search query")
    sp_search.add_argument("--config", "-c", metavar="PATH")
    sp_search.add_argument(
        "--source", "-s", metavar="NAME", action="append",
        help="Source to query. Optional when only one source is configured. "
             "Repeat to fuse a subset (-s api -s docs); use 'ALL' for every "
             "source.",
    )
    sp_search.add_argument("--top-k", "-k", type=int, default=None)
    sp_search.add_argument(
        "--mode", choices=["hybrid", "dense", "sparse"], default=None,
        help="Per-call retrieval mode override (only used for direct search, "
             "not cross-source).",
    )
    sp_search.add_argument("--ext", action="append", metavar="EXT")
    sp_search.add_argument("--glob", metavar="PATTERN")
    sp_search.add_argument("--path-contains", metavar="SUBSTRING")
    sp_search.add_argument(
        "--outline", action="store_true",
        help="Return signatures + first doc line instead of full bodies — "
             "cheap triage for a broad query or a large --top-k.",
    )
    sp_search.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit the raw hits as JSON instead of text (for scripts).",
    )

    _add_query_parsers(sub)

    sp_status = sub.add_parser("status", help="Show RAG status per source")
    sp_status.add_argument("--config", "-c", metavar="PATH")
    sp_status.add_argument(
        "--source", "-s", metavar="NAME",
        help="Inspect only this source. Default: all sources.",
    )

    sp_list = sub.add_parser("list-sources", help="Enumerate configured sources")
    sp_list.add_argument("--config", "-c", metavar="PATH")

    sp_graph = sub.add_parser(
        "graph",
        help="Build, query, inspect, or export the per-source knowledge "
             "graph layer (opt-in via `graph: { enabled: true }` in the "
             "source config).",
    )
    graph_sub = sp_graph.add_subparsers(dest="graph_command", metavar="GRAPH_COMMAND")

    sp_graph_build = graph_sub.add_parser(
        "build", help="Rebuild the graph layer for a source"
    )
    sp_graph_build.add_argument("--config", "-c", metavar="PATH")
    sp_graph_build.add_argument(
        "--source", "-s", metavar="NAME",
        help="Source whose graph to rebuild. Optional when only one source "
             "has the graph layer enabled.",
    )
    sp_graph_build.add_argument(
        "--force", action="store_true",
        help="Wipe state and rebuild from scratch (default: SHA-incremental).",
    )

    sp_graph_status = graph_sub.add_parser(
        "status", help="Show graph layer status (nodes/edges/by-language/...)"
    )
    sp_graph_status.add_argument("--config", "-c", metavar="PATH")
    sp_graph_status.add_argument(
        "--source", "-s", metavar="NAME",
        help="Show status for a single source. Default: all sources with "
             "the graph layer enabled.",
    )

    sp_graph_query = graph_sub.add_parser(
        "query",
        help="Ask the graph a question (callers, callees, subclasses, "
             "shortest_path, overview, ...) — the CLI face of the "
             "`graph_query` MCP tool.",
    )
    sp_graph_query.add_argument("--config", "-c", metavar="PATH")
    sp_graph_query.add_argument(
        "--source", "-s", metavar="NAME",
        help="Source to query. Optional when only one has the graph layer.",
    )
    sp_graph_query.add_argument(
        "--op", "--operation", dest="op", metavar="OP", required=True,
        choices=list(_GRAPH_OPERATIONS),
        help="Operation to run: " + " | ".join(_GRAPH_OPERATIONS) + ".",
    )
    sp_graph_query.add_argument(
        "--symbol", metavar="NAME",
        help="Symbol the operation acts on (required for callers, callees, "
             "subclasses, superclasses, imports, neighbors, shortest_path). "
             "Matching is fuzzy: case-insensitive substring.",
    )
    sp_graph_query.add_argument(
        "--target", metavar="NAME",
        help="Destination symbol for --op shortest_path.",
    )
    sp_graph_query.add_argument(
        "--relation", metavar="REL", dest="relation_filter",
        help="For --op neighbors: keep only this edge relation "
             "(calls | imports | imports_from | contains | inherits).",
    )
    sp_graph_query.add_argument(
        "--depth", type=int, default=1,
        help="For --op neighbors: hops to traverse (default 1).",
    )
    sp_graph_query.add_argument(
        "--limit", type=int, default=50,
        help="Maximum edges/results to return (default 50).",
    )
    sp_graph_query.add_argument(
        "--max-hops", type=int, default=8,
        help="For --op shortest_path: longest path to search (default 8).",
    )
    sp_graph_query.add_argument(
        "--top-n", type=int, default=10,
        help="For --op overview / surprising_connections (default 10).",
    )
    sp_graph_query.add_argument(
        "--min-community-size", type=int, default=3,
        help="For --op overview: smallest cluster to report (default 3).",
    )
    sp_graph_query.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit the raw result (edges, path, counts) as JSON instead of "
             "the text an AI client would receive.",
    )

    sp_graph_export = graph_sub.add_parser(
        "export",
        help="Write a self-contained graph view (single offline .html) for a "
             "symbol's blast radius or a file's dependencies.",
    )
    sp_graph_export.add_argument("--config", "-c", metavar="PATH")
    sp_graph_export.add_argument(
        "--source", "-s", metavar="NAME",
        help="Source to read. Optional when only one has the graph layer.",
    )
    grp = sp_graph_export.add_mutually_exclusive_group(required=True)
    grp.add_argument("--symbol", metavar="NAME", help="Render a symbol's blast radius.")
    grp.add_argument("--module", metavar="FILE", help="Render a file's import/dependent hub.")
    sp_graph_export.add_argument(
        "--depth", type=int, default=2, help="Call-graph hops for --symbol (1-6, default 2).",
    )
    sp_graph_export.add_argument(
        "--out", "-o", metavar="PATH",
        help="Output file. Default: <reports_path or storage/reports>/<name>.html",
    )

    # --------------------------------------------------------------
    # `lynx source <cmd>` — add/remove sources without the web UI.
    # Writes config.json through the same validator the UI uses, so a
    # headless or scripted install never needs a browser.
    # --------------------------------------------------------------
    sp_source = sub.add_parser(
        "source",
        help="Add or remove a source in config.json (the CLI equivalent of "
             "the web UI's guided form).",
    )
    source_sub = sp_source.add_subparsers(dest="source_command", metavar="SOURCE_COMMAND")

    sp_src_add = source_sub.add_parser(
        "add",
        help="Add a source. The config is validated before it is written; "
             "on failure the existing file is left untouched.",
    )
    sp_src_add.add_argument("name", help="Name for the new source (letters, digits, underscore).")
    sp_src_add.add_argument("--config", "-c", metavar="PATH")
    sp_src_add.add_argument(
        "--type", "-t", choices=list(_SOURCE_TYPES), dest="source_type",
        help="Source type. Required unless --block is used.",
    )
    sp_src_add.add_argument(
        "--block", metavar="JSON", dest="block_json",
        help="Raw JSON source block, bypassing the per-type flags below. "
             "Escape hatch for fields the flags don't cover. Codebase blocks "
             "still receive the default ignore list unless they set "
             "`ignored_path_fragments` themselves.",
    )
    sp_src_add.add_argument(
        "--build", action="store_true",
        help="Index the source immediately after adding it "
             "(equivalent to a following `lynx build --source NAME`).",
    )
    sp_src_add.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Report the outcome as JSON instead of text (for scripts).",
    )

    g_code = sp_src_add.add_argument_group("codebase options")
    g_code.add_argument("--path", metavar="DIR", help="Directory to index (codebase, pdf).")
    g_code.add_argument(
        "--ext", action="append", metavar="EXT",
        help="File extension to index, repeatable (e.g. --ext .cs --ext .py). "
             "Omitted: the folder is scanned and its top extensions are used.",
    )
    g_code.add_argument(
        "--ignore", action="append", metavar="FRAGMENT",
        help="Path fragment to skip, repeatable (e.g. --ignore /Library/). "
             "Omitted: the standard VCS/deps/build ignore list is applied.",
    )
    g_code.add_argument(
        "--graph", action="store_true", help="Enable the knowledge graph layer.",
    )
    g_code.add_argument(
        "--no-watcher", action="store_true",
        help="Don't watch the folder for changes (watcher is on by default "
             "for codebase sources, off for pdf).",
    )
    g_code.add_argument(
        "--watcher-debounce", type=float, metavar="SECONDS",
        help="Seconds to coalesce rapid edits before re-indexing.",
    )
    g_code.add_argument(
        "--no-git", action="store_true",
        help="Disable git integration (commit-based staleness detection).",
    )

    g_web = sp_src_add.add_argument_group("webdoc options")
    g_web.add_argument("--url", metavar="URL", help="Crawl starting point (webdoc).")
    g_web.add_argument("--max-depth", type=int, metavar="N", help="Crawl depth (default 3).")
    g_web.add_argument("--max-pages", type=int, metavar="N", help="Page cap (default 500).")
    g_web.add_argument(
        "--request-delay", type=float, metavar="SECONDS",
        help="Politeness delay between requests (default 0.5).",
    )
    g_web.add_argument(
        "--include-url", action="append", metavar="PATTERN",
        help="Only crawl URLs matching this pattern, repeatable.",
    )
    g_web.add_argument(
        "--exclude-url", action="append", metavar="PATTERN",
        help="Skip URLs matching this pattern, repeatable.",
    )
    g_web.add_argument(
        "--render-js", action="store_true",
        help="Render pages with headless Chromium (needs the `webdoc-js` extra).",
    )
    g_web.add_argument(
        "--allow-cross-origin", action="store_true",
        help="Follow links off the starting host (default: same origin only).",
    )

    g_pdf = sp_src_add.add_argument_group("pdf options")
    g_pdf.add_argument(
        "--no-recursive", action="store_true",
        help="Only read PDFs directly in --path, not in sub-folders.",
    )
    g_pdf.add_argument(
        "--file-glob", metavar="PATTERN", help="PDF match pattern (default `**/*.pdf`).",
    )
    g_pdf.add_argument(
        "--extractor", choices=["auto", "pypdf", "pymupdf"],
        help="PDF text extraction backend (default auto).",
    )
    g_pdf.add_argument(
        "--watcher", action="store_true",
        help="Watch the folder for new/changed PDFs (off by default: "
             "re-extraction costs 10-30s per file).",
    )

    sp_src_rm = source_sub.add_parser(
        "remove",
        help="Remove a source from config.json. The index on disk is kept "
             "unless --purge is given.",
    )
    sp_src_rm.add_argument("name", help="Source to remove.")
    sp_src_rm.add_argument("--config", "-c", metavar="PATH")
    sp_src_rm.add_argument(
        "--purge", action="store_true",
        help="Also delete the source's index directory under storage_path.",
    )
    sp_src_rm.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt.",
    )
    sp_src_rm.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Report the outcome as JSON instead of text (for scripts).",
    )

    # --------------------------------------------------------------
    # `lynx manager <cmd>` — LynxManager (setup wizard, doctor, install,
    # web UI). All four sub-commands lazy-import the manager package so
    # `lynx serve` doesn't pay the FastAPI / huggingface_hub import cost
    # when only the MCP server is needed.
    # --------------------------------------------------------------
    sp_manager = sub.add_parser(
        "manager",
        help="LynxManager: interactive setup wizard, diagnostic, "
             "extras/model installer, and local web UI.",
    )
    manager_sub = sp_manager.add_subparsers(
        dest="manager_command", metavar="MANAGER_COMMAND",
    )

    sp_mgr_init = manager_sub.add_parser(
        "init",
        help="Bootstrap a fresh Lynx install: write a default config.json "
             "and pre-download the embedding model. Sources are added "
             "afterwards via `lynx manager ui` (guided form).",
    )
    sp_mgr_init.add_argument(
        "--output", "-o", metavar="PATH", default="config.json",
        help="Where to write the generated config (default ./config.json).",
    )
    sp_mgr_init.add_argument(
        "--non-interactive", action="store_true",
        help="Skip prompts; overwrite any existing config; do not offer to "
             "launch the UI. Useful in scripts / CI.",
    )
    sp_mgr_init.add_argument(
        "--skip-model-download", action="store_true",
        help="Don't pre-download the embedding model. It will be fetched "
             "lazily on the first `lynx serve` query instead.",
    )

    sp_mgr_doctor = manager_sub.add_parser(
        "doctor",
        help="Run diagnostic checks: HF cache, drift, paths, deps, "
             "watcher health. Exit code = 0 ok, 1 warn, 2 error.",
    )
    sp_mgr_doctor.add_argument("--config", "-c", metavar="PATH")
    sp_mgr_doctor.add_argument(
        "--json", action="store_true",
        help="Output results as JSON instead of colored text.",
    )
    sp_mgr_doctor.add_argument(
        "--heal-wal", metavar="SOURCE",
        help="Surgically heal SOURCE's wedged index WAL (writes left stuck "
             "by a killed process make ChromaDB hang on every open). Purges "
             "the stuck writes and queues the affected files for re-indexing "
             "— no full rebuild. Stop all Lynx processes first.",
    )
    sp_mgr_doctor.add_argument(
        "--heal-coverage", metavar="SOURCE",
        help="Re-queue SOURCE's files that the SHA cache lists as indexed but "
             "that are absent from the index (a failed insert used to be "
             "recorded as success, so those files were skipped forever and "
             "were invisible to search). Drops just those cache entries — no "
             "full rebuild. Stop all Lynx processes first.",
    )

    sp_mgr_install = manager_sub.add_parser(
        "install",
        help="Manage optional extras (pip) and HuggingFace model downloads.",
    )
    install_group = sp_mgr_install.add_mutually_exclusive_group()
    install_group.add_argument(
        "--list", action="store_true",
        help="List available optional extras and which are installed.",
    )
    install_group.add_argument(
        "--model", metavar="MODEL_NAME", nargs="?", const="__default__",
        help="Download a HuggingFace model into the local cache. With no "
             "value, downloads the embedding model from the active config.",
    )
    install_group.add_argument(
        "--from-archive", metavar="PATH_OR_URL",
        help="Import a model archive into the local HF cache (offline / "
             "air-gapped). Accepts a local path or a direct http(s) URL that "
             "serves the file with no auth/interstitial (e.g. a public GitHub "
             "Release asset). The archive must contain the `models--ORG--NAME/` "
             "layout produced by --export-archive.",
    )
    install_group.add_argument(
        "--export-archive", metavar="PATH",
        help="Zip the cached model's HF directory to PATH so it can be shared "
             "(copy it to the offline machine, or host it as a public download) "
             "and imported elsewhere with --from-archive.",
    )
    install_group.add_argument(
        "extra", nargs="?", metavar="EXTRA",
        help="Optional extra to install via pip (e.g. `pdf-fast`). "
             "Equivalent to `pip install lynx[<extra>]`.",
    )
    sp_mgr_install.add_argument(
        "--with-reranker", action="store_true",
        help="When used with --model, also download the reranker model.",
    )
    sp_mgr_install.add_argument(
        "--model-name", metavar="NAME",
        help="Model the archive is for (used with --from-archive / "
             "--export-archive). Defaults to the config's embedding model.",
    )
    sp_mgr_install.add_argument(
        "--config", "-c", metavar="PATH",
        help="Config file to read for model name detection (used by --model).",
    )

    sp_mgr_feedback = manager_sub.add_parser(
        "feedback",
        help="Summarize the local feedback log (reports agents filed when the "
             "index couldn't answer). Read-only; nothing leaves your machine.",
    )
    sp_mgr_feedback.add_argument("--config", "-c", metavar="PATH")
    sp_mgr_feedback.add_argument(
        "--limit", type=int, default=10,
        help="How many recent reports to show (default 10).",
    )
    sp_mgr_feedback.add_argument(
        "--json", action="store_true",
        help="Output the summary as JSON instead of colored text.",
    )

    sp_mgr_ui = manager_sub.add_parser(
        "ui",
        help="Launch the local web UI (FastAPI + HTMX). Listens only on "
             "127.0.0.1; opens your browser automatically.",
    )
    sp_mgr_ui.add_argument("--config", "-c", metavar="PATH")
    sp_mgr_ui.add_argument(
        "--port", type=int, default=8765,
        help="Port to listen on (default 8765, falls back to next free).",
    )
    sp_mgr_ui.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default 127.0.0.1 — localhost-only by design).",
    )
    sp_mgr_ui.add_argument(
        "--no-browser", action="store_true",
        help="Don't open the browser automatically.",
    )

    sp_mig = sub.add_parser(
        "migrate-config",
        help="Convert a v1 config.json to the v2 schema",
    )
    sp_mig.add_argument(
        "--input", "-i", metavar="PATH", required=True,
        help="v1 config.json to read",
    )
    sp_mig.add_argument(
        "--output", "-o", metavar="PATH",
        help="Where to write the v2 config. Default: alongside input as "
             "<input>.v2.json",
    )
    sp_mig.add_argument(
        "--source-name", default="codebase",
        help="Name to give the migrated source in the v2 schema (default: 'codebase')",
    )

    return parser
