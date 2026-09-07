"""The text every MCP client reads for each tool: descriptions and parameter
descriptions, kept short on purpose.

Tool definitions are sent once per session and then sit in the model's
context for every turn, so each character here is paid for on every
request. The rule of thumb that shaped these strings: say what the tool
answers and the one thing a model tends to get wrong when calling it, and
nothing else. The catalog of sources, the escalation ladder and the score
scale live once in the handshake `instructions` (see `server.py`), and the
long playbook is the `lynx://guide` resource, fetched only when wanted.
Arguments are not re-listed in prose: the input schema already carries
them, with the descriptions below.

Measured against the previous descriptions on the same 17 tools: 32.5k
characters of `tools/list` down to about a third.
"""
from __future__ import annotations

TOOL_DESCRIPTIONS = {
    "search": (
        "Hybrid semantic + lexical search over the indexed code and docs. "
        "Describe what the code does in plain words ('method that clamps camera "
        "zoom'), not an identifier. Returns ranked chunks with file:line, symbol "
        "and score; a hybrid score near 0.03 is a strong match. Omit `source` to "
        "search every source at once. outline=true returns signatures only, for "
        "cheap triage of broad queries."
    ),
    "deep_search": (
        "Escalation for when `search` came back weak or empty: runs 2-4 genuinely "
        "different phrasings of the same need and returns the first set that "
        "passes the quality threshold, or the strongest weak set with a warning. "
        "Slower than `search`; do not start here."
    ),
    "graph_query": (
        "Raw access to the code knowledge graph (calls, inheritance, imports), "
        "for the questions the dedicated tools do not cover: find_usages, impact "
        "and describe_symbol answer the common ones with the results already "
        "shaped. operation: callers | callees | subclasses | superclasses | "
        "imports | neighbors | shortest_path | overview | surprising_connections "
        "| status. `symbol` is matched as a case-insensitive substring; results "
        "carry file:line."
    ),
    "find_definition": (
        "Jump to where a symbol is defined, when you already know its name. "
        "AST-precise when the graph layer is on, BM25 fallback otherwise; each "
        "hit says which. Use `search` instead when you can only describe what "
        "the code does, and describe_symbol when you also want the callers and "
        "the tests in the same call."
    ),
    "find_usages": (
        "Every use of a symbol: calls (from the graph when on) plus textual "
        "references (generics, decorators, imports, docs), definition excluded. "
        "Answers 'who uses X' at one hop; `impact` walks the chain further and "
        "adds the tests."
    ),
    "find_tests_for": (
        "List the tests that mention a symbol, searched under conventional test "
        "paths (tests/, spec/, __tests__/, *_test.*, *.spec.*, *Test.cs, "
        "*Tests.cs); test_path_pattern replaces them for another layout. Use it "
        "when the tests are all you want, before or after a change; "
        "describe_symbol returns the same tests bundled with the definition and "
        "the callers, and impact returns them for a whole blast radius."
    ),
    "find_similar": (
        "Code semantically similar to a snippet you already have, dense search "
        "only: use it before writing a function, to check whether something like "
        "it exists. `search` is the one to call when you can describe the need in "
        "words instead. Snippets are cut at 2000 chars."
    ),
    "describe_symbol": (
        "One-shot context for a symbol: definition, who calls it, what it calls, "
        "and its tests, in a single call. The fastest way to understand a function "
        "or class before changing it, and cheaper than find_definition, find_usages "
        "and find_tests_for one after the other. Call data needs the graph layer; "
        "definition and tests always work."
    ),
    "impact": (
        "Answer 'what breaks if I change this': everything that reaches a symbol "
        "transitively through the call graph, with hop distance, plus the tests "
        "to re-run. Use find_usages for the direct, one-hop answer; use this "
        "before a risky edit, when the indirect callers are the point. "
        "max_depth trades reach for noise: 2 stays close to the change, 6 on a "
        "hub symbol can return most of the codebase. Transitive callers need the "
        "graph layer."
    ),
    "module_summary": (
        "A file as a unit: the symbols it defines, what it imports, and which files "
        "depend on it (via the call graph). Read it before editing a file you don't "
        "know; repo_overview does the same for the whole repository, and "
        "describe_symbol for one symbol inside the file."
    ),
    "repo_overview": (
        "Orientation for an unfamiliar codebase: languages by file count, "
        "frameworks, manifests, likely entry points, and build/test/run commands. "
        "Filesystem scan, no index needed. Call once per session."
    ),
    "export_graph": (
        "Write a self-contained offline HTML view of a symbol's blast radius "
        "(mode=symbol) or a file's imports and dependents (mode=module), for a "
        "human to open or attach to a PR. Returns the file path."
    ),
    "search_diff": (
        "Search only the files added or modified against a base branch "
        "(auto-detected main / master / develop). For code review: what else in my "
        "change uses this formula? Returns the base, the modified files and the hits."
    ),
    "feedback": (
        "Report that the index could not answer you, before giving up. Appended to "
        "a local log (never uploaded) so the index owner can tune sources and "
        "filters."
    ),
    "list_sources": (
        "List which sources exist and what each one is: name, type, path, chunk "
        "count and drift flag. Read from config and metadata, no index opened. "
        "Call it first when you do not know the source names a `source` argument "
        "expects; for how fresh one index is, and whether to rebuild it, use "
        "get_rag_status instead."
    ),
    "get_rag_status": (
        "Index state for one or all sources: freshness, chunk count, last update, "
        "drift. Check it before considering a rebuild."
    ),
    "update_source_index": (
        "Force a full rebuild of a source's index. Expensive and blocking; the "
        "watcher keeps the index current, so use it only after a big merge or a "
        "drift warning."
    ),
}

# Parameter descriptions, keyed "tool.param". Shared parameters use the
# "*.param" key.
PARAM_DESCRIPTIONS = {
    "*.source": "Source name. Omit when only one source applies.",
    "*.sources": "Source name, list of names, or omit for all sources, fused.",
    "*.file_glob": "Restrict to paths matching this glob, e.g. `*.cs`.",
    "*.extensions": "Restrict to these extensions, e.g. ['.cs'].",
    "*.path_contains": "Restrict to paths containing this substring.",
    "*.top_k": "Max results; default from config.",
    "*.limit": "Max results.",

    "search.query": "What the code does, in plain words, not an identifier.",
    "search.outline": "Signatures only, no bodies: cheap triage for broad queries.",

    "deep_search.queries": "2-4 genuinely different phrasings, tried in order.",
    "deep_search.mode": "dense | sparse | hybrid (single source only).",
    "deep_search.min_score": "Quality threshold override.",
    "deep_search.min_results": "Minimum results for a variant to count as strong.",
    "deep_search.return_all_variants": "Include per-variant diagnostics (single source only).",

    "update_source_index.source": "Source to rebuild.",
    "update_source_index.force": "Rebuild even without new git commits.",
    "get_rag_status.source": "Source to inspect; omit for all.",

    "feedback.trying_to_do": "What you were trying to find or answer.",
    "feedback.tried": "Tools and queries you already tried.",
    "feedback.stuck": "Where exactly you got stuck.",

    "graph_query.operation": "One of the operations listed in the tool description.",
    "graph_query.symbol": "Symbol the operation acts on (not needed by overview / status).",
    "graph_query.target": "Destination symbol for shortest_path.",
    "graph_query.relation_filter": "neighbors: calls | inherits | imports | imports_from | contains.",
    "graph_query.depth": "neighbors: hops out (1-6).",
    "graph_query.limit": "Max edges.",
    "graph_query.max_hops": "shortest_path: max path length.",
    "graph_query.top_n": "overview / surprising_connections: items to return.",
    "graph_query.min_community_size": "overview: minimum community size.",

    "find_definition.symbol": "Identifier, e.g. `MyClass` or `MyClass.handleClick`.",
    "find_usages.symbol": "Identifier to find the uses of.",
    "find_tests_for.symbol": "Identifier to find tests for.",
    "find_tests_for.test_path_pattern": "Regex for test file paths, replacing the defaults.",
    "find_similar.snippet": "Code block to match; cut at 2000 chars.",

    "describe_symbol.symbol": "Identifier, e.g. `MyClass` or `MyClass.handleClick`.",
    "describe_symbol.callers_limit": "Max callers.",
    "describe_symbol.callees_limit": "Max callees.",
    "describe_symbol.tests_limit": "Max tests.",

    "impact.symbol": "Identifier whose blast radius to compute.",
    "impact.max_depth": "Call-graph hops to walk (1-6).",
    "impact.tests_limit": "Max tests.",

    "module_summary.file": "File path or fragment, e.g. `VoxelWorld.cs`.",
    "module_summary.limit": "Max symbols to list.",

    "export_graph.target": "Symbol (mode=symbol) or file path / fragment (mode=module).",
    "export_graph.mode": "symbol | module.",
    "export_graph.depth": "Hops for symbol mode (1-6).",
    "export_graph.out": "Output path; default: the reports dir.",

    "search_diff.query": "What the code does, in plain words.",
    "search_diff.base": "Base branch; default auto-detected.",
    "search_diff.top_k": "Max hits.",
}


def desc(tool: str) -> str:
    return TOOL_DESCRIPTIONS[tool]


def param(key: str) -> str:
    return PARAM_DESCRIPTIONS[key]
