"""Tool profiles: which MCP tools a session gets, and why that is a setting.

Every tool definition an MCP server exposes rides along in the client's
context on every turn. Lynx's seventeen tools cost about 8,100 tokens per
session before this module existed, against a saving of roughly 2,400 tokens
per retrieval compared to agentic grep: a session that searched three times
had not yet paid for its own tool list. Most sessions need a handful of tools,
so the tool set is layered:

  core      the five you reach for in an ordinary coding session
  standard  core plus the precise lookups, the escalation search, the graph
            query, the repo orientation and the diff-scoped search (default)
  full      everything, including maintenance and export

A profile is a ceiling on what gets registered, never a promise: a tool that
needs the graph layer or git integration is still registered only when a
source supports it. `include` and `exclude` fine-tune a profile by name.

Standard library only: `config.py` imports this at load time.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

# Every tool `run_server` can register, in the order the docs list them.
ALL_TOOLS: Tuple[str, ...] = (
    "search",
    "deep_search",
    "graph_query",
    "find_definition",
    "find_usages",
    "find_tests_for",
    "find_similar",
    "describe_symbol",
    "impact",
    "module_summary",
    "repo_overview",
    "export_graph",
    "search_diff",
    "feedback",
    "list_sources",
    "get_rag_status",
    "update_source_index",
)

CORE_TOOLS: Tuple[str, ...] = (
    "search",
    "describe_symbol",
    "find_usages",
    "impact",
    "feedback",
)

STANDARD_TOOLS: Tuple[str, ...] = CORE_TOOLS + (
    "find_definition",
    "deep_search",
    "graph_query",
    "repo_overview",
    "search_diff",
)

PROFILES = {
    "core": CORE_TOOLS,
    "standard": STANDARD_TOOLS,
    "full": ALL_TOOLS,
}

DEFAULT_PROFILE = "standard"

# Tools whose registration depends on what the configured sources support.
# Mirrors the gating in `server.run_server`; `tests/test_tool_profiles.py`
# asserts the two agree, so this table cannot drift silently.
_NEEDS_CODEBASE = frozenset({
    "find_definition", "find_usages", "find_tests_for", "find_similar",
    "describe_symbol", "impact", "repo_overview", "module_summary",
    "export_graph", "search_diff",
})
_NEEDS_GRAPH = frozenset({"graph_query", "module_summary", "export_graph"})
_NEEDS_GIT = frozenset({"search_diff"})


class ToolProfileError(ValueError):
    """An unknown profile or tool name in the configuration."""


def validate_profile(profile: str) -> str:
    if profile not in PROFILES:
        raise ToolProfileError(
            f"unknown tool profile {profile!r}; choose one of {', '.join(PROFILES)}"
        )
    return profile


def validate_tool_names(names: Iterable[str], *, what: str) -> Tuple[str, ...]:
    out = []
    for n in names:
        if n not in ALL_TOOLS:
            raise ToolProfileError(
                f"unknown tool {n!r} in tools.{what}; known tools: {', '.join(ALL_TOOLS)}"
            )
        if n not in out:
            out.append(n)
    return tuple(out)


def available_tools(*, has_codebase: bool, has_graph: bool, has_git: bool) -> List[str]:
    """The tools `run_server` would register for these capabilities, before
    any profile is applied."""
    out = []
    for name in ALL_TOOLS:
        if name in _NEEDS_CODEBASE and not has_codebase:
            continue
        if name in _NEEDS_GRAPH and not has_graph:
            continue
        if name in _NEEDS_GIT and not has_git:
            continue
        out.append(name)
    return out


def select_tools(
    available: Iterable[str],
    profile: str = DEFAULT_PROFILE,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> Tuple[List[str], List[str]]:
    """Split `available` into (kept, dropped) for a profile.

    `include` adds tools to the profile, `exclude` removes them; exclude wins
    when a name is in both. Order follows `ALL_TOOLS`.
    """
    validate_profile(profile)
    wanted: Set[str] = set(PROFILES[profile])
    wanted.update(validate_tool_names(include, what="include"))
    wanted.difference_update(validate_tool_names(exclude, what="exclude"))
    avail = set(available)
    kept = [n for n in ALL_TOOLS if n in avail and n in wanted]
    dropped = [n for n in ALL_TOOLS if n in avail and n not in wanted]
    # Anything registered under a name this module does not know is kept:
    # a profile must never silently hide a tool it cannot reason about.
    kept += sorted(avail - set(ALL_TOOLS))
    return kept, dropped
