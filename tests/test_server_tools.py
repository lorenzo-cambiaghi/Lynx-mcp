"""Unit tests for the fixed MCP tool surface in server.py.

The core property under test: the number of registered tools is CONSTANT
in the number of sources. The per-source naming scheme this replaced
(~17 tools per source) blew client tool limits as soon as a user added a
second or third source.

Uses a fake manager (no embedding model, no ChromaDB) — these tests are
about registration and routing, not retrieval quality.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mcp.server.fastmcp import FastMCP

from lynx.server import (
    _build_guide,
    _build_instructions,
    _register_search_tools,
    _register_global_tools,
    _register_graph_tools,
    _register_combined_tools,
    _resolve_source,
)


class FakeBackend:
    def __init__(self, type_name="codebase", graph=None, git=False, path="/x"):
        self.type_name = type_name
        self.graph = graph
        self.source_config = {
            "path": path,
            "git_integration": {"enabled": git},
        }


class FakeManager:
    """Records calls; returns canned results shaped like the real ones."""

    def __init__(self, backends, storage_path="/tmp/lynx-test-storage"):
        self.backends = backends
        self.config = SimpleNamespace(
            search=SimpleNamespace(default_top_k=5),
            storage_path=storage_path,
        )
        self.calls = []

    def get(self, source):
        if source not in self.backends:
            raise KeyError(f"Unknown source {source!r}")
        return self.backends[source]

    _HIT = {
        "file": "a.py", "symbol_name": "add", "symbol_kind": "function_definition",
        "language": "python", "start_line": 1, "end_line": 3, "score": 0.9,
        "content": 'def add(a, b):\n    """Add a and b."""\n    return a + b',
    }

    def search(self, source, query, top_k=5, **kw):
        self.calls.append(("search", source, query))
        return [dict(self._HIT)]

    def search_all(self, query, top_k=5, **kw):
        self.calls.append(("search_all", query))
        return [dict(self._HIT, source="one")]

    def deep_search(self, source, queries, top_k=5, **kw):
        self.calls.append(("deep_search", source))
        return {
            "results": [], "variants_tried": len(queries),
            "winning_variant_index": None, "all_weak": True,
        }

    def deep_search_all(self, queries, top_k=5, **kw):
        self.calls.append(("deep_search_all",))
        return {
            "results": [], "variants_tried": len(queries),
            "winning_variant_index": None, "all_weak": True,
        }

    def find_definition(self, source, symbol, limit=10):
        self.calls.append(("find_definition", source, symbol))
        return []


def _register_everything(manager):
    mcp = FastMCP("test")
    _register_search_tools(mcp, manager)
    _register_global_tools(mcp, manager)
    if any(b.graph is not None for b in manager.backends.values()):
        _register_graph_tools(mcp, manager)
    if any(b.type_name == "codebase" for b in manager.backends.values()):
        has_graph = any(b.graph is not None for b in manager.backends.values())
        _register_combined_tools(mcp, manager, has_graph=has_graph)
    return mcp


def _tools(mcp):
    return mcp._tool_manager._tools


def test_tool_count_constant_in_number_of_sources():
    one = FakeManager({"a": FakeBackend(graph=object(), git=True)})
    three = FakeManager({
        "a": FakeBackend(graph=object(), git=True),
        "b": FakeBackend(graph=object(), git=True),
        "c": FakeBackend(type_name="webdoc"),
    })
    n_one = len(_tools(_register_everything(one)))
    n_three = len(_tools(_register_everything(three)))
    assert n_one == n_three, "tool count must not grow with sources"
    assert n_three <= 17, f"tool surface too large: {n_three}"


def test_expected_fixed_tool_names():
    mgr = FakeManager({"a": FakeBackend(graph=object(), git=True)})
    names = set(_tools(_register_everything(mgr)))
    assert names == {
        "search", "deep_search", "list_sources", "update_source_index",
        "get_rag_status", "feedback", "graph_query", "find_definition",
        "find_usages", "find_tests_for", "find_similar", "describe_symbol",
        "impact", "module_summary", "repo_overview", "export_graph",
        "search_diff",
    }


def test_conditional_tools_skipped_without_capability():
    # webdoc-only config: no graph_query, no find_*, no search_diff.
    mgr = FakeManager({"docs": FakeBackend(type_name="webdoc")})
    names = set(_tools(_register_everything(mgr)))
    assert names == {
        "search", "deep_search", "list_sources", "update_source_index",
        "get_rag_status", "feedback",
    }


def test_codebase_without_graph_omits_graph_only_tools():
    # A codebase source WITHOUT the graph layer: the graph-only tools
    # (graph_query, module_summary, export_graph) must NOT register; the
    # search-backed comprehension tools still do.
    mgr = FakeManager({"code": FakeBackend(graph=None, git=False)})
    names = set(_tools(_register_everything(mgr)))
    assert "module_summary" not in names
    assert "export_graph" not in names
    assert "graph_query" not in names
    assert "search_diff" not in names          # git off
    # ...but these degrade gracefully without the graph and stay available:
    assert {"find_definition", "describe_symbol", "impact", "repo_overview"} <= names


def test_descriptions_carry_no_source_catalog_the_instructions_do():
    """The source names used to be repeated in every tool description. They
    now appear once, in the handshake instructions, and the descriptions
    stay short (see tool_docs.py for why that matters per session)."""
    mgr = FakeManager({
        "mygame": FakeBackend(graph=object()),
        "docs": FakeBackend(type_name="webdoc"),
    })
    tools = _tools(_register_everything(mgr))
    for name in ("search", "deep_search"):
        desc = tools[name].description
        # "docs" is an ordinary word the description may use; the source
        # name that could only come from the catalog is "mygame".
        assert desc and "mygame" not in desc
    text = _build_instructions(mgr)
    assert "mygame" in text and "docs" in text


def test_search_routes_to_one_source_or_all():
    mgr = FakeManager({"a": FakeBackend(), "b": FakeBackend()})
    tools = _tools(_register_everything(mgr))

    tools["search"].fn(query="q", source="a")
    assert mgr.calls[-1] == ("search", "a", "q")

    tools["search"].fn(query="q")  # no source → fan out
    assert mgr.calls[-1] == ("search_all", "q")

    out = tools["search"].fn(query="q", source="nope")
    assert "Error" in out and "nope" in out


def test_search_outline_mode_drops_body_keeps_signature():
    mgr = FakeManager({"a": FakeBackend()})
    tools = _tools(_register_everything(mgr))

    full = tools["search"].fn(query="q", source="a")
    outline = tools["search"].fn(query="q", source="a", outline=True)

    # full carries the body; outline does not
    assert "return a + b" in full
    assert "return a + b" not in outline
    # outline surfaces the signature + the doc line, and labels itself
    assert "def add(a, b)" in outline
    assert "Add a and b." in outline
    assert "OUTLINE" in outline
    # both still cite file + symbol so the agent can fetch the body on demand
    assert "a.py" in outline and "add" in outline


def test_find_definition_defaults_single_codebase_source():
    mgr = FakeManager({
        "code": FakeBackend(),
        "docs": FakeBackend(type_name="webdoc"),
    })
    tools = _tools(_register_everything(mgr))
    tools["find_definition"].fn(symbol="Foo")  # source omitted
    assert mgr.calls[-1] == ("find_definition", "code", "Foo")


def test_find_definition_ambiguous_source_lists_candidates():
    mgr = FakeManager({"a": FakeBackend(), "b": FakeBackend()})
    tools = _tools(_register_everything(mgr))
    out = tools["find_definition"].fn(symbol="Foo")
    assert "Error" in out and "a" in out and "b" in out


def test_tool_annotations():
    mgr = FakeManager({"a": FakeBackend(graph=object(), git=True)})
    tools = _tools(_register_everything(mgr))
    # Tools that are not pure reads: index rebuild, feedback log append, and
    # export_graph (writes a graph view file to disk).
    rebuild_or_write = {"update_source_index", "feedback", "export_graph"}
    for name, tool in tools.items():
        ann = tool.annotations
        assert ann is not None, f"{name} has no annotations"
        if name in rebuild_or_write:
            assert ann.readOnlyHint is False, name
        else:
            assert ann.readOnlyHint is True, name
        assert ann.destructiveHint is False, name
    # Only the index rebuild may touch the network (webdoc crawl).
    assert tools["update_source_index"].annotations.openWorldHint is True
    assert tools["feedback"].annotations.openWorldHint is False


def test_instructions_embed_catalog_and_playbook():
    mgr = FakeManager({
        "mygame": FakeBackend(graph=object()),
        "docs": FakeBackend(type_name="webdoc"),
    })
    text = _build_instructions(mgr)
    assert "mygame" in text and "docs" in text
    assert "search" in text
    assert "lynx://guide" in text
    assert "graph_query" in text  # graph source present → structural hint
    # No graph anywhere → no structural hint.
    text2 = _build_instructions(FakeManager({"a": FakeBackend()}))
    assert "graph_query" not in text2


def test_guide_is_generated_from_sources():
    mgr = FakeManager({"mygame": FakeBackend(graph=object(), git=True)})
    guide = _build_guide(mgr)
    assert "mygame" in guide
    assert "search" in guide
    assert "search_diff" in guide  # git-enabled → diff section present


def test_feedback_tool_writes_local_jsonl(tmp_path):
    mgr = FakeManager({"a": FakeBackend()}, storage_path=str(tmp_path))
    tools = _tools(_register_everything(mgr))
    out = tools["feedback"].fn(
        trying_to_do="find the damage formula",
        tried="search + deep_search",
        stuck="no chunk mentions damage",
    )
    assert "recorded locally" in out
    log = tmp_path / "_feedback" / "feedback.jsonl"
    assert log.exists()
    import json as _json
    rec = _json.loads(log.read_text(encoding="utf-8").strip())
    assert rec["trying_to_do"] == "find the damage formula"
    assert rec["sources"] == ["a"]


# ---------------------------------------------------------------------------
# _resolve_source unit tests
# ---------------------------------------------------------------------------


def test_resolve_source_explicit_valid():
    mgr = FakeManager({"a": FakeBackend(), "b": FakeBackend()})
    assert _resolve_source(mgr, "b") == "b"


def test_resolve_source_unknown_raises_with_available():
    mgr = FakeManager({"a": FakeBackend()})
    with pytest.raises(ValueError, match="unknown source"):
        _resolve_source(mgr, "zzz")


def test_resolve_source_predicate_mismatch():
    mgr = FakeManager({"a": FakeBackend(type_name="webdoc"), "b": FakeBackend()})
    with pytest.raises(ValueError, match="does not support"):
        _resolve_source(mgr, "a", predicate=lambda b: b.type_name == "codebase")


def test_resolve_source_none_unambiguous():
    mgr = FakeManager({"a": FakeBackend(type_name="webdoc"), "b": FakeBackend()})
    got = _resolve_source(mgr, None, predicate=lambda b: b.type_name == "codebase")
    assert got == "b"


def test_resolve_source_none_no_candidates():
    mgr = FakeManager({"a": FakeBackend(type_name="webdoc")})
    with pytest.raises(ValueError, match="no configured"):
        _resolve_source(mgr, None, predicate=lambda b: b.type_name == "codebase")
