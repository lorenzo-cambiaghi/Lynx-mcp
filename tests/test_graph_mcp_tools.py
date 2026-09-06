"""Verifies that `_register_graph_tools` exposes the single `graph_query`
tool, that its description carries the operation catalog + source names,
and that each operation calls through to SourceManager / GraphLayer.

Reuses the stubbed-RAG trick from test_graph_integration so we don't
have to load the HuggingFace embedding model. Pytest-style: the manager
and the FastMCP instance are built once per module via a fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_codebase(root: Path, files: dict) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _stub_rag():
    """Same trick as test_graph_integration — stub CodebaseRAG so we don't
    need HuggingFace running."""
    from lynx.rag_manager import CodebaseRAG

    class _StubCollection:
        def count(self):
            return 0

    class _StubVS:
        def __init__(self):
            self._collection = _StubCollection()

    def stub_init(self, **kwargs):
        self.codebase_path = Path(kwargs["codebase_path"])
        self.storage_path = Path(kwargs["rag_storage_path"])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata = {"last_commit": None, "last_update": None}
        self.vector_store = _StubVS()

    CodebaseRAG.__init__ = stub_init
    CodebaseRAG.check_config_drift = lambda self: None
    CodebaseRAG.drift_status_text = lambda self: "No drift."
    CodebaseRAG.needs_update = lambda self: False


@pytest.fixture(scope="module")
def graph_manager(tmp_path_factory):
    """A REAL SourceManager over a tiny synthetic codebase, graph built.

    Real classes all the way down — SourceManager, CodebaseBackend,
    GraphLayer — with only CodebaseRAG stubbed, so no HuggingFace model is
    needed. This is what lets a test exercise the actual attribute chain
    the dispatch layer reaches through, instead of a hand-shaped fake that
    would keep passing after a rename.

    Codebase: util.py defines `helper` + an inheritance chain
    (Base <- Derived) used to exercise subclasses / superclasses.
    """
    tmp = tmp_path_factory.mktemp("lynx-mcp-tools")
    code = tmp / "code"
    code.mkdir()
    _make_codebase(code, {
        "util.py": (
            "def helper(x):\n    return x + 1\n"
            "class Base:\n    pass\n"
        ),
        "main.py": (
            "from util import helper, Base\n"
            "def go():\n    return helper(1)\n"
            "class Derived(Base):\n    pass\n"
        ),
    })

    _stub_rag()

    from lynx.config import load_config
    cfg_path = tmp / "config.json"
    cfg_path.write_text(json.dumps({
        "config_version": 2,
        "storage_path": str(tmp / "storage"),
        "sources": {
            "demo": {
                "type": "codebase",
                "path": str(code),
                "supported_extensions": [".py"],
                "graph": {"enabled": True},
                "watcher": {"enabled": False},
            }
        }
    }), encoding="utf-8")
    cfg = load_config(cfg_path)

    from lynx.source_manager import SourceManager
    mgr = SourceManager(cfg)
    mgr.get("demo").graph.rebuild(force=True)
    return mgr


@pytest.fixture(scope="module")
def graph_mcp(graph_manager):
    """FastMCP with the graph tools registered over `graph_manager`."""
    from mcp.server.fastmcp import FastMCP
    from lynx.server import _register_graph_tools

    mcp = FastMCP("test")
    _register_graph_tools(mcp, graph_manager)
    return mcp


def _tool(graph_mcp):
    # FastMCP exposes registered tools via _tool_manager._tools (dict)
    return graph_mcp._tool_manager._tools["graph_query"]


def test_single_graph_query_tool_with_rich_description(graph_mcp):
    registered = list(graph_mcp._tool_manager._tools.keys())
    assert registered == ["graph_query"], registered

    # The description MUST be non-empty (f-string "docstrings" silently
    # produce empty descriptions) and must carry the operation catalog. The
    # source names are not in it any more: they are stated once in the
    # handshake instructions instead of once per tool (see tool_docs.py).
    desc = _tool(graph_mcp).description
    assert desc and len(desc) > 100
    assert "demo" not in desc
    for op in (
        "callers", "callees", "subclasses", "superclasses", "imports",
        "neighbors", "shortest_path", "overview", "surprising_connections",
        "status",
    ):
        assert op in desc, f"operation {op!r} missing from description"


def test_callers_operation(graph_mcp):
    out = _tool(graph_mcp).fn(operation="callers", symbol="helper")
    assert "Callers of 'helper'" in out
    assert "helper" in out


def test_source_defaults_to_only_graph_source(graph_mcp):
    # `source` omitted → resolves to 'demo' since it's the only graph source.
    out = _tool(graph_mcp).fn(operation="status")
    assert "Graph status for 'demo'" in out


def test_unknown_operation_lists_valid_ones(graph_mcp):
    out = _tool(graph_mcp).fn(operation="explode", symbol="helper")
    assert "unknown operation" in out
    assert "callers" in out


def test_missing_symbol_is_reported(graph_mcp):
    out = _tool(graph_mcp).fn(operation="callers")
    assert "requires `symbol`" in out


def test_overview_operation(graph_mcp):
    out = _tool(graph_mcp).fn(operation="overview", top_n=5, min_community_size=2)
    assert "Architectural overview" in out
    assert "God nodes" in out
    assert "Communities" in out


def test_shortest_path_operation(graph_mcp):
    tool = _tool(graph_mcp)
    # No path → user-friendly text, not a stack trace.
    out = tool.fn(operation="shortest_path", symbol="helper", target="nonexistent", max_hops=5)
    assert "No directed path" in out or "Error" in out
    # Real path
    out = tool.fn(operation="shortest_path", symbol="go", target="helper", max_hops=5)
    assert "Path from 'go'" in out
    assert "helper" in out


def test_subclasses_and_superclasses(graph_mcp):
    tool = _tool(graph_mcp)

    out = tool.fn(operation="subclasses", symbol="Base")
    assert "Subclasses of 'Base'" in out
    assert "Derived" in out, "cross-file inheritance not resolved"
    assert "inherits" in out

    out = tool.fn(operation="superclasses", symbol="Derived")
    assert "Superclasses of 'Derived'" in out
    assert "Base" in out


# ----------------------------------------------------------------------
# The "unknown symbol" hint, against the REAL object graph.
#
# `_seed_exists` reaches through `manager.get(source).graph.graph` and
# swallows every exception, so if any link in that chain is renamed the
# hint silently stops appearing and nothing fails. The unit tests use a
# fake shaped like the chain — they would keep passing against a broken
# real API. These run on the actual SourceManager / CodebaseBackend /
# GraphLayer, which is the only way the chain stays pinned.
# ----------------------------------------------------------------------


def test_seed_exists_reaches_the_real_graph(graph_manager):
    from lynx.graph.dispatch import _seed_exists

    # A function the graph knows, one it doesn't.
    assert _seed_exists(graph_manager, "demo", "helper") is True
    assert _seed_exists(graph_manager, "demo", "definitelyNotHere") is False
    # None means "couldn't check" — it must not be the answer here, or the
    # hint is dead while every test still passes.
    assert _seed_exists(graph_manager, "demo", "helper") is not None


def test_seed_exists_counts_file_nodes_only_for_imports(graph_manager):
    """`get_imports` falls back to file nodes; every other operation goes
    through `find_symbols`, which skips them. The check has to mirror that.

    Counting files for everything looked like the safe default — it can
    only stay quiet, never wrongly claim absence — but it silenced the hint
    exactly where it earns its keep: on `main`, `config`, `utils`, the
    names that collide with filenames and get mistyped.
    """
    from lynx.graph.dispatch import _seed_exists

    # `imports` accepts a file path.
    assert _seed_exists(graph_manager, "demo", "main.py", allow_files=True) is True
    # `callers` does not: there is no *symbol* called main.py.
    assert _seed_exists(graph_manager, "demo", "main.py", allow_files=False) is False


def test_symbol_colliding_with_a_filename_is_reported_missing(graph_manager):
    """The regression this behaviour exists for: the codebase has `main.py`
    but no function `main`, and `--op callers --symbol main` used to report
    a match and print a bare `(no results)`."""
    from lynx.graph.dispatch import query_graph

    res = query_graph(graph_manager, "demo", "callers", symbol="main")
    assert res.matched is False
    assert "nothing in this graph is called 'main'" in res.text

    # ...while the operation that really does take a file path still works.
    res = query_graph(graph_manager, "demo", "imports", symbol="main.py")
    assert res.matched is not False


def test_unknown_symbol_hint_appears_on_the_real_graph(graph_manager):
    from lynx.graph.dispatch import query_graph

    res = query_graph(graph_manager, "demo", "callers", symbol="ghostSymbol")
    assert res.ok is True          # a real answer about nothing
    assert res.matched is False
    assert "nothing in this graph is called 'ghostSymbol'" in res.text
    assert res.data["matched"] is False


def test_known_symbol_without_callers_gets_no_hint(graph_manager):
    """`go` is called by nobody, but it exists — the two empty answers must
    stay distinguishable on the real graph, not just on the fake."""
    from lynx.graph.dispatch import query_graph

    res = query_graph(graph_manager, "demo", "callers", symbol="go")
    assert res.data["count"] == 0
    assert res.matched is True
    assert "nothing in this graph" not in res.text


def test_json_payload_shape_on_the_real_graph(graph_manager):
    """The keys `lynx graph query --json` promises, produced by real
    objects rather than by the fake that was built to match them."""
    from lynx.graph.dispatch import query_graph

    res = query_graph(graph_manager, "demo", "callers", symbol="helper")
    assert res.data["operation"] == "callers"
    assert res.data["source"] == "demo"
    assert res.data["matched"] is True
    assert res.data["count"] >= 1
    edge = res.data["edges"][0]
    assert edge["source"]["label"] and edge["target"]["label"]
    assert edge["relation"] == "calls"

    err = query_graph(graph_manager, "demo", "callers")
    assert err.ok is False
    assert err.data["operation"] == "callers" and err.data["source"] == "demo"


# ----------------------------------------------------------------------
# The web UI's playground, against the REAL manager.
#
# `test_playground.py` drives every panel, but through a fake shaped like
# the manager — which is the failure mode this file exists to catch: a
# hand-built stub keeps passing after the real API moves under it. The
# panels claim to render exactly what the MCP tool returns, so the honest
# assertion is to run both against the same real graph and compare.
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def playground_client(graph_manager, tmp_path_factory):
    from fastapi.testclient import TestClient

    from lynx.manager.ui.app import create_app

    tmp = tmp_path_factory.mktemp("pg-real")
    cfg = tmp / "config.json"
    cfg.write_text(json.dumps({
        "config_version": 2,
        "storage_path": str(tmp / "storage"),
        "sources": {},
    }), encoding="utf-8")
    app = create_app(cfg)
    app.state.manager = graph_manager
    return TestClient(app)


def _unescape(html: str) -> str:
    return (html.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"')
                .replace("&#x27;", "'").replace("&#39;", "'"))


@pytest.mark.parametrize("operation,symbol", [
    ("callers", "helper"),
    ("callees", "go"),
    ("subclasses", "Base"),
    ("superclasses", "Derived"),
    ("status", ""),
])
def test_playground_graph_panel_matches_the_tool(playground_client,
                                                 graph_manager,
                                                 operation, symbol):
    """Byte-for-byte, on a real graph: the panel is the tool's own text."""
    from lynx.graph.dispatch import query_graph

    r = playground_client.post("/api/playground/graph_query", data={
        "source": "demo", "operation": operation, "symbol": symbol,
    })

    assert r.status_code == 200, r.text
    expected = query_graph(graph_manager, "demo", operation,
                           symbol=symbol or None).text
    rendered = _unescape(r.text)
    for line in expected.splitlines():
        assert line.strip() in rendered, f"{operation}: missing {line.strip()!r}"


def test_playground_describe_symbol_matches_the_tool(playground_client,
                                                     graph_manager):
    from lynx._format import _format_describe_symbol

    r = playground_client.post("/api/playground/describe_symbol",
                               data={"source": "demo", "symbol": "helper"})

    assert r.status_code == 200
    expected = _format_describe_symbol(
        "helper", graph_manager.describe_symbol("demo", "helper"))
    rendered = _unescape(r.text)
    assert "CALLED BY" in expected          # guard the fixture itself
    for line in expected.splitlines():
        assert line.strip() in rendered


def test_playground_find_definition_hits_the_real_graph(playground_client):
    r = playground_client.post("/api/playground/find_definition",
                               data={"source": "demo", "symbol": "helper",
                                     "limit": "5"})

    assert r.status_code == 200
    assert "util.py" in r.text


def test_playground_export_writes_a_real_view(playground_client, graph_manager):
    """The whole chain on real objects: render a blast radius, write it,
    serve it back through the download endpoint."""
    r = playground_client.post("/api/playground/export_graph",
                               data={"source": "demo", "target": "helper",
                                     "mode": "symbol", "depth": "2"})

    assert r.status_code == 200, r.text
    import re
    m = re.search(r'href="(/api/reports/[^"]+)"', r.text)
    assert m, r.text

    got = playground_client.get(m.group(1))
    assert got.status_code == 200
    assert "helper" in got.text
    assert got.text.lstrip().lower().startswith("<!doctype html")
