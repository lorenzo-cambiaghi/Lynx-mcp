"""Tool profiles: what a session gets, and what it costs in context.

Two things are asserted rather than claimed. First, that the profile table
in `tool_profiles.py` agrees with what `run_server`'s registrars actually
register for each capability mix, so the handshake `instructions` (built
from the table, before FastMCP exists) never name a tool the client will not
see. Second, the size of the tool list itself: every character of
`tools/list` rides in the model's context on every turn, and the budget
below is what stops the descriptions from growing back.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.fastmcp import FastMCP

from lynx import tool_profiles as tp
from lynx.server import (
    LynxServer,
    _build_guide,
    _build_instructions,
    _register_combined_tools,
    _register_global_tools,
    _register_graph_tools,
    _register_search_tools,
    apply_tool_profile,
    hidden_tool_message,
    package_version,
    resolve_profile,
)


# ---------------------------------------------------------------------------
# fakes: only what the registrars read while building the tool list
# ---------------------------------------------------------------------------

class _Backend:
    def __init__(self, type_name="codebase", graph=True, git=True):
        self.type_name = type_name
        self.graph = object() if graph else None
        self.source_config = {"path": "/tmp/demo", "git_integration": {"enabled": git}}


class _Manager:
    def __init__(self, backends):
        self.backends = backends

    class _Config:
        storage_path = "/tmp/storage"
        reports_path = None

    config = _Config()

    def list_sources(self):
        return [{"name": n, "type": b.type_name, "chunk_count": 1}
                for n, b in self.backends.items()]


def _register_all(manager):
    """The same four registrar calls `run_server` makes."""
    mcp = FastMCP("profiles")
    _register_search_tools(mcp, manager)
    _register_global_tools(mcp, manager)
    if any(b.graph is not None for b in manager.backends.values()):
        _register_graph_tools(mcp, manager)
    if any(b.type_name == "codebase" for b in manager.backends.values()):
        has_graph = any(b.graph is not None for b in manager.backends.values())
        _register_combined_tools(mcp, manager, has_graph=has_graph)
    return mcp


def _wire(mcp) -> str:
    """The tools/list payload as the SDK puts it on the wire (no null keys)."""
    tools = asyncio.run(mcp.list_tools())
    return json.dumps([t.model_dump(mode="json", exclude_none=True) for t in tools])


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

def test_profiles_nest_and_full_is_everything():
    core, standard, full = (set(tp.PROFILES[p]) for p in ("core", "standard", "full"))
    assert core < standard < full
    assert full == set(tp.ALL_TOOLS)
    assert (len(core), len(standard), len(full)) == (5, 10, 17)
    assert tp.DEFAULT_PROFILE == "standard"


@pytest.mark.parametrize("graph,git", [(True, True), (True, False), (False, True), (False, False)])
def test_available_tools_mirrors_what_the_registrars_register(graph, git):
    mgr = _Manager({"demo": _Backend(graph=graph, git=git)})
    registered = set(_register_all(mgr)._tool_manager._tools)
    mirrored = set(tp.available_tools(has_codebase=True, has_graph=graph, has_git=git))
    assert registered == mirrored


def test_available_tools_without_a_codebase_source():
    mgr = _Manager({"docs": _Backend(type_name="webdoc", graph=False, git=False)})
    registered = set(_register_all(mgr)._tool_manager._tools)
    assert registered == set(tp.available_tools(has_codebase=False, has_graph=False, has_git=False))
    assert registered == {"search", "deep_search", "feedback", "list_sources",
                          "get_rag_status", "update_source_index"}


def test_select_tools_include_exclude_and_unknown_names():
    avail = list(tp.ALL_TOOLS)
    kept, dropped = tp.select_tools(avail, "core")
    assert kept == [n for n in tp.ALL_TOOLS if n in tp.CORE_TOOLS]
    assert set(kept) | set(dropped) == set(avail)

    kept, _ = tp.select_tools(avail, "core", include=["graph_query"])
    assert "graph_query" in kept
    kept, _ = tp.select_tools(avail, "full", exclude=["update_source_index", "feedback"])
    assert "update_source_index" not in kept and "feedback" not in kept
    # exclude wins over include
    kept, _ = tp.select_tools(avail, "core", include=["graph_query"], exclude=["graph_query"])
    assert "graph_query" not in kept
    # a tool the profile does not cover but a source cannot provide is simply absent
    kept, dropped = tp.select_tools(["search", "feedback"], "full")
    assert kept == ["search", "feedback"] and dropped == []

    with pytest.raises(tp.ToolProfileError):
        tp.select_tools(avail, "turbo")
    with pytest.raises(tp.ToolProfileError):
        tp.select_tools(avail, "core", include=["grep"])


def test_unknown_registered_names_are_never_hidden():
    kept, dropped = tp.select_tools(["search", "some_plugin_tool"], "core")
    assert "some_plugin_tool" in kept and dropped == []


# ---------------------------------------------------------------------------
# applying a profile to a live FastMCP
# ---------------------------------------------------------------------------

def test_apply_tool_profile_prunes_the_registered_tools():
    mgr = _Manager({"demo": _Backend()})
    mcp = _register_all(mgr)
    assert len(mcp._tool_manager._tools) == 17
    kept, dropped = apply_tool_profile(mcp, "core")
    assert set(mcp._tool_manager._tools) == set(kept) == set(tp.CORE_TOOLS)
    assert len(dropped) == 12


def test_instructions_and_guide_name_only_the_tools_the_client_has():
    mgr = _Manager({"demo": _Backend()})
    core = list(tp.CORE_TOOLS)
    text = _build_instructions(mgr, core, "core")
    assert "Tool profile 'core' (5 of 17 tools)" in text
    assert "not loaded: deep_search" in text
    assert "Escalate to `deep_search`" not in text
    assert "find_usages / describe_symbol" in text
    assert "lynx://guide" in text

    guide = _build_guide(mgr, core, "core")
    assert "`find_definition(" not in guide and "`search_diff(" not in guide
    assert "`describe_symbol(" in guide and "`impact(" in guide
    assert "## Tool profile" in guide

    full = list(tp.ALL_TOOLS)
    assert "Tool profile" not in _build_instructions(mgr, full, "full")
    assert "## Tool profile" not in _build_guide(mgr, full, "full")


def test_resolve_profile_precedence(monkeypatch):
    class _Cfg:
        class tools:
            profile = "core"
    monkeypatch.delenv("LYNX_TOOL_PROFILE", raising=False)
    assert resolve_profile(_Cfg) == "core"
    monkeypatch.setenv("LYNX_TOOL_PROFILE", "full")
    assert resolve_profile(_Cfg) == "full"
    assert resolve_profile(_Cfg, "standard") == "standard"
    with pytest.raises(tp.ToolProfileError):
        resolve_profile(_Cfg, "turbo")


def test_config_tools_block(tmp_path):
    from lynx.config import load_config

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "config_version": 2, "sources": {},
        "tools": {"profile": "core", "include": ["graph_query"], "exclude": ["feedback"]},
    }), encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.tools.profile == "core"
    assert cfg.tools.include == ("graph_query",)
    assert cfg.tools.exclude == ("feedback",)

    cfg_path.write_text(json.dumps({"config_version": 2, "sources": {}}), encoding="utf-8")
    assert load_config(cfg_path).tools.profile == "standard"

    cfg_path.write_text(json.dumps({
        "config_version": 2, "sources": {}, "tools": {"profile": "turbo"},
    }), encoding="utf-8")
    with pytest.raises(SystemExit):
        load_config(cfg_path)


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------

# Wire size of tools/list for a codebase source with graph + git (every tool
# registered), after `apply_tool_profile`. Measured 2026-09-07: 5,176 / 11,295 /
# 16,526 characters, once every overlapping tool named its sibling; 4,783 /
# 10,704 / 15,279 before that, and about 31,000 at full before the profiles
# existed. Headroom is now thin: ~120 characters on core, ~270 on full. Raise a
# ceiling only with a reason written next to it.
_WIRE_BUDGET = {"core": 5_300, "standard": 11_800, "full": 16_800}
_MAX_DESCRIPTION = 450


@pytest.mark.parametrize("profile", ["core", "standard", "full"])
def test_tools_list_stays_within_budget(profile):
    mgr = _Manager({"demo": _Backend()})
    mcp = _register_all(mgr)
    apply_tool_profile(mcp, profile)
    wire = _wire(mcp)
    assert len(wire) <= _WIRE_BUDGET[profile], (
        f"tools/list for {profile!r} is {len(wire)} chars, budget {_WIRE_BUDGET[profile]}")
    for tool in asyncio.run(mcp.list_tools()):
        assert len(tool.description or "") <= _MAX_DESCRIPTION, tool.name
        schema = json.dumps(tool.inputSchema)
        # A string-valued "title" is pydantic's generated label; a parameter that
        # happens to be named title would render as `"title": {`, and is fine.
        assert '"title": "' not in schema, f"{tool.name}: schema titles are dead weight"


def test_calling_a_hidden_tool_names_the_profile_and_the_fix():
    from mcp.server.fastmcp.exceptions import ToolError

    mgr = _Manager({"demo": _Backend()})
    mcp = LynxServer("t")
    _register_search_tools(mcp, mgr)
    _register_global_tools(mcp, mgr)
    _, dropped = apply_tool_profile(mcp, "core")
    assert "list_sources" in dropped
    mcp.hidden_tools = {n: hidden_tool_message(n, "core") for n in dropped}
    with pytest.raises(ToolError) as exc:
        asyncio.run(mcp.call_tool("list_sources", {}))
    msg = str(exc.value)
    assert "not loaded in tool profile 'core'" in msg
    assert "--profile full" in msg and "tools.include" in msg
    # a genuinely unknown name still fails the SDK way
    with pytest.raises(ToolError):
        asyncio.run(mcp.call_tool("no_such_tool", {}))


def test_server_reports_the_package_version_not_the_sdk_version():
    import importlib.metadata

    mcp = LynxServer("t")
    opts = mcp._mcp_server.create_initialization_options()
    assert opts.server_version == package_version()
    assert opts.server_version != importlib.metadata.version("mcp")
