"""The handshake no longer waits for the indexes: what the tools see meanwhile.

Everything here runs without an index. The `ManagerHandle` is exercised with
a fake manager handed over by hand or by `start_loader` with an injected
factory, and the tools are the real registered ones, called directly.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from lynx import startup
from lynx.server import (
    _build_instructions,
    _capabilities,
    _register_combined_tools,
    _register_global_tools,
    _source_catalog,
)
from lynx.startup import ManagerHandle, ManagerNotReady, PreviewBackend, preview_backends
from lynx.tool_profiles import available_tools


def _config(sources, timeout=600):
    return SimpleNamespace(
        sources=sources,
        loading_timeout_seconds=timeout,
        storage_path=Path("/tmp/storage"),
        reports_path=None,
        search=SimpleNamespace(default_top_k=5),
    )


SOURCES = {
    "game": {"type": "codebase", "path": Path("C:/repo/game"),
             "graph": {"enabled": True}, "git_integration": {"enabled": True}},
    "lib": {"type": "codebase", "path": Path("C:/repo/lib"),
            "graph": {"enabled": False}, "git_integration": {"enabled": False}},
    "docs": {"type": "webdoc", "url": "https://docs.example.com"},
}


class _RealBackend:
    """Shaped like a loaded backend, for comparison with the preview."""

    def __init__(self, cfg):
        self.type_name = cfg["type"]
        self.source_config = cfg
        self.graph = object() if (cfg.get("graph") or {}).get("enabled") else None


class _FakeManager:
    def __init__(self, config):
        self.config = config
        self.backends = {n: _RealBackend(c) for n, c in config.sources.items()}
        self.broken = {}
        self.watchers_started = False

    def start_watchers(self):
        self.watchers_started = True

    def list_sources(self):
        return [{"name": n, "type": b.type_name, "chunk_count": 42} for n, b in self.backends.items()]


# ---------------------------------------------------------------------------
# the preview
# ---------------------------------------------------------------------------

def test_preview_backends_mirror_the_config():
    pv = preview_backends(_config(SOURCES))
    assert set(pv) == {"game", "lib", "docs"}
    assert pv["game"].type_name == "codebase" and pv["game"].graph is not None
    assert pv["lib"].graph is None            # graph not enabled
    assert pv["docs"].type_name == "webdoc" and pv["docs"].graph is None
    assert pv["docs"].source_config["url"] == "https://docs.example.com"
    assert isinstance(pv["game"], PreviewBackend)


def test_preview_yields_the_same_capabilities_and_catalog_as_loaded_backends():
    cfg = _config(SOURCES)
    handle = ManagerHandle(cfg, wait_seconds=0)
    real = _FakeManager(cfg)
    assert _capabilities(handle) == _capabilities(real)
    assert available_tools(**_capabilities(handle)) == available_tools(**_capabilities(real))
    assert _source_catalog(handle) == _source_catalog(real)
    assert "game" in _source_catalog(handle) and "docs.example.com" in _source_catalog(handle)


# ---------------------------------------------------------------------------
# the handle
# ---------------------------------------------------------------------------

def test_handle_answers_backends_and_config_without_waiting_then_delegates():
    cfg = _config(SOURCES)
    handle = ManagerHandle(cfg, wait_seconds=0)
    assert handle.config is cfg
    assert not handle.ready
    assert set(handle.backends) == set(SOURCES) and handle.broken == {}

    real = _FakeManager(cfg)
    handle.set_manager(real)
    assert handle.ready
    assert handle.backends is real.backends
    assert handle.list_sources() == real.list_sources()


def test_handle_reports_the_loading_state_instead_of_blocking():
    handle = ManagerHandle(_config(SOURCES), wait_seconds=0.05, clock=lambda: 100.0)
    handle._started = 88.0
    handle.set_phase("opening 3 sources")
    t0 = time.perf_counter()
    with pytest.raises(ManagerNotReady) as exc:
        handle.search("q")
    assert time.perf_counter() - t0 < 2
    msg = str(exc.value)
    assert "still opening" in msg and "opening 3 sources" in msg
    assert "12s so far" in msg and "Retry" in msg
    assert "longer than" not in msg


def test_handle_warns_when_the_load_outlives_the_configured_timeout():
    handle = ManagerHandle(_config(SOURCES, timeout=10), wait_seconds=0, clock=lambda: 1000.0)
    handle._started = 0.0
    with pytest.raises(ManagerNotReady) as exc:
        handle.search("q")
    assert "longer than loading_timeout_seconds (10s)" in str(exc.value)


def test_handle_reports_a_failed_load():
    handle = ManagerHandle(_config(SOURCES), wait_seconds=5)
    handle.set_error("ValueError: path does not exist: C:/repo/game")
    t0 = time.perf_counter()
    with pytest.raises(ManagerNotReady) as exc:
        handle.get("game")
    assert time.perf_counter() - t0 < 1  # the event is set: no waiting on failure
    assert "could not open" in str(exc.value) and "path does not exist" in str(exc.value)
    assert "lynx manager doctor" in str(exc.value)


def test_private_lookups_never_wait():
    handle = ManagerHandle(_config(SOURCES), wait_seconds=30)
    t0 = time.perf_counter()
    with pytest.raises(AttributeError):
        handle.__deepcopy__
    assert time.perf_counter() - t0 < 1


def test_wait_seconds_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("LYNX_TOOL_WAIT_SECONDS", "3.5")
    assert ManagerHandle(_config(SOURCES))._wait_seconds == 3.5
    monkeypatch.setenv("LYNX_TOOL_WAIT_SECONDS", "nonsense")
    assert ManagerHandle(_config(SOURCES))._wait_seconds == startup.DEFAULT_TOOL_WAIT_SECONDS
    monkeypatch.delenv("LYNX_TOOL_WAIT_SECONDS")
    assert ManagerHandle(_config(SOURCES))._wait_seconds == startup.DEFAULT_TOOL_WAIT_SECONDS


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------

def test_loader_hands_the_manager_over_and_starts_the_watchers():
    cfg = _config(SOURCES)
    handle = ManagerHandle(cfg, wait_seconds=5)
    logged = []
    thread = startup.start_loader(cfg, handle, factory=_FakeManager, log=logged.append)
    thread.join(timeout=5)
    assert handle.ready and not handle.failed
    assert handle.list_sources()[0]["chunk_count"] == 42
    assert handle._manager.watchers_started
    assert any("indexes ready" in m for m in logged)


def test_loader_records_a_failure_for_the_tools_to_report():
    cfg = _config(SOURCES)
    handle = ManagerHandle(cfg, wait_seconds=5)
    logged = []

    def boom(_cfg):
        raise RuntimeError("chroma exploded")

    startup.start_loader(cfg, handle, factory=boom, log=logged.append).join(timeout=5)
    assert handle.failed and not handle.ready
    with pytest.raises(ManagerNotReady) as exc:
        handle.search("q")
    assert "RuntimeError: chroma exploded" in str(exc.value)
    assert any("failed to load" in m for m in logged)


def test_loader_runs_while_the_handle_is_already_answering():
    """The whole point: registration and the handshake happen before the
    factory returns, and a call issued meanwhile succeeds once it does."""
    cfg = _config(SOURCES)
    handle = ManagerHandle(cfg, wait_seconds=5)
    release = threading.Event()

    def slow_factory(c):
        release.wait(5)
        return _FakeManager(c)

    startup.start_loader(cfg, handle, factory=slow_factory)
    # Nothing below blocks on the index.
    assert set(handle.backends) == set(SOURCES)
    text = _build_instructions(handle, ["search", "feedback"], "core")
    assert "game" in text and "open in the background" in text

    result = {}

    def call():
        result["v"] = handle.list_sources()

    t = threading.Thread(target=call)
    t.start()
    release.set()
    t.join(5)
    assert result["v"][0]["name"] == "game"


# ---------------------------------------------------------------------------
# the tools, called too early
# ---------------------------------------------------------------------------

def test_tools_answer_with_the_loading_state_then_with_results():
    cfg = _config(SOURCES)
    handle = ManagerHandle(cfg, wait_seconds=0.05)
    mcp = FastMCP("startup")
    _register_global_tools(mcp, handle)
    _register_combined_tools(mcp, handle, has_graph=True)
    tools = mcp._tool_manager._tools

    early = tools["list_sources"].fn()
    assert "still opening" in early and "Retry" in early

    # Validation that only needs the config still answers at once.
    unknown = tools["find_definition"].fn(symbol="X", source="nope")
    assert "unknown source" in unknown and "still opening" not in unknown

    handle.set_manager(_FakeManager(cfg))
    later = tools["list_sources"].fn()
    assert "game" in later and "42" in later
