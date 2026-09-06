"""Answer the MCP handshake first, open the indexes after.

Until 1.8 `lynx serve` waited for every index to open before it would read
a single byte of JSON-RPC: embedding model, Chroma stores, BM25 corpora, the
graph, the integrity probe. With two indexes that was about 10 seconds; on
a first run it is the whole index build, minutes on a large repository.
Claude Code gives an MCP server 30 seconds to answer `initialize` by default,
and a server that misses that window is reported dead with no useful error.

Everything the handshake needs is already in the config file: which sources
exist, their type and path, whether the graph layer or git integration is on.
So the tools are registered against a `ManagerHandle` built from the config
alone, the transport starts at once, and the real `SourceManager` is
constructed in a background thread. The handle answers `.backends` and
`.config` immediately, from a preview of the sources; any other attribute
(that is, every operation a tool performs) waits for the real manager, for a
bounded time, and otherwise raises `ManagerNotReady` with a message that says
what is happening and asks the caller to retry. The tools already turn
exceptions into text, so a call that arrives too early comes back as a
sentence, not a timeout.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

# How long a tool call waits for the indexes before answering with the
# loading state. Short of the client's own per-call timeout (60 s in the MCP
# SDKs, and Claude Code's default), long enough that a call arriving a few
# seconds early simply succeeds. LYNX_TOOL_WAIT_SECONDS overrides it.
DEFAULT_TOOL_WAIT_SECONDS = 20.0


class ManagerNotReady(RuntimeError):
    """Raised by a tool that needs the indexes while they are still opening,
    or after the load failed. The message is written for the model."""


class PreviewBackend:
    """A source as the config describes it, before its index is open.

    Exposes exactly what the registrars and the handshake read from a real
    backend: `name`, `type_name`, `source_config` (the validated dict) and
    `graph`, non-None when the source opts into the graph layer. Nothing
    here touches disk.
    """

    def __init__(self, name: str, source_config: dict) -> None:
        self.name = name
        self.type_name = source_config.get("type", "")
        self.source_config = source_config
        graph_cfg = source_config.get("graph") or {}
        self.graph = object() if (self.type_name == "codebase" and graph_cfg.get("enabled")) else None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PreviewBackend({self.name!r}, type={self.type_name!r})"


def preview_backends(config) -> Dict[str, PreviewBackend]:
    return {name: PreviewBackend(name, cfg) for name, cfg in config.sources.items()}


class ManagerHandle:
    """Stand-in for the `SourceManager` while it loads, and a thin proxy after.

    `backends` and `config` never block. Everything else is looked up on the
    real manager, after waiting up to `wait_seconds` for it to exist.
    """

    def __init__(self, config, *, wait_seconds: Optional[float] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self._clock = clock
        self._started = clock()
        self._ready = threading.Event()
        self._manager: Any = None
        self._error: Optional[str] = None
        self._phase = "starting"
        self._preview = preview_backends(config)
        if wait_seconds is None:
            wait_seconds = _wait_seconds_from_env()
        self._wait_seconds = float(wait_seconds)

    # -- state --------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._manager is not None

    @property
    def failed(self) -> bool:
        return self._error is not None

    @property
    def phase(self) -> str:
        return self._phase

    def elapsed(self) -> float:
        return self._clock() - self._started

    def set_phase(self, text: str) -> None:
        self._phase = text

    def set_manager(self, manager) -> None:
        self._manager = manager
        self._ready.set()

    def set_error(self, error: str) -> None:
        self._error = error
        self._ready.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the load finished (well or badly). True when a manager
        is available."""
        self._ready.wait(timeout)
        return self._manager is not None

    # -- what never blocks --------------------------------------------------

    @property
    def backends(self):
        return self._manager.backends if self._manager is not None else self._preview

    @property
    def broken(self) -> dict:
        return self._manager.broken if self._manager is not None else {}

    # -- what waits ---------------------------------------------------------

    def status_text(self) -> str:
        if self._error is not None:
            return (
                f"Lynx could not open its indexes: {self._error} Fix the "
                "configuration or run `lynx manager doctor`, then restart the server."
            )
        secs = int(self.elapsed())
        text = (
            f"Lynx is still opening its indexes ({self._phase}, {secs}s so far). "
            "Retry this call in a few seconds."
        )
        limit = getattr(self.config, "loading_timeout_seconds", None)
        if limit and secs > limit:
            text += (
                f" This is taking longer than loading_timeout_seconds ({limit}s): "
                "a first-run index build on a large repository can take minutes; "
                "the server log on stderr shows the progress."
            )
        return text

    def __getattr__(self, name: str):
        # Only reached for names not set on the handle itself. Dunder and
        # private lookups (copy, pickle, introspection) must fail fast rather
        # than wait for the indexes.
        if name.startswith("_"):
            raise AttributeError(name)
        if self._manager is None:
            self._ready.wait(self._wait_seconds)
            if self._manager is None:
                raise ManagerNotReady(self.status_text())
        return getattr(self._manager, name)


def _wait_seconds_from_env() -> float:
    raw = os.environ.get("LYNX_TOOL_WAIT_SECONDS")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return DEFAULT_TOOL_WAIT_SECONDS


def _default_factory(config):
    # Decide HF offline mode BEFORE the heavy imports freeze the env flags
    # (see config.configure_hf_offline for the rationale).
    from .config import configure_hf_offline
    configure_hf_offline(config)
    from .source_manager import SourceManager
    return SourceManager(config)


def start_loader(config, handle: ManagerHandle, *,
                 factory: Callable[[Any], Any] = _default_factory,
                 log=None) -> threading.Thread:
    """Build the real manager in a daemon thread and hand it to `handle`.

    `factory` exists so tests can run this without an index. Progress and the
    outcome go to `log` (stderr by default), never to stdout, which is the
    JSON-RPC channel.
    """
    log = log or (lambda msg: print(msg, file=sys.stderr))
    n = len(config.sources)

    def _run():
        try:
            handle.set_phase(f"opening {n} source{'s' if n != 1 else ''}")
            manager = factory(config)
            handle.set_phase("starting the file watchers")
            try:
                manager.start_watchers()
            except Exception as e:  # a watcher is a convenience, not the index
                log(f"[server] failed to start watchers: {e}")
            handle.set_manager(manager)
            log(f"[server] indexes ready after {handle.elapsed():.1f}s")
        except Exception as e:
            handle.set_error(f"{type(e).__name__}: {e}")
            log(f"[server] source manager failed to load: {e}")

    thread = threading.Thread(target=_run, name="lynx-loader", daemon=True)
    thread.start()
    return thread
