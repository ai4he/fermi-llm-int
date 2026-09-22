"""A small synchronous event bus.

Hooks let a plugin observe or adjust the platform without replacing a
component: log every run, push progress to Slack, add a header to each
export, record metrics. Listeners never block the request on failure — an
exception is captured and reported through :attr:`HookBus.errors`.

Event names are listed in :data:`EVENTS`; core emits exactly these. A plugin
may emit its own names too (use a ``vendor.`` prefix).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

log = logging.getLogger(__name__)

EVENTS = (
    'app.configured',        # (ctx)            after all components loaded
    'app.shutdown',          # (ctx)
    'session.created',       # (session)
    'session.loaded',        # (session)
    'session.saved',         # (session)
    'chat.request',          # (session, message, model_id)
    'chat.generated',        # (session, turn)  after the model, before guardrails
    'chat.completed',        # (session, turn)  after the guardrail chain
    'guardrail.applied',     # (session, name, changed)
    'run.review',            # (session, preview)
    'run.started',           # (session, run_id)
    'run.progress',          # (session_dir, record)
    'run.finished',          # (session, result)
    'export.built',          # (session, artifact)
    'model.selected',        # (session, model_id)
)


@dataclass
class _Listener:
    priority: int
    order: int
    fn: Callable[..., Any]
    name: str


class HookBus:
    def __init__(self) -> None:
        self._listeners: Dict[str, List[_Listener]] = {}
        self._counter = 0
        self._lock = threading.RLock()
        self.errors: List[Tuple[str, str]] = []

    def on(self, event: str, fn: Callable[..., Any] = None, *,
           priority: int = 100, name: str = ''):
        """Subscribe. Usable as a decorator when ``fn`` is omitted."""
        def _do(target):
            with self._lock:
                self._counter += 1
                self._listeners.setdefault(event, []).append(
                    _Listener(priority, self._counter, target,
                              name or getattr(target, '__name__', 'listener')))
            return target
        return _do if fn is None else _do(fn)

    def off(self, event: str, fn: Callable[..., Any]) -> None:
        with self._lock:
            self._listeners[event] = [
                l for l in self._listeners.get(event, []) if l.fn is not fn]

    def emit(self, event: str, **payload) -> List[Any]:
        """Notify listeners in priority order; never raises."""
        with self._lock:
            listeners = sorted(self._listeners.get(event, []),
                               key=lambda l: (l.priority, l.order))
        results = []
        for listener in listeners:
            try:
                results.append(listener.fn(**payload))
            except Exception as exc:                      # noqa: BLE001
                self.errors.append((event, f'{listener.name}: {exc}'))
                log.warning('hook %s failed in %s: %s',
                            event, listener.name, exc)
        return results

    def listeners(self) -> Dict[str, List[str]]:
        with self._lock:
            return {event: [l.name for l in sorted(
                        items, key=lambda l: (l.priority, l.order))]
                    for event, items in self._listeners.items() if items}


HOOKS = HookBus()
