"""Shared mutable state for one running instance.

The monolith kept these as module globals in ``server.py``; every route and
helper reached for them directly, which is exactly what made the file
impossible to split. They now live in one small object that modules receive
instead of import from each other:

* ``ctx``      — the :class:`~fermi_llm.core.context.AppContext` (components)
* ``sessions`` — in-memory cache of loaded tasks
* ``websockets``/``pipeline_runs`` — per-instance connection bookkeeping

Anything that needs a component asks ``RUNTIME.ctx``; nothing imports a
component module directly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class Runtime:
    def __init__(self) -> None:
        self.ctx: Optional[Any] = None
        self.sessions: Dict[str, Any] = {}
        self.websockets: Dict[str, Any] = {}
        self.pipeline_runs: Dict[str, Any] = {}

    # -- wiring -----------------------------------------------------------
    def bind(self, ctx) -> 'Runtime':
        self.ctx = ctx
        return self

    @property
    def settings(self):
        return self.ctx.settings

    @property
    def hooks(self):
        return self.ctx.hooks

    @property
    def models(self):
        """The model service (was the ``model_manager`` global)."""
        return self.ctx.models

    @property
    def agent(self):
        """The active intent analyzer (was the ``agent`` global)."""
        return self.ctx.intent

    @property
    def knowledge(self):
        from . import kinds
        return self.ctx.build_stack(kinds.KNOWLEDGE_SOURCE)

    @property
    def sessions_dir(self) -> str:
        return self.ctx.settings.sessions_dir

    @property
    def max_active_pipelines(self) -> int:
        return self.ctx.settings.max_active_pipelines

    def reset(self) -> None:
        self.sessions.clear()
        self.websockets.clear()
        self.pipeline_runs.clear()


#: The instance-wide runtime, bound by :func:`fermi_llm.app.create_app`.
RUNTIME = Runtime()
