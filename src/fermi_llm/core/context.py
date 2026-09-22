"""The application context: the object every component receives.

``AppContext`` is the seam between core and components. A component never
imports another component; it asks the context for what it needs. That is
what makes a module replaceable: swap the registration, and every consumer
follows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import kinds
from .config import Settings, load_settings
from .contracts import check_contract
from .hooks import HOOKS, HookBus
from .registry import REGISTRY, Registry

log = logging.getLogger(__name__)


@dataclass
class AppContext:
    settings: Settings = field(default_factory=load_settings)
    registry: Registry = REGISTRY
    hooks: HookBus = HOOKS
    services: Dict[str, Any] = field(default_factory=dict)
    load_report: Any = None
    _singletons: Dict[str, Any] = field(default_factory=dict, repr=False)

    # -- singleton kinds --------------------------------------------------
    def _singleton(self, kind: str, configured: str = '') -> Any:
        if kind not in self._singletons:
            reg = self.registry.selected(kind, configured or None)
            instance = reg.factory(self)
            check_contract(kind, instance)
            self._singletons[kind] = instance
            log.debug('%s -> %s', kind, reg.name)
        return self._singletons[kind]

    @property
    def sessions(self):
        """The active session store."""
        return self._singleton(kinds.SESSION_STORE, self.settings.session_store)

    @property
    def executor(self):
        """The active execution backend."""
        return self._singleton(kinds.EXECUTION_BACKEND,
                               self.settings.execution_backend)

    @property
    def intent(self):
        """The active intent analyzer."""
        return self._singleton(kinds.INTENT_ANALYZER,
                               self.settings.intent_analyzer)

    @property
    def models(self):
        """The model service (all providers combined)."""
        if 'models' not in self.services:
            from ..components.models.service import ModelService
            self.services['models'] = ModelService(self)
        return self.services['models']

    # -- stacked kinds ----------------------------------------------------
    def build_stack(self, kind: str) -> List[Any]:
        """Instantiate every active component of a stacked kind, in order."""
        cache_key = f'stack:{kind}'
        if cache_key not in self.services:
            instances = []
            for reg in self.registry.active(kind):
                try:
                    instance = reg.factory(self)
                    check_contract(kind, instance)
                    instances.append(instance)
                except Exception as exc:                  # noqa: BLE001
                    log.warning('component %s failed to build: %s',
                                reg.label, exc)
                    if self.load_report is not None:
                        self.load_report.add_failure(reg.label, exc)
            self.services[cache_key] = instances
        return self.services[cache_key]

    def invalidate(self, kind: Optional[str] = None) -> None:
        """Drop cached instances (used by tests and hot reloads)."""
        if kind is None:
            self.services.clear()
            self._singletons.clear()
            return
        self.services.pop(f'stack:{kind}', None)
        self._singletons.pop(kind, None)

    def component_settings(self, kind: str, name: str) -> Dict[str, Any]:
        return self.settings.for_component(kind, name)

    def describe(self) -> Dict[str, Any]:
        """What is plugged in right now (served at ``/api/platform``)."""
        return {
            'components': self.registry.diagnostics(),
            'active': self.registry.snapshot(),
            'hooks': self.hooks.listeners(),
            'plugins_loaded': getattr(self.load_report, 'loaded', []),
            'plugin_failures': getattr(self.load_report, 'failures', []),
            'settings': {
                'project_dir': self.settings.project_dir,
                'sessions_dir': self.settings.sessions_dir,
                'plugin_paths': self.settings.plugin_paths,
                'execution_backend': self.settings.execution_backend,
                'skin': self.settings.skin,
            },
        }


def build_context(settings: Optional[Settings] = None,
                  registry: Optional[Registry] = None,
                  hooks: Optional[HookBus] = None) -> AppContext:
    """Create a context and load every plugin source into it."""
    from .loader import load_all
    ctx = AppContext(settings=settings or load_settings(),
                     registry=registry or REGISTRY,
                     hooks=hooks or HOOKS)
    ctx.settings.ensure_dirs()
    ctx.load_report = load_all(ctx, ctx.registry)
    ctx.hooks.emit('app.configured', ctx=ctx)
    log.info('fermi-llm context ready: %s', ctx.load_report.summary())
    return ctx
