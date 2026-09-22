"""The component registry.

One process-wide registry holds every component, keyed by ``(kind, name)``.
Components declare how they relate to what is already there:

``priority``
    Lower runs first for stacked kinds (guardrails, validators, exporters...).
    Core components sit at 100, 200, 300... so a plugin can slot in between
    without editing core.
``replaces``
    Names this component takes over from. The replaced component stays
    registered but inactive, so a deployment can revert by disabling the
    replacement instead of reinstalling anything.
``requires``
    Names that must be active for this one to activate. A component whose
    requirement is missing is reported by :meth:`Registry.diagnostics`
    rather than raising, so one broken plugin cannot take down the app.

Nothing here imports FastAPI, FermiPy or any component: the registry is the
one module every other module may depend on.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .errors import DuplicateComponent, UnknownComponent
from .kinds import ALL_KINDS


@dataclass(frozen=True)
class Registration:
    """One registered component."""

    kind: str
    name: str
    factory: Callable[..., Any]
    priority: int = 100
    replaces: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = 'unknown'
    enabled: bool = True

    @property
    def label(self) -> str:
        return f'{self.kind}:{self.name}'


class Registry:
    """Thread-safe store of registrations with replace/stack semantics."""

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Registration]] = {
            kind: {} for kind in ALL_KINDS}
        self._disabled: Dict[str, set] = {kind: set() for kind in ALL_KINDS}
        self._lock = threading.RLock()

    # -- registration -----------------------------------------------------
    def register(self, kind: str, name: str, factory: Callable[..., Any] = None,
                 *, priority: int = 100, replaces: Iterable[str] = (),
                 requires: Iterable[str] = (),
                 metadata: Optional[Dict[str, Any]] = None,
                 source: str = 'unknown', override: bool = False):
        """Register ``factory`` under ``(kind, name)``.

        Usable directly or as a decorator when ``factory`` is omitted.
        """
        if kind not in self._items:
            raise UnknownComponent(
                f'unknown component kind {kind!r}; known kinds: '
                + ', '.join(sorted(self._items)))

        def _do(target: Callable[..., Any]) -> Callable[..., Any]:
            reg = Registration(
                kind=kind, name=name, factory=target, priority=priority,
                replaces=tuple(replaces), requires=tuple(requires),
                metadata=dict(metadata or {}), source=source)
            with self._lock:
                existing = self._items[kind].get(name)
                if existing is not None and not override:
                    raise DuplicateComponent(
                        f'{kind}:{name} already registered by '
                        f'{existing.source!r}; register with a different name, '
                        f'or declare replaces=("{name}",) from your own name')
                self._items[kind][name] = reg
            return target

        if factory is None:
            return _do
        return _do(factory)

    def unregister(self, kind: str, name: str) -> None:
        with self._lock:
            self._items.get(kind, {}).pop(name, None)

    # -- enable / disable -------------------------------------------------
    def disable(self, kind: str, name: str) -> None:
        """Turn a component off without removing it (config driven)."""
        with self._lock:
            self._disabled.setdefault(kind, set()).add(name)

    def enable(self, kind: str, name: str) -> None:
        with self._lock:
            self._disabled.setdefault(kind, set()).discard(name)

    def is_enabled(self, kind: str, name: str) -> bool:
        return name not in self._disabled.get(kind, set())

    # -- lookup -----------------------------------------------------------
    def get(self, kind: str, name: str) -> Registration:
        try:
            return self._items[kind][name]
        except KeyError:
            raise UnknownComponent(
                f'no {kind} named {name!r}; available: '
                + (', '.join(self.names(kind)) or '(none)')) from None

    def find(self, kind: str, name: str) -> Optional[Registration]:
        return self._items.get(kind, {}).get(name)

    def create(self, kind: str, name: str, *args, **kwargs) -> Any:
        """Instantiate one component."""
        return self.get(kind, name).factory(*args, **kwargs)

    def active(self, kind: str) -> List[Registration]:
        """Enabled components of ``kind``, replacements applied, in run order."""
        with self._lock:
            items = list(self._items.get(kind, {}).values())
        enabled = [r for r in items if self.is_enabled(kind, r.name)]
        superseded = {n for r in enabled for n in r.replaces}
        unmet = self._unmet_requirements(kind, enabled, superseded)
        live = [r for r in enabled
                if r.name not in superseded and r.name not in unmet]
        return sorted(live, key=lambda r: (r.priority, r.name))

    def all(self, kind: str) -> List[Registration]:
        """Every registration of ``kind``, including disabled and replaced."""
        with self._lock:
            items = list(self._items.get(kind, {}).values())
        return sorted(items, key=lambda r: (r.priority, r.name))

    def names(self, kind: str) -> List[str]:
        return [r.name for r in self.all(kind)]

    def create_active(self, kind: str, *args, **kwargs) -> List[Any]:
        return [r.factory(*args, **kwargs) for r in self.active(kind)]

    def selected(self, kind: str, name: Optional[str]) -> Registration:
        """The one component to use for a singleton kind.

        ``name`` comes from configuration; without it the highest-priority
        (lowest number) active component wins, which keeps a deployment
        working when a plugin replaces the core default.
        """
        if name:
            return self.get(kind, name)
        live = self.active(kind)
        if not live:
            raise UnknownComponent(f'no {kind} component is active')
        return live[0]

    # -- introspection ----------------------------------------------------
    def _unmet_requirements(self, kind, enabled, superseded) -> set:
        present = {r.name for r in enabled if r.name not in superseded}
        present |= superseded  # a replaced name is still "provided"
        return {r.name for r in enabled
                if any(dep not in present for dep in r.requires)}

    def diagnostics(self) -> List[Dict[str, Any]]:
        """Human/agent readable report: what is active, replaced or blocked."""
        report: List[Dict[str, Any]] = []
        for kind in ALL_KINDS:
            items = self.all(kind)
            if not items:
                continue
            enabled = [r for r in items if self.is_enabled(kind, r.name)]
            superseded = {n for r in enabled for n in r.replaces}
            unmet = self._unmet_requirements(kind, enabled, superseded)
            for reg in items:
                if not self.is_enabled(kind, reg.name):
                    state = 'disabled'
                elif reg.name in superseded:
                    state = 'replaced'
                elif reg.name in unmet:
                    state = 'blocked'
                else:
                    state = 'active'
                report.append({
                    'kind': kind, 'name': reg.name, 'state': state,
                    'priority': reg.priority, 'source': reg.source,
                    'replaces': list(reg.replaces),
                    'requires': list(reg.requires),
                    'metadata': reg.metadata,
                })
        return report

    def snapshot(self) -> Dict[str, List[str]]:
        return {kind: [r.name for r in self.active(kind)]
                for kind in ALL_KINDS if self.all(kind)}


#: The process-wide registry. Components register against this at import time.
REGISTRY = Registry()


def component(kind: str, name: str, **kwargs):
    """Decorator form: ``@component(kinds.EXPORTER, 'notebook')``."""
    return REGISTRY.register(kind, name, **kwargs)
