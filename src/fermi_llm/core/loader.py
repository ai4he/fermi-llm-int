"""Plugin discovery.

Three sources, in this order, each one able to replace what came before:

1. **Built-in components** — ``fermi_llm.components.*``, imported by
   :func:`load_builtins`. These are ordinary plugins that happen to ship in
   this repository; nothing in core special-cases them.
2. **Installed distributions** — any package exposing the
   ``fermi_llm.plugins`` entry point. This is how another lab ships its
   modules as a pip package without touching this repo.
3. **Directories** — every path in ``settings.plugin_paths`` (default
   ``plugins/``). A plugin is a directory or a ``.py`` file exposing
   ``register(registry, ctx)``; useful while developing, and for
   site-specific code that is never published.

A failing plugin is recorded in :attr:`LoadReport.failures` and skipped; the
platform still starts. That is deliberate: a half-installed module from one
institute must not stop everyone else's deployment.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import pkgutil
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import PluginLoadError
from .registry import REGISTRY, Registry

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = 'fermi_llm.plugins'

#: Built-in component packages, imported in order. Import order does not
#: define run order — ``priority`` on each registration does.
BUILTIN_PACKAGES = (
    'fermi_llm.components.storage',
    'fermi_llm.components.auth',
    'fermi_llm.components.models',
    'fermi_llm.components.intent',
    'fermi_llm.components.guardrails',
    'fermi_llm.components.validators',
    'fermi_llm.components.datasources',
    'fermi_llm.components.execution',
    'fermi_llm.components.pipelines',
    'fermi_llm.components.exporters',
    'fermi_llm.components.knowledge',
    'fermi_llm.components.ui',
)


@dataclass
class LoadReport:
    loaded: List[str] = field(default_factory=list)
    failures: List[Dict[str, str]] = field(default_factory=list)

    def add_failure(self, source: str, exc: BaseException) -> None:
        self.failures.append({
            'source': source,
            'error': f'{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc(limit=5),
        })
        log.warning('plugin %s failed to load: %s', source, exc)

    def summary(self) -> str:
        parts = [f'{len(self.loaded)} plugin sources loaded']
        if self.failures:
            parts.append(f'{len(self.failures)} failed: '
                         + ', '.join(f['source'] for f in self.failures))
        return '; '.join(parts)


def _import_submodules(package_name: str, report: LoadReport) -> None:
    package = importlib.import_module(package_name)
    report.loaded.append(package_name)
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith('_'):
            continue
        full = f'{package_name}.{info.name}'
        try:
            importlib.import_module(full)
            report.loaded.append(full)
        except Exception as exc:                          # noqa: BLE001
            report.add_failure(full, exc)


def load_builtins(report: Optional[LoadReport] = None) -> LoadReport:
    report = report or LoadReport()
    for package in BUILTIN_PACKAGES:
        try:
            _import_submodules(package, report)
        except Exception as exc:                          # noqa: BLE001
            report.add_failure(package, exc)
    return report


def load_entry_points(ctx: Any, report: LoadReport,
                      registry: Registry = REGISTRY) -> None:
    try:
        from importlib.metadata import entry_points
    except ImportError:                                   # pragma: no cover
        return
    try:
        found = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:                                     # pragma: no cover
        found = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore
    for entry in found:
        try:
            target = entry.load()
            _invoke_register(target, registry, ctx)
            report.loaded.append(f'entry_point:{entry.name}')
        except Exception as exc:                          # noqa: BLE001
            report.add_failure(f'entry_point:{entry.name}', exc)


def _invoke_register(target: Any, registry: Registry, ctx: Any) -> None:
    """Call a plugin's ``register`` in whichever shape it provides."""
    fn = getattr(target, 'register', target)
    if not callable(fn):
        raise PluginLoadError(
            'plugin must expose a callable register(registry, ctx)')
    try:
        fn(registry, ctx)
    except TypeError:
        fn(registry)                      # register(registry) is also fine


def _load_path_plugin(path: str, registry: Registry, ctx: Any,
                      report: LoadReport) -> None:
    name = os.path.basename(path).removesuffix('.py')
    module_file = path
    if os.path.isdir(path):
        for candidate in ('plugin.py', '__init__.py'):
            if os.path.isfile(os.path.join(path, candidate)):
                module_file = os.path.join(path, candidate)
                break
        else:
            return
    spec = importlib.util.spec_from_file_location(
        f'fermi_llm_plugin_{name}', module_file)
    if spec is None or spec.loader is None:
        raise PluginLoadError(f'cannot import {module_file}')
    module = importlib.util.module_from_spec(spec)
    # A directory plugin may import its own submodules by relative path.
    plugin_dir = os.path.dirname(module_file)
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _invoke_register(module, registry, ctx)
    report.loaded.append(f'path:{path}')


def load_paths(paths, ctx: Any, report: LoadReport,
               registry: Registry = REGISTRY) -> None:
    for root in paths:
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            if entry.startswith(('_', '.')):
                continue
            full = os.path.join(root, entry)
            if not (os.path.isdir(full) or entry.endswith('.py')):
                continue
            try:
                _load_path_plugin(full, registry, ctx, report)
            except Exception as exc:                      # noqa: BLE001
                report.add_failure(f'path:{full}', exc)


def apply_config(settings, registry: Registry = REGISTRY) -> None:
    """Disable components the deployment turned off in ``plugins.toml``."""
    from .kinds import ALL_KINDS
    for kind in ALL_KINDS:
        for reg in registry.all(kind):
            if settings.is_disabled(kind, reg.name):
                registry.disable(kind, reg.name)
            else:
                registry.enable(kind, reg.name)


def load_all(ctx: Any, registry: Registry = REGISTRY) -> LoadReport:
    """Load built-ins, entry points and path plugins, then apply config."""
    report = LoadReport()
    load_builtins(report)
    load_entry_points(ctx, report, registry)
    load_paths(getattr(ctx.settings, 'plugin_paths', []), ctx, report, registry)
    apply_config(ctx.settings, registry)
    return report
