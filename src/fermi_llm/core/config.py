"""Settings and per-deployment plugin configuration.

Environment variables keep the names the monolith used, so an existing
deployment (``configs/env.sh``) keeps working unchanged. Anything that used
to be hardcoded in a module is a field here instead.

The optional ``configs/plugins.toml`` selects which components a deployment
runs::

    [plugins]
    paths = ["plugins", "/opt/otherlab/fermi_modules"]

    [components]
    disable = ["guardrail:full_data_defaults"]
    execution_backend = "local_subprocess"
    skin = "classic"

    [settings."vendor:my_exporter"]
    include_raw = true
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:                                    # Python 3.11+
    import tomllib
except ModuleNotFoundError:             # pragma: no cover
    tomllib = None


def _env(name: str, default: str = '') -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _package_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_project_dir() -> str:
    """The checkout root: three levels up from ``src/fermi_llm/core``."""
    return os.path.dirname(os.path.dirname(_package_root()))


@dataclass
class Settings:
    """Everything the platform reads from the environment, in one place."""

    project_dir: str = field(default_factory=lambda: _env(
        'FERMI_LLM_PROJECT_DIR') or _default_project_dir())
    host: str = field(default_factory=lambda: _env('HOST', '0.0.0.0'))
    port: int = field(default_factory=lambda: _env_int('PORT', 8765))

    # -- paths ------------------------------------------------------------
    sessions_dir: str = ''
    data_dir: str = ''
    lat_data_dir: str = ''
    fermi_data_dir: str = ''
    results_dir: str = ''
    rag_dir: str = ''
    configs_dir: str = ''
    static_dir: str = ''
    plugin_paths: List[str] = field(default_factory=list)

    # -- behaviour --------------------------------------------------------
    default_model: str = field(
        default_factory=lambda: _env('FERMI_LLM_DEFAULT_MODEL'))
    max_active_pipelines: int = field(
        default_factory=lambda: _env_int('FERMI_LLM_MAX_ACTIVE_PIPELINES', 4))
    confirm_threshold_min: float = field(
        default_factory=lambda: _env_float(
            'FERMI_LLM_CONFIRM_THRESHOLD_MIN', 60))
    fit_timeout: int = field(
        default_factory=lambda: _env_int('FERMI_LLM_FIT_TIMEOUT', 1800))
    default_time_range_days: int = field(
        default_factory=lambda: _env_int(
            'FERMI_LLM_DEFAULT_TIME_RANGE_DAYS', 365))

    # -- component selection ---------------------------------------------
    execution_backend: str = field(
        default_factory=lambda: _env('FERMI_LLM_EXECUTION_BACKEND'))
    session_store: str = field(
        default_factory=lambda: _env('FERMI_LLM_SESSION_STORE'))
    intent_analyzer: str = field(
        default_factory=lambda: _env('FERMI_LLM_INTENT_ANALYZER'))
    skin: str = field(default_factory=lambda: _env('FERMI_LLM_SKIN'))
    disabled: List[str] = field(default_factory=list)
    enabled_only: List[str] = field(default_factory=list)
    component_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        root = self.project_dir
        self.configs_dir = self.configs_dir or os.path.join(root, 'configs')
        self.data_dir = self.data_dir or (
            _env('FERMI_LLM_DATA_DIR') or os.path.join(root, 'data'))
        self.lat_data_dir = self.lat_data_dir or (
            _env('FERMI_LLM_LAT_DATA_DIR') or os.path.join(root, 'lat_data'))
        self.fermi_data_dir = self.fermi_data_dir or _env(
            'FERMI_LLM_FERMI_DATA_DIR', '/workspace/fermillm/fermi-data')
        self.results_dir = self.results_dir or (
            _env('FERMI_LLM_RESULTS_DIR') or os.path.join(root, 'results'))
        self.static_dir = self.static_dir or os.path.join(
            _package_root(), 'web', 'static')
        self.sessions_dir = self.sessions_dir or (
            _env('FERMI_LLM_SESSIONS_DIR')
            or os.path.join(root, 'webapp', 'sessions'))
        self.rag_dir = self.rag_dir or (
            _env('FERMI_LLM_RAG_DIR')
            or os.path.join(root, 'webapp', 'rag_docs'))
        if not self.plugin_paths:
            env_paths = _env('FERMI_LLM_PLUGIN_PATH')
            self.plugin_paths = (
                [p for p in env_paths.split(os.pathsep) if p]
                or [os.path.join(root, 'plugins')])

    # -- helpers ----------------------------------------------------------
    def config_file(self, name: str, env_var: str = '') -> str:
        """Path of a file in ``configs/``, overridable by ``env_var``."""
        if env_var:
            override = _env(env_var)
            if override:
                return override
        return os.path.join(self.configs_dir, name)

    def for_component(self, kind: str, name: str) -> Dict[str, Any]:
        """Per-component settings block from ``plugins.toml``."""
        return dict(self.component_settings.get(f'{kind}:{name}', {}))

    def is_disabled(self, kind: str, name: str) -> bool:
        label = f'{kind}:{name}'
        if self.enabled_only and label not in self.enabled_only:
            return True
        return label in self.disabled

    def ensure_dirs(self) -> None:
        for path in (self.sessions_dir, self.rag_dir, self.results_dir):
            os.makedirs(path, exist_ok=True)


def load_settings(config_path: Optional[str] = None) -> Settings:
    """Build :class:`Settings` from the environment plus ``plugins.toml``."""
    settings = Settings()
    path = config_path or os.path.join(settings.configs_dir, 'plugins.toml')
    data: Dict[str, Any] = {}
    if tomllib and os.path.isfile(path):
        with open(path, 'rb') as handle:
            data = tomllib.load(handle)

    plugins = data.get('plugins') or {}
    extra_paths = [p for p in plugins.get('paths', []) if p]
    for raw in extra_paths:
        resolved = raw if os.path.isabs(raw) else os.path.join(
            settings.project_dir, raw)
        if resolved not in settings.plugin_paths:
            settings.plugin_paths.append(resolved)

    components = data.get('components') or {}
    settings.disabled = list(components.get('disable', []))
    settings.enabled_only = list(components.get('only', []))
    settings.execution_backend = (settings.execution_backend
                                  or components.get('execution_backend', ''))
    settings.session_store = (settings.session_store
                              or components.get('session_store', ''))
    settings.intent_analyzer = (settings.intent_analyzer
                                or components.get('intent_analyzer', ''))
    settings.skin = settings.skin or components.get('skin', '')
    settings.component_settings = dict(data.get('settings') or {})
    return settings
