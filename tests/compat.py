"""A name map from the old monolith to the split modules.

The tests were written against ``server.py`` and ``model_manager.py``. Rather
than rewrite every assertion (and lose the regression value of tests that
predate the refactor), they look names up through this map, which is also the
answer to "where did this function go?" for anyone porting their own code.

Every entry is a real module — nothing is re-exported for convenience, and a
name that no longer exists raises an error naming the modules that were
searched.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import List

#: Searched in order when resolving an attribute of the old ``server`` module.
SERVER_MODULES = (
    'fermi_llm.core.session',
    'fermi_llm.core.progress',
    'fermi_llm.core.jsonutil',
    'fermi_llm.core.runtime',
    'fermi_llm.services.chat',
    'fermi_llm.components.pipelines.review',
    'fermi_llm.components.pipelines.execute',
    'fermi_llm.components.exporters.run_bundle',
    'fermi_llm.components.exporters.notebook',
    'fermi_llm.components.exporters.artifacts_zip',
    'fermi_llm.components.guardrails.yamlops',
    'fermi_llm.components.intent.rule_based',
    'fermi_llm.components.knowledge.rag_docs',
    'fermi_llm.components.auth.tokens',
    'fermi_llm.components.auth.google',
    'fermi_llm.components.ui.demo_samples',
    'fermi_llm.fermipy.targets',
    'fermi_llm.fermipy.estimates',
    'fermi_llm.api.runs',
    'fermi_llm.api.auth',
    'fermi_llm.api.artifacts',
    'fermi_llm.api.meta',
    'fermi_llm.api.sessions',
    'fermi_llm.api.chat',
    'fermi_llm.api.knowledge',
    'fermi_llm.api.ui',
    'fermi_llm.services.runs',
    'fermi_llm.components.models.prompting',
)

#: Searched in order for the old ``model_manager`` module.
MODEL_MANAGER_MODULES = (
    # Providers first: each one owns the module-level constants its backend
    # reads, so patching (e.g. CODEX_BIN) reaches the code under test.
    'fermi_llm.components.models.providers.openai_api',
    'fermi_llm.components.models.providers.gemini',
    'fermi_llm.components.models.providers.codex_cli',
    'fermi_llm.components.models.providers.vllm_remote',
    'fermi_llm.components.models.providers.vllm_local',
    'fermi_llm.components.models.providers.huggingface_local',
    'fermi_llm.components.models.providers.template',
    'fermi_llm.components.models.prompting',
    'fermi_llm.components.models.base',
    'fermi_llm.components.models.service',
    'fermi_llm.components.models.settings',
)


class _Facade:
    """Resolves attributes across a list of modules, first match wins."""

    def __init__(self, label: str, modules):
        self._label = label
        self._modules: List[ModuleType] = [importlib.import_module(m)
                                           for m in modules]

    #: Globals the monolith kept at module level and the platform now keeps
    #: on the runtime. Resolved dynamically so tests see live state.
    RUNTIME_ALIASES = {
        'sessions': lambda rt: rt.sessions,
        'active_websockets': lambda rt: rt.websockets,
        'active_pipeline_runs': lambda rt: rt.pipeline_runs,
        'SESSIONS_DIR': lambda rt: rt.sessions_dir,
        'model_manager': lambda rt: rt.models,
        'agent': lambda rt: rt.agent,
        'MAX_ACTIVE_PIPELINES': lambda rt: rt.max_active_pipelines,
        # The catalogue is now the union of every provider's models.
        'MODEL_REGISTRY': lambda rt: {
            spec.id: dict(spec.options, name=spec.name, backend=spec.backend,
                          group=spec.group, description=spec.description,
                          recommended=spec.recommended,
                          requires_gpu=spec.requires_gpu)
            for provider in rt.models.providers
            for spec in provider.list_models()},
    }

    def __getattr__(self, name):
        if name in self.RUNTIME_ALIASES:
            from fermi_llm.core.runtime import RUNTIME
            return self.RUNTIME_ALIASES[name](RUNTIME)
        for module in self._modules:
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(
            f'{name!r} is not defined in any module of the {self._label} '
            f'group. Searched: '
            + ', '.join(m.__name__ for m in self._modules))

    #: Attributes of the facade itself, not of the modules behind it.
    _OWN = ('_label', '_modules')

    def __setattr__(self, name, value):
        # Tests monkeypatch module attributes; write to the owning module
        # (including private names like ``_terminate_pipeline_worker``).
        if name in self._OWN:
            return super().__setattr__(name, value)
        for module in self._modules:
            if hasattr(module, name):
                setattr(module, name, value)
                return
        super().__setattr__(name, value)

    def module_of(self, name: str) -> str:
        """Where a name lives now (used in the migration docs and by agents)."""
        for module in self._modules:
            if hasattr(module, name):
                return module.__name__
        return ''


def server():
    return _Facade('server', SERVER_MODULES)


def model_manager():
    facade = _Facade('model_manager', MODEL_MANAGER_MODULES)
    object.__setattr__(facade, 'ModelManager', ModelManager)
    return facade


class ModelManager:
    """The old façade over every backend, for tests that drive one directly.

    The platform itself has no such object: a provider owns its backend and
    :class:`~fermi_llm.components.models.service.ModelService` routes to it.
    This adapter exists so backend-level regression tests (payload shape,
    retry behaviour, Codex thread handling) keep running unchanged.
    """

    _PREFIX_TO_BACKEND = {
        '_generate_openai': 'openai', '_generate_gemini': 'gemini',
        '_generate_codex': 'codex', '_codex_ready': 'codex',
        '_generate_vllm_remote': 'vllm_remote',
        '_generate_vllm_local': 'vllm_local',
        '_local_vllm_served_model': 'vllm_local',
        '_generate_hf': 'huggingface', '_load_hf_model': 'huggingface',
        '_generate_vllm': 'vllm',
        '_openai_key': 'openai', '_gemini_keys': 'gemini',
        '_gemini_key_idx': 'gemini', '_clemson_key': 'vllm_remote',
    }

    def __init__(self, ctx=None):
        from fermi_llm.core.runtime import RUNTIME
        object.__setattr__(self, '_ctx', ctx or RUNTIME.ctx)
        object.__setattr__(self, '_service', self._ctx.models)
        providers = {p.backend: p for p in self._service.providers}
        object.__setattr__(self, '_providers', providers)

    def _provider_for(self, name):
        backend = self._PREFIX_TO_BACKEND.get(name)
        return self._providers.get(backend) if backend else None

    def __getattr__(self, name):
        provider = self._provider_for(name)
        if provider is not None and hasattr(provider, name):
            return getattr(provider, name)
        service = object.__getattribute__(self, '_service')
        if hasattr(service, name):
            return getattr(service, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        provider = self._provider_for(name)
        if provider is not None:
            setattr(provider, name, value)
            return
        object.__setattr__(self, name, value)
