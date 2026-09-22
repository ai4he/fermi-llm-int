"""The model service: one facade over every registered provider.

Core code (chat, script review) talks to this object and never to a backend.
Adding a model means registering a provider; nothing here changes.

The public surface is deliberately the same three calls the monolithic
``ModelManager`` exposed — ``list_models``, ``generate``, ``complete`` — so
porting a caller is a rename, not a rewrite.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from ...core import kinds
from ...core.contracts import GenerationRequest, GenerationResult, ModelSpec
from .prompting import (DEFAULT_MODEL_ID, SEMANTIC_INTENT_MODELS, build_prompt)

log = logging.getLogger(__name__)


class ModelService:
    """Aggregates providers and routes every request to the right one."""

    def __init__(self, ctx):
        self.ctx = ctx
        self._providers: List[Any] = []
        self._by_backend: Dict[str, Any] = {}
        self._specs: Dict[str, ModelSpec] = {}
        self._train_data: Optional[List[dict]] = None
        self.reload()

    # -- wiring -----------------------------------------------------------
    def reload(self) -> None:
        """Rebuild the catalogue from the registry (after a plugin change)."""
        self._providers = self.ctx.build_stack(kinds.MODEL_PROVIDER)
        self._by_backend = {}
        self._specs = {}
        for provider in self._providers:
            backend = getattr(provider, 'backend', '') or type(provider).__name__
            self._by_backend.setdefault(backend, provider)
            for spec in provider.list_models():
                if spec.id in self._specs:
                    log.warning('model id %r offered twice; keeping %s',
                                spec.id, self._specs[spec.id].backend)
                    continue
                self._specs[spec.id] = spec
        log.info('model service: %d providers, %d models',
                 len(self._providers), len(self._specs))

    @property
    def providers(self) -> List[Any]:
        return list(self._providers)

    def provider_for(self, model_id: str):
        spec = self.spec(model_id)
        provider = self._by_backend.get(spec.backend)
        if provider is None:
            raise ValueError(f'no provider registered for backend '
                             f'{spec.backend!r} (model {model_id!r})')
        return provider

    def spec(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError:
            raise ValueError(f'Unknown model: {model_id}') from None

    def has(self, model_id: str) -> bool:
        return model_id in self._specs

    @property
    def default_model_id(self) -> str:
        configured = self.ctx.settings.default_model or DEFAULT_MODEL_ID
        if configured in self._specs:
            return configured
        for spec in self._specs.values():
            if spec.recommended:
                return spec.id
        return next(iter(self._specs), DEFAULT_MODEL_ID)

    # -- catalogue --------------------------------------------------------
    def list_models(self) -> List[Dict[str, Any]]:
        """The model picker payload, in registration order."""
        models = []
        for provider in self._providers:
            for spec in provider.list_models():
                if spec.id not in self._specs:
                    continue
                available, status, detail = provider.availability(spec)
                models.append({
                    'id': spec.id,
                    'name': spec.name,
                    'group': spec.group,
                    'backend': spec.backend,
                    'description': spec.description,
                    'requires_gpu': spec.requires_gpu,
                    'available': available,
                    'status': status,
                    'status_detail': detail,
                    'loaded': provider.is_loaded(spec.id),
                    'recommended': spec.recommended,
                })
        return models

    @property
    def gpu_available(self) -> bool:
        return any(getattr(p, 'gpu_available', False) for p in self._providers)

    @property
    def gpu_info(self) -> str:
        from .base import gpu_info
        return gpu_info()[1]

    # -- prompting --------------------------------------------------------
    @property
    def train_data(self) -> List[dict]:
        if self._train_data is None:
            path = os.path.join(self.ctx.settings.data_dir, 'train.json')
            try:
                with open(path) as handle:
                    self._train_data = json.load(handle)
            except Exception:                             # noqa: BLE001
                self._train_data = []
            log.info('few-shot examples: %d', len(self._train_data))
        return self._train_data

    def _prompt_builder(self, request: GenerationRequest, **overrides) -> str:
        params = dict(
            n_shots=request.n_shots, context=request.context,
            include_intent=request.include_intent,
            structured_envelope=(request.include_intent
                                 and request.spec.backend == 'openai'),
        )
        params.update(overrides)
        return build_prompt(request.user_prompt, self.train_data, **params)

    # -- generation -------------------------------------------------------
    def generate(self, model_id: str, user_prompt: str,
                 temperature: float = 0.1, n_shots: int = 3,
                 progress_cb=None, context=None,
                 codex_thread_id: Optional[str] = None
                 ) -> Tuple[str, str, dict]:
        """Generate YAML + Python. Returns ``(yaml, python, metadata)``."""
        spec = self.spec(model_id)
        provider = self.provider_for(model_id)

        # In edit mode the session's files are the reference; few-shot
        # examples actively hurt (models copy the example's target).
        edit_mode = bool(context and (context.get('yaml') or context.get('python')))
        if edit_mode:
            n_shots = 0

        request = GenerationRequest(
            model_id=model_id, spec=spec, user_prompt=user_prompt,
            context=context, temperature=temperature, n_shots=n_shots,
            include_intent=model_id in SEMANTIC_INTENT_MODELS,
            progress=progress_cb, thread_id=codex_thread_id,
            train_examples=self.train_data,
            prompt_builder=self._prompt_builder,
        )
        self.ctx.hooks.emit('model.selected', session=None, model_id=model_id)

        result = provider.generate(request)
        metadata = {
            'model_id': model_id,
            'model_name': spec.name,
            'backend': spec.backend,
            'edit_mode': edit_mode,
        }
        metadata.update(result.metadata)
        if result.semantic_intent:
            metadata['semantic_intent'] = result.semantic_intent
        elif request.include_intent:
            metadata['semantic_intent_missing'] = True
        return result.yaml, result.python, metadata

    def complete(self, model_id: str, prompt: str,
                 response_schema: Optional[dict] = None) -> Optional[str]:
        """Free-form completion used by the script-review agent.

        Returns ``None`` when the backend cannot answer an arbitrary prompt
        (``template`` has no model; a task-specific LoRA would answer in the
        wrong format), which callers treat as "fall back to heuristics".
        """
        spec = self.spec(model_id)
        provider = self.provider_for(model_id)
        if getattr(provider, 'uses_raw_prompt', False) or spec.backend == 'template':
            return None
        try:
            return provider.complete(spec, prompt, response_schema=response_schema)
        except TypeError:
            return provider.complete(spec, prompt)

    # -- lifecycle --------------------------------------------------------
    def shutdown(self) -> None:
        for provider in self._providers:
            try:
                provider.shutdown()
            except Exception as exc:                      # noqa: BLE001
                log.warning('provider %s shutdown failed: %s', provider, exc)
