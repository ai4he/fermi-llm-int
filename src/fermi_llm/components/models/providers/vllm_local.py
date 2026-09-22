"""Local vLLM provider.

Moved verbatim out of the monolithic ``ModelManager`` so this backend can be
disabled, replaced or copied as a starting point for another lab's provider.
Register one of these per backend; the model picker is the union of every
active provider's :meth:`list_models`.
"""

import json
import os
import re
import subprocess
import time
from typing import Optional

from ....core import kinds
from ....core.contracts import GenerationRequest, GenerationResult, ModelSpec
from ....core.registry import REGISTRY
from ..base import BaseProvider, read_key_file
from .. import settings as cfg
from ..prompting import (SEMANTIC_GENERATION_SCHEMA, SEMANTIC_INTENT_MODELS,
                         extract_python, extract_semantic_intent, extract_yaml)

# Deployment-level settings for this backend (same env-var names as before).
LOCAL_VLLM_BASE_URL = cfg.LOCAL_VLLM_BASE_URL
LOCAL_VLLM_KEY = cfg.LOCAL_VLLM_KEY


MODELS = {
    # --- Generic local vLLM (model-agnostic) ---
    'local-vllm': {
        'name': 'Local Model (vLLM)',
        'group': 'Local vLLM',
        'backend': 'vllm_local',
        'base_url': LOCAL_VLLM_BASE_URL,
        'max_new_tokens': 4096,
        'requires_gpu': False,   # served by an already-running local vLLM
        'description': 'Whatever model the local vLLM server is currently serving.',
        'recommended': False,
    },
}


class LocalVLLMProvider(BaseProvider):
    """Whatever model an OpenAI-compatible vLLM on this host is serving."""

    backend = 'vllm_local'
    MODELS = MODELS

    @staticmethod
    def _local_vllm_served_model(base_url, timeout=1.5):
        """Return the id of the model the local vLLM is currently serving.

        Queries {base_url}/models and returns the first served model id, or
        None if the server is unreachable or serving nothing. This is what
        makes the 'Local Model' entry model-agnostic.
        """
        import urllib.request, urllib.error
        url = f'{base_url.rstrip("/")}/models'
        headers = {}
        if LOCAL_VLLM_KEY:
            headers['Authorization'] = f'Bearer {LOCAL_VLLM_KEY}'
        try:
            req = urllib.request.Request(url, headers=headers, method='GET')
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = json.loads(resp.read().decode())
            models = data.get('data') or []
            return models[0]['id'] if models else None
        except Exception:
            return None

    def availability(self, spec):
        served = self._local_vllm_served_model(self.model_info(spec.id)['base_url'])
        if served:
            return True, 'available', 'Local vLLM server is running'
        return False, 'not_running', 'No local vLLM server detected'

    def _generate_vllm_local(self, model_id: str, prompt: str,
                             temperature: float = 0.1, top_p: float = 0.95) -> str:
        """Call the local OpenAI-compatible vLLM server with a schema-guided
        prompt, targeting whatever model it is currently serving."""
        info = self.model_info(model_id)
        base_url = info['base_url']
        served = self._local_vllm_served_model(base_url)
        if not served:
            raise RuntimeError(
                f"Local vLLM server at {base_url} is not reachable or is not "
                f"serving a model. Start a vLLM server on this host first."
            )
        payload = {
            'model': served,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': max(temperature, 0.01),
            'top_p': top_p,
            'max_tokens': info.get('max_new_tokens', 4096),
            # Ask the model to skip its chain-of-thought and emit the YAML/Python
            # directly. For reasoning models (Qwen3, etc.) this is a large speedup
            # and avoids burning the token budget on reasoning; templates without
            # a thinking mode simply ignore the extra kwarg.
            'chat_template_kwargs': {'enable_thinking': False},
        }
        import urllib.request, urllib.error
        headers = {'Content-Type': 'application/json'}
        if LOCAL_VLLM_KEY:
            headers['Authorization'] = f'Bearer {LOCAL_VLLM_KEY}'
        req = urllib.request.Request(
            f'{base_url.rstrip("/")}/chat/completions',
            data=json.dumps(payload).encode(), headers=headers, method='POST',
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Local vLLM HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Local vLLM unreachable: {e.reason}")
        data = json.loads(resp.read().decode())
        if not data.get('choices'):
            raise RuntimeError(f"Empty response from local vLLM: {data}")
        msg = data['choices'][0]['message']
        content = msg.get('content') or ''
        reasoning = msg.get('reasoning_content') or msg.get('reasoning') or ''
        return (reasoning + '\n' + content) if reasoning else content

    def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.time()
        raw = self._generate_vllm_local(request.model_id, request.build_prompt(),
                                        temperature=request.temperature)
        return self.finish(request, raw, start)

    def complete(self, spec, prompt, **kwargs) -> str:
        return self._generate_vllm_local(spec.id, prompt,
                                         temperature=kwargs.get('temperature', 0.1))


REGISTRY.register(kinds.MODEL_PROVIDER, 'vllm_local', LocalVLLMProvider,
                  priority=230, source='core',
                  metadata={'label': 'Local vLLM server'})
