"""OpenAI API provider.

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
OPENAI_BASE_URL = cfg.OPENAI_BASE_URL
OPENAI_KEY_FILE = cfg.OPENAI_KEY_FILE
OPENAI_REASONING_EFFORT = cfg.OPENAI_REASONING_EFFORT


MODELS = {
    # --- OpenAI API ---
    'gpt-5.6-luna': {
        'name': 'ChatGPT Luna 5.6',
        'group': 'OpenAI API',
        'backend': 'openai',
        'openai_model': 'gpt-5.6-luna',
        'base_url': OPENAI_BASE_URL,
        'key_file': OPENAI_KEY_FILE,
        'reasoning_effort': OPENAI_REASONING_EFFORT,
        # Reasoning tokens count against this budget, so it must be generous
        # enough to hold the chain-of-thought AND the YAML+Python output.
        'max_new_tokens': 16000,
        'requires_gpu': False,
        'description': "OpenAI cloud API - GPT-5.6 Luna reasoning model "
                       "(reasoning effort: high).",
        'recommended': True,
    },
}


class OpenAIProvider(BaseProvider):
    """OpenAI-compatible chat/completions (the ChatGPT Luna entry)."""

    backend = 'openai'
    MODELS = MODELS

    def __init__(self, ctx):
        super().__init__(ctx)
        self._openai_key = self._load_openai_key()

    @staticmethod
    def _load_openai_key():
        """Load the OpenAI API key (single bearer token)."""
        try:
            with open(OPENAI_KEY_FILE) as f:
                key = f.read().strip()
            return key or None
        except Exception:
            return None

    def availability(self, spec):
        if not self._openai_key:
            return False, 'no_api_key', 'No OpenAI API key configured'
        return True, 'available', ''

    def _generate_openai(self, model_id: str, prompt: str,
                         response_schema: Optional[dict] = None) -> str:
        """Call the OpenAI chat/completions API for a GPT-5.x reasoning model.

        GPT-5.x reasoning models differ from the vLLM/Gemini paths:
          * they require `max_completion_tokens` (legacy `max_tokens` is rejected),
          * they reject any non-default `temperature`, so we send none,
          * they accept `reasoning_effort`; we use the per-model setting (high),
            which spends hidden reasoning tokens against the completion budget —
            hence the generous max_new_tokens on the registry entry.
        The schema-guided few-shot `prompt` (already built by build_prompt) is
        sent as a single user turn, matching the local-vLLM path.
        """
        info = self.model_info(model_id)
        if not self._openai_key:
            raise RuntimeError(f"No OpenAI API key configured for {model_id}")

        base_url = info['base_url']
        openai_model = info['openai_model']
        payload = {
            'model': openai_model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_completion_tokens': info.get('max_new_tokens', 16000),
        }
        if response_schema is not None:
            payload['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'fermipy_script_review',
                    'strict': True,
                    'schema': response_schema,
                },
            }
        elif model_id in SEMANTIC_INTENT_MODELS:
            payload['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'fermipy_semantic_generation',
                    'strict': True,
                    'schema': SEMANTIC_GENERATION_SCHEMA,
                },
            }
        effort = info.get('reasoning_effort')
        if effort:
            payload['reasoning_effort'] = effort

        import urllib.request, urllib.error
        req = urllib.request.Request(
            f'{base_url.rstrip("/")}/chat/completions',
            data=json.dumps(payload).encode(),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self._openai_key}',
            },
            method='POST',
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors='replace')[:500]
            raise RuntimeError(f"OpenAI HTTP {e.code}: {err_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenAI API unreachable: {e.reason}")
        data = json.loads(resp.read().decode())
        if not data.get('choices'):
            raise RuntimeError(f"Empty response from OpenAI API: {data}")
        choice = data['choices'][0]
        content = (choice.get('message') or {}).get('content') or ''
        if not content and choice.get('finish_reason') == 'length':
            raise RuntimeError(
                "OpenAI response truncated before any content was returned "
                "(reasoning consumed the token budget). Increase max_new_tokens "
                "or lower reasoning_effort."
            )
        return content

    def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt = request.build_prompt()
        request.report('api_call',
                       f'Sending prompt to {request.spec.name} '
                       f'(reasoning_effort={OPENAI_REASONING_EFFORT})...')
        start = time.time()
        raw = self._generate_openai(request.model_id, prompt)
        if not request.include_intent:
            return self.finish(request, raw, start)
        # Structured generation: this backend answers with a strict JSON
        # envelope {intent, yaml, python} rather than markdown code blocks,
        # so the interpretation travels with the files it produced.
        return self.finish_envelope(request, raw, start)

    def complete(self, spec, prompt, **kwargs) -> str:
        return self._generate_openai(spec.id, prompt, raw_prompt=True,
                                     temperature=kwargs.get('temperature', 0.1))


REGISTRY.register(kinds.MODEL_PROVIDER, 'openai', OpenAIProvider,
                  priority=210, source='core',
                  metadata={'label': 'OpenAI API'})
