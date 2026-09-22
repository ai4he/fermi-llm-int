"""Gemini API provider.

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
API_KEYS_FILE = cfg.API_KEYS_FILE


MODELS = {
    # --- Google Gemini API ---
    'gemini-3.5-flash': {
        'name': 'Gemini 3.5 Flash',
        'group': 'Gemini API',
        'backend': 'gemini',
        'gemini_model': 'gemini-3.5-flash',
        'requires_gpu': False,
        'description': 'Google cloud API - latest Gemini Flash (default)',
        'recommended': True,
    },
    'gemini-2.5-flash-lite': {
        'name': 'Gemini 2.5 Flash Lite',
        'group': 'Gemini API',
        'backend': 'gemini',
        'gemini_model': 'gemini-2.5-flash-lite',
        'requires_gpu': False,
        'description': 'Google cloud API - fast & free tier',
    },
    'gemini-2.0-flash': {
        'name': 'Gemini 2.0 Flash',
        'group': 'Gemini API',
        'backend': 'gemini',
        'gemini_model': 'gemini-2.0-flash',
        'requires_gpu': False,
        'description': 'Google cloud API - balanced',
    },
    'gemini-3.1-flash-lite': {
        'name': 'Gemini 3.1 Flash Lite',
        'group': 'Gemini API',
        'backend': 'gemini',
        'gemini_model': 'gemini-3.1-flash-lite-preview',
        'requires_gpu': False,
        'description': 'Latest Gemini - 100% Level 3 pass rate',
    },
}


class GeminiProvider(BaseProvider):
    """Google Gemini REST API (keys rotate on quota errors)."""

    backend = 'gemini'
    MODELS = MODELS

    def __init__(self, ctx):
        super().__init__(ctx)
        self._gemini_keys = self._load_gemini_keys()
        self._gemini_key_idx = 0

    @staticmethod
    def _load_gemini_keys():
        try:
            with open(API_KEYS_FILE) as f:
                return json.load(f)
        except Exception:
            return []

    def availability(self, spec):
        if not self._gemini_keys:
            return False, 'no_api_key', 'No Gemini API keys configured'
        return True, 'available', ''

    def _generate_gemini(self, model_id, prompt, max_tokens=4096,
                         temperature=0.1, top_p=0.95):
        """Generate text via Google Gemini API with key rotation."""
        import google.generativeai as genai

        info = self.model_info(model_id)
        gemini_model_name = info['gemini_model']

        for attempt in range(len(self._gemini_keys)):
            key_idx = (self._gemini_key_idx + attempt) % len(self._gemini_keys)
            key = self._gemini_keys[key_idx]
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel(gemini_model_name)
                config = genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    top_p=top_p,
                )
                response = model.generate_content(
                    prompt,
                    generation_config=config,
                    request_options={'timeout': 60},
                )
                self._gemini_key_idx = key_idx
                return response.text
            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                    print(f"[ModelManager] Gemini key {key_idx} rate limited, trying next...")
                    time.sleep(2)
                    continue
                else:
                    raise RuntimeError(f"Gemini API error: {error_str[:300]}")

        raise RuntimeError("All Gemini API keys exhausted (rate limited)")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt = request.build_prompt()
        request.report('api_call', f'Sending prompt to {request.spec.name}...')
        start = time.time()
        raw = self._generate_gemini(request.model_id, prompt,
                                    temperature=request.temperature)
        return self.finish(request, raw, start)

    def complete(self, spec, prompt, **kwargs) -> str:
        return self._generate_gemini(spec.id, prompt,
                                     temperature=kwargs.get('temperature', 0.1))


REGISTRY.register(kinds.MODEL_PROVIDER, 'gemini', GeminiProvider,
                  priority=200, source='core',
                  metadata={'label': 'Google Gemini API'})
