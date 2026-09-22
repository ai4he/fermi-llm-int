"""Remote vLLM provider.

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
CLEMSON_VLLM_BASE_URL = cfg.CLEMSON_VLLM_BASE_URL
CLEMSON_VLLM_KEY_FILE = cfg.CLEMSON_VLLM_KEY_FILE


MODELS = {
    # --- Clemson RCD vLLM (OpenAI-compatible remote endpoint) ---
    'qwen3.6-27b-lora-fermipy-clemson': {
        'name': 'Qwen3.6-27B-FP8 + FermiPy LoRA (Clemson RCD)',
        'group': 'Clemson vLLM',
        'backend': 'vllm_remote',
        'remote_model': 'Qwen3.6-27B-LoRA-fermipy',   # name as registered in the remote vLLM
        'base_url': CLEMSON_VLLM_BASE_URL,
        'key_file': CLEMSON_VLLM_KEY_FILE,
        'max_input_tokens': 16384,
        'max_new_tokens': 4096,
        'enable_thinking': False,
        'requires_gpu': False,                          # remote, no local GPU needed
        'description': 'Task-specific fine-tune trained in this project (remote vLLM, no local GPU needed)',
        'recommended': False,
    },
}


CLEMSON_REMOTE_CHAT_MODELS = cfg.CLEMSON_REMOTE_CHAT_MODELS

for _rid, _label in CLEMSON_REMOTE_CHAT_MODELS:
    MODELS.setdefault(_rid, {
        'name': f'{_label} (Clemson vLLM)',
        'group': 'Clemson vLLM',
        'backend': 'vllm_remote',
        'remote_model': _rid,
        'base_url': cfg.CLEMSON_VLLM_BASE_URL,
        'key_file': cfg.CLEMSON_VLLM_KEY_FILE,
        'max_input_tokens': 16384,
        'max_new_tokens': 4096,
        'enable_thinking': False,
        'requires_gpu': False,
        'description': 'Served by the Clemson RCD vLLM endpoint (remote, no local GPU).',
        'recommended': False,
    })


class RemoteVLLMProvider(BaseProvider):
    """An OpenAI-compatible vLLM endpoint hosted elsewhere (Clemson RCD).

    The fine-tuned LoRA entry deliberately skips the few-shot prompt: it was
    trained on the canonical chat, so the prefix would shift the input
    distribution. ``uses_raw_prompt`` marks that.
    """

    backend = 'vllm_remote'
    MODELS = MODELS
    uses_raw_prompt = True

    def __init__(self, ctx):
        super().__init__(ctx)
        self._clemson_key = self._load_clemson_key()

    @staticmethod
    def _load_clemson_key():
        """Load the bearer token for the Clemson RCD vLLM endpoint."""
        try:
            with open(CLEMSON_VLLM_KEY_FILE) as f:
                key = f.read().strip()
            return key or None
        except Exception:
            return None

    def availability(self, spec):
        if not self._clemson_key:
            return False, 'no_api_key', 'No Clemson vLLM API key configured'
        return True, 'available', ''

    def _generate_vllm_remote(self, model_id: str, user_prompt: str,
                              temperature: float = 0.1, top_p: float = 0.95) -> str:
        """Call a remote OpenAI-compatible vLLM endpoint.

        For models trained as instruction-following chat models (such as the
        Qwen3.6-27B-LoRA-fermipy adapter), we send the canonical system + user
        chat used at training time. Few-shot prompting is intentionally skipped:
        the LoRA already encodes the schema.
        """
        info = self.model_info(model_id)
        if not self._clemson_key:
            raise RuntimeError(f"No API key configured for {model_id}")

        base_url = info['base_url']
        remote_model = info['remote_model']
        max_new_tokens = info.get('max_new_tokens', 4096)
        enable_thinking = info.get('enable_thinking', False)

        # Canonical training-time chat. SYSTEM matches what the LoRA was trained with.
        system_msg = (
            "You are a FermiPy expert. Given a natural-language description of a "
            "Fermi-LAT gamma-ray analysis, generate two artifacts:\n"
            "1. A YAML configuration file for FermiPy's GTAnalysis.\n"
            "2. A Python script that uses the FermiPy API to perform the analysis.\n\n"
            "Return EXACTLY one fenced block of each, in this order, with no extra prose:\n\n"
            "### YAML Configuration:\n```yaml\n<yaml>\n```\n\n"
            "### Python Script:\n```python\n<python>\n```"
        )
        user_msg = (
            "Generate the FermiPy YAML configuration and Python script for the "
            "following analysis:\n\n" + user_prompt.rstrip()
        )

        payload = {
            'model': remote_model,
            'messages': [
                {'role': 'system', 'content': system_msg},
                {'role': 'user',   'content': user_msg},
            ],
            'temperature': max(temperature, 0.01),
            'top_p': top_p,
            'max_tokens': max_new_tokens,
            # Disable Qwen's auto-injected <think> block — the LoRA was trained
            # to emit YAML/Python directly after the user turn.
            'chat_template_kwargs': {'enable_thinking': enable_thinking},
        }
        body = json.dumps(payload).encode()
        import urllib.request, urllib.error
        req = urllib.request.Request(
            f'{base_url.rstrip("/")}/chat/completions',
            data=body,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self._clemson_key}',
            },
            method='POST',
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors='replace')[:500]
            raise RuntimeError(f"Clemson vLLM HTTP {e.code}: {err_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Clemson vLLM unreachable: {e.reason}")
        data = json.loads(resp.read().decode())
        if not data.get('choices'):
            raise RuntimeError(f"Empty response from Clemson vLLM: {data}")
        msg = data['choices'][0]['message']
        # When thinking is disabled, the response is in `content`; when thinking
        # is on, vLLM may put it in `reasoning`. We concatenate both to be safe.
        content = msg.get('content') or ''
        reasoning = msg.get('reasoning') or ''
        text = (reasoning + '\n' + content) if reasoning else content
        return text

    def generate(self, request: GenerationRequest) -> GenerationResult:
        info = self.model_info(request.model_id)
        request.report('api_call',
                       f'Sending prompt to remote vLLM ({info["remote_model"]} '
                       f'@ {info["base_url"]})...')
        start = time.time()
        raw = self._generate_vllm_remote(request.model_id, request.user_prompt,
                                         temperature=request.temperature)
        result = self.finish(request, raw, start)
        result.metadata['method'] = 'vllm_remote_lora'
        return result

    def complete(self, spec, prompt, **kwargs) -> str:
        return self._generate_vllm_remote(spec.id, prompt,
                                          temperature=kwargs.get('temperature', 0.1))


REGISTRY.register(kinds.MODEL_PROVIDER, 'vllm_remote', RemoteVLLMProvider,
                  priority=100, source='core',
                  metadata={'label': 'Clemson RCD vLLM'})
