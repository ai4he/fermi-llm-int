"""Managed local vLLM provider.

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
import threading
import signal
import sys

# Deployment-level settings for this backend (same env-var names as before).
CACHE_DIR = cfg.CACHE_DIR
VLLM_DIR = cfg.VLLM_DIR


MODELS = {
    # --- Local models (vLLM backend) ---
    'qwen3.5-0.8b-vllm': {
        'name': 'Qwen3.5-0.8B (vLLM)',
        'group': 'Local Models (vLLM)',
        'backend': 'vllm',
        'path': f'{CACHE_DIR}/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17',
        'hf_id': 'Qwen/Qwen3.5-0.8B',
        'max_input_tokens': 12288,
        'requires_gpu': True,
        'description': 'vLLM server - high throughput',
    },
    'qwen3.5-35b-a3b-vllm': {
        'name': 'Qwen3.5-35B-A3B (vLLM)',
        'group': 'Local Models (vLLM)',
        'backend': 'vllm',
        'path': f'{CACHE_DIR}/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307',
        'hf_id': 'Qwen/Qwen3.5-35B-A3B',
        'max_input_tokens': 10240,
        'requires_gpu': True,
        'description': 'vLLM server - best quality + throughput',
    },
}


class ManagedVLLMProvider(BaseProvider):
    """Starts and stops a vLLM server for a local snapshot on demand."""

    backend = 'vllm'
    MODELS = MODELS
    requires_gpu = True

    def __init__(self, ctx):
        super().__init__(ctx)
        self._vllm_process = None
        self._vllm_model_id = None
        self._vllm_port = 8100
        self._vllm_lock = threading.Lock()

    def availability(self, spec):
        if not self.gpu_available:
            return False, 'no_gpu', 'Requires GPU'
        path = self.model_info(spec.id).get('path', '')
        if path and not os.path.exists(path):
            return False, 'not_downloaded', 'Model weights not found'
        return True, 'available', ''

    def is_loaded(self, model_id):
        return self._vllm_model_id == model_id

    def _start_vllm_server(self, model_id):
        """Start a vLLM OpenAI-compatible server for the given model."""
        info = self.model_info(model_id)
        if self._vllm_model_id == model_id and self._vllm_process is not None:
            if self._vllm_process.poll() is None:
                return  # already running

        with self._vllm_lock:
            self._stop_vllm_server()

            model_path = info['path']
            port = self._vllm_port

            # Use vLLM venv
            vllm_python = os.path.join(VLLM_DIR, '.venv', 'bin', 'python')
            if not os.path.exists(vllm_python):
                vllm_python = 'python'  # fallback

            cmd = [
                vllm_python, '-m', 'vllm.entrypoints.openai.api_server',
                '--model', model_path,
                '--port', str(port),
                '--trust-remote-code',
                '--max-model-len', str(info.get('max_input_tokens', 8192) + 4096),
            ]

            print(f"[ModelManager] Starting vLLM server: {info['name']} on port {port}")
            self._vllm_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            self._vllm_model_id = model_id

            # Wait for server to be ready (up to 120s)
            import urllib.request
            for i in range(60):
                time.sleep(2)
                if self._vllm_process.poll() is not None:
                    stderr = self._vllm_process.stderr.read().decode()[:500]
                    raise RuntimeError(f"vLLM server exited: {stderr}")
                try:
                    req = urllib.request.urlopen(
                        f'http://localhost:{port}/health', timeout=2,
                    )
                    if req.status == 200:
                        print(f"[ModelManager] vLLM server ready on port {port}")
                        return
                except Exception:
                    pass

            raise RuntimeError("vLLM server failed to start within 120s")

    def _stop_vllm_server(self):
        """Stop the vLLM server process."""
        if self._vllm_process is not None:
            print(f"[ModelManager] Stopping vLLM server: {self._vllm_model_id}")
            try:
                os.killpg(os.getpgid(self._vllm_process.pid), signal.SIGTERM)
                self._vllm_process.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._vllm_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._vllm_process = None
            self._vllm_model_id = None

    def _generate_vllm(self, model_id, prompt, max_new_tokens=4096,
                       temperature=0.1, top_p=0.95):
        """Generate text via vLLM OpenAI-compatible API."""
        import urllib.request

        self._start_vllm_server(model_id)
        info = self.model_info(model_id)
        port = self._vllm_port

        payload = json.dumps({
            'model': info['path'],
            'prompt': prompt,
            'max_tokens': max_new_tokens,
            'temperature': max(temperature, 0.01),
            'top_p': top_p,
        }).encode()

        req = urllib.request.Request(
            f'http://localhost:{port}/v1/completions',
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        resp = urllib.request.urlopen(req, timeout=300)
        result = json.loads(resp.read().decode())
        return result['choices'][0]['text']

    def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.time()
        raw = self._generate_vllm(request.model_id, request.build_prompt(),
                                  temperature=request.temperature,
                                  progress_cb=request.progress)
        return self.finish(request, raw, start)

    def complete(self, spec, prompt, **kwargs) -> str:
        return self._generate_vllm(spec.id, prompt,
                                   temperature=kwargs.get('temperature', 0.1))

    def shutdown(self):
        self._stop_vllm_server()


REGISTRY.register(kinds.MODEL_PROVIDER, 'vllm', ManagedVLLMProvider,
                  priority=250, source='core',
                  metadata={'label': 'Managed local vLLM'})
