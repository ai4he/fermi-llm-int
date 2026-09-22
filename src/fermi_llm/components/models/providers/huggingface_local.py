"""Local HuggingFace provider.

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


MODELS = {
    # --- Local models (HuggingFace transformers) ---
    'qwen3.5-0.8b': {
        'name': 'Qwen3.5-0.8B',
        'group': 'Local Models',
        'backend': 'huggingface',
        'path': f'{CACHE_DIR}/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17',
        'hf_id': 'Qwen/Qwen3.5-0.8B',
        'quantize': False,
        'max_input_tokens': 12288,
        'requires_gpu': True,
        'description': 'Smallest & fastest (0.8B params)',
    },
    'qwen3.5-27b': {
        'name': 'Qwen3.5-27B',
        'group': 'Local Models',
        'backend': 'huggingface',
        'path': f'{CACHE_DIR}/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd',
        'hf_id': 'Qwen/Qwen3.5-27B',
        'quantize': False,
        'max_input_tokens': 10240,
        'requires_gpu': True,
        'description': 'Mid-size (27B params)',
    },
    'qwen3.5-35b-a3b': {
        'name': 'Qwen3.5-35B-A3B',
        'group': 'Local Models',
        'backend': 'huggingface',
        'path': f'{CACHE_DIR}/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307',
        'hf_id': 'Qwen/Qwen3.5-35B-A3B',
        'quantize': False,
        'max_input_tokens': 10240,
        'requires_gpu': True,
        'description': 'Best local MoE: 88% param acc, 97% API cov (requires GPU)',
    },
    'qwen3.5-122b-a10b': {
        'name': 'Qwen3.5-122B-A10B',
        'group': 'Local Models',
        'backend': 'huggingface',
        'path': f'{CACHE_DIR}/models--Qwen--Qwen3.5-122B-A10B/snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99',
        'hf_id': 'Qwen/Qwen3.5-122B-A10B',
        'quantize': True,
        'max_input_tokens': 6144,
        'requires_gpu': True,
        'description': 'Largest MoE, 4-bit quantized (122B, 10B active)',
    },
}


class HuggingFaceProvider(BaseProvider):
    """Local transformers snapshots, loaded on demand into GPU memory."""

    backend = 'huggingface'
    MODELS = MODELS
    requires_gpu = True

    def __init__(self, ctx):
        super().__init__(ctx)
        self._hf_model = None
        self._hf_tokenizer = None
        self._hf_model_id = None
        self._hf_lock = threading.Lock()

    def availability(self, spec):
        if not self.gpu_available:
            return False, 'no_gpu', 'Requires GPU'
        path = self.model_info(spec.id).get('path', '')
        if path and not os.path.exists(path):
            return False, 'not_downloaded', 'Model weights not found'
        return True, 'available', ''

    def is_loaded(self, model_id):
        return self._hf_model_id == model_id

    def _load_hf_model(self, model_id):
        """Load a HuggingFace model into GPU memory."""
        info = self.model_info(model_id)
        if self._hf_model_id == model_id and self._hf_model is not None:
            return  # already loaded

        with self._hf_lock:
            # Unload previous model
            self._unload_hf_model()

            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            print(f"[ModelManager] Loading HF model: {info['name']}...")
            start = time.time()

            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                info['path'], trust_remote_code=True,
            )
            if self._hf_tokenizer.pad_token is None:
                self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

            model_kwargs = {
                'trust_remote_code': True,
                'device_map': 'auto',
            }
            if info.get('quantize'):
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs['quantization_config'] = bnb_config
            else:
                model_kwargs['torch_dtype'] = torch.bfloat16

            self._hf_model = AutoModelForCausalLM.from_pretrained(
                info['path'], **model_kwargs,
            )
            self._hf_model.eval()
            self._hf_model_id = model_id

            elapsed = time.time() - start
            print(f"[ModelManager] Loaded {info['name']} in {elapsed:.1f}s")

    def _unload_hf_model(self):
        """Free GPU memory from current model."""
        if self._hf_model is not None:
            print(f"[ModelManager] Unloading HF model: {self._hf_model_id}")
            del self._hf_model
            self._hf_model = None
            self._hf_tokenizer = None
            self._hf_model_id = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            import gc
            gc.collect()

    def _generate_hf(self, model_id, prompt, max_new_tokens=4096,
                     temperature=0.1, top_p=0.95):
        """Generate text using a HuggingFace model."""
        import torch

        self._load_hf_model(model_id)
        info = self.model_info(model_id)
        max_input = info.get('max_input_tokens', 8192)

        tokenizer = self._hf_tokenizer
        model = self._hf_model

        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        if input_ids.shape[1] > max_input:
            input_ids = input_ids[:, :max_input]
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text

    def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.time()
        raw = self._generate_hf(request.model_id, request.build_prompt(),
                                temperature=request.temperature,
                                progress_cb=request.progress)
        return self.finish(request, raw, start)

    def complete(self, spec, prompt, **kwargs) -> str:
        return self._generate_hf(spec.id, prompt,
                                 temperature=kwargs.get('temperature', 0.1))

    def shutdown(self):
        self._unload_hf_model()


REGISTRY.register(kinds.MODEL_PROVIDER, 'huggingface', HuggingFaceProvider,
                  priority=240, source='core',
                  metadata={'label': 'Local HuggingFace weights'})
