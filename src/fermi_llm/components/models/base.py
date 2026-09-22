"""Shared behaviour for model providers.

A provider only has to describe its models and produce text; parsing the
reply into YAML + Python + semantic intent is identical for every backend and
lives here, so a new provider is usually fewer than fifty lines.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.contracts import (GenerationRequest, GenerationResult, ModelProvider,
                               ModelSpec)
from .prompting import (SEMANTIC_INTENT_MODELS, extract_python,
                        extract_semantic_intent, extract_yaml)
from ...fermipy.diffuse_models import normalize_diffuse_yaml


def read_key_file(path: str) -> str:
    """Return a single-line secret from ``path``; '' when absent."""
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return ''


def gpu_available() -> bool:
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name',
                              '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:                                     # noqa: BLE001
        return False


def gpu_info() -> Tuple[int, str]:
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5)
        rows = [r.strip() for r in out.stdout.splitlines() if r.strip()]
        return len(rows), '; '.join(r.replace(',', '') for r in rows)
    except Exception:                                     # noqa: BLE001
        return 0, 'none'


class BaseProvider(ModelProvider):
    """Convenience base: model catalogue, availability, reply parsing."""

    backend: str = ''
    MODELS: Dict[str, dict] = {}
    #: True when the backend is fine-tuned on raw user text (skip few-shot).
    uses_raw_prompt: bool = False
    requires_gpu: bool = False

    def __init__(self, ctx=None):
        self.ctx = ctx
        self._gpu_available = None

    # -- catalogue --------------------------------------------------------
    @property
    def gpu_available(self) -> bool:
        if self._gpu_available is None:
            self._gpu_available = gpu_available()
        return self._gpu_available

    def model_info(self, model_id: str) -> dict:
        return self.MODELS[model_id]

    def list_models(self) -> Sequence[ModelSpec]:
        specs = []
        for model_id, info in self.MODELS.items():
            specs.append(ModelSpec(
                id=model_id,
                name=info.get('name', model_id),
                backend=self.backend,
                group=info.get('group', 'Models'),
                description=info.get('description', ''),
                recommended=bool(info.get('recommended')),
                requires_gpu=bool(info.get('requires_gpu')),
                supports_semantic_intent=model_id in SEMANTIC_INTENT_MODELS,
                options=dict(info),
            ))
        return specs

    # -- availability -----------------------------------------------------
    def availability(self, spec: ModelSpec) -> Tuple[bool, str, str]:
        """(available, status, detail) — the picker shows all three."""
        return True, 'available', ''

    def is_available(self, spec: ModelSpec) -> bool:
        return self.availability(spec)[0]

    def is_loaded(self, model_id: str) -> bool:
        return False

    # -- reply handling ---------------------------------------------------
    def finish(self, request: GenerationRequest, raw_text: str,
               started: Optional[float] = None) -> GenerationResult:
        """Parse a raw reply the same way for every backend."""
        result = GenerationResult(raw_text=raw_text or '')
        if started is not None:
            elapsed = time.time() - started
            result.metadata['gen_time'] = round(elapsed, 1)
            request.report('response',
                           f'Response received in {elapsed:.1f}s '
                           f'({len(raw_text or ""):,} chars)')
        request.report('parsing',
                       'Extracting YAML configuration and Python script...')
        result.yaml = extract_yaml(raw_text)
        result.python = extract_python(raw_text)
        if request.include_intent:
            result.semantic_intent = extract_semantic_intent(raw_text)
        result.yaml, normalized = normalize_diffuse_yaml(result.yaml)
        if normalized:
            result.metadata['diffuse_normalized'] = normalized
        result.metadata.update({
            'method': 'llm',
            'raw_length': len(raw_text or ''),
            'has_yaml': bool(result.yaml),
            'has_python': bool(result.python),
        })
        request.report('done',
                       'Extraction complete — YAML: '
                       f'{"found" if result.yaml else "missing"}, Python: '
                       f'{"found" if result.python else "missing"}')
        return result

    def finish_envelope(self, request: GenerationRequest, raw_text: str,
                        started: Optional[float] = None) -> GenerationResult:
        """Parse a strict JSON envelope ``{intent, yaml, python}``.

        Used by backends asked for structured output. A malformed envelope is
        recorded in ``metadata['parse_error']`` and yields empty files, which
        the chat flow handles by falling back to template generation.
        """
        import json

        from .prompting import _normalise_semantic_intent

        result = GenerationResult(raw_text=raw_text or '')
        if started is not None:
            elapsed = time.time() - started
            result.metadata['gen_time'] = round(elapsed, 1)
            request.report('response',
                           f'Response received in {elapsed:.1f}s '
                           f'({len(raw_text or ""):,} chars)')
        request.report('parsing',
                       'Extracting YAML configuration and Python script...')
        try:
            envelope = json.loads(raw_text)
        except (TypeError, ValueError):
            envelope = {}
            result.metadata['parse_error'] = 'invalid structured generation JSON'
        result.yaml = str(envelope.get('yaml') or '')
        result.python = str(envelope.get('python') or '')
        result.semantic_intent = _normalise_semantic_intent(
            envelope.get('intent'))
        result.yaml, normalized = normalize_diffuse_yaml(result.yaml)
        if normalized:
            result.metadata['diffuse_normalized'] = normalized
        result.metadata.update({
            'method': 'llm',
            'raw_length': len(raw_text or ''),
            'has_yaml': bool(result.yaml),
            'has_python': bool(result.python),
        })
        request.report('done',
                       'Extraction complete — YAML: '
                       f'{"found" if result.yaml else "missing"}, Python: '
                       f'{"found" if result.python else "missing"}')
        return result

    def shutdown(self) -> None:
        """Release anything the backend holds (GPU, subprocess, socket)."""
