"""Template (no-LLM) provider.

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


MODELS = {
    # --- Template (no LLM) ---
    'template': {
        'name': 'Template (No LLM)',
        'group': 'Built-in',
        'backend': 'template',
        'requires_gpu': False,
        'description': 'Rule-based generation, no model needed',
    },
}


class TemplateProvider(BaseProvider):
    """No LLM at all: the rule-based intent analyzer writes the files.

    Kept as a first-class provider because it is the fallback every
    deployment can rely on, and the simplest example of the contract.
    """

    backend = 'template'
    MODELS = MODELS

    def availability(self, spec):
        return True, 'available', 'Rule-based, no model required'

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.report('template', 'Using template-based generation (no LLM)')
        return GenerationResult(metadata={'method': 'template'})

    def complete(self, spec, prompt, **kwargs) -> str:
        return ''


REGISTRY.register(kinds.MODEL_PROVIDER, 'template', TemplateProvider,
                  priority=900, source='core',
                  metadata={'label': 'Template (no LLM)'})
