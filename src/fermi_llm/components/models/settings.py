"""Endpoints, key-file locations and runtime knobs for the built-in providers.

Every value keeps the environment-variable name the monolith used, so an
existing ``configs/env.sh`` needs no edits. A provider plugin should read its
own settings from ``ctx.component_settings(kinds.MODEL_PROVIDER, name)``
instead of adding entries here.
"""

import os
import shutil as _shutil
import sys


def _project_dir():
    return os.environ.get(
        'FERMI_LLM_PROJECT_DIR',
        os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     '..', '..', '..', '..')))


PROJECT_DIR = _project_dir()
CACHE_DIR = os.environ.get(
    'FERMI_LLM_CACHE_DIR',
    os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
)

VLLM_DIR = os.environ.get(
    'FERMI_LLM_VLLM_DIR',
    os.path.join(os.path.expanduser('~'), '.cache', 'vllm')
)

API_KEYS_FILE = os.environ.get(
    'FERMI_LLM_GEMINI_KEYS_FILE',
    os.path.join(PROJECT_DIR, 'configs', 'gemini_keys.json')
)

CLEMSON_VLLM_BASE_URL = 'https://llm.rcd.clemson.edu/v1'

CLEMSON_VLLM_KEY_FILE = os.environ.get(
    'FERMI_LLM_CLEMSON_KEY_FILE',
    os.path.join(PROJECT_DIR, 'configs', 'clemson_vllm_key.txt')
)

LOCAL_VLLM_BASE_URL = os.environ.get('FERMI_LLM_LOCAL_VLLM_URL',
                                     'http://localhost:8000/v1')

LOCAL_VLLM_KEY = os.environ.get('FERMI_LLM_LOCAL_VLLM_KEY', '').strip()

OPENAI_BASE_URL = os.environ.get('FERMI_LLM_OPENAI_URL',
                                 'https://api.openai.com/v1')

OPENAI_KEY_FILE = os.environ.get(
    'FERMI_LLM_OPENAI_KEY_FILE',
    os.path.join(PROJECT_DIR, 'configs', 'openai_key.txt')
)

OPENAI_REASONING_EFFORT = os.environ.get(
    'FERMI_LLM_OPENAI_REASONING_EFFORT', 'high').strip()

CODEX_BIN = (_shutil.which('codex')
             or os.path.expanduser('~/.local/bin/codex'))

CODEX_MODEL = os.environ.get('FERMI_LLM_CODEX_MODEL', 'gpt-5.6-luna')

CODEX_SANDBOX = os.environ.get('FERMI_LLM_CODEX_SANDBOX', 'read-only')

CODEX_CWD = os.environ.get('FERMI_LLM_CODEX_CWD',
                           os.path.join(PROJECT_DIR, '.codex_scratch'))

CODEX_TIMEOUT = int(os.environ.get('FERMI_LLM_CODEX_TIMEOUT', '600'))

CLEMSON_REMOTE_CHAT_MODELS = [
    ('deepseek-v4-pro',            'DeepSeek V4 Pro'),
    ('gemma-4-31b',                'Gemma 4 31B'),
    ('glm-5.1-fp8',                'GLM 5.1 (FP8)'),
    ('gptoss-120b',                'GPT-OSS 120B'),
    ('gptoss-20b',                 'GPT-OSS 20B'),
    ('leanstral-2603',             'Leanstral 2603'),
    ('qwen3-30b-a3b-instruct-fp8', 'Qwen3 30B-A3B Instruct (FP8)'),
    ('qwen3-omni-30b-a3b',         'Qwen3 Omni 30B-A3B'),
    ('qwen3.5-9b',                 'Qwen3.5 9B'),
    ('qwen3.6-27b-fp8',            'Qwen3.6 27B (FP8)'),
    ('qwen3.6-35b-a3b-fp8',        'Qwen3.6 35B-A3B (FP8)'),
]

