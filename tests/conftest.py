"""Test bootstrap: a real context, with tasks written to a temp directory.

Tests exercise the same components the server runs. The only difference is
where sessions land, so a test run never touches a deployment's data.
"""

from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault('FERMI_LLM_PROJECT_DIR', ROOT)
_SESSIONS = os.environ.setdefault(
    'FERMI_LLM_SESSIONS_DIR',
    os.path.join(tempfile.gettempdir(), 'fermi_llm_test_sessions'))
os.makedirs(_SESSIONS, exist_ok=True)

from fermi_llm.core.context import build_context      # noqa: E402
from fermi_llm.core.runtime import RUNTIME            # noqa: E402

CTX = RUNTIME.ctx or build_context()
RUNTIME.bind(CTX)


def context():
    return CTX
