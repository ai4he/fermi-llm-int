"""Codex CLI provider.

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
from typing import Optional, Tuple

from ....core import kinds
from ....core.contracts import GenerationRequest, GenerationResult, ModelSpec
from ....core.registry import REGISTRY
from ..base import BaseProvider, read_key_file
from .. import settings as cfg
from ..prompting import (SEMANTIC_GENERATION_SCHEMA, SEMANTIC_INTENT_MODELS,
                         extract_python, extract_semantic_intent, extract_yaml)

# Deployment-level settings for this backend (same env-var names as before).
CODEX_BIN = cfg.CODEX_BIN
CODEX_CWD = cfg.CODEX_CWD
CODEX_MODEL = cfg.CODEX_MODEL
CODEX_SANDBOX = cfg.CODEX_SANDBOX
CODEX_TIMEOUT = cfg.CODEX_TIMEOUT


MODELS = {
    # --- OpenAI Codex CLI (session-based Luna) ---
    'gpt-5.6-luna-codex': {
        'name': 'ChatGPT Luna 5.6 (Codex session)',
        'group': 'OpenAI Codex',
        'backend': 'codex',
        'codex_model': CODEX_MODEL,
        'requires_gpu': False,
        'description': "GPT-5.6 Luna via the local Codex CLI. Keeps one Codex "
                       "conversation per task, so follow-up messages resume the "
                       "same session (reusing all prior context) instead of "
                       "starting from scratch.",
        'recommended': False,
    },
}


class CodexProvider(BaseProvider):
    """The local Codex CLI, one conversation (thread) per task."""

    backend = 'codex'
    MODELS = MODELS

    @staticmethod
    def _codex_ready():
        """Whether the local Codex CLI is usable: binary present + logged in.

        Returns (ok: bool, detail: str). Auth is a ChatGPT login stored in
        $CODEX_HOME/auth.json (default ~/.codex/auth.json).
        """
        if not CODEX_BIN or not os.path.exists(CODEX_BIN):
            return False, 'Codex CLI not found on this host'
        codex_home = os.environ.get('CODEX_HOME',
                                    os.path.expanduser('~/.codex'))
        auth_file = os.path.join(codex_home, 'auth.json')
        # An empty auth.json (aborted login) is as unusable as a missing one;
        # treating it as logged-in only defers the failure to generation time.
        try:
            if not os.path.getsize(auth_file):
                return False, 'Codex CLI is not logged in'
        except OSError:
            return False, 'Codex CLI is not logged in'
        return True, 'Codex CLI ready (per-task session)'

    def availability(self, spec):
        ok, detail = self._codex_ready()
        return (True, 'available', detail) if ok else (False, 'not_running', detail)

    def _generate_codex(self, model_id: str, prompt: str,
                        thread_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Generate via the local Codex CLI, one conversation per task.

        Returns (text, thread_id). On the first turn of a task `thread_id` is
        None, so a fresh `codex exec` session is started and its thread id
        (captured from the JSONL `thread.started` event) is returned. On
        follow-ups the caller passes that id back and we `codex exec resume
        <thread_id> <prompt>`, so the model keeps ALL prior context (the earlier
        request, the generated YAML/Python, and any corrections) instead of
        regenerating from scratch. If a stored thread can no longer be resumed
        (archived/expired), we transparently fall back to a fresh session.
        """
        import subprocess, tempfile, uuid as _uuid
        info = self.model_info(model_id)
        codex_model = info['codex_model']

        ok, detail = self._codex_ready()
        if not ok:
            raise RuntimeError(detail)

        os.makedirs(CODEX_CWD, exist_ok=True)
        last_file = os.path.join(tempfile.gettempdir(),
                                 f'codex_last_{_uuid.uuid4().hex}.txt')
        common = ['--json', '--skip-git-repo-check', '-m', codex_model,
                  '-o', last_file]

        def _run(cmd):
            try:
                return subprocess.run(
                    cmd, cwd=CODEX_CWD, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=CODEX_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"Codex CLI timed out after {CODEX_TIMEOUT}s")

        proc = None
        # Resume the task's existing conversation when we have one.
        if thread_id:
            # `resume` does not accept -s/--sandbox; set it via -c instead.
            cmd = ([CODEX_BIN, 'exec', 'resume', thread_id] + common +
                   ['-c', f'sandbox_mode="{CODEX_SANDBOX}"', prompt])
            proc = _run(cmd)
            if proc.returncode != 0:
                # Thread gone/unresumable — start fresh below.
                thread_id = None
                proc = None

        if proc is None:
            cmd = ([CODEX_BIN, 'exec'] + common +
                   ['-s', CODEX_SANDBOX, prompt])
            proc = _run(cmd)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Codex exec failed ({proc.returncode}): "
                    f"{(proc.stderr or '')[:400]}")

        # Capture (possibly new) thread id from the JSONL event stream.
        new_tid = thread_id
        for line in (proc.stdout or '').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue
            if evt.get('type') == 'thread.started' and evt.get('thread_id'):
                new_tid = evt['thread_id']

        # The agent's final message was written to last_file by `-o`.
        text = ''
        try:
            with open(last_file) as f:
                text = f.read()
        except Exception:
            text = ''
        finally:
            try:
                os.remove(last_file)
            except Exception:
                pass

        if not text.strip():
            raise RuntimeError(
                "Codex returned an empty message "
                f"(stderr: {(proc.stderr or '')[:300]})")
        return text, new_tid

    def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.time()
        raw, thread_id = self._generate_codex(
            request.model_id, request.build_prompt(),
            progress_cb=request.progress, thread_id=request.thread_id)
        result = self.finish(request, raw, start)
        result.metadata['codex_thread_id'] = thread_id
        return result

    def complete(self, spec, prompt, **kwargs) -> str:
        raw, _ = self._generate_codex(spec.id, prompt)
        return raw


REGISTRY.register(kinds.MODEL_PROVIDER, 'codex', CodexProvider,
                  priority=220, source='core',
                  metadata={'label': 'Codex CLI session'})
