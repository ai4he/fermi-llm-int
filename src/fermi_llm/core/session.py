"""The task (session) model.

A session is the unit of work the UI calls a *task*: the chat history, the
current YAML/Python, the pending run and everything a run produced. Where it
is stored is a plugin decision — this class holds the shape, the active
``session_store`` component holds the bytes.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

from .runtime import RUNTIME


class AnalysisSession:
    """Isolated session for a single user/analysis."""

    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.session_dir = RUNTIME.ctx.sessions.create(self.session_id)
        self.chat_history = []
        self.current_yaml = ''
        self.current_python = ''
        self.current_prompt = ''
        self.validation_log = []
        self.pipeline_status = 'idle'  # idle, generating, validating, running, complete, error
        self.pipeline_result = None
        self.pending_run = None
        self.created_at = datetime.now().isoformat()
        # Incremental-run bookkeeping: the exact (repaired) YAML that was
        # actually executed by the last successful run, which test source it
        # used, and whether a reloadable ROI snapshot exists in this
        # session's fermipy_workdir (see run_pipeline_isolated). Used to
        # decide whether a follow-up run can reuse the fit via gta.load_roi()
        # instead of redoing gta.setup() + gta.optimize() + gta.fit().
        self.last_executed_yaml = ''
        self.last_run_test_idx = None
        self.last_run_roi_ready = False
        # Codex CLI conversation id for this task (backend='codex'). Persisted so
        # follow-up messages resume the SAME Codex session instead of starting a
        # new one. A brand-new AnalysisSession (i.e. a new task) starts with none,
        # which is exactly why a new task gets a fresh Codex conversation.
        self.codex_thread_id = None
        # Task ownership (Google Sign-In). owner = Google 'sub' of the signed-in
        # user who created/claimed this task; None for guest tasks. title is a
        # short human label shown in the "My Tasks" list.
        self.owner = None
        self.owner_email = None
        self.title = ''
        # Validation Agent script review (see _prepare_run_review). A change
        # the validator proposes waits here until the user accepts/declines
        # it; script_reviews remembers decisions per (YAML, script) digest so
        # the same content is not re-reviewed by the LLM.
        self.script_proposal = None
        self.script_reviews = {}

    def to_dict(self):
        return {
            'session_id': self.session_id,
            'chat_history': self.chat_history,
            'current_yaml': self.current_yaml,
            'current_python': self.current_python,
            'current_prompt': self.current_prompt,
            'validation_log': self.validation_log,
            'pipeline_status': self.pipeline_status,
            'pipeline_result': self.pipeline_result,
            'pending_run': self.pending_run,
            'created_at': self.created_at,
            'last_executed_yaml': self.last_executed_yaml,
            'last_run_test_idx': self.last_run_test_idx,
            'last_run_roi_ready': self.last_run_roi_ready,
            'codex_thread_id': self.codex_thread_id,
            'owner': self.owner,
            'owner_email': self.owner_email,
            'title': self.title,
            'script_proposal': self.script_proposal,
            'script_reviews': self.script_reviews,
        }

    def save(self):
        """Persist through the active session store, then fire the hook."""
        RUNTIME.ctx.sessions.save(self.session_id, self.to_dict())
        RUNTIME.hooks.emit('session.saved', session=self)

    @classmethod
    def load(cls, session_id):
        data = RUNTIME.ctx.sessions.load(session_id)
        if data is None:
            return None
        s = cls(session_id)
        s.chat_history = data.get('chat_history', [])
        s.current_yaml = data.get('current_yaml', '')
        s.current_python = data.get('current_python', '')
        s.current_prompt = data.get('current_prompt', '')
        s.validation_log = data.get('validation_log', [])
        s.pipeline_status = data.get('pipeline_status', 'idle')
        s.pipeline_result = data.get('pipeline_result')
        s.pending_run = data.get('pending_run')
        s.created_at = data.get('created_at', '')
        s.last_executed_yaml = data.get('last_executed_yaml', '')
        s.last_run_test_idx = data.get('last_run_test_idx')
        s.last_run_roi_ready = data.get('last_run_roi_ready', False)
        s.codex_thread_id = data.get('codex_thread_id')
        s.owner = data.get('owner')
        s.owner_email = data.get('owner_email')
        s.title = data.get('title', '')
        s.script_proposal = data.get('script_proposal')
        s.script_reviews = data.get('script_reviews') or {}
        return s


def new_session_id() -> str:
    return uuid.uuid4().hex[:8]
