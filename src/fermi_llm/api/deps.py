"""Shared dependencies for the routers."""

from __future__ import annotations

from fastapi import HTTPException

from ..core.runtime import RUNTIME
from ..core.session import AnalysisSession


def get_session_or_404(session_id: str):
    """Load a task from the in-memory cache or the session store."""
    session = RUNTIME.sessions.get(session_id) or AnalysisSession.load(session_id)
    if not session:
        raise HTTPException(404, 'Session not found')
    RUNTIME.sessions[session_id] = session
    return session


def current_user(request):
    from ..components.auth.tokens import _current_user
    return _current_user(request)
