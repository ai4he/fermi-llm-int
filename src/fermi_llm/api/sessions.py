"""Task lifecycle: create, read and edit the current files."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse,
                               StreamingResponse)

from ..core.jsonutil import sanitize_for_json
from ..core.runtime import RUNTIME
from ..core.session import AnalysisSession
from ..components.auth.tokens import _current_user
from ..components.exporters.run_bundle import ensure_session_run_exports
from ..components.pipelines.review import run_content_digest
from .deps import get_session_or_404

router = APIRouter()

@router.post("/api/session/create")
async def create_session(request: Request):
    session = AnalysisSession()
    # A signed-in user owns the tasks they create, so they appear in "My Tasks".
    user = _current_user(request)
    if user:
        session.owner = user['sub']
        session.owner_email = user.get('email')
    RUNTIME.sessions[session.session_id] = session
    session.save()
    return {"session_id": session.session_id, "owner": session.owner}


@router.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = get_session_or_404(session_id)
    ensure_session_run_exports(session)
    return sanitize_for_json(session.to_dict())


@router.post("/api/session/{session_id}/update_code")
async def update_code(session_id: str, request: Request):
    body = await request.json()
    session = get_session_or_404(session_id)
    RUNTIME.sessions[session_id] = session

    if 'yaml' in body:
        session.current_yaml = body['yaml']
    if 'python' in body:
        session.current_python = body['python']
    proposal = session.script_proposal
    if proposal and session.current_python != proposal.get('script') and (
            run_content_digest(session.current_yaml, session.current_python)
            != proposal.get('base_digest')):
        # The user edited the script themselves; the proposal is stale.
        session.script_proposal = None
    if session.pending_run:
        digest = run_content_digest(
            session.current_yaml, session.current_python)
        if digest != session.pending_run.get('digest'):
            session.pending_run = None
            if session.pipeline_status == 'review_required':
                session.pipeline_status = 'idle'
    session.save()
    return {"status": "ok"}