"""Sign-in, ownership and the per-user task list."""

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
from ..core import kinds
from ..core.runtime import RUNTIME
from ..core.session import AnalysisSession
from ..components.auth.tokens import (_current_user, _list_user_sessions,
                                      _make_auth_token)
from .deps import get_session_or_404

router = APIRouter()


def _providers():
    """Active sign-in providers, keyed by id."""
    return {p.id: p for p in RUNTIME.ctx.build_stack(kinds.AUTH_PROVIDER)}

@router.get("/api/auth/config")
async def auth_config():
    """Public config for every configured provider.

    The legacy ``enabled``/``client_id`` fields describe Google, which is
    what the shipped frontend renders; ``providers`` lists everything that
    is registered, so a plugin's login button can be rendered too. With
    nothing configured the app stays in guest-only mode.
    """
    providers = {}
    for provider_id, provider in _providers().items():
        if provider_id == 'guest':
            continue
        try:
            providers[provider_id] = provider.public_config()
        except Exception:                                 # noqa: BLE001
            continue
    google = providers.get('google', {})
    return {"enabled": bool(google.get('client_id')),
            "client_id": google.get('client_id', ''),
            "providers": providers}


@router.post("/api/auth/google")
async def auth_google(request: Request):
    """Exchange a Google ID token (from the Sign-In button) for our own signed
    session token."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    credential = (body or {}).get('credential', '')
    if not credential:
        raise HTTPException(400, "Missing Google credential")
    provider = _providers().get('google')
    if provider is None or not provider.is_configured():
        raise HTTPException(400, "Google sign-in is not configured")
    try:
        identity = provider.verify(credential)
    except Exception as e:                                # noqa: BLE001
        raise HTTPException(401, f"Google sign-in failed: {e}")
    user = {'sub': identity.subject, 'email': identity.email,
            'name': identity.name, 'picture': identity.picture}
    return {"token": _make_auth_token(user), "user": user}


@router.get("/api/auth/me")
async def auth_me(request: Request):
    user = _current_user(request)
    if not user:
        return {"authenticated": False}
    return {"authenticated": True,
            "user": {"sub": user.get('sub'), "email": user.get('email'),
                     "name": user.get('name'), "picture": user.get('picture')}}


@router.get("/api/my/sessions")
async def my_sessions(request: Request):
    """List the signed-in user's saved tasks (for the 'My Tasks' list)."""
    user = _current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    return {"sessions": _list_user_sessions(user['sub'])}


@router.post("/api/session/{session_id}/claim")
async def claim_session(session_id: str, request: Request):
    """Attach an unowned (guest) task to the signed-in user — used when a guest
    signs in mid-task so their current work shows up in 'My Tasks'."""
    user = _current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    session = get_session_or_404(session_id)
    if not session.owner:
        session.owner = user['sub']
        session.owner_email = user.get('email')
        RUNTIME.sessions[session_id] = session
        session.save()
    return {"owner": session.owner, "claimed": session.owner == user['sub']}