"""Signed cookie tokens shared by every auth provider.

The token is a compact HMAC envelope: the identity a provider verified, the
issue time, and a signature over both. Providers never mint their own format,
so adding a login provider does not touch session handling or the cookie.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets as _secrets
import time

from ...core.runtime import RUNTIME


def _configs_dir() -> str:
    return RUNTIME.settings.configs_dir


def _load_or_create_auth_secret():
    """HMAC secret for signing our session tokens; persisted so tokens survive
    server restarts. Auto-generated on first run."""
    env = os.environ.get('FERMI_LLM_AUTH_SECRET', '').strip()
    if env:
        return env.encode()
    path = os.path.join(_configs_dir(), 'auth_secret.txt')
    try:
        with open(path) as f:
            s = f.read().strip()
        if s:
            return s.encode()
    except Exception:
        pass
    s = _secrets.token_hex(32)
    try:
        os.makedirs(_configs_dir(), exist_ok=True)
        with open(path, 'w') as f:
            f.write(s)
        os.chmod(path, 0o600)
    except Exception:
        pass
    return s.encode()


_AUTH_SECRET = None
_AUTH_TTL_SECONDS = 30 * 24 * 3600  # 30 days


def auth_secret() -> bytes:
    """Lazily loaded so importing this module needs no filesystem."""
    global _AUTH_SECRET
    if _AUTH_SECRET is None:
        _AUTH_SECRET = _load_or_create_auth_secret()
    return _AUTH_SECRET


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')


def _b64u_dec(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + '=' * (-len(s) % 4))


def _make_auth_token(user: dict) -> str:
    """Compact HMAC-signed token: b64url(payload).b64url(hmac_sha256)."""
    now = int(time.time())
    payload = {
        'sub': user['sub'], 'email': user.get('email'),
        'name': user.get('name'), 'picture': user.get('picture'),
        'iat': now, 'exp': now + _AUTH_TTL_SECONDS,
    }
    pj = _b64u(json.dumps(payload, separators=(',', ':')).encode())
    sig = _b64u(hmac.new(auth_secret(), pj.encode(), hashlib.sha256).digest())
    return f"{pj}.{sig}"


def _verify_auth_token(token: str):
    """Return the payload dict if the token is valid and unexpired, else None."""
    try:
        pj, sig = token.split('.', 1)
        expected = _b64u(hmac.new(auth_secret(), pj.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64u_dec(pj))
        if int(payload.get('exp', 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def _current_user(request: 'Request'):
    """Identify the caller from a Bearer token or cookie. None => guest."""
    token = ''
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        token = auth[7:].strip()
    if not token:
        token = request.cookies.get('fermi_auth', '')
    if not token:
        return None
    return _verify_auth_token(token)


def _list_user_sessions(owner_sub: str):
    """All saved tasks owned by this user, newest first (light metadata only)."""
    out = []
    try:
        entries = os.listdir(RUNTIME.sessions_dir)
    except Exception:
        return out
    for sid in entries:
        fp = os.path.join(RUNTIME.sessions_dir, sid, 'session.json')
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get('owner') != owner_sub:
            continue
        prompt = (d.get('current_prompt') or '').strip()
        out.append({
            'session_id': d.get('session_id', sid),
            'title': (d.get('title') or prompt or 'Untitled task')[:100],
            'created_at': d.get('created_at', ''),
            'status': d.get('pipeline_status', 'idle'),
            'has_config': bool((d.get('current_yaml') or '').strip()),
        })
    out.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return out
