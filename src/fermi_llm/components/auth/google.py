"""Google Sign-In (optional).

Configured by dropping ``configs/google_oauth.json`` (or the downloaded
``client_secret_*.json``) in place, or by setting
``FERMI_LLM_GOOGLE_CLIENT_ID``. Absent configuration simply hides the button:
guest mode always works, which keeps the app usable at a workshop with no
internet.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

from ...core import kinds
from ...core.contracts import Identity
from ...core.registry import REGISTRY
from ...core.runtime import RUNTIME

GOOGLE_CLIENT_ID = os.environ.get('FERMI_LLM_GOOGLE_CLIENT_ID', '').strip()
# Optional allow-list, e.g. "clemson.edu,nasa.gov"; empty means any account.
GOOGLE_ALLOWED_DOMAINS = [d.strip().lower() for d in os.environ.get(
    'FERMI_LLM_GOOGLE_ALLOWED_DOMAINS', '').split(',') if d.strip()]


def _configs_dir() -> str:
    return RUNTIME.settings.configs_dir


def _load_google_oauth_config():
    """Client id + optional domain allow-list. Env overrides the JSON file."""
    client_id = os.environ.get('FERMI_LLM_GOOGLE_CLIENT_ID', '').strip()
    allowed = []
    path = os.environ.get('FERMI_LLM_GOOGLE_OAUTH_FILE',
                          os.path.join(_configs_dir(), 'google_oauth.json'))
    try:
        with open(path) as f:
            data = json.load(f)
        client_id = client_id or (data.get('client_id') or '').strip()
        allowed = data.get('allowed_domains') or []
    except Exception:
        pass
    # Fallback: accept a Google Cloud Console client-secret download directly
    # (configs/client_secret_*.apps.googleusercontent.com.json, "web" format),
    # so no separate google_oauth.json is required.
    if not client_id:
        import glob
        for p in sorted(glob.glob(os.path.join(_configs_dir(), 'client_secret_*.json'))):
            try:
                with open(p) as f:
                    gd = json.load(f)
                node = gd.get('web') or gd.get('installed') or {}
                cid = (node.get('client_id') or '').strip()
                if cid:
                    client_id = cid
                    break
            except Exception:
                continue
    return client_id, [d.lower() for d in allowed]


def _verify_google_credential(credential: str) -> dict:
    """Validate a Google ID token (JWT) via Google's tokeninfo endpoint and
    return a normalized user dict. Raises on any failure."""
    if not GOOGLE_CLIENT_ID:
        raise RuntimeError("Google sign-in is not configured on this server")
    import requests as _rq
    resp = _rq.get('https://oauth2.googleapis.com/tokeninfo',
                   params={'id_token': credential}, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError("Google rejected the credential")
    info = resp.json()
    if info.get('aud') != GOOGLE_CLIENT_ID:
        raise RuntimeError("Token was not issued for this application")
    if info.get('iss') not in ('accounts.google.com', 'https://accounts.google.com'):
        raise RuntimeError("Unexpected token issuer")
    if str(info.get('email_verified', '')).lower() != 'true':
        raise RuntimeError("Google account email is not verified")
    if GOOGLE_ALLOWED_DOMAINS:
        domain = (info.get('hd') or (info.get('email') or '').split('@')[-1]).lower()
        if domain not in GOOGLE_ALLOWED_DOMAINS:
            raise RuntimeError("This Google account's domain is not allowed")
    return {
        'sub': info['sub'],
        'email': info.get('email'),
        'name': info.get('name') or info.get('email') or 'User',
        'picture': info.get('picture'),
    }


class GoogleAuthProvider:
    """Verifies a Google Identity Services credential."""

    id = 'google'
    label = 'Google'

    def __init__(self, ctx):
        self.ctx = ctx

    def is_configured(self) -> bool:
        return bool(_load_google_oauth_config().get('client_id'))

    def public_config(self):
        cfg = _load_google_oauth_config()
        return {'enabled': bool(cfg.get('client_id')),
                'client_id': cfg.get('client_id', '')}

    def verify(self, credential: str) -> Identity:
        info = _verify_google_credential(credential)
        return Identity(subject=info.get('sub', ''),
                        email=info.get('email', ''),
                        name=info.get('name', ''),
                        picture=info.get('picture', ''),
                        provider=self.id, claims=info)


REGISTRY.register(kinds.AUTH_PROVIDER, 'google', GoogleAuthProvider,
                  priority=100, source='core',
                  metadata={'label': 'Google Sign-In'})
