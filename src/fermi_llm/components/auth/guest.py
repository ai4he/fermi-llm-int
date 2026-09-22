"""Guest access: no login, tasks live in the browser's local list.

Always registered. The app must stay fully usable without any identity
provider configured, so nothing in core may assume a signed-in user.
"""

from __future__ import annotations

from ...core import kinds
from ...core.contracts import Identity
from ...core.registry import REGISTRY


class GuestAuthProvider:
    id = 'guest'
    label = 'Continue as guest'

    def __init__(self, ctx):
        self.ctx = ctx

    def is_configured(self) -> bool:
        return True

    def public_config(self):
        return {'enabled': True}

    def verify(self, credential: str) -> Identity:
        return Identity(subject='', provider='guest')


REGISTRY.register(kinds.AUTH_PROVIDER, 'guest', GuestAuthProvider,
                  priority=900, source='core',
                  metadata={'label': 'Guest (no sign-in)'})
