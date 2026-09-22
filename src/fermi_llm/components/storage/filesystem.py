"""One directory per task under ``sessions/`` — the default store.

Everything a run produces (progress, artifacts, exports, the FermiPy work
directory) lives inside the session directory, so this store is also what
makes a session portable: copy the directory, get the task.

Replace this component to keep tasks in object storage or a database; the
contract is small on purpose (see ``docs/module-types.md``).
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional

from ...core import kinds
from ...core.registry import REGISTRY


class FilesystemSessionStore:
    """Sessions as directories; ``session.json`` holds the state."""

    name = 'filesystem'

    def __init__(self, ctx):
        self.ctx = ctx
        self.root = ctx.settings.sessions_dir
        os.makedirs(self.root, exist_ok=True)

    def path(self, session_id: str) -> str:
        return os.path.join(self.root, session_id)

    def create(self, session_id: str) -> str:
        directory = self.path(session_id)
        os.makedirs(directory, exist_ok=True)
        return directory

    def state_file(self, session_id: str) -> str:
        return os.path.join(self.path(session_id), 'session.json')

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            with open(self.state_file(session_id)) as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, session_id: str, data: Dict[str, Any]) -> None:
        self.create(session_id)
        with open(self.state_file(session_id), 'w') as handle:
            json.dump(data, handle, indent=2, default=str)

    def delete(self, session_id: str) -> None:
        shutil.rmtree(self.path(session_id), ignore_errors=True)

    def list_ids(self) -> List[str]:
        try:
            return sorted(entry for entry in os.listdir(self.root)
                          if os.path.isdir(os.path.join(self.root, entry)))
        except OSError:
            return []

    def exists(self, session_id: str) -> bool:
        return os.path.isfile(self.state_file(session_id))


REGISTRY.register(kinds.SESSION_STORE, 'filesystem', FilesystemSessionStore,
                  priority=100, source='core',
                  metadata={'label': 'One directory per task'})
