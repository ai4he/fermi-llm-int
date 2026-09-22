"""Fill in real data paths and a time range for non-bundled targets.

A target outside the three bundled demonstrations runs on the weekly all-sky
archive. Resolving that at chat time (instead of at run time) is what lets
the scientist see the actual file list and date range in the YAML before
committing to a long run.
"""

from __future__ import annotations

from ...core import kinds
from ...core.contracts import ChatTurn
from ...core.registry import REGISTRY
from .base import BaseGuardrail
from .yamlops import _apply_full_data_defaults


class FullDataDefaultsGuardrail(BaseGuardrail):
    name = 'full_data_defaults'
    label = 'Resolve archive paths and time range for non-bundled targets'

    def apply(self, turn: ChatTurn) -> ChatTurn:
        before = turn.yaml
        turn.yaml = _apply_full_data_defaults(
            turn.yaml, turn.response_parts, turn.session.session_dir,
            turn.user_message)
        if turn.yaml != before:
            turn.note(self.name, 'applied full-dataset defaults')
        return turn


REGISTRY.register(kinds.GUARDRAIL, 'full_data_defaults',
                  FullDataDefaultsGuardrail, priority=400, source='core',
                  metadata={'label': FullDataDefaultsGuardrail.label})
