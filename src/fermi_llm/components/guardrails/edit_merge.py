"""Carry over sections a regenerated configuration silently dropped.

A follow-up message ("switch to a log-parabola") must not lose the target,
the time range or a requested product just because the model rewrote the
file from memory. Only sections the new YAML omits are restored, so an
intentional removal still goes through.
"""

from __future__ import annotations

from ...core import kinds
from ...core.contracts import ChatTurn
from ...core.registry import REGISTRY
from .base import BaseGuardrail
from .yamlops import _merge_edit_yaml


class EditMergeGuardrail(BaseGuardrail):
    name = 'edit_merge'
    label = 'Carry over dropped sections on edits'

    def applies(self, turn: ChatTurn) -> bool:
        return bool((turn.yaml or '').strip()) and self.has_previous_yaml(turn)

    def apply(self, turn: ChatTurn) -> ChatTurn:
        previous = self.edit_context(turn)['yaml']
        turn.yaml, carried = _merge_edit_yaml(
            turn.yaml, previous, turn.user_message,
            semantic_intent=turn.semantic_intent)
        if carried:
            turn.response_parts.append(
                f"\n**Consistency check:** the regenerated YAML dropped "
                f"`{'`, `'.join(carried)}` from the current configuration — "
                f"carried over unchanged.")
            turn.note(self.name, f'carried {", ".join(carried)}')
        return turn


REGISTRY.register(kinds.GUARDRAIL, 'edit_merge', EditMergeGuardrail,
                  priority=100, source='core',
                  metadata={'label': EditMergeGuardrail.label})
