"""Base class for guardrails."""

from __future__ import annotations

from ...core.contracts import ChatTurn, Guardrail


class BaseGuardrail(Guardrail):
    """Adds the small amount of plumbing every guardrail repeats."""

    name = ''
    #: Human-readable, shown in the platform view and the docs table.
    label = ''

    def __init__(self, ctx=None):
        self.ctx = ctx

    # -- helpers used by subclasses ---------------------------------------
    @staticmethod
    def edit_context(turn: ChatTurn):
        return turn.metadata.get('edit_context') or {}

    @staticmethod
    def has_previous_yaml(turn: ChatTurn) -> bool:
        return bool((BaseGuardrail.edit_context(turn).get('yaml') or '').strip())

    def applies(self, turn: ChatTurn) -> bool:
        return bool((turn.yaml or '').strip())

    def apply(self, turn: ChatTurn) -> ChatTurn:      # pragma: no cover
        raise NotImplementedError
