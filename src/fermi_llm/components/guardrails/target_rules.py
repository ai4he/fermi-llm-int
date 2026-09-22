"""Keep the analysis target the one the scientist actually named.

Two failures, two guardrails:

``target_enforcement``
    The message names a source, so ``selection.target`` is forced to it (and
    a hallucinated designation is corrected in the Python script too). A
    non-catalog target with explicit coordinates is added to ``model.sources``.
``target_preservation``
    The message does *not* name a source, so a target change the model made
    on its own is reverted.

They are mutually exclusive by design: the second only runs when the first
did not, which the shared ``target_enforced`` flag records.
"""

from __future__ import annotations

from ...core import kinds
from ...core.contracts import ChatTurn
from ...core.registry import REGISTRY
from .base import BaseGuardrail
from .yamlops import (_enforce_requested_target, _ensure_custom_target,
                      _preserve_target_on_edit)


class TargetEnforcementGuardrail(BaseGuardrail):
    name = 'target_enforcement'
    label = 'Force the requested target into the configuration'

    def applies(self, turn: ChatTurn) -> bool:
        analysis = turn.analysis or {}
        return bool(analysis.get('target')) and bool((turn.yaml or '').strip())

    def apply(self, turn: ChatTurn) -> ChatTurn:
        analysis = turn.analysis or {}
        target = analysis['target']
        turn.yaml, previous, changed = _enforce_requested_target(turn.yaml, target)
        turn.metadata['target_enforced'] = True
        if changed:
            # The script carries the same wrong name — fix it there too, so
            # the displayed code never references a nonexistent source.
            if previous and turn.python and previous in turn.python:
                turn.python = turn.python.replace(previous, target)
            turn.response_parts.append(
                f"\n**Consistency check:** set `selection.target` to "
                f"`{target}` as named in your request"
                + (f" (the generated YAML/script had `{previous}`)." if previous
                   else " (the generated YAML omitted it)."))
            turn.note(self.name, f'target -> {target}')

        custom = analysis.get('custom_target')
        if custom:
            turn.yaml = _ensure_custom_target(turn.yaml, custom)
            turn.response_parts.append(
                f"\n**Custom target:** added {target} to "
                f"model.sources as a point source at "
                f"RA={custom['ra']:.6f}, Dec={custom['dec']:.6f}. The 4FGL "
                f"catalog is retained for surrounding background sources.")
            turn.note(self.name, 'custom target injected')
        return turn


class TargetPreservationGuardrail(BaseGuardrail):
    name = 'target_preservation'
    label = 'Revert an unrequested target switch'

    def applies(self, turn: ChatTurn) -> bool:
        if turn.metadata.get('target_enforced'):
            return False
        return bool((turn.yaml or '').strip()) and self.has_previous_yaml(turn)

    def apply(self, turn: ChatTurn) -> ChatTurn:
        previous_yaml = self.edit_context(turn)['yaml']
        turn.yaml, reverted_from, kept = _preserve_target_on_edit(
            turn.yaml, previous_yaml, turn.user_message,
            semantic_intent=turn.semantic_intent)
        if reverted_from:
            if turn.python and reverted_from and kept:
                turn.python = turn.python.replace(reverted_from, kept)
            turn.response_parts.append(
                f"\n**Consistency check:** the model tried to switch the "
                f"analysis target to `{reverted_from}`, which your message "
                f"didn't ask for — the current target `{kept}` was preserved.")
            turn.note(self.name, f'reverted {reverted_from} -> {kept}')
        return turn


REGISTRY.register(kinds.GUARDRAIL, 'target_enforcement',
                  TargetEnforcementGuardrail, priority=200, source='core',
                  metadata={'label': TargetEnforcementGuardrail.label})
REGISTRY.register(kinds.GUARDRAIL, 'target_preservation',
                  TargetPreservationGuardrail, priority=210, source='core',
                  metadata={'label': TargetPreservationGuardrail.label})
