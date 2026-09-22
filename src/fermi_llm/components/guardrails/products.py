"""Make the configuration contain the products that were asked for.

The plan from the intent analyzer lists SED, light curve, TS map, residual
map or PS map. FermiPy only produces a product whose section exists in the
YAML, so a missing section silently means "no plot" — the single most
confusing failure for a user who asked for one. This guardrail adds the
missing sections (and removes the ones a message asked to drop).
"""

from __future__ import annotations

from ...core import kinds
from ...core.contracts import ChatTurn
from ...core.registry import REGISTRY
from .base import BaseGuardrail
from .yamlops import _reconcile_planned_products


class ProductReconciliationGuardrail(BaseGuardrail):
    name = 'product_reconciliation'
    label = 'Add or drop product sections to match the plan'

    def apply(self, turn: ChatTurn) -> ChatTurn:
        analysis = turn.analysis or {}
        turn.yaml, injected, notes = _reconcile_planned_products(
            turn.yaml, analysis.get('analyses', []), turn.user_message,
            semantic_intent=turn.semantic_intent)
        if injected:
            turn.response_parts.append(
                "\n**Consistency check:** the generated YAML was missing the "
                f"section(s) `{'`, `'.join(injected)}` required by the "
                "planned analyses — default section(s) were added so the "
                "requested product(s) actually run.")
            turn.note(self.name, f'added {", ".join(injected)}')
        for note in notes:
            turn.response_parts.append("\n**Consistency check:** " + note)
            turn.note(self.name, note)
        return turn


REGISTRY.register(kinds.GUARDRAIL, 'product_reconciliation',
                  ProductReconciliationGuardrail, priority=300, source='core',
                  metadata={'label': ProductReconciliationGuardrail.label})
