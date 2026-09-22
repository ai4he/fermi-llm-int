"""Pin the diffuse models to the files installed on this machine.

``gll_iem_v07.fits`` and the matching isotropic template must be absolute
paths that exist in *this* ScienceTools installation, and the isotropic file
has to match the event type. Models write bare filenames from their training
data, which fail at run time with an opaque error, so the paths are rewritten
here — the same resolver the validator uses.
"""

from __future__ import annotations

from ...core import kinds
from ...core.contracts import ChatTurn
from ...core.registry import REGISTRY
from ...fermipy.diffuse_models import normalize_diffuse_yaml
from .base import BaseGuardrail


class DiffusePathsGuardrail(BaseGuardrail):
    name = 'diffuse_paths'
    label = 'Pin galdiff/isodiff to the installed files'

    def apply(self, turn: ChatTurn) -> ChatTurn:
        turn.yaml, changes = normalize_diffuse_yaml(turn.yaml)
        if changes:
            turn.response_parts.append(
                "\n**Environment check:** pinned `model.galdiff` and "
                "`model.isodiff` to the installed Fermi diffuse-model files.")
            turn.note(self.name, 'pinned diffuse paths')
        return turn


REGISTRY.register(kinds.GUARDRAIL, 'diffuse_paths', DiffusePathsGuardrail,
                  priority=500, source='core',
                  metadata={'label': DiffusePathsGuardrail.label})
