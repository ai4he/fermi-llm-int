"""The three bundled demonstration sources.

Mrk 421, the Vela pulsar and the Crab ship with their own photon and
spacecraft files under ``lat_data/``, so a fresh checkout can run a real
analysis end to end with no archive access. Anything this source does not
claim falls through to the next one.
"""

from __future__ import annotations

from typing import Any, Dict

from ...core import kinds
from ...core.registry import REGISTRY
from ...fermipy.targets import test_idx_for_target


class BundledLATDataSource:
    """Resolves a target to one of the bundled demo datasets."""

    name = 'bundled_lat'
    label = 'Bundled demo data (Mrk 421, Vela, Crab)'

    def __init__(self, ctx):
        self.ctx = ctx

    def handles(self, target: str, spec: Dict[str, Any]) -> bool:
        return test_idx_for_target(target) is not None

    def resolve(self, target: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'mode': 'bundled',
            'test_idx': test_idx_for_target(target),
            'yaml': spec.get('yaml', ''),
            'notes': [],
            'resolved_source': None,
        }


REGISTRY.register(kinds.DATA_SOURCE, 'bundled_lat', BundledLATDataSource,
                  priority=100, source='core',
                  metadata={'label': BundledLATDataSource.label})
