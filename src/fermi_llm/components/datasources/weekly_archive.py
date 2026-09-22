"""The weekly all-sky archive: any 4FGL source, or an explicit position.

Builds the event-file list and the time range for a target that is not one of
the bundled demos, resolving the name through the 4FGL catalog first. A
deployment with its own archive layout replaces this component; one with an
*additional* archive registers another and orders it by priority.
"""

from __future__ import annotations

from typing import Any, Dict

from ...core import kinds
from ...core.registry import REGISTRY

_RESOLVED_KEYS = ('target', 'name', 'ra', 'dec', 'class1', 'catalog',
                  'catalogued', 'target_origin', 'tmin', 'tmax', 'n_weeks',
                  'extended')


class WeeklyArchiveDataSource:
    """Photon files from ``FERMI_LLM_FERMI_DATA_DIR``."""

    name = 'weekly_archive'
    label = 'Weekly all-sky archive (any 4FGL source)'

    def __init__(self, ctx):
        self.ctx = ctx

    def handles(self, target: str, spec: Dict[str, Any]) -> bool:
        # Last resort: claims everything the bundled source did not.
        return True

    def resolve(self, target: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        from ...fermipy.data_registry import build_full_data_spec
        test_idx, yaml_text, notes = build_full_data_spec(
            spec.get('yaml', ''), spec.get('session_dir', ''))
        return {
            'mode': 'full',
            'test_idx': test_idx,
            'yaml': yaml_text,
            'notes': notes,
            'resolved_source': {k: test_idx.get(k) for k in _RESOLVED_KEYS},
        }


REGISTRY.register(kinds.DATA_SOURCE, 'weekly_archive', WeeklyArchiveDataSource,
                  priority=900, source='core',
                  metadata={'label': WeeklyArchiveDataSource.label})
