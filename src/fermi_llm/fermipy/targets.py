"""Target naming: aliases, bundled demo sources and ROI coordinates.

The three bundled demonstration sources (Mrk 421, Vela, the Crab) ship with
their own photon files; every other target is resolved against the 4FGL
catalog and the weekly archive. Keeping the alias tables here means a data
source plugin can reuse them instead of re-implementing name matching.
"""

from __future__ import annotations

import re
import yaml

# Common-name aliases for the three bundled sources (see run_pipeline_isolated
# for how they map onto the bundled photon/spacecraft data).
_SOURCE_ALIASES = {
    0: ['4fgl j1104.4+3812', 'mrk421', 'mrk 421', 'markarian 421', 'markarian421'],
    1: ['4fgl j0835.3-4510', 'vela', 'vela pulsar', 'psr j0835-4510'],
    2: ['4fgl j0534.5+2200', '4fgl j0534.5+2201', 'crab', 'crab nebula', 'crab pulsar'],
}


# ROI center coordinates (deg, J2000) for the three bundled sources, keyed by
# their 4FGL designation prefix. Used when enforcing a requested target so the
# ROI actually centers on it.
_SOURCE_COORDS = {
    '4fglj1104.4+3812': (166.114, 38.209),   # Mrk 421
    '4fglj0835.3-4510': (128.836, -45.176),  # Vela
    '4fglj0534.5+2200': (83.633, 22.015),    # Crab
    '4fglj0534.5+2201': (83.633, 22.015),
}


def _norm_name(s):
    return re.sub(r'[^a-z0-9.+-]', '', (s or '').lower())


def _target_is_bundled(target):
    """True when the target maps onto one of the three bundled demo sources."""
    tnorm = _norm_name(target)
    if not tnorm:
        return False
    for aliases in _SOURCE_ALIASES.values():
        anorm = [_norm_name(a) for a in aliases]
        if any(a in tnorm or tnorm in a for a in anorm if a):
            return True
    return False


def _yaml_target(yaml_str):
    """selection.target from a YAML string, or ''."""
    try:
        cfg = yaml.safe_load(yaml_str) or {}
        return str((cfg.get('selection') or {}).get('target') or '')
    except Exception:
        return ''


def _bundled_idx_for_target(target):
    """Bundled demo index (0-2) for a target, or None."""
    tnorm = _norm_name(target)
    if not tnorm:
        return None
    for idx, aliases in _SOURCE_ALIASES.items():
        anorm = [_norm_name(a) for a in aliases]
        if any(a in tnorm or tnorm in a for a in anorm if a):
            return idx
    return None


def _coords_for_target(target_name):
    """(ra, dec) for a target: bundled table first, then the 4FGL catalog."""
    coords = _SOURCE_COORDS.get(_norm_name(target_name))
    if coords:
        return coords
    try:
        from . import source_resolver
        res = source_resolver.resolve(target_name)
        if res.get('found') and res.get('ra') is not None:
            return (res['ra'], res['dec'])
    except Exception:
        pass
    return None


def _test_idx_for_target(target):
    normalized = (target or '').replace(' ', '').lower()
    for idx, aliases in _SOURCE_ALIASES.items():
        if any(alias.replace(' ', '') in normalized
               or normalized in alias.replace(' ', '')
               for alias in aliases if alias):
            return idx
    return None


# Public aliases (the underscore names are kept for moved code).
SOURCE_ALIASES = _SOURCE_ALIASES
SOURCE_COORDS = _SOURCE_COORDS
norm_name = _norm_name
yaml_target = _yaml_target
target_is_bundled = _target_is_bundled
bundled_idx_for_target = _bundled_idx_for_target
coords_for_target = _coords_for_target
test_idx_for_target = _test_idx_for_target
