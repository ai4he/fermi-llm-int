"""Example: an institute's own pre-run rule.

Blocks a run whose energy range falls outside the band this (imaginary)
collaboration trusts. Registered in the ``yaml`` stage, so it runs next to
the FermiPy levels and blocks on the same terms.

What it demonstrates: the ``validator`` contract, and that a site policy can
be enforced without touching the review pipeline.
"""

from __future__ import annotations

from typing import Any, Dict

import yaml as _yaml

from fermi_llm.core import kinds
from fermi_llm.core.contracts import ValidationResult

MIN_EMIN_MEV = 50.0
MAX_EMAX_MEV = 3.0e6


class StrictEnergyValidator:
    name = 'strict_energy_range'
    label = 'Energy range within the collaboration band'
    stage = 'yaml'

    def __init__(self, ctx):
        self.ctx = ctx
        options = ctx.component_settings(kinds.VALIDATOR, self.name)
        self.min_emin = float(options.get('min_emin_mev', MIN_EMIN_MEV))
        self.max_emax = float(options.get('max_emax_mev', MAX_EMAX_MEV))

    def validate(self, artifact: Dict[str, Any]) -> ValidationResult:
        try:
            config = _yaml.safe_load(artifact.get('yaml') or '') or {}
            selection = config.get('selection') or {}
            emin = float(selection.get('emin'))
            emax = float(selection.get('emax'))
        except (TypeError, ValueError, _yaml.YAMLError):
            # Not this validator's job to complain about malformed YAML.
            return ValidationResult(ok=True, stage=self.stage)

        errors = []
        if emin < self.min_emin:
            errors.append(f'emin {emin:g} MeV is below the agreed '
                          f'{self.min_emin:g} MeV floor')
        if emax > self.max_emax:
            errors.append(f'emax {emax:g} MeV is above the agreed '
                          f'{self.max_emax:g} MeV ceiling')
        return ValidationResult(ok=not errors, stage=self.stage, errors=errors,
                                data={'emin': emin, 'emax': emax})


def register(registry, ctx=None):
    registry.register(kinds.VALIDATOR, 'strict_energy_range',
                      StrictEnergyValidator, priority=150,
                      source='example:strict_energy_validator',
                      metadata={'label': StrictEnergyValidator.label,
                                'stage': 'yaml'})
