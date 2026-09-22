"""The deterministic FermiPy validation levels, as registered components.

Level 1 parses the YAML against the FermiPy schema, Level 2 constructs a
``GTAnalysis`` without touching data, Level 3 runs ``gta.setup()``. The run
pipeline calls these directly because their results have dedicated slots in
the run record; registering them as well gives plugins a place to sit *next
to* them in the same stage, and gives the platform view something to show.
"""

from __future__ import annotations

from typing import Any, Dict

from ...core import kinds
from ...core.contracts import ValidationResult
from ...core.registry import REGISTRY


class _LevelValidator:
    stage = 'yaml'
    level = 1

    def __init__(self, ctx):
        self.ctx = ctx

    def _call(self, yaml_text):                           # pragma: no cover
        raise NotImplementedError

    def validate(self, artifact: Dict[str, Any]) -> ValidationResult:
        yaml_text = artifact.get('yaml') or ''
        if not yaml_text.strip():
            return ValidationResult(ok=False, stage=self.stage,
                                    errors=['empty configuration'])
        data = self._call(yaml_text) or {}
        error = data.get('error')
        return ValidationResult(ok=not error, stage=self.stage,
                                errors=[error] if error else [], data=data)


class YamlSchemaValidator(_LevelValidator):
    """Level 1: the YAML parses and FermiPy accepts every key."""

    name = 'fermipy_level1'
    label = 'Level 1 — YAML schema'

    def _call(self, yaml_text):
        from ...fermipy.run_execution_validated import validate_level1
        return validate_level1(yaml_text)


class GTAnalysisInitValidator(_LevelValidator):
    """Level 2: ``GTAnalysis(config)`` constructs."""

    name = 'fermipy_level2'
    label = 'Level 2 — GTAnalysis init'

    def _call(self, yaml_text):
        from ...fermipy.run_execution_validated import validate_level2
        return validate_level2(yaml_text)


REGISTRY.register(kinds.VALIDATOR, 'fermipy_level1', YamlSchemaValidator,
                  priority=100, source='core',
                  metadata={'label': YamlSchemaValidator.label,
                            'stage': 'yaml'})
REGISTRY.register(kinds.VALIDATOR, 'fermipy_level2', GTAnalysisInitValidator,
                  priority=110, source='core',
                  metadata={'label': GTAnalysisInitValidator.label,
                            'stage': 'yaml'})
