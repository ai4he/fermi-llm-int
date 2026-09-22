"""Static review of the model's ``analysis.py`` before it may run.

Deterministic only: an import allow-list, GTAnalysis methods and keyword
options read from the installed FermiPy, call order (setup → fit →
products), and agreement with the reviewed YAML. Findings marked ``error``
block the run; everything else is shown to the user.

This is defence in depth, not a sandbox — the execution backend is what
isolates the process.
"""

from __future__ import annotations

from typing import Any, Dict

from ...core import kinds
from ...core.contracts import ValidationResult
from ...core.registry import REGISTRY


class ScriptStaticValidator:
    name = 'script_static'
    label = 'Script safety and FermiPy API check'
    stage = 'script'

    def __init__(self, ctx):
        self.ctx = ctx

    def validate(self, artifact: Dict[str, Any]) -> ValidationResult:
        from ...fermipy import script_validator as sv
        script = artifact.get('python') or ''
        if not script.strip():
            return ValidationResult(ok=True, stage=self.stage)
        review = sv.static_review(script, artifact.get('yaml') or '')
        findings = review.get('findings') or []
        errors = [f['message'] for f in findings if f.get('severity') == 'error']
        warnings = [f['message'] for f in findings
                    if f.get('severity') != 'error']
        return ValidationResult(ok=not review.get('blocking'), stage=self.stage,
                                errors=errors, warnings=warnings, data=review)


REGISTRY.register(kinds.VALIDATOR, 'script_static', ScriptStaticValidator,
                  priority=200, source='core',
                  metadata={'label': ScriptStaticValidator.label,
                            'stage': 'script'})
