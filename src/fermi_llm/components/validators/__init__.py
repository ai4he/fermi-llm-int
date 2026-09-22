"""Checks that run before an analysis is allowed to execute.

Validators are grouped by ``stage`` (``yaml``, ``script``, ``run``) and
stacked within a stage. Core registers the FermiPy levels and the script
review; a plugin validator in the same stage runs alongside them, and an
``ok=False`` result with errors blocks the run exactly like a core failure.
"""

from . import fermipy_levels, script_static  # noqa: F401


def run_stage(ctx, stage, artifact, skip=()):
    """Run every active validator of ``stage`` and merge the results."""
    from ...core import kinds
    errors, warnings, data = [], [], {}
    for validator in ctx.build_stack(kinds.VALIDATOR):
        if validator.stage != stage or validator.name in skip:
            continue
        try:
            result = validator.validate(artifact)
        except Exception as exc:                          # noqa: BLE001
            warnings.append(f'{validator.name} failed to run: {exc}')
            continue
        data[validator.name] = result.data
        if not result.ok:
            errors.extend(f'{validator.name}: {e}' for e in result.errors)
        warnings.extend(f'{validator.name}: {w}' for w in result.warnings)
    return {'ok': not errors, 'errors': errors, 'warnings': warnings,
            'data': data}
