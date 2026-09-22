"""Pre-execution review: repair, validate and freeze what will run.

Clicking *Run Pipeline* never starts an analysis. It produces the exact two
files that would run — the repaired YAML and the reviewed ``analysis.py`` —
plus an approval token bound to their content digest. The second click
carries that token and runs those bytes. Editing anything in between voids
the token, so what a scientist approved is what executes.

This stage is also where the script Validation Agent runs (deterministic AST
checks plus an LLM fidelity review) and where an unchanged configuration can
offer to reuse a previous fit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets as _secrets
import uuid
import yaml

from fastapi import HTTPException

from ...core.runtime import RUNTIME
from ...fermipy.estimates import expected_sed_bin_count, fit_timeout_for
from ...components.models.prompting import DEFAULT_MODEL_ID
from ...fermipy.targets import test_idx_for_target, yaml_target
from ..exporters.run_bundle import write_run_exports

def _build_analysis_python(yaml_str, run_mode='full'):
    """Build the canonical script that the isolated worker will execute."""
    try:
        config = yaml.safe_load(yaml_str) or {}
    except Exception:
        config = {}
    if not isinstance(config, dict):
        config = {}

    selection = config.get('selection') or {}
    target = str(selection.get('target') or '')
    lines = [
        '"""Execute the reviewed FermiPy analysis without hidden rewrites."""',
        '',
        'import os',
        'from pathlib import Path',
        '',
        'import yaml',
        'from fermipy.gtanalysis import GTAnalysis',
        '',
        'RUN_DIR = Path(__file__).resolve().parent',
        "WORK_DIR = Path(os.environ.get('FERMI_LLM_WORK_DIR', RUN_DIR))",
        "config_path = Path(os.environ.get(",
        "    'FERMI_LLM_CONFIG_PATH', RUN_DIR / 'final_config.yaml'))",
        'WORK_DIR.mkdir(parents=True, exist_ok=True)',
        'os.chdir(WORK_DIR)',
        "with config_path.open() as stream:",
        '    config = yaml.safe_load(stream) or {}',
        '',
        'gta = GTAnalysis(str(config_path))',
        f'target = {target!r}',
    ]

    if run_mode == 'incremental':
        lines += [
            '',
            '# Reuse the previously approved fit; do not silently fall back.',
            "gta.load_roi('fit_model')",
            'optimize_ok = True',
            'fit_result = None',
        ]
    else:
        lines += ['', 'gta.setup()']

    # The default fitting procedure: free the diffuse backgrounds and the
    # target, then optimize and fit once. Nothing beyond that, since this
    # script is only built when the model supplied none and the user
    # therefore stated no fitting directions of their own.
    model_cfg = config.get('model') or {}
    lines += ['', '# Default fit: free the diffuse backgrounds and the target.']
    for diffuse in ('galdiff', 'isodiff'):
        if model_cfg.get(diffuse):
            lines.append(f"gta.free_source({diffuse!r})")
    if target:
        lines.append('gta.free_source(target)')
    if run_mode != 'incremental':
        lines += [
            '',
            'gta.optimize()',
            'optimize_ok = True',
            'fit_result = gta.fit()',
            "gta.write_roi('fit_model', make_plots=False)",
        ]

    lines += [
        '',
        '# Run each requested product with the reviewed YAML options. A failing',
        '# product is recorded and skipped so it cannot discard the completed fit.',
        'product_results = {}',
        'product_errors = {}',
        '',
        '',
        'def run_product(name, func, *args, **kwargs):',
        '    try:',
        '        product_results[name] = func(*args, **kwargs)',
        '    except Exception as exc:',
        '        product_errors[name] = f"{type(exc).__name__}: {exc}"',
        '        print(f"{name} skipped: {product_errors[name]}")',
        '',
        '',
        'def psmap_options():',
        '    # psmap compares data and model cubes: write the model cube first.',
        '    options = dict(config.get("psmap") or {})',
        '    if not options.get("cmap") or not options.get("mmap"):',
        '        gta.write_model_map(model_name="psmap_model")',
        '        options["cmap"] = ":".join(',
        '            c.files["ccube"] for c in gta.components)',
        '        options["mmap"] = ":".join(',
        '            os.path.join(gta.workdir, "mcube_psmap_model%s.fits"',
        '                         % c.config["file_suffix"])',
        '            for c in gta.components)',
        '    selection = config.get("selection") or {}',
        '    for key in ("emin", "emax"):',
        '        if key not in options and selection.get(key) is not None:',
        '            options[key] = float(selection[key])',
        '    return options',
        '',
        '',
        'if "sed" in config:',
        '    run_product("sed", gta.sed, target, **(config["sed"] or {}))',
        '',
        'if "tsmap" in config:',
        '    run_product("tsmap", gta.tsmap, "fit_ts", **(config["tsmap"] or {}))',
        '',
        'if "residmap" in config:',
        '    run_product("residmap", gta.residmap, "fit_resid",',
        '                **(config["residmap"] or {}))',
        '',
        'if "lightcurve" in config:',
        '    run_product("lightcurve", gta.lightcurve, target,',
        '                **(config["lightcurve"] or {}))',
        '',
        'if "psmap" in config:',
        '    run_product("psmap", lambda: gta.psmap(**psmap_options()))',
        '',
        '# The worker reads this in-memory state only to serialize results.',
        'analysis_state = {',
        '    "gta": gta,',
        '    "target": target,',
        '    "fit_result": fit_result,',
        '    "optimize_ok": optimize_ok,',
        '    "product_results": product_results,',
        '    "product_errors": product_errors,',
        '}',
        '',
        "print('Analysis complete')",
        "print(fit_result)",
        '',
    ]
    script = '\n'.join(lines)
    compile(script, 'analysis.py', 'exec')
    return script


def _run_content_digest(yaml_text, python_text):
    """Bind one approval to the exact YAML and Python text."""
    payload = json.dumps(
        {'yaml': yaml_text or '', 'python': python_text or ''},
        sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def _session_requests(session):
    """Every user instruction of the task, oldest first."""
    return [str(m.get('content') or '') for m in session.chat_history
            if m.get('role') == 'user' and m.get('content')]


def _script_review_llm(session):
    """(model_id, prompt->text callable or None, note) for the script review.

    Uses the model the user last generated with, so the review runs on the
    backend they chose (and have credentials for).
    """
    from ...fermipy.script_validator import REVIEW_SCHEMA
    models = RUNTIME.models
    model_id = models.default_model_id
    for message in reversed(session.chat_history):
        if message.get('role') == 'user' and message.get('model'):
            model_id = message['model']
            break
    # A backend that cannot answer a free-form prompt (no LLM, or a LoRA
    # trained only to emit YAML+Python) reviews heuristically instead.
    backend = models.spec(model_id).backend if models.has(model_id) else None
    if backend in (None, 'template', 'vllm_remote'):
        return model_id, None, (
            f'{model_id} cannot review scripts, so request fidelity was '
            f'checked heuristically (static checks still apply).')

    def call(prompt):
        text = RUNTIME.models.complete(model_id, prompt,
                                      response_schema=REVIEW_SCHEMA)
        if text is None:
            raise RuntimeError(f'{model_id} returned no review')
        return text
    return model_id, call, None


def _fermipy_outdir(session_dir, yaml_text):
    """Where FermiPy writes for this YAML inside the session work dir."""
    cfg = {}
    try:
        cfg = yaml.safe_load(yaml_text or '') or {}
    except yaml.YAMLError:
        pass
    outdir = ((cfg.get('fileio') or {}) if isinstance(cfg, dict) else {}).get(
        'outdir') or 'output'
    work_dir = os.path.join(session_dir, 'fermipy_workdir')
    return outdir if os.path.isabs(outdir) else os.path.join(work_dir, outdir)


def _incremental_eligible(session, final_yaml, test_idx):
    """True when the previous fit can be reloaded for this YAML."""
    from ...fermipy.run_execution_validated import core_sections_equal
    if not (session.last_run_roi_ready and session.last_executed_yaml.strip()
            and session.last_run_test_idx == test_idx):
        return False
    if _fermipy_outdir(session.session_dir, final_yaml) != _fermipy_outdir(
            session.session_dir, session.last_executed_yaml):
        return False
    snapshot = os.path.join(
        _fermipy_outdir(session.session_dir, final_yaml), 'fit_model.npy')
    return (os.path.isfile(snapshot)
            and core_sections_equal(final_yaml, session.last_executed_yaml))


def _prepare_run_review(session):
    """Repair the YAML, review the model's analysis.py, stage files for approval.

    Returns one of three responses (nothing is executed here):

    * ``script_changes_proposed`` -- the Validation Agent wants to change the
      script (LLM revision, deterministic fix, or reuse of the previous fit);
      the user must accept or decline via /script_review first.
    * ``script_blocked`` -- deterministic errors (safety, invalid FermiPy API,
      call order) and no acceptable fix; the script cannot run as is.
    * ``review_required`` -- the exact YAML + script that the second
      /run_pipeline POST will execute, with an approval token.
    """
    from ...fermipy.run_execution_validated import (
        repair_config, validate_level1, validate_level2)

    submitted_yaml = session.current_yaml
    target = yaml_target(submitted_yaml)

    # Which files this target runs on is a data-source decision: the bundled
    # demo sets claim their three targets, the weekly archive claims the rest,
    # and a site plugin can claim any of them first.
    from ..datasources import resolve_data
    resolved = resolve_data(RUNTIME.ctx, target, submitted_yaml,
                            session.session_dir)
    test_idx = resolved['test_idx']
    submitted_yaml = resolved.get('yaml') or submitted_yaml
    data_mode = resolved['mode']
    resolution_notes = resolved.get('notes') or []
    resolved_source = resolved.get('resolved_source')

    from ...fermipy import script_validator as _sv
    final_yaml, repairs = repair_config(
        submitted_yaml, test_idx, iteration=0,
        sed_bins=_sv.sed_bins_requested(_session_requests(session)))
    if not final_yaml:
        raise HTTPException(400, 'Validator could not produce a reviewable YAML')

    level1 = validate_level1(final_yaml)
    level2 = validate_level2(final_yaml) if level1.get('fermipy_load') else {}
    if not level1.get('fermipy_load') or not level2.get('gta_init'):
        detail = level1.get('error') or level2.get('error') or 'unknown error'
        raise HTTPException(
            400, f'Pre-execution validation failed; nothing was run: {detail}')

    # Any *additional* yaml-stage validator (site rules, an institute's
    # calibration policy) blocks here on the same terms as the core levels.
    from ..validators import run_stage as _run_validator_stage
    extra = _run_validator_stage(
        RUNTIME.ctx, 'yaml', {'yaml': final_yaml, 'session': session},
        skip=('fermipy_level1', 'fermipy_level2'))
    if not extra['ok']:
        raise HTTPException(
            400, 'Pre-execution validation failed; nothing was run: '
                 + '; '.join(extra['errors'])[:500])

    # ---- Review the model's own analysis.py (it is executed as written) ----
    from ...fermipy import script_validator

    script = session.current_python or ''
    session.current_yaml = final_yaml
    repair_history = [{'iteration': 0, 'repairs': repairs}] if repairs else []
    allow_incremental = _incremental_eligible(session, final_yaml, test_idx)
    base_digest = _run_content_digest(final_yaml, script)
    prior = session.script_reviews.get(base_digest) or {}
    llm_note = None

    if not script.strip():
        review = {
            'findings': [], 'blocking': True, 'llm_used': False,
            'llm_error': None, 'summary': '', 'static': {},
            'proposal': {
                'kind': 'fix',
                'script': _build_analysis_python(final_yaml),
                'reason': ('No Python script was generated. Proposed: the '
                           'standard setup -> optimize -> fit script with the '
                           'products the YAML configures.'),
                'findings': [], 'blocking': False,
            },
        }
    else:
        llm = None
        if not prior:
            model_id, llm, llm_note = _script_review_llm(session)
        review = script_validator.review_script(
            script, final_yaml, requests=_session_requests(session), llm=llm,
            allow_incremental=allow_incremental)
        if review.get('llm_error'):
            llm_note = (f'The LLM review failed ({review["llm_error"]}); '
                        f'request fidelity was checked heuristically.')
        if prior:
            # Already reviewed (and possibly decided): reuse the LLM's
            # findings instead of asking it again for identical content.
            review['findings'] = script_validator._finalize({
                'findings': script_validator._dedupe(
                    review['findings'] + list(prior.get('llm_findings') or []))
            })['findings']
            review['summary'] = prior.get('summary', '')
        else:
            session.script_reviews[base_digest] = {
                'llm_findings': [f for f in review['findings']
                                 if f.get('source') == 'llm'],
                'summary': review.get('summary', ''),
                'decision': None,
            }
            while len(session.script_reviews) > 50:
                session.script_reviews.pop(next(iter(session.script_reviews)))

    proposal = review.get('proposal')
    if proposal and proposal['kind'] in (prior.get('declined_kinds') or []):
        proposal = None
    review_info = {
        'findings': review['findings'],
        'summary': review.get('summary', ''),
        'llm_used': review.get('llm_used', False),
        'llm_note': llm_note,
        'repairs': repairs,
        'resolution_notes': resolution_notes,
    }

    if proposal:
        proposal_id = uuid.uuid4().hex[:12]
        session.script_proposal = {
            'id': proposal_id,
            'kind': proposal['kind'],
            'script': proposal['script'],
            'reason': proposal['reason'],
            'base_digest': base_digest,
            'findings': proposal['findings'],
        }
        session.pending_run = None
        session.pipeline_status = 'script_review'
        session.save()
        return {
            'status': 'script_changes_proposed',
            'proposal_id': proposal_id,
            'proposal_kind': proposal['kind'],
            'reason': proposal['reason'],
            'blocking': review['blocking'],
            'yaml': final_yaml,
            'python': script,
            'proposed_python': proposal['script'],
            'proposal_findings': proposal['findings'],
            **review_info,
            'message': (
                'The Validation Agent proposes changes to analysis.py. Review '
                'the diff, then Apply or Discard it. Nothing has run yet.'),
        }

    session.script_proposal = None
    if review['blocking']:
        session.pending_run = None
        session.pipeline_status = 'script_blocked'
        session.save()
        return {
            'status': 'script_blocked',
            'yaml': final_yaml,
            'python': script,
            **review_info,
            'message': (
                'analysis.py cannot be executed until the errors below are '
                'fixed. Edit the script or ask for a corrected version.'),
        }

    static = review.get('static') or {}
    run_mode = script_validator.infer_run_mode(static)
    config_ref = static.get('config_ref')
    final_python = script
    digest = base_digest
    run_id = uuid.uuid4().hex
    approval_token = _secrets.token_urlsafe(24)
    preview = {
        'status': 'review_required',
        'success': False,
        'run_id': run_id,
        'run_mode': run_mode,
        'final_yaml': final_yaml,
        'final_python': final_python,
        'repair_history': repair_history,
        # Edges in the script override the count the YAML implies, exactly
        # as the post-run contract does; otherwise the review panel promises
        # a bin count the approved script will not produce.
        'expected_sed_bins': (script_validator.script_sed_bin_count(final_python)
                              or expected_sed_bin_count(final_yaml)),
    }
    write_run_exports(preview, session.session_dir, run_id)
    script_path = os.path.join(
        session.session_dir, 'runs', run_id, 'analysis.py')

    session.current_python = final_python
    session.pipeline_status = 'review_required'
    session.pipeline_result = None
    session.pending_run = {
        'approval_token': approval_token,
        'digest': digest,
        'run_id': run_id,
        'run_mode': run_mode,
        'test_idx': test_idx,
        'script_path': script_path,
        'config_aliases': [config_ref] if config_ref else [],
        'final_yaml': final_yaml,
        'final_python': final_python,
        'data_mode': data_mode,
        'resolved_source': resolved_source,
        'resolution_notes': resolution_notes,
        'repair_history': repair_history,
        'review_findings': review['findings'],
        'expected_sed_bins': preview['expected_sed_bins'],
        'downloads': preview.get('downloads', {}),
    }
    session.save()
    return {
        'status': 'review_required',
        'approval_token': approval_token,
        'run_id': run_id,
        'run_mode': run_mode,
        'yaml': final_yaml,
        'python': final_python,
        'expected_sed_bins': preview['expected_sed_bins'],
        'downloads': preview.get('downloads', {}),
        **review_info,
        'message': (
            'The Validation Agent reviewed analysis.py; it will be executed '
            'exactly as shown, with the YAML shown. Click Run Pipeline again '
            'to start.'),
    }


prepare_run_review = _prepare_run_review
build_analysis_python = _build_analysis_python
run_content_digest = _run_content_digest
