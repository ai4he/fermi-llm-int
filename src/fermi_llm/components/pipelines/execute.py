"""Worker-side execution of an approved analysis.

``run_pipeline_isolated`` is what a dedicated worker process runs: it
re-validates the approved files, executes the reviewed script through the
active execution backend, then collects fit results, products and science
checks. It writes progress as JSON lines, so a browser that reconnects (or
an operator tailing the file) sees the same stream the run produced.

Nothing in here imports the web layer: the module must be usable in a bare
subprocess with no FastAPI, which is also what makes a batch or cluster
runner possible.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
import yaml

from ...core.contracts import ExecutionJob
from ...core.progress import write_progress
from ...core.runtime import RUNTIME
from ...fermipy.estimates import (expected_sed_bin_count, fit_timeout_for,
                                  _sed_bins_from_script_calls)
import uuid

from ...fermipy.targets import (_SOURCE_ALIASES, test_idx_for_target,
                                yaml_target)
from ..exporters.run_bundle import write_run_exports
from .review import run_content_digest

def _science_results_from_l4(l4_result, target_unmatched):
    """Build the 'science_results' summary dict from a Level 4 result.

    Shared by the full-run and incremental-run code paths in
    ``run_pipeline_isolated`` so both produce an identical shape.
    """
    return {
        'target_ts': l4_result.get('target_ts'),
        'target_flux': l4_result.get('target_flux'),
        'target_flux_error': l4_result.get('target_flux_error'),
        'target_flux_ul95': l4_result.get('target_flux_ul95'),
        'target_eflux_ul95': l4_result.get('target_eflux_ul95'),
        'detection_status': l4_result.get('detection_status'),
        'detection_threshold_ts': l4_result.get('detection_threshold_ts'),
        'spectral_index': l4_result.get('spectral_index'),
        'spectral_index_error': l4_result.get('spectral_index_error'),
        'catalog_flux': l4_result.get('catalog_flux'),
        'catalog_check_applicable':
            l4_result.get('catalog_check_applicable', False),
        'flux_ratio': l4_result.get('flux_ratio'),
        'fit_quality': l4_result.get('fit_quality'),
        'convergence_ok': l4_result.get('convergence_ok'),
        'sources_summary': l4_result.get('sources_summary', []),
        'loglike': l4_result.get('loglike'),
        'artifacts': l4_result.get('artifacts', []),
        'sed_data': l4_result.get('sed_data'),
        'requested_products': l4_result.get('requested_products', []),
        'target_spectral_model': l4_result.get('target_spectral_model'),
        'target_warning': l4_result.get('target_warning'),
        'target_unmatched': target_unmatched,
        'roi_sources': l4_result.get('roi_sources', []),
    }


def _run_validation_with_progress(yaml_str, test_idx, session_dir, max_iter=5,
                                  work_dir=None):
    """Run the iterative validation-repair loop with progress updates.

    ``work_dir``: when given, Level-3 setup runs there and keeps its
    outputs (ltcube, srcmaps, ...) so the subsequent Level-4 run with the
    same config reuses them instead of redoing gta.setup() -- essential
    for full-dataset runs where setup can take a long time.
    """
    from ...fermipy.run_execution_validated import (
        repair_config, validate_level1, validate_level2, validate_level3,
        get_test_meta,
    )

    meta = get_test_meta(test_idx)
    evfile = meta.get('evfile', '')
    scfile = meta.get('scfile', '')

    all_results = []
    repair_history = []
    current_yaml = yaml_str
    prev_error = None

    for iteration in range(max_iter):
        write_progress(session_dir, 'repair', 'active',
                       f'Validation Agent: Repair iteration {iteration + 1}/{max_iter} -- diagnosing and applying targeted repairs...',
                       agent='validation')

        # Apply repair
        repaired_yaml, repairs = repair_config(
            current_yaml, test_idx,
            iteration=iteration, prev_error=prev_error
        )

        if not repaired_yaml:
            repair_history.append({
                'iteration': iteration, 'repairs': repairs,
                'result': 'repair failed - empty yaml'
            })
            write_progress(session_dir, 'repair', 'fail',
                           f'Validation Agent: Repair iteration {iteration + 1} failed: empty YAML',
                           agent='validation')
            break

        if repairs:
            repair_history.append({'iteration': iteration, 'repairs': repairs})
            write_progress(session_dir, 'repair', 'active',
                           f'Validation Agent: Applied {len(repairs)} repairs: {", ".join(repairs[:3])}{"..." if len(repairs) > 3 else ""}',
                           agent='validation')

        # Level 1: YAML parsing
        write_progress(session_dir, 'level1', 'active',
                       f'Validation Agent: [Iter {iteration + 1}] Level 1 -- Parsing YAML + ConfigManager validation...',
                       agent='validation')
        l1 = validate_level1(repaired_yaml)

        if not l1.get('fermipy_load'):
            all_results.append({'level1': l1, 'level2': {}, 'level3': {}})
            prev_error = l1.get('error', 'Level 1 failed')
            current_yaml = repaired_yaml
            write_progress(session_dir, 'level1', 'fail',
                           f'Validation Agent: Level 1 failed: {l1.get("error", "unknown")[:150]}',
                           agent='validation')
            continue

        write_progress(session_dir, 'level1', 'pass',
                       'Validation Agent: Level 1 passed -- YAML is structurally valid',
                       agent='validation')

        # Level 2: GTAnalysis instantiation
        write_progress(session_dir, 'level2', 'active',
                       f'Validation Agent: [Iter {iteration + 1}] Level 2 -- Instantiating GTAnalysis (target / ROI model check)...',
                       agent='validation')
        l2 = validate_level2(repaired_yaml)

        if not l2.get('gta_init'):
            all_results.append({'level1': l1, 'level2': l2, 'level3': {}})
            prev_error = l2.get('error', 'Level 2 failed')
            current_yaml = repaired_yaml
            write_progress(session_dir, 'level2', 'fail',
                           f'Validation Agent: Level 2 failed: {l2.get("error", "unknown")[:150]}',
                           agent='validation')
            continue

        write_progress(session_dir, 'level2', 'pass',
                       'Validation Agent: Level 2 passed -- GTAnalysis initialized, target and ROI model loaded',
                       agent='validation')

        # Level 3: gta.setup() with real data
        write_progress(session_dir, 'level3', 'active',
                       f'Validation Agent: [Iter {iteration + 1}] Level 3 -- Running gta.setup() with real Fermi-LAT data '
                       f'(gtselect -> gtmktime -> gtltcube -> gtsrcmaps)...',
                       agent='validation')
        l3 = validate_level3(repaired_yaml, evfile, scfile, work_dir=work_dir)

        iter_result = {'level1': l1, 'level2': l2, 'level3': l3}
        all_results.append(iter_result)

        if l3.get('setup_ok'):
            write_progress(session_dir, 'level3', 'pass',
                           f'Validation Agent: Level 3 passed! gta.setup() completed. '
                           f'{len(l3.get("sources", []))} sources in ROI.',
                           agent='validation')
            return repaired_yaml, all_results, repair_history

        prev_error = l3.get('error', 'Level 3 failed')
        current_yaml = repaired_yaml
        write_progress(session_dir, 'level3', 'fail',
                       f'Validation Agent: Level 3 failed (iteration {iteration + 1}): {l3.get("error", "unknown")[:150]}',
                       agent='validation')

    return current_yaml, all_results, repair_history


def _attempt_incremental_run(yaml_str, test_idx, session_dir, prev_run_state, artifact_dir):
    """Try to reuse a previous successful run's fitted ROI via gta.load_roi()
    instead of redoing gta.setup() + gta.optimize() + gta.fit().

    Only viable when the incoming YAML's core (setup-affecting) sections --
    data, selection, binning, model, components, gtlike -- are identical to
    the previous successful run's, and a saved ROI snapshot exists in that
    run's work directory. Sections that only affect which post-fit products
    get computed (sed, lightcurve, tsmap, residmap, psmap, plotting) never
    block incremental reuse.

    Returns a dict of fields to merge into the pipeline result on success
    (including 'run_mode': 'incremental'), or None if incremental reuse is
    not viable -- the caller should then fall back to a normal full run.
    """
    from ...fermipy.run_execution_validated import (
        repair_config, validate_level1, validate_level2, core_sections_equal,
        get_test_meta,
    )
    from ...fermipy.run_level4_validation import validate_level4

    workdir = prev_run_state.get('workdir')
    prev_yaml = prev_run_state.get('yaml', '')
    roi_prefix = prev_run_state.get('roi_prefix', 'fit_model')

    if not workdir or not os.path.isdir(workdir) or not prev_yaml:
        return None
    if prev_run_state.get('test_idx') is not None and prev_run_state['test_idx'] != test_idx:
        return None

    # Cheap repair pass (no gta.setup(), just dict/YAML manipulation and a
    # FITS header read) to get the config that would actually be executed,
    # for a fair comparison against the previous run's executed config.
    repaired_yaml, repairs = repair_config(yaml_str, test_idx, iteration=0)
    if not repaired_yaml:
        return None
    if not core_sections_equal(repaired_yaml, prev_yaml):
        return None

    write_progress(session_dir, 'incremental', 'active',
                   'Core setup parameters (data/selection/binning/model/components/gtlike) '
                   'match the previous run -- attempting an incremental run via gta.load_roi() '
                   '(skips gta.setup() + gta.optimize() + gta.fit())...',
                   agent='validation')

    l1 = validate_level1(repaired_yaml)
    if not l1.get('fermipy_load'):
        return None
    l2 = validate_level2(repaired_yaml)
    if not l2.get('gta_init'):
        return None

    meta = get_test_meta(test_idx)
    evfile = meta.get('evfile', '')
    scfile = meta.get('scfile', '')
    if not evfile or not scfile:
        return None

    l4 = validate_level4(
        repaired_yaml, evfile, scfile, test_idx,
        fit_timeout=fit_timeout_for(test_idx), artifact_dir=artifact_dir,
        work_dir=workdir, incremental=True, roi_prefix=roi_prefix,
    )
    if l4.get('incremental_failed') or not l4.get('fit_ok'):
        write_progress(session_dir, 'incremental', 'fail',
                       f"Incremental run failed ({str(l4.get('error', 'unknown'))[:150]}); "
                       f"falling back to a full run.",
                       agent='validation')
        return None

    write_progress(session_dir, 'incremental', 'pass',
                   'Incremental run succeeded -- reused the previous fit via gta.load_roi(), '
                   'only (re)computed the requested products.',
                   agent='validation')

    return {
        'final_yaml': repaired_yaml,
        'repair_history': [{'iteration': 0, 'repairs': repairs}] if repairs else [],
        'validation_results': [],
        'level1': l1,
        'level2': l2,
        'level3': {
            'level': 3, 'setup_ok': True,
            'note': 'skipped gta.setup(); reused the previous run via gta.load_roi()',
            'sources': [s.get('name') for s in l4.get('sources_summary', [])[:10]],
        },
        'level4': l4,
        'success': True,
        'run_mode': 'incremental',
        'incremental_info': {
            'reused_workdir': workdir,
            'skipped_stages': ['setup', 'optimize', 'fit'],
            'note': ('Core YAML sections (data, selection, binning, model, components, '
                     'gtlike) were unchanged from the previous successful run, so the '
                     'previously fitted ROI model was reloaded with gta.load_roi() instead '
                     'of re-running gta.setup()/gta.optimize()/gta.fit().'),
        },
    }


def run_pipeline_isolated(session_dir, yaml_str, python_str, test_idx=None,
                          prev_run_state=None, run_id=None,
                          approved_script_path=None, approved_digest=None,
                          approved_run_mode='full', approved_config_aliases=()):
    """Run FermiPy pipeline in an isolated process.

    Runs Levels 1-3 (repair + validation) and, if Level 3 passes,
    Level 4 (gta.optimize() + gta.fit()) for science-quality results.

    If ``prev_run_state`` is given (see the /run_pipeline endpoint) and the
    incoming YAML's core setup sections match the previous successful run,
    an incremental run is attempted first: it reuses the previous fit via
    gta.load_roi() and only (re)computes the requested products, instead of
    redoing gta.setup() + gta.optimize() + gta.fit() (which can take
    30 minutes to 2+ hours). Any failure during the incremental attempt
    falls back automatically to a full run.

    Writes progress updates to pipeline_progress.jsonl for SSE streaming.
    """
    # Give each run its own process group.  The abort endpoint can then stop
    # this worker and any ScienceTools children it launched without touching
    # other sessions (or another webapp instance on the same host).
    run_id = run_id or uuid.uuid4().hex
    process_group_isolated = False
    try:
        os.setsid()
        process_group_isolated = True
    except (AttributeError, OSError):
        pass
    with open(os.path.join(session_dir, 'pipeline_worker.json'), 'w') as f:
        json.dump({
            'run_id': run_id,
            'pid': os.getpid(),
            'process_group_isolated': process_group_isolated,
            'started_at': time.time(),
        }, f)

    from ...fermipy.run_execution_validated import (
        repair_config, validate_level1, validate_level2, validate_level3,
        validate_with_iterative_repair, extract_yaml, extract_python,
    )

    # The run_pipeline endpoint already cleared the progress file and wrote
    # the 'run_started' line (runtime estimate for the frontend ETA bar) --
    # do not clear it again here.

    result = {
        'status': 'running',
        'steps': [],
        'final_yaml': yaml_str,
        'submitted_python': python_str,
        'final_python': python_str if approved_script_path else None,
        'level1': None,
        'level2': None,
        'level3': None,
        'level4': None,
        'success': False,
        'run_mode': 'full',
        'test_idx': None,
        'run_id': run_id,
        'approved_digest': approved_digest,
    }
    # Match on the 4FGL designation AND on common-name aliases, since the
    # prompt/LLM frequently uses the popular name ("Markarian 421", "Crab",
    # "Vela") rather than the catalogue id. Only the three bundled sources
    # have data available. (Module-level: also used by the chat handler's
    # target-preservation guard.)
    SOURCE_ALIASES = _SOURCE_ALIASES

    write_progress(session_dir, 'init', 'active',
                   'Handing off to Validation Agent for iterative repair and validation...',
                   agent='master')

    target = ''
    if test_idx is None:
        try:
            parsed = yaml.safe_load(yaml_str)
            target = (parsed.get('selection', {}).get('target', '') or '')
            tnorm = target.replace(' ', '').lower()
            for idx, aliases in SOURCE_ALIASES.items():
                if any(a.replace(' ', '') in tnorm or tnorm in a.replace(' ', '')
                       for a in aliases if a):
                    test_idx = idx
                    break
        except Exception:
            pass

    if test_idx is None:
        # Not one of the three bundled demo sources: resolve the target
        # against the 4FGL catalog and run on the full weekly dataset
        # (FERMI_LLM_FERMI_DATA_DIR). There is deliberately NO fallback to
        # bundled data here -- silently fitting a different source's data
        # buries the real problem inside a confusing science result. An
        # unresolvable target aborts the run with an explicit error.
        write_progress(session_dir, 'source_resolution', 'active',
                       f'Resolving "{target or "(no target)"}" against the '
                       f'4FGL catalog and the weekly data registry...',
                       agent='validation')
        try:
            from ...fermipy.data_registry import build_full_data_spec, DataRegistryError
            spec, yaml_str, notes = build_full_data_spec(yaml_str, session_dir)
            test_idx = spec
            result['final_yaml'] = yaml_str
            result['data_mode'] = 'full'
            result['resolved_source'] = {
                k: spec.get(k) for k in
                ('target', 'name', 'ra', 'dec', 'class1', 'catalog',
                 'catalogued', 'target_origin', 'tmin', 'tmax', 'n_weeks',
                 'extended')}
            for note in notes:
                write_progress(session_dir, 'source_resolution', 'active',
                               f'Data resolution: {note}', agent='validation')
            write_progress(session_dir, 'source_resolution', 'pass',
                           f'Resolved {spec["target"]} ({spec["name"]}): '
                           f'{spec["n_weeks"]} weekly photon files + merged '
                           f'spacecraft file (full-dataset run).',
                           agent='validation')
        except Exception as e:
            is_registry_err = type(e).__name__ in (
                'DataRegistryError', 'SourceNotFoundError', 'NoDataError')
            msg = (str(e) if is_registry_err else
                   f'data resolution failed unexpectedly: '
                   f'{type(e).__name__}: {str(e)[:300]}')
            result['status'] = 'error'
            result['error'] = f'Source/data resolution failed: {msg}'
            result['success'] = False
            write_progress(session_dir, 'source_resolution', 'fail',
                           f'Source/data resolution failed: {msg[:400]}',
                           agent='validation')
            write_progress(session_dir, 'done', 'error',
                           'Pipeline aborted: the requested source could not '
                           'be mapped onto available data.', agent='system')
            with open(os.path.join(session_dir, 'pipeline_result.json'), 'w') as f:
                json.dump(result, f, indent=2, default=str)
            return result
    else:
        # Preflight resolves arbitrary/full-data targets to a dict before
        # this worker starts.  Preserve that mode instead of relabelling the
        # already-resolved run as one of the three bundled demonstrations.
        result['data_mode'] = (
            'full' if isinstance(test_idx, dict) else 'bundled')

    result['test_idx'] = test_idx
    artifact_dir = os.path.join(session_dir, 'artifacts')
    persistent_workdir = os.path.join(session_dir, 'fermipy_workdir')

    # Approved runs are immutable: validate the reviewed YAML/Python, execute
    # that exact analysis.py, and never repair or replace it in the worker.
    if approved_script_path:
        actual_digest = run_content_digest(yaml_str, python_str)
        try:
            with open(approved_script_path) as stream:
                on_disk_python = stream.read()
        except Exception as exc:
            on_disk_python = ''
            result['error'] = f'approved analysis.py unavailable: {exc}'
        if actual_digest != approved_digest or on_disk_python != python_str:
            result['status'] = 'approval_mismatch'
            result['error'] = (
                'The reviewed YAML/Python changed before execution; '
                'nothing was run. Review the regenerated files again.')
        else:
            from ...fermipy import script_validator
            l1 = validate_level1(yaml_str)
            l2 = validate_level2(yaml_str) if l1.get('fermipy_load') else {}
            result['level1'] = l1
            result['level2'] = l2
            # Re-check the script in the worker too: the review endpoint is
            # not the only way a script can reach this point.
            static = script_validator.static_review(python_str, yaml_str)
            if not l1.get('fermipy_load') or not l2.get('gta_init'):
                result['status'] = 'validation_failed'
                result['error'] = (
                    l1.get('error') or l2.get('error')
                    or 'approved configuration failed pre-execution validation')
            elif static['blocking']:
                result['status'] = 'validation_failed'
                result['error'] = 'analysis.py failed the safety/API checks: ' + '; '.join(
                    f['message'] for f in static['findings']
                    if f['severity'] == 'error')[:1500]
            else:
                from ...fermipy.run_execution_validated import get_test_meta
                meta = get_test_meta(test_idx)
                result['run_mode'] = approved_run_mode
                run_dir = os.path.dirname(approved_script_path)
                write_progress(
                    session_dir, 'level3', 'active',
                    'Executing the reviewed analysis.py exactly as shown, in an '
                    'isolated interpreter...', agent='validation')
                fit_timeout = fit_timeout_for(test_idx)
                # Where this runs is a plugin decision: the active
                # execution_backend receives the job and returns the same
                # Level-4 result dict, whether it ran here or on a cluster.
                job = ExecutionJob(
                    session_dir=session_dir, run_id=run_id or '',
                    yaml_text=yaml_str, python_text=python_str,
                    work_dir=persistent_workdir,
                    config_path=os.path.join(run_dir, 'final_config.yaml'),
                    script_path=approved_script_path,
                    timeout=fit_timeout + 600,
                    options=dict(
                        yaml_str=yaml_str, evfile=meta.get('evfile', ''),
                        scfile=meta.get('scfile', ''), test_idx=test_idx,
                        fit_timeout=fit_timeout, artifact_dir=artifact_dir,
                        work_dir=persistent_workdir,
                        incremental=(approved_run_mode == 'incremental'),
                        analysis_script=approved_script_path,
                        approved_digest=approved_digest,
                        config_aliases=list(approved_config_aliases or ()),
                        stdout_path=os.path.join(run_dir, 'script_output.txt'),
                    ))
                l4_result = RUNTIME.ctx.executor.execute(job).data
                result['level4'] = l4_result
                expected_bins = expected_sed_bin_count(yaml_str)
                sed_data = l4_result.get('sed_data') or {}
                actual_bins = len(sed_data.get('e_ctr') or [])
                expected_bins = _sed_bins_from_script_calls(
                    expected_bins, l4_result.get('script_calls'))
                result['expected_sed_bins'] = expected_bins
                result['actual_sed_bins'] = (
                    actual_bins if 'sed' in (l4_result.get(
                        'requested_products') or []) else None)
                bin_mismatch = bool(
                    result['expected_sed_bins'] is not None
                    and result['actual_sed_bins'] is not None
                    and result['actual_sed_bins'] != expected_bins)
                l4_result['sed_bin_count_ok'] = not bin_mismatch
                result['level3'] = {
                    'level': 3,
                    'setup_ok': bool(l4_result.get('setup_ok')),
                    'error': l4_result.get('setup_error'),
                    'sources': l4_result.get('setup_sources', []),
                    'n_sources': l4_result.get('setup_n_sources'),
                    'traceback': l4_result.get('traceback', ''),
                }
                script_ok = bool(l4_result.get('setup_ok')) and not (
                    l4_result.get('script_error'))
                result['success'] = script_ok and not bin_mismatch
                if result['level3']['setup_ok']:
                    write_progress(
                        session_dir, 'level3', 'pass',
                        'analysis.py set up the ROI (setup/load_roi).',
                        agent='validation')
                if bin_mismatch:
                    result['status'] = 'intent_mismatch'
                    result['error'] = (
                        f'SED bin-count mismatch: reviewed configuration '
                        f'implies {expected_bins}, execution produced '
                        f'{result["actual_sed_bins"]}.')
                    write_progress(
                        session_dir, 'level4', 'fail', result['error'],
                        agent='validation')
                elif result['success'] and l4_result.get('level4_pass'):
                    result['status'] = 'complete_l4'
                    result['science_results'] = _science_results_from_l4(
                        l4_result, result.get('target_unmatched', False))
                    write_progress(
                        session_dir, 'level4', 'pass',
                        'analysis.py completed and passed the science-quality '
                        'checks.',
                        data=result['science_results'], agent='validation')
                elif result['success']:
                    result['status'] = 'complete'
                    write_progress(
                        session_dir, 'level4', 'pass',
                        'analysis.py completed without errors.',
                        agent='validation')
                else:
                    result['status'] = 'execution_failed'
                    result['error'] = l4_result.get('error')
                    write_progress(
                        session_dir, 'level4', 'fail',
                        f'analysis.py failed: {str(result["error"])[:400]}',
                        agent='validation')

        if result.get('success'):
            try:
                write_run_exports(result, session_dir, run_id)
            except Exception as exc:
                result['export_error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
            result['artifacts_zip'] = (
                f"/api/session/{os.path.basename(session_dir)}/artifacts.zip")
        write_progress(session_dir, 'done', result.get('status', 'error'),
                       'Approved script execution complete', agent='system')
        with open(os.path.join(session_dir, 'pipeline_result.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)
        return result

    # ---- Try an incremental run first (reuse previous fit via load_roi) ----
    did_incremental = False
    if prev_run_state:
        try:
            inc = _attempt_incremental_run(
                yaml_str, test_idx, session_dir, prev_run_state, artifact_dir
            )
        except Exception as e:
            inc = None
            write_progress(session_dir, 'incremental', 'fail',
                           f'Incremental attempt raised {type(e).__name__}: {str(e)[:150]}; '
                           f'falling back to a full run.',
                           agent='validation')
        if inc is not None:
            result.update(inc)
            did_incremental = True

    if not did_incremental:
        # Run iterative repair + validation (Levels 1-3)
        write_progress(session_dir, 'repair', 'active',
                       'Validation Agent: Starting iterative validate-repair loop (up to 5 iterations)...',
                       agent='validation')

        try:
            # We instrument the validation loop with progress updates.
            # Level 3 runs inside the persistent session workdir so the
            # Level-4 stage below reuses its setup products (ltcube,
            # srcmaps) instead of redoing gta.setup() from scratch.
            final_yaml, val_results, repair_hist = _run_validation_with_progress(
                yaml_str, test_idx, session_dir,
                work_dir=persistent_workdir,
            )
            result['final_yaml'] = final_yaml
            result['repair_history'] = repair_hist
            result['validation_results'] = val_results

            # Check Level 3 success
            for vr in val_results:
                l3 = vr.get('level3', {})
                if l3.get('setup_ok'):
                    result['success'] = True
                    result['level3'] = l3
                    result['status'] = 'complete'
                    break

            if not result['success']:
                result['status'] = 'validation_failed'
                if val_results:
                    last = val_results[-1]
                    result['level1'] = last.get('level1')
                    result['level2'] = last.get('level2')
                    result['level3'] = last.get('level3')
                write_progress(session_dir, 'level3', 'fail',
                               'Validation Agent: Level 3 validation failed after all repair iterations',
                               agent='validation')

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            write_progress(session_dir, 'error', 'fail', f'Error: {str(e)[:200]}', agent='system')

        # Level 4: Run optimize + fit if Level 3 passed
        if result['success'] and result.get('final_yaml'):
            write_progress(session_dir, 'level4', 'active',
                           'Validation Agent: Level 3 passed! Starting Level 4: gta.optimize() + gta.fit()...',
                           agent='validation')
            try:
                from ...fermipy.run_level4_validation import validate_level4, CATALOG_REFERENCE
                from ...fermipy.run_execution_validated import get_test_meta

                meta = get_test_meta(test_idx)
                evfile = meta.get('evfile', '')
                scfile = meta.get('scfile', '')

                if evfile and scfile:
                    write_progress(session_dir, 'level4_optimize', 'active',
                                   'Validation Agent: Running likelihood optimization (gta.optimize + gta.fit)...',
                                   agent='validation')

                    # The persistent session workdir already holds the
                    # products of the successful Level-3 setup for exactly
                    # this config (validate_level3 wiped it at the start of
                    # each repair iteration), so Level 4 reuses the cached
                    # ltcube/srcmaps instead of redoing gta.setup(). Do NOT
                    # wipe it here. It also serves a later incremental run.
                    os.makedirs(persistent_workdir, exist_ok=True)

                    l4_result = validate_level4(
                        result['final_yaml'], evfile, scfile, test_idx,
                        fit_timeout=fit_timeout_for(test_idx),
                        artifact_dir=artifact_dir,
                        work_dir=persistent_workdir,
                    )
                    result['level4'] = l4_result

                    if l4_result.get('level4_pass'):
                        result['status'] = 'complete_l4'
                        result['science_results'] = _science_results_from_l4(
                            l4_result, result.get('target_unmatched', False)
                        )
                        write_progress(session_dir, 'level4', 'pass',
                                       'Validation Agent: Level 4 passed! Science-quality results ready.',
                                       data=result['science_results'], agent='validation')
                    else:
                        result['status'] = 'complete'
                        write_progress(session_dir, 'level4', 'partial',
                                       f"Validation Agent: Level 4 did not fully pass: {(l4_result.get('error') or 'science checks did not pass (fit itself converged)')[:200]}",
                                       agent='validation')
            except Exception as e:
                # Keep any L4 results already collected; a failure in the
                # reporting path must not erase a completed fit.
                if not result.get('level4'):
                    result['level4'] = {
                        'level4_pass': False,
                        'error': f'{type(e).__name__}: {str(e)[:300]}',
                    }
                write_progress(session_dir, 'level4', 'fail',
                               f'Validation Agent: Level 4 error: {str(e)[:200]}',
                               agent='validation')
    else:
        # Incremental path already ran Level 4 (load_roi + requested
        # products) inside _attempt_incremental_run; build status/
        # science_results the same way a full run would.
        l4_result = result.get('level4') or {}
        if l4_result.get('level4_pass'):
            result['status'] = 'complete_l4'
            result['science_results'] = _science_results_from_l4(
                l4_result, result.get('target_unmatched', False)
            )
            write_progress(session_dir, 'level4', 'pass',
                           'Validation Agent: Incremental run passed! Science-quality results ready.',
                           data=result['science_results'], agent='validation')
        else:
            result['status'] = 'complete'
            write_progress(session_dir, 'level4', 'partial',
                           f"Validation Agent: Incremental run did not fully pass: "
                           f"{(l4_result.get('error') or 'science checks did not confirm')[:200]}",
                           agent='validation')

    if result.get('success'):
        try:
            write_run_exports(result, session_dir, run_id)
        except Exception as e:
            # Export failure must never erase a completed science result.
            result['export_error'] = f'{type(e).__name__}: {str(e)[:300]}'

    if result.get('success'):
        result['artifacts_zip'] = f"/api/session/{os.path.basename(session_dir)}/artifacts.zip"

    # Final status
    write_progress(session_dir, 'done', result.get('status', 'error'),
                   'Pipeline execution complete', agent='system')

    # Save result
    with open(os.path.join(session_dir, 'pipeline_result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result
