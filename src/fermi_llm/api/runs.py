"""Starting, watching and aborting a run.

The two-click contract lives here: the first POST returns a review, the
second carries the approval token and starts a worker. Each run gets its own
single-worker process pool and its own process group, so aborting one run
cannot disturb another user's.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import hmac
import signal
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState

from ..core.jsonutil import sanitize_for_json
from ..core.progress import write_progress
from ..core.runtime import RUNTIME
from ..core.session import AnalysisSession
from ..components.exporters.run_bundle import ensure_session_run_exports
from ..components.pipelines.execute import run_pipeline_isolated
from ..components.pipelines.review import (build_analysis_python,
                                           prepare_run_review as _prepare_run_review,
                                           run_content_digest)
from ..fermipy.estimates import estimate_run_duration
from ..fermipy.targets import target_is_bundled, yaml_target
from .deps import get_session_or_404

router = APIRouter()


@router.post("/api/session/{session_id}/script_review")
async def script_review_decision(session_id: str, request: Request):
    """Accept or decline the Validation Agent's proposed analysis.py change."""
    session = get_session_or_404(session_id)
    RUNTIME.sessions[session_id] = session
    body = await request.json()
    decision = str(body.get('decision') or '')
    proposal = session.script_proposal
    if decision not in ('accept', 'decline'):
        raise HTTPException(400, "decision must be 'accept' or 'decline'")
    if not proposal or proposal.get('id') != body.get('proposal_id'):
        raise HTTPException(409, "No matching script proposal is pending; "
                                 "click Run Pipeline to review again")

    if decision == 'accept':
        session.current_python = proposal['script']
        # The accepted text was produced by the validator itself: record it
        # so the next review does not send it back to the LLM.
        session.script_reviews[run_content_digest(
            session.current_yaml, proposal['script'])] = {
                'decision': 'accepted', 'kind': proposal['kind'],
                'llm_findings': [], 'summary': proposal.get('reason', ''),
        }
    else:
        entry = session.script_reviews.setdefault(
            proposal['base_digest'], {'llm_findings': [], 'summary': ''})
        entry['decision'] = 'declined'
        entry.setdefault('declined_kinds', [])
        if proposal['kind'] not in entry['declined_kinds']:
            entry['declined_kinds'].append(proposal['kind'])
    session.script_proposal = None
    session.pending_run = None
    session.pipeline_status = 'idle'
    session.save()
    return {"status": "ok", "decision": decision,
            "yaml": session.current_yaml, "python": session.current_python}


@router.post("/api/session/{session_id}/run_pipeline")
async def run_pipeline(session_id: str, request: Request):
    session = get_session_or_404(session_id)
    RUNTIME.sessions[session_id] = session

    active = RUNTIME.pipeline_runs.get(session_id)
    if active and (active.get('aborted') or not active['future'].done()):
        raise HTTPException(409, "A pipeline is already running for this session")
    if active:
        RUNTIME.pipeline_runs.pop(session_id, None)
    active_count = sum(
        1 for item in RUNTIME.pipeline_runs.values()
        if item.get('aborted') or not item['future'].done())
    if active_count >= RUNTIME.max_active_pipelines:
        raise HTTPException(
            429, "The pipeline worker limit is currently full; try again shortly")

    if not session.current_yaml.strip():
        raise HTTPException(400, "No YAML configuration to run")

    try:
        body = await request.json()
    except Exception:
        body = {}
    confirmed = bool(body.get('confirm'))
    approval_token = str(body.get('approval_token') or '')

    pending = session.pending_run or {}
    current_digest = run_content_digest(
        session.current_yaml, session.current_python)
    approved = bool(
        approval_token
        and hmac.compare_digest(
            approval_token, str(pending.get('approval_token') or ''))
        and hmac.compare_digest(
            current_digest, str(pending.get('digest') or '')))
    if not approved:
        # The review may call the LLM; keep the event loop responsive.
        preview = await asyncio.to_thread(_prepare_run_review, session)
        RUNTIME.hooks.emit('run.review', session=session, preview=preview)
        return sanitize_for_json(preview)

    # Everything below operates on the immutable files the user reviewed.
    final_yaml = pending['final_yaml']
    final_python = pending['final_python']
    test_idx = pending['test_idx']
    run_id = pending['run_id']

    # Long runs (e.g. a 10-year full-dataset analysis) need an explicit
    # go-ahead: estimate the wall-clock time and, above the threshold,
    # return a confirmation request instead of silently starting hours of
    # computation. The frontend re-POSTs with {"confirm": true}.
    target = yaml_target(final_yaml)
    full_data = bool(target) and not target_is_bundled(target)
    duration_est = estimate_run_duration(final_yaml,
                                          full_data=full_data)
    threshold_min = float(os.environ.get('FERMI_LLM_CONFIRM_THRESHOLD_MIN', 60))
    # Bundled demo runs never need confirmation: whatever time range the
    # config claims, execution is clamped to the one-week bundled files.
    if duration_est and not confirmed and full_data \
            and duration_est['total_hi_min'] > threshold_min:
        hi = duration_est['total_hi_min']
        hi_txt = f"{hi / 60:.1f} hours" if hi >= 90 else f"{hi:.0f} minutes"
        return {
            "status": "confirmation_required",
            "estimate": duration_est,
            "threshold_minutes": threshold_min,
            "message": (
                f"This analysis is estimated to take up to {hi_txt} "
                f"({duration_est['text']}) Do you want to start it? "
                f"Reduce selection.tmin/tmax to analyze a shorter time "
                f"range if this is more than you intended."),
        }

    # Clear previous progress file
    progress_file = os.path.join(session.session_dir, 'pipeline_progress.jsonl')
    if os.path.exists(progress_file):
        os.remove(progress_file)
    # Clear previous result file
    result_file = os.path.join(session.session_dir, 'pipeline_result.json')
    if os.path.exists(result_file):
        os.remove(result_file)

    # Do not expose a stale result while this new run is active. The session
    # endpoint is used to rebuild the UI after task switches/page reloads;
    # keeping the prior result here made the client render "aborted" or
    # "complete" controls on top of a genuinely running pipeline.
    session.pipeline_result = None
    session.pipeline_status = 'running'
    session.pending_run = None
    session.save()

    # First progress line carries the runtime estimate so the frontend can
    # draw a time-based progress bar (and recover it after a page reload,
    # since SSE replays the progress file from the start).
    if duration_est:
        write_progress(
            session.session_dir, 'run_started', 'active',
            f"Pipeline launched -- estimated "
            f"{duration_est['total_lo_min']:.0f}-"
            f"{duration_est['total_hi_min']:.0f} min.",
            data={'estimate': duration_est, 'started_at': time.time()})

    # If the last run succeeded and saved a reloadable ROI model, hand its
    # location to the isolated process so it can attempt an incremental run
    # (gta.load_roi() instead of gta.setup()+optimize()+fit()) when this
    # request's core YAML sections are unchanged. See run_pipeline_isolated /
    # _attempt_incremental_run in this file for the eligibility check.
    prev_run_state = None
    if session.last_run_roi_ready and session.last_executed_yaml.strip():
        prev_run_state = {
            'yaml': session.last_executed_yaml,
            'test_idx': session.last_run_test_idx,
            'workdir': os.path.join(session.session_dir, 'fermipy_workdir'),
            'roi_prefix': 'fit_model',
        }

    # Give every run a dedicated executor/worker.  Besides isolating FermiPy,
    # this lets /abort_pipeline terminate exactly one run without breaking a
    # shared process pool or affecting another user's analysis.
    loop = asyncio.get_event_loop()
    run_executor = ProcessPoolExecutor(max_workers=1)
    future = loop.run_in_executor(
        run_executor,
        run_pipeline_isolated,
        session.session_dir,
        final_yaml,
        final_python,
        test_idx,
        prev_run_state,
        run_id,
        pending['script_path'],
        pending['digest'],
        pending.get('run_mode', 'full'),
        pending.get('config_aliases') or [],
    )
    RUNTIME.pipeline_runs[session_id] = {
        'run_id': run_id,
        'future': future,
        'executor': run_executor,
        'started_at': time.time(),
        'aborted': False,
    }

    asyncio.ensure_future(_handle_pipeline_result(
        session_id, future, run_id, run_executor))
    RUNTIME.hooks.emit('run.started', session=session, run_id=run_id)

    return {"status": "started",
            "run_id": run_id,
            "estimate": duration_est,
            "approved_digest": pending['digest'],
            "message": (
                "Approved analysis.py started. The worker verified and is "
                "executing the exact YAML/Python shown during review.")}


def _pipeline_worker_info(session_dir, run_id):
    """Return trusted PID metadata written by this run's worker."""
    path = os.path.join(session_dir, 'pipeline_worker.json')
    try:
        with open(path) as f:
            info = json.load(f)
        pid = int(info.get('pid', 0))
        if info.get('run_id') == run_id and pid > 1:
            return info
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return None


def _signal_pipeline_worker(pid, isolated_group, sig):
    """Signal a run only; never signal the web server's process group."""
    if isolated_group:
        os.killpg(pid, sig)
    else:
        os.kill(pid, sig)


async def _terminate_pipeline_worker(session_dir, entry):
    """Stop one run's worker and its children, escalating after two seconds."""
    future = entry['future']
    run_executor = entry['executor']
    run_id = entry['run_id']

    # The worker writes its PID immediately on startup.  Allow a short race
    # window for an abort pressed just after the HTTP start response.
    info = None
    for _ in range(20):
        info = _pipeline_worker_info(session_dir, run_id)
        if info or future.done():
            break
        await asyncio.sleep(0.05)

    # If startup is still between pool creation and metadata creation, the
    # dedicated executor itself provides a safe one-run-only PID fallback.
    if not info:
        processes = getattr(run_executor, '_processes', {}) or {}
        live_pids = [pid for pid, proc in processes.items()
                     if pid > 1 and proc.is_alive()]
        if len(live_pids) == 1:
            info = {
                'pid': live_pids[0],
                # os.setsid() may not have run yet, so signal only the worker.
                'process_group_isolated': False,
            }

    terminated = False
    if info:
        pid = int(info['pid'])
        isolated = bool(info.get('process_group_isolated'))
        try:
            _signal_pipeline_worker(pid, isolated, signal.SIGTERM)
            terminated = True
        except ProcessLookupError:
            pass

        # Let FermiPy/ScienceTools unwind first, then force-stop a stubborn
        # process group so the UI's aborted state is truthful.
        for _ in range(20):
            await asyncio.sleep(0.1)
            try:
                _signal_pipeline_worker(pid, isolated, 0)
            except ProcessLookupError:
                break
        else:
            try:
                _signal_pipeline_worker(pid, isolated, signal.SIGKILL)
                terminated = True
            except ProcessLookupError:
                pass

    future.cancel()
    run_executor.shutdown(wait=False, cancel_futures=True)
    return terminated


@router.post("/api/session/{session_id}/abort_pipeline")
async def abort_pipeline(session_id: str):
    """Abort the currently running pipeline for one analysis session."""
    session = get_session_or_404(session_id)
    RUNTIME.sessions[session_id] = session

    entry = RUNTIME.pipeline_runs.get(session_id)
    if not entry or entry['future'].done():
        raise HTTPException(409, "No active pipeline run to abort")

    # Mark first so the completion task cannot race in and overwrite the
    # aborted result with an exception from the terminated worker.
    entry['aborted'] = True
    terminated = await _terminate_pipeline_worker(session.session_dir, entry)
    RUNTIME.hooks.emit('run.finished', session=session,
                       result={'status': 'aborted', 'success': False})
    result = {
        'status': 'aborted',
        'success': False,
        'aborted': True,
        'error': 'Run aborted by user.',
        'final_yaml': session.current_yaml,
        'run_id': entry['run_id'],
        'worker_terminated': terminated,
    }

    result_file = os.path.join(session.session_dir, 'pipeline_result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    write_progress(session.session_dir, 'done', 'aborted',
                   'Pipeline aborted by user.', agent='system')

    session.pipeline_status = 'aborted'
    session.pipeline_result = result
    session.last_run_roi_ready = False
    msg = "**Pipeline aborted.** The running analysis was stopped at your request."
    session.chat_history.append({
        'role': 'assistant',
        'content': msg,
        'timestamp': datetime.now().isoformat(),
        'pipeline_result': True,
    })
    session.save()
    if RUNTIME.pipeline_runs.get(session_id) is entry:
        RUNTIME.pipeline_runs.pop(session_id, None)

    ws = RUNTIME.websockets.get(session_id)
    if ws and ws.client_state == WebSocketState.CONNECTED:
        await ws.send_json(sanitize_for_json({
            'type': 'pipeline_result', 'result': result, 'message': msg,
        }))
    return sanitize_for_json({'status': 'aborted', 'result': result})


@router.get("/api/session/{session_id}/pipeline_stream")
async def pipeline_stream(session_id: str):
    """Server-Sent Events endpoint for real-time pipeline progress.

    Reads pipeline_progress.jsonl and streams new entries as SSE events.
    Falls back gracefully if the pipeline finishes before client connects.
    """
    session = get_session_or_404(session_id)

    async def event_generator():
        progress_file = os.path.join(session.session_dir, 'pipeline_progress.jsonl')
        result_file = os.path.join(session.session_dir, 'pipeline_result.json')
        lines_read = 0
        idle_count = 0
        max_idle = 600  # 10 minutes max (600 * 1s)

        while idle_count < max_idle:
            # Read new progress lines
            new_lines = []
            if os.path.exists(progress_file):
                try:
                    with open(progress_file) as f:
                        all_lines = f.readlines()
                    new_lines = all_lines[lines_read:]
                    lines_read = len(all_lines)
                except Exception:
                    pass

            for line in new_lines:
                line = line.strip()
                if line:
                    # Check if this is a 'done' event - if so, wait for result file
                    try:
                        parsed = json.loads(line)
                        if parsed.get('step') == 'done':
                            # Wait briefly for result file to be written
                            await asyncio.sleep(1)
                            if os.path.exists(result_file):
                                try:
                                    with open(result_file) as f:
                                        result = json.load(f)
                                    yield f"data: {json.dumps(sanitize_for_json({'step': 'result', 'status': result.get('status', 'unknown'), 'result': result}), default=str)}\n\n"
                                except Exception:
                                    yield f"data: {line}\n\n"
                                return  # End the generator
                            else:
                                yield f"data: {line}\n\n"
                                continue
                    except (json.JSONDecodeError, KeyError):
                        pass
                    yield f"data: {line}\n\n"
                    idle_count = 0

            # Check if pipeline is done (result file written after progress)
            if os.path.exists(result_file):
                # Small delay to ensure file is fully written
                await asyncio.sleep(0.5)
                try:
                    with open(result_file) as f:
                        result = json.load(f)
                    yield f"data: {json.dumps(sanitize_for_json({'step': 'result', 'status': result.get('status', 'unknown'), 'result': result}), default=str)}\n\n"
                except Exception:
                    pass
                break

            # Send heartbeat every 15 seconds to keep connection alive
            # (ngrok-friendly: minimal traffic)
            if idle_count > 0 and idle_count % 15 == 0:
                yield f": heartbeat\n\n"

            idle_count += 1
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _script_run_message(result):
    """Chat summary of an executed analysis.py (reviewed-script runs)."""
    status = result.get('status')
    l4 = result.get('level4') or {}
    headers = {
        'complete_l4': '**analysis.py completed and passed the science-quality checks.**',
        'complete': '**analysis.py completed.**',
        'intent_mismatch': '**analysis.py ran, but the output does not match the reviewed configuration.**',
        'approval_mismatch': '**Nothing was run: the files changed after review.**',
        'validation_failed': '**Nothing was run: validation failed.**',
    }
    lines = [headers.get(status, '**analysis.py failed.**'), '']
    if status not in ('complete_l4', 'complete') and result.get('error'):
        lines += [f"**Error:** {str(result['error'])[:600]}", '']
        if l4.get('script_traceback'):
            lines += ['```', l4['script_traceback'][-1500:], '```', '']
    sci = result.get('science_results') or {}
    fmt = lambda v, spec: format(v, spec) if isinstance(v, (int, float)) else 'N/A'
    if sci:
        lines.append(f"**Target TS:** {fmt(sci.get('target_ts'), '.1f')} | "
                     f"**Flux:** {fmt(sci.get('target_flux'), '.2e')} ph/cm2/s | "
                     f"**Index:** {fmt(sci.get('spectral_index'), '.2f')}")
    elif l4.get('target_warning'):
        lines.append(l4['target_warning'])
    elif status == 'complete' and l4.get('error'):
        lines.append(f"Science-quality checks did not pass: {str(l4['error'])[:300]}")
    produced = l4.get('requested_products') or []
    failed = l4.get('product_errors') or {}
    if produced or failed:
        items = [f"{p} (failed: {str(failed[p])[:120]})" if p in failed
                 else p for p in produced]
        items += [f"{p} (failed: {str(e)[:120]})" for p, e in failed.items()
                  if p not in produced]
        lines.append('**Products:** ' + ', '.join(items))
    calls = l4.get('script_calls') or []
    if calls:
        lines.append(f"**FermiPy calls executed:** {len(calls)} "
                     f"(see the script trace in Results).")
    if (l4.get('script_stdout') or '').strip():
        lines.append('The script printed output; it is shown in Results.')
    return '\n'.join(lines)


async def _handle_pipeline_result(session_id, future, run_id, run_executor):
    """Handle pipeline completion."""
    try:
        result = await future
        entry = RUNTIME.pipeline_runs.get(session_id)
        if (not entry or entry.get('run_id') != run_id
                or entry.get('aborted')):
            return
        session = RUNTIME.sessions.get(session_id)
        if session:
            session.pipeline_status = result.get('status', 'error')
            session.pipeline_result = result
            RUNTIME.hooks.emit('run.finished', session=session, result=result)
            if result.get('final_yaml'):
                session.current_yaml = result['final_yaml']
            if result.get('final_python'):
                session.current_python = result['final_python']

            # Remember what was actually executed so the *next* run can be
            # considered for incremental reuse (gta.load_roi()). A saved ROI
            # snapshot only exists once a fit has actually completed --
            # either just now (full run) or reused from before (incremental).
            if result.get('success'):
                session.last_executed_yaml = result.get('final_yaml', '') or session.current_yaml
                session.last_run_test_idx = result.get('test_idx')
                session.last_run_roi_ready = bool((result.get('level4') or {}).get('fit_ok'))
            else:
                session.last_run_roi_ready = False

            incremental_note = ''
            if result.get('run_mode') == 'incremental':
                incremental_note = (
                    "**(Incremental run — reused the previous fit via `gta.load_roi()`, "
                    "only recomputed the requested products)**\n\n"
                )

            if result.get('approved_digest'):
                msg = incremental_note + _script_run_message(result)
            elif result.get('status') == 'complete_l4':
                sci = result.get('science_results', {})
                ts_str = f"{sci['target_ts']:.1f}" if sci.get('target_ts') is not None else 'N/A'
                flux_str = f"{sci['target_flux']:.2e}" if sci.get('target_flux') is not None else 'N/A'
                flux_err_str = f"{sci['target_flux_error']:.2e}" if sci.get('target_flux_error') is not None else ''
                idx_str = f"{sci['spectral_index']:.2f}" if sci.get('spectral_index') is not None else 'N/A'
                idx_err_str = f"{sci['spectral_index_error']:.2f}" if sci.get('spectral_index_error') is not None else ''
                conv_str = 'Yes' if sci.get('convergence_ok') else 'No'

                msg = (incremental_note +
                       "**Pipeline Complete -- Level 4 (Science Quality) Passed!**\n\n"
                       f"**Fit Convergence:** {conv_str}\n"
                       f"**Target TS:** {ts_str}\n"
                       f"**Target Flux:** {flux_str}")
                if flux_err_str:
                    msg += f" +/- {flux_err_str}"
                msg += f" ph/cm2/s\n**Spectral Index:** {idx_str}"
                if idx_err_str:
                    msg += f" +/- {idx_err_str}"
                msg += "\n\n"

                top_src = sci.get('sources_summary', [])[:5]
                if top_src:
                    msg += "**Top Sources by TS:**\n"
                    for s in top_src:
                        msg += f"- {s['name']}: TS={s['ts']:.1f}, Flux={s['flux']:.2e}\n"
                    msg += "\n"

                msg += ("The Validator Agent repaired, validated, and ran the full "
                        "likelihood fit (optimize + fit) successfully.")
            elif result.get('success'):
                msg = (incremental_note +
                       "**Pipeline Complete!** Level 3 validation passed (gta.setup() successful).\n\n"
                       f"**Sources found:** {result.get('level3', {}).get('sources', [])}\n\n")
                l4 = result.get('level4')
                if l4 and not l4.get('level4_pass'):
                    l4_err = l4.get('error', 'unknown')
                    msg += (f"**Level 4 (fit):** Did not fully pass -- {l4_err}\n\n"
                            "The configuration is valid and can be executed. "
                            "The fit may need manual tuning for science-quality results.")
                else:
                    msg += "The Validator Agent iteratively repaired and validated the configuration."
            else:
                error = result.get('level3', {}).get('error', result.get('error', 'Unknown error'))
                msg = (f"**Pipeline Failed.** The validation agent could not achieve a successful run.\n\n"
                       f"**Last error:** {error}\n\n"
                       "You can modify the YAML/Python and try again, or ask me for help fixing the issue.")

            session.chat_history.append({
                "role": "assistant",
                "content": msg,
                "timestamp": datetime.now().isoformat(),
                "pipeline_result": True,
            })
            session.save()

            # Notify via websocket (keep as fallback)
            ws = RUNTIME.websockets.get(session_id)
            if ws and ws.client_state == WebSocketState.CONNECTED:
                await ws.send_json(sanitize_for_json({
                    "type": "pipeline_result",
                    "result": result,
                    "message": msg,
                }))
    except asyncio.CancelledError:
        # Expected when /abort_pipeline cancels this run's future.
        pass
    except Exception as e:
        entry = RUNTIME.pipeline_runs.get(session_id)
        if entry and entry.get('run_id') == run_id and entry.get('aborted'):
            return
        session = RUNTIME.sessions.get(session_id)
        if session:
            session.pipeline_status = 'error'
            session.pipeline_result = {'error': str(e)}
            session.save()
    finally:
        entry = RUNTIME.pipeline_runs.get(session_id)
        if (entry and entry.get('run_id') == run_id
                and not entry.get('aborted')):
            RUNTIME.pipeline_runs.pop(session_id, None)
        worker_file = os.path.join(
            (RUNTIME.sessions.get(session_id).session_dir if RUNTIME.sessions.get(session_id)
             else os.path.join(RUNTIME.sessions_dir, session_id)),
            'pipeline_worker.json')
        try:
            info = _pipeline_worker_info(os.path.dirname(worker_file), run_id)
            if info and os.path.exists(worker_file):
                os.remove(worker_file)
        except OSError:
            pass
        try:
            run_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


@router.get("/api/session/{session_id}/pipeline_status")
async def pipeline_status(session_id: str):
    session = get_session_or_404(session_id)

    result_file = os.path.join(session.session_dir, 'pipeline_result.json')
    if os.path.exists(result_file):
        result = ensure_session_run_exports(session)
        if result is None:
            with open(result_file) as f:
                result = json.load(f)
        return sanitize_for_json({"status": result.get('status', 'unknown'), "result": result})

    return {"status": session.pipeline_status}