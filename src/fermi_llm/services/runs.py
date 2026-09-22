"""Run bookkeeping that belongs to the web process.

Only one job: stopping cleanly. Workers run in their own process groups, so
a graceful shutdown has to terminate them explicitly — otherwise an analysis
outlives the server that owns it and nobody ever collects the result.
"""

from __future__ import annotations

import json
import os

from ..core.runtime import RUNTIME
from ..core.session import AnalysisSession


async def shutdown_active_runs():
    """Terminate every in-flight run and record why it stopped."""
    from ..api.runs import _terminate_pipeline_worker

    # Dedicated workers call setsid(), so explicitly terminate them during a
    # graceful web-server shutdown rather than leaving analyses orphaned.
    for session_id, entry in list(RUNTIME.pipeline_runs.items()):
        if entry['future'].done():
            continue
        entry['aborted'] = True
        session = RUNTIME.sessions.get(session_id) or AnalysisSession.load(session_id)
        try:
            await _terminate_pipeline_worker(
                session.session_dir if session else
                os.path.join(RUNTIME.sessions_dir, session_id), entry)
        except Exception:
            pass
        if session:
            result = {
                'status': 'aborted', 'success': False, 'aborted': True,
                'error': 'Run stopped because the server shut down.',
                'final_yaml': session.current_yaml,
                'run_id': entry['run_id'],
            }
            session.pipeline_status = 'aborted'
            session.pipeline_result = result
            session.last_run_roi_ready = False
            session.save()
            try:
                with open(os.path.join(session.session_dir,
                                       'pipeline_result.json'), 'w') as f:
                    json.dump(result, f, indent=2, default=str)
            except OSError:
                pass
        RUNTIME.pipeline_runs.pop(session_id, None)
