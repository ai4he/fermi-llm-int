"""Example: observing the platform without changing it.

Subscribes to lifecycle events and appends one line per run to a log file.
The same shape forwards runs to a lab dashboard, a metrics backend or a
Slack channel.

What it demonstrates: the hook bus. Hooks never block a request — an
exception here is captured and reported, and the run continues.
"""

from __future__ import annotations

import json
import os
from datetime import datetime


def register(registry, ctx=None):
    if ctx is None:
        return
    path = os.path.join(ctx.settings.project_dir, 'run-events.log')

    def append(kind, payload):
        with open(path, 'a') as handle:
            handle.write(json.dumps({
                'at': datetime.now().isoformat(), 'event': kind,
                **payload}, default=str) + '\n')

    @ctx.hooks.on('run.started', name='run_logger.started')
    def _started(session=None, run_id=None, **_):
        append('run.started', {'session': getattr(session, 'session_id', ''),
                               'run_id': run_id})

    @ctx.hooks.on('run.finished', name='run_logger.finished')
    def _finished(session=None, result=None, **_):
        append('run.finished', {
            'session': getattr(session, 'session_id', ''),
            'status': (result or {}).get('status'),
            'success': (result or {}).get('success')})
