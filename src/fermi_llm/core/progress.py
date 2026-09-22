"""Progress reporting.

Every stage of a run appends one JSON line to the session's
``pipeline_progress.jsonl``; the SSE endpoint tails that file and the
WebSocket channel mirrors it. Writing to a file (rather than an in-memory
queue) is what lets an isolated worker process report progress, and what
lets a reconnecting browser catch up.

The ``run.progress`` hook fires for every event, which is the supported way
to forward progress somewhere else (a lab's dashboard, Slack, a log system)
without touching this module.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

from .runtime import RUNTIME


def _monitor_log_path():
    return os.path.join(RUNTIME.settings.project_dir, 'webapp', 'monitor.log')


def write_monitor(agent, detail, session_id=''):
    """Append a line to the unified monitor log (tail -f friendly)."""
    ts = datetime.now().strftime('%H:%M:%S')
    tag = f'[{agent.upper():10s}]'
    sid = f' ({session_id[:8]})' if session_id else ''
    line = f'{ts} {tag}{sid}  {detail}\n'
    try:
        with open(_monitor_log_path(), 'a') as f:
            f.write(line)
    except Exception:
        pass


def write_progress(session_dir, step, status, detail='', data=None, agent='system'):
    """Write a progress update to a file that SSE can read.

    agent: 'master' for Master Agent, 'validation' for Validation Agent, 'system' for system messages
    """
    progress_file = os.path.join(session_dir, 'pipeline_progress.jsonl')
    entry = {
        'step': step,
        'status': status,
        'detail': detail,
        'agent': agent,
        'timestamp': time.time(),
    }
    if data:
        entry['data'] = data
    with open(progress_file, 'a') as f:
        f.write(json.dumps(entry, default=str) + '\n')
    # Also write to unified monitor log
    session_id = os.path.basename(session_dir)
    write_monitor(agent, detail, session_id)
    if RUNTIME.ctx is not None:
        RUNTIME.hooks.emit('run.progress', session_dir=session_dir,
                           record=entry)


def write_chat_progress(session_dir, stage, detail, agent='master'):
    """Write a progress update for the chat/generation phase (Master Agent)."""
    progress_file = os.path.join(session_dir, 'chat_progress.jsonl')
    entry = {
        'stage': stage,
        'detail': detail,
        'agent': agent,
        'timestamp': time.time(),
    }
    with open(progress_file, 'a') as f:
        f.write(json.dumps(entry, default=str) + '\n')
    # Also write to unified monitor log
    session_id = os.path.basename(session_dir)
    write_monitor(agent, detail, session_id)
