"""Run the reviewed script in a fresh interpreter on this machine.

The default backend. It executes the approved ``analysis.py`` through
``fermi_llm.fermipy.script_runner``: a new process with a scrubbed
environment, resource limits, and a recorder that captures the script's own
GTAnalysis calls so results can be read back without the script having to
write a result file.

A cluster/HPC/cloud backend implements the same ``execute`` and is selected
with ``execution_backend = "..."`` in ``plugins.toml``.
"""

from __future__ import annotations

import os

from ...core import kinds
from ...core.contracts import ExecutionJob, ExecutionResult
from ...core.registry import REGISTRY


class LocalSubprocessBackend:
    """Level-4 execution in a subprocess on the host running the web app."""

    name = 'local_subprocess'
    label = 'Local subprocess (this machine)'

    def __init__(self, ctx):
        self.ctx = ctx

    def supports(self, job: ExecutionJob) -> bool:
        return True

    def execute(self, job: ExecutionJob) -> ExecutionResult:
        """Run Level 4 for one approved script.

        ``job.options`` is the keyword payload for ``validate_level4``; the
        runner serialises it, starts the child and reads the result back.
        """
        from ...fermipy.script_runner import run_level4_subprocess
        data = run_level4_subprocess(
            dict(job.options), job_dir=os.path.dirname(job.script_path),
            timeout=job.timeout)
        ok = bool(data.get('fit_ok') or data.get('level4_pass'))
        return ExecutionResult(ok=ok,
                               status='complete' if ok else 'execution_failed',
                               error=data.get('error'), data=data)


REGISTRY.register(kinds.EXECUTION_BACKEND, 'local_subprocess',
                  LocalSubprocessBackend, priority=100, source='core',
                  metadata={'label': LocalSubprocessBackend.label})
