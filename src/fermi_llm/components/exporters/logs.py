"""Plain-text logs: what the model did, and what FermiPy did.

Two exporters rather than one because reviewers ask two different questions.
The LLM log is the provenance of the configuration (every message, the model
behind each, the prompt analysis). The FermiPy log is the provenance of the
result (validation steps, repairs, the executed YAML and FermiPy's own log).
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from ...core import kinds
from ...core.contracts import ExportArtifact
from ...core.registry import REGISTRY


class LLMLogExporter:
    name = 'llm_log'
    label = 'LLM interaction log'
    media_type = 'text/plain'

    def __init__(self, ctx):
        self.ctx = ctx

    def available(self, session) -> bool:
        return True

    def build(self, session) -> ExportArtifact:
        session_id = session.session_id

        bar = "=" * 70
        lines = [
            "FermiLLM — LLM interaction log",
            f"Session:   {session_id}",
            f"Exported:  {datetime.now().isoformat()}",
            "",
            bar,
            "CHAT HISTORY",
            bar,
        ]
        for m in session.chat_history:
            header = f"[{m.get('timestamp', '')}] {m.get('role', '?').upper()}"
            if m.get('model'):
                header += f"  (model: {m['model']})"
            lines += ["", header, "-" * len(header), m.get('content', '')]
            if m.get('analysis'):
                lines += ["", "(prompt analysis: "
                          + json.dumps(m['analysis'], default=str) + ")"]
        pipeline_result = session.pipeline_result or {}
        submitted_python = (pipeline_result.get('submitted_python')
                            or session.current_python or '')
        lines += ["", bar, "CURRENT YAML CONFIGURATION", bar,
                  session.current_yaml or "(empty)",
                  "", bar, "SUBMITTED PYTHON SCRIPT", bar,
                  submitted_python or "(empty)", ""]
        return ExportArtifact(
            filename=f'fermillm_{session_id}_llm.log',
            media_type=self.media_type,
            content="\n".join(lines).encode())


class FermiPyLogExporter:
    name = 'fermipy_log'
    label = 'FermiPy execution log'
    media_type = 'text/plain'

    def __init__(self, ctx):
        self.ctx = ctx

    def available(self, session) -> bool:
        return True

    def build(self, session) -> ExportArtifact:
        session_id = session.session_id

        bar = "=" * 70
        lines = [
            "FermiLLM — FermiPy execution log",
            f"Session:   {session_id}",
            f"Exported:  {datetime.now().isoformat()}",
        ]

        progress_file = os.path.join(session.session_dir, 'pipeline_progress.jsonl')
        if os.path.exists(progress_file):
            lines += ["", bar, "PIPELINE PROGRESS (validation & repair agent)", bar]
            with open(progress_file) as f:
                for raw in f:
                    try:
                        rec = json.loads(raw)
                    except Exception:
                        continue
                    try:
                        ts = datetime.fromtimestamp(rec.get('timestamp', 0)).isoformat()
                    except Exception:
                        ts = str(rec.get('timestamp', ''))
                    lines.append(f"[{ts}] [{rec.get('agent', '-')}] "
                                 f"{rec.get('step', '')}/{rec.get('status', '')}: "
                                 f"{rec.get('detail', '')}")

        result_file = os.path.join(session.session_dir, 'pipeline_result.json')
        if os.path.exists(result_file):
            try:
                with open(result_file) as f:
                    res = json.load(f)
            except Exception:
                res = {}
            repairs = [r for it in (res.get('repair_history') or [])
                       for r in (it.get('repairs') or [])]
            if repairs:
                lines += ["", bar, "REPAIRS APPLIED BY THE VALIDATION AGENT", bar]
                lines += [f"- {r}" for r in repairs]
            if (res.get('final_yaml') or '').strip():
                lines += ["", bar, "EXECUTED (REPAIRED) YAML", bar, res['final_yaml']]

        outdir = os.path.join(session.session_dir, 'fermipy_workdir', 'output')
        if os.path.isdir(outdir):
            for fn in sorted(os.listdir(outdir)):
                if not fn.endswith('.log'):
                    continue
                lines += ["", bar, f"FERMIPY LOG: {fn}", bar]
                try:
                    with open(os.path.join(outdir, fn), errors='replace') as f:
                        lines.append(f.read())
                except Exception as e:
                    lines.append(f"(could not read: {e})")

        if len(lines) <= 3:
            lines += ["", "(no pipeline run in this session yet)"]

        return ExportArtifact(
            filename=f'fermillm_{session_id}_fermipy.log',
            media_type=self.media_type,
            content="\n".join(lines).encode())


REGISTRY.register(kinds.EXPORTER, 'llm_log', LLMLogExporter,
                  priority=300, source='core',
                  metadata={'label': LLMLogExporter.label})
REGISTRY.register(kinds.EXPORTER, 'fermipy_log', FermiPyLogExporter,
                  priority=310, source='core',
                  metadata={'label': FermiPyLogExporter.label})
