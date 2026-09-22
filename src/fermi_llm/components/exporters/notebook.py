"""Jupyter notebook export.

The notebook is the "take it home" format: prompt, final YAML, the executed
script and the produced figures in one file a collaborator can rerun. Built
from the finished run when there is one, otherwise from the current panels.
"""

from __future__ import annotations

import base64
import json
import os
import re
import yaml
from datetime import datetime

from ...core import kinds
from ...core.contracts import ExportArtifact
from ...core.registry import REGISTRY
from ...fermipy.targets import yaml_target as _yaml_target
from ..guardrails.yamlops import _build_run_outcome_summary


def _nb_cell(cell_type, source):
    """One nbformat v4 cell. `source` is a str; stored as a list of lines."""
    cell = {"cell_type": cell_type, "metadata": {},
            "source": source.splitlines(keepends=True) or [""]}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _notebook_filename(session):
    target = _yaml_target(session.current_yaml or '') or 'analysis'
    slug = re.sub(r'[^A-Za-z0-9.+_-]+', '_', target).strip('_') or 'analysis'
    return f"fermipy_{slug}_{session.session_id}.ipynb"


def _build_notebook(session):
    """Build a self-contained Jupyter notebook (nbformat v4) reproducing this
    task's analysis: the request, a cell that writes config.yaml, the generated
    Python script, the last run's results, and any produced plots embedded
    inline. Returns the notebook as a JSON-serialisable dict.
    """
    import base64

    final_result = session.pipeline_result or {}
    yaml_txt = final_result.get('final_yaml') or session.current_yaml or ''
    py_txt = final_result.get('final_python') or session.current_python or ''
    prompt = (session.current_prompt or '').strip()
    target = _yaml_target(yaml_txt) or ''
    try:
        generated = datetime.now().isoformat(timespec='seconds')
    except Exception:
        generated = datetime.now().isoformat()

    cells = []

    # --- Title / request ---
    title = ["# FermiPy Analysis Notebook", ""]
    if target:
        title.append(f"**Target:** `{target}`  ")
    title.append(f"**Generated:** {generated}  ")
    title.append(f"**Session:** `{session.session_id}`")
    if prompt:
        title += ["", "**Request**", "", "> " + prompt.replace("\n", "\n> ")]
    cells.append(_nb_cell("markdown", "\n".join(title)))

    # --- How to run ---
    cells.append(_nb_cell("markdown",
        "## How to run\n\n"
        "Run the cells top-to-bottom in an environment with **FermiPy** "
        "installed and the Fermi-LAT data files referenced in the config "
        "available. The first code cell writes `config.yaml`; the analysis "
        "cell then drives `GTAnalysis` from it."))

    # --- config.yaml writer (keeps the notebook self-contained) ---
    cells.append(_nb_cell("markdown", "## Configuration (`config.yaml`)"))
    cells.append(_nb_cell("code",
        'config_yaml = r"""\n' + yaml_txt.rstrip('\n') + '\n"""\n'
        'with open("config.yaml", "w") as f:\n'
        '    f.write(config_yaml)\n'
        'print("Wrote config.yaml")'))

    # --- Analysis script ---
    cells.append(_nb_cell("markdown", "## Analysis"))
    cells.append(_nb_cell("code", py_txt.rstrip('\n') or
                          "# No Python script was generated for this analysis."))

    # --- Last run results (reuses the same summary fed back to the models) ---
    summary = _build_run_outcome_summary(session)
    if summary:
        cells.append(_nb_cell("markdown",
            "## Last run results\n\n```\n" + summary + "\n```"))

    # --- Produced plots, embedded inline so the notebook is shareable ---
    art_dir = os.path.join(session.session_dir, 'artifacts')
    embedded = []
    if os.path.isdir(art_dir):
        for fn in sorted(os.listdir(art_dir)):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in ('.png', '.jpg', '.jpeg'):
                continue
            fpath = os.path.join(art_dir, fn)
            try:
                if os.path.getsize(fpath) > 8 * 1024 * 1024:
                    continue  # keep the notebook a sane size
                with open(fpath, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('ascii')
            except Exception:
                continue
            mime = 'image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/png'
            embedded.append(f"### {fn}\n\n"
                            f"![{fn}](data:{mime};base64,{b64})")
    if embedded:
        cells.append(_nb_cell("markdown",
            "## Output plots\n\n" + "\n\n".join(embedded)))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "fermi_llm": {"session_id": session.session_id, "target": target,
                          "generated": generated},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class NotebookExporter:
    name = 'notebook'
    label = 'Jupyter notebook'
    media_type = 'application/x-ipynb+json'

    def __init__(self, ctx):
        self.ctx = ctx

    def available(self, session) -> bool:
        return bool((session.current_yaml or '').strip()
                    or (session.current_python or '').strip())

    def build(self, session) -> ExportArtifact:
        notebook = _build_notebook(session)
        return ExportArtifact(
            filename=_notebook_filename(session),
            media_type=self.media_type,
            content=json.dumps(notebook, indent=1).encode())


REGISTRY.register(kinds.EXPORTER, 'notebook', NotebookExporter,
                  priority=200, source='core',
                  metadata={'label': NotebookExporter.label})
