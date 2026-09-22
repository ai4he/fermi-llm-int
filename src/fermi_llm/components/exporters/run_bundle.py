"""The reproducibility bundle: the exact YAML and Python a run executed.

Written at review time and again when a run succeeds, so the files a user
downloads are byte-identical to what ran — including the manifest with both
SHA-256 digests. This is what makes a result reproducible off-platform:
`python analysis.py` with the shipped config reproduces the analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from datetime import datetime

from ...core import kinds
from ...core.contracts import ExportArtifact
from ...core.registry import REGISTRY
from ...core.runtime import RUNTIME


def _build_analysis_python(*args, **kwargs):
    """Late import: the review stage owns script generation."""
    from ..pipelines.review import build_analysis_python
    return build_analysis_python(*args, **kwargs)


def _write_run_exports(result, session_dir, run_id):
    """Persist the final YAML and matching runnable Python for one run."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id or ''):
        raise ValueError('invalid run id for export')
    yaml_text = result.get('final_yaml') or ''
    if not yaml_text.strip():
        return

    python_text = result.get('final_python') or _build_analysis_python(
        yaml_text, run_mode=result.get('run_mode', 'full'))
    run_dir = os.path.join(session_dir, 'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)

    yaml_path = os.path.join(run_dir, 'final_config.yaml')
    python_path = os.path.join(run_dir, 'analysis.py')
    manifest_path = os.path.join(run_dir, 'manifest.json')
    bundle_path = os.path.join(run_dir, 'analysis_files.zip')

    with open(yaml_path, 'w') as f:
        f.write(yaml_text.rstrip() + '\n')
    with open(python_path, 'w') as f:
        f.write(python_text)

    manifest = {
        'session_id': os.path.basename(session_dir),
        'run_id': run_id,
        'generated_at': datetime.now().isoformat(),
        'status': result.get('status'),
        'run_mode': result.get('run_mode', 'full'),
        'files': ['final_config.yaml', 'analysis.py'],
        'yaml_sha256': hashlib.sha256(yaml_text.encode()).hexdigest(),
        'python_sha256': hashlib.sha256(python_text.encode()).hexdigest(),
        'note': ('analysis.py is the reviewed script executed by the isolated '
                 'worker; server-side code only validates and serializes it.'),
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(yaml_path, 'final_config.yaml')
        zf.write(python_path, 'analysis.py')
        zf.write(manifest_path, 'manifest.json')

    session_id = os.path.basename(session_dir)
    base = f'/api/session/{session_id}/runs/{run_id}'
    result['final_python'] = python_text
    result['downloads'] = {
        'yaml': f'{base}/final_config.yaml',
        'python': f'{base}/analysis.py',
        'bundle': f'{base}/analysis_files.zip',
    }


def _ensure_session_run_exports(session):
    """Backfill exports for a completed session created before this feature."""
    result = session.pipeline_result
    result_path = os.path.join(session.session_dir, 'pipeline_result.json')
    if not result and os.path.isfile(result_path):
        try:
            with open(result_path) as f:
                result = json.load(f)
        except Exception:
            return None
    if not isinstance(result, dict):
        return None

    run_id = result.get('run_id')
    if (result.get('success') and result.get('final_yaml') and run_id
            and (not result.get('final_python') or not result.get('downloads'))):
        try:
            _write_run_exports(result, session.session_dir, run_id)
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            result['export_error'] = f'{type(e).__name__}: {str(e)[:300]}'

    session.pipeline_result = result
    if result.get('final_yaml'):
        session.current_yaml = result['final_yaml']
    if result.get('final_python'):
        session.current_python = result['final_python']
    session.save()
    return result


write_run_exports = _write_run_exports
ensure_session_run_exports = _ensure_session_run_exports


class RunBundleExporter:
    """Serves the zip of ``final_config.yaml`` + ``analysis.py`` + manifest."""

    name = 'run_bundle'
    label = 'Analysis files (YAML + Python)'
    media_type = 'application/zip'

    def __init__(self, ctx):
        self.ctx = ctx

    def _run_dir(self, session):
        result = session.pipeline_result or {}
        run_id = result.get('run_id')
        if not run_id or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
            return None
        path = os.path.join(session.session_dir, 'runs', run_id)
        return path if os.path.isdir(path) else None

    def available(self, session) -> bool:
        return self._run_dir(session) is not None

    def build(self, session) -> ExportArtifact:
        ensure_session_run_exports(session)
        run_dir = self._run_dir(session)
        if run_dir is None:
            raise FileNotFoundError('this task has no completed run to export')
        return ExportArtifact(
            filename=f'fermillm_{session.session_id}_analysis_files.zip',
            media_type=self.media_type,
            path=os.path.join(run_dir, 'analysis_files.zip'))


REGISTRY.register(kinds.EXPORTER, 'run_bundle', RunBundleExporter,
                  priority=100, source='core',
                  metadata={'label': RunBundleExporter.label})
