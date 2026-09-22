"""Artifacts and exports.

Two kinds of download: files a run wrote (figures, FITS) and artifacts an
exporter builds (notebook, bundle, logs, archive). The exporter list is
served so the UI can render whatever is registered — including a plugin's
format — without a frontend change. The historical per-format URLs are kept
so existing links and scripts keep working.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from starlette.background import BackgroundTask

from ..core import kinds
from ..core.runtime import RUNTIME
from ..core.session import AnalysisSession
from ..components.exporters.run_bundle import ensure_session_run_exports
from .deps import get_session_or_404

router = APIRouter()


def _exporters():
    return {e.name: e for e in RUNTIME.ctx.build_stack(kinds.EXPORTER)}


def _respond(artifact):
    """Turn an ExportArtifact into a FastAPI response."""
    cleanup = artifact.headers.get('x-cleanup')
    background = None
    if cleanup:
        background = BackgroundTask(
            lambda: os.path.exists(cleanup) and os.remove(cleanup))
    if artifact.path:
        return FileResponse(artifact.path, media_type=artifact.media_type,
                            filename=artifact.filename, background=background)
    return Response(
        content=artifact.content or b'', media_type=artifact.media_type,
        headers={'Content-Disposition':
                 f'attachment; filename="{artifact.filename}"'})


@router.get('/api/session/{session_id}/exports')
async def list_exports(session_id: str):
    """Every export this task can produce right now."""
    session = get_session_or_404(session_id)
    out = []
    for name, exporter in _exporters().items():
        try:
            available = exporter.available(session)
        except Exception:                                 # noqa: BLE001
            available = False
        out.append({'name': name,
                    'label': getattr(exporter, 'label', name),
                    'media_type': exporter.media_type,
                    'available': available,
                    'url': f'/api/session/{session_id}/export/{name}'})
    return {'exports': out}


@router.get('/api/session/{session_id}/export/{name}')
async def build_export(session_id: str, name: str):
    session = get_session_or_404(session_id)
    exporter = _exporters().get(name)
    if exporter is None:
        raise HTTPException(404, f'No exporter named {name!r}')
    try:
        artifact = exporter.build(session)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None
    RUNTIME.hooks.emit('export.built', session=session, artifact=artifact)
    return _respond(artifact)


# -- historical URLs, kept working ----------------------------------------

@router.get('/api/session/{session_id}/notebook')
async def get_notebook(session_id: str):
    return await build_export(session_id, 'notebook')


@router.get('/api/session/{session_id}/artifacts.zip')
async def download_artifacts_zip(session_id: str):
    return await build_export(session_id, 'artifacts_zip')


@router.get('/api/session/{session_id}/logs/llm')
async def download_llm_log(session_id: str):
    return await build_export(session_id, 'llm_log')


@router.get('/api/session/{session_id}/logs/fermipy')
async def download_fermipy_log(session_id: str):
    return await build_export(session_id, 'fermipy_log')


@router.get("/api/session/{session_id}/artifact/{name}")
async def get_artifact(session_id: str, name: str):
    """Serve a generated output artifact (plot image) for a session.

    Filenames are sanitised to the basename to prevent path traversal.
    """
    session = get_session_or_404(session_id)

    safe_name = os.path.basename(name)
    artifact_path = os.path.join(session.session_dir, 'artifacts', safe_name)
    if not os.path.isfile(artifact_path):
        raise HTTPException(404, "Artifact not found")

    ext = os.path.splitext(safe_name)[1].lower()
    media = {
        '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.svg': 'image/svg+xml', '.pdf': 'application/pdf',
        '.json': 'application/json', '.xml': 'application/xml',
        '.yaml': 'application/x-yaml', '.yml': 'application/x-yaml',
        '.log': 'text/plain',
    }.get(ext, 'application/octet-stream')
    return FileResponse(artifact_path, media_type=media)


@router.get('/api/session/{session_id}/runs/{run_id}/{filename}')
async def download_run_export(session_id: str, run_id: str, filename: str):
    session = get_session_or_404(session_id)
    if not session:
        raise HTTPException(404, 'Session not found')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id or ''):
        raise HTTPException(404, 'Run export not found')

    allowed = {
        'final_config.yaml': 'application/x-yaml',
        'analysis.py': 'text/x-python',
        'analysis_files.zip': 'application/zip',
    }
    media_type = allowed.get(filename)
    if not media_type:
        raise HTTPException(404, 'Run export not found')

    run_dir = os.path.realpath(os.path.join(session.session_dir, 'runs', run_id))
    expected_root = os.path.realpath(os.path.join(session.session_dir, 'runs'))
    if not run_dir.startswith(expected_root + os.sep):
        raise HTTPException(404, 'Run export not found')
    export_path = os.path.join(run_dir, filename)
    if not os.path.isfile(export_path):
        raise HTTPException(404, 'Run export not found')
    return FileResponse(export_path, media_type=media_type, filename=filename)