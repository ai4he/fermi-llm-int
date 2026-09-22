"""Metadata endpoints: models, versions, demo samples and the platform view."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import signal
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse,
                               StreamingResponse)

from ..core import kinds
from ..core.jsonutil import sanitize_for_json
from ..core.runtime import RUNTIME
from ..core.session import AnalysisSession
from ..components.ui.demo_samples import DEMO_SAMPLES
from .deps import get_session_or_404

router = APIRouter()

@router.get("/api/models")
async def list_models():
    """List available models with status."""
    return {
        "models": RUNTIME.models.list_models(),
        "default_model": RUNTIME.models.default_model_id,
        "gpu_available": RUNTIME.models.gpu_available,
        "gpu_info": RUNTIME.models.gpu_info,
    }


@router.get("/api/versions")
async def get_versions():
    """Report the FermiPy / Fermi ScienceTools versions in use (for the UI)."""
    fermipy_v = None
    fermitools_v = None
    try:
        import fermipy
        fermipy_v = getattr(fermipy, '__version__', None)
    except Exception:
        pass
    # fermitools is a conda package without standard dist metadata; read the
    # version from the conda-meta record in the active environment.
    try:
        import sys as _sys, glob as _glob
        # Match only the main package (fermitools-<digit>...), not fermitools-data.
        recs = _glob.glob(os.path.join(_sys.prefix, 'conda-meta', 'fermitools-[0-9]*.json'))
        if recs:
            base = os.path.basename(recs[0])[len('fermitools-'):]
            fermitools_v = base.split('-')[0]
    except Exception:
        pass
    return {"fermipy": fermipy_v, "fermitools": fermitools_v,
            "python": platform.python_version()}


@router.get("/api/demo_samples")
async def get_demo_samples():
    """Return curated demo samples for presentations."""
    # Check model availability and annotate samples
    available_models = {m['id']: m['available'] for m in RUNTIME.models.list_models()}
    # Samples come from every UI extension that offers them, so a lab can
    # ship its own demo prompts as a plugin.
    catalogue = []
    for extension in RUNTIME.ctx.build_stack(kinds.UI_EXTENSION):
        catalogue.extend(getattr(extension, 'samples', lambda: [])())
    samples = []
    for s in catalogue:
        sample = dict(s)
        sample['model_available'] = available_models.get(s['model'], False)
        samples.append(sample)
    return {"samples": samples}


@router.get('/api/platform')
async def platform_view():
    """What is plugged in right now.

    The honest answer to "which modules is this deployment running?" —
    every component, its state (active, replaced, disabled, blocked), where
    it came from, and any plugin that failed to load. The UI shows it under
    *Platform*; an agent integrating a new module reads it to confirm the
    registration landed.
    """
    return sanitize_for_json(RUNTIME.ctx.describe())
