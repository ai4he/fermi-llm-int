"""What the browser needs in order to assemble itself.

Two endpoints, both driven by the registry: the active skin's CSS, and the
manifest list the frontend loader walks. Plugin assets are served from each
extension's own ``asset_root`` under ``/static/plugins/<name>/``.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from ..core import kinds
from ..core.runtime import RUNTIME

router = APIRouter()


def _active_skin():
    ctx = RUNTIME.ctx
    registration = ctx.registry.selected(kinds.SKIN, ctx.settings.skin or None)
    return registration.factory(ctx)


@router.get('/api/ui/skin.css')
async def skin_css():
    """Concatenate the active skin's stylesheets into one response."""
    skin = _active_skin()
    chunks = []
    for path in skin.stylesheets():
        try:
            with open(path) as handle:
                chunks.append(f'/* {os.path.basename(path)} */\n' + handle.read())
        except OSError as exc:
            raise HTTPException(500, f'skin asset missing: {exc}') from None
    return PlainTextResponse('\n\n'.join(chunks), media_type='text/css')


@router.get('/api/ui/extensions')
async def ui_extensions():
    """Manifests of every active UI extension, in load order."""
    out = []
    for extension in RUNTIME.ctx.build_stack(kinds.UI_EXTENSION):
        try:
            manifest = dict(extension.manifest())
        except Exception:                                 # noqa: BLE001
            continue
        manifest.setdefault('name', extension.name)
        # Samples travel on /api/demo_samples; keep this payload small.
        manifest.pop('samples', None)
        out.append(manifest)
    return {'extensions': out,
            'skin': _active_skin().name}


@router.get('/static/plugins/{name}/{asset:path}')
async def plugin_asset(name: str, asset: str):
    """Fallback asset route.

    Extensions present at startup get a real static mount (see
    ``create_app``); this serves the rest — one registered later, or one
    whose asset directory appeared after boot.
    """
    for extension in RUNTIME.ctx.build_stack(kinds.UI_EXTENSION):
        if extension.name != name:
            continue
        root = extension.asset_root() if hasattr(extension, 'asset_root') else None
        if not root:
            break
        full = os.path.realpath(os.path.join(root, asset))
        if not full.startswith(os.path.realpath(root) + os.sep):
            raise HTTPException(404, 'Not found')
        if os.path.isfile(full):
            return FileResponse(full)
        break
    raise HTTPException(404, 'Not found')
