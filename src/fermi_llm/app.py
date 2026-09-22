"""The application factory.

``create_app`` is deliberately small: build the context (which loads every
plugin), bind the runtime, mount the routers each component contributes, and
serve the frontend. Everything a feature needs is registered, not imported
here — adding a module must never require editing this file.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .core import kinds
from .core.config import Settings, load_settings
from .core.context import AppContext, build_context
from .core.runtime import RUNTIME
from .api import (artifacts, auth, chat, knowledge, meta, runs, sessions,
                  sockets, ui)

log = logging.getLogger(__name__)

#: Routers that ship with the platform, mounted in this order.
CORE_ROUTERS = (meta.router, auth.router, sessions.router, chat.router,
                runs.router, artifacts.router, knowledge.router, ui.router,
                sockets.router)


def create_app(settings: Settings = None, ctx: AppContext = None) -> FastAPI:
    """Build the FastAPI application for one deployment."""
    ctx = ctx or build_context(settings or load_settings())
    RUNTIME.bind(ctx)

    app = FastAPI(title='Fermi LLM', version=__version__)
    app.state.ctx = ctx

    # Plugin assets mount *before* /static: routes match in order, so the
    # broad /static mount would otherwise shadow /static/plugins/<name>/.
    for extension in ctx.build_stack(kinds.UI_EXTENSION):
        root = getattr(extension, 'asset_root', lambda: None)()
        if not root or not os.path.isdir(root):
            continue
        app.mount(f'/static/plugins/{extension.name}',
                  StaticFiles(directory=root),
                  name=f'plugin-{extension.name}')
        log.info('mounted assets for ui extension %s', extension.name)

    static_dir = ctx.settings.static_dir
    app.mount('/static', StaticFiles(directory=static_dir), name='static')

    for router in CORE_ROUTERS:
        app.include_router(router)

    # Plugins may add routes of their own. A failing extension is logged and
    # skipped: one lab's experimental endpoint must not stop the server.
    for extension in ctx.build_stack(kinds.API_EXTENSION):
        try:
            app.include_router(extension.router(ctx),
                               prefix=getattr(extension, 'prefix', ''))
            log.info('mounted api extension %s', extension.name)
        except Exception as exc:                          # noqa: BLE001
            log.warning('api extension %s failed: %s', extension.name, exc)
            if ctx.load_report is not None:
                ctx.load_report.add_failure(f'api_extension:{extension.name}', exc)

    @app.get('/')
    async def index():                                    # noqa: ANN202
        return FileResponse(os.path.join(static_dir, 'index.html'))

    @app.on_event('shutdown')
    async def _shutdown():                                # noqa: ANN202
        from .services.runs import shutdown_active_runs
        await shutdown_active_runs()
        ctx.models.shutdown()
        ctx.hooks.emit('app.shutdown', ctx=ctx)

    log.info('fermi-llm %s ready on %s', __version__, ctx.settings.project_dir)
    return app


def main() -> None:
    """Entry point used by ``scripts/run_webapp.sh`` and ``python -m``."""
    import uvicorn

    logging.basicConfig(
        level=os.environ.get('FERMI_LLM_LOG_LEVEL', 'INFO'),
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s')
    settings = load_settings()
    app = create_app(settings)
    host, port = settings.host, settings.port
    print(f'Starting Fermi LLM on {host}:{port}')
    print(f'Open http://localhost:{port} in your browser')
    import socket
    try:
        socket.getaddrinfo(host, port)
    except socket.gaierror:
        host = '127.0.0.1'
    uvicorn.run(app, host=host, port=port, log_level='info')


if __name__ == '__main__':
    main()
