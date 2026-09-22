"""The WebSocket fallback channel.

Progress is delivered over SSE; this channel stays as the notification path
for clients that keep a socket open, and as the hook other front ends use.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core.runtime import RUNTIME

router = APIRouter()


# WebSocket for real-time updates (fallback)
@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    RUNTIME.websockets[session_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
    except WebSocketDisconnect:
        RUNTIME.websockets.pop(session_id, None)