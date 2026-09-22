"""Chat: one message in, a reviewed configuration out.

The endpoint stays thin on purpose — everything it does lives in
``fermi_llm.services.chat`` so the same flow can be driven by a test, a
batch script or another front end.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..core.runtime import RUNTIME
from ..services.chat import handle_message
from .deps import get_session_or_404

router = APIRouter()


@router.post('/api/session/{session_id}/chat')
async def chat(session_id: str, request: Request):
    body = await request.json()
    message = body.get('message', '')
    selected_model = body.get('model') or RUNTIME.models.default_model_id
    session = get_session_or_404(session_id)
    return await handle_message(session, message, selected_model)


@router.get("/api/session/{session_id}/chat_stream")
async def chat_stream(session_id: str):
    """SSE endpoint for real-time Master Agent (chat/generation) progress.

    The frontend opens this BEFORE sending the /chat POST so it can show
    each stage: prompt analysis, RAG lookup, model loading, inference, parsing.
    """
    session = get_session_or_404(session_id)

    async def event_generator():
        chat_progress_file = os.path.join(session.session_dir, 'chat_progress.jsonl')
        lines_read = 0
        idle_count = 0
        max_idle = 300  # 5 minutes max

        while idle_count < max_idle:
            new_lines = []
            if os.path.exists(chat_progress_file):
                try:
                    with open(chat_progress_file) as f:
                        all_lines = f.readlines()
                    new_lines = all_lines[lines_read:]
                    lines_read = len(all_lines)
                except Exception:
                    pass

            for line in new_lines:
                line = line.strip()
                if line:
                    yield f"data: {line}\n\n"
                    idle_count = 0
                    try:
                        parsed = json.loads(line)
                        if parsed.get('stage') == 'complete':
                            return
                    except Exception:
                        pass

            if idle_count > 0 and idle_count % 10 == 0:
                yield f": heartbeat\n\n"

            idle_count += 1
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )