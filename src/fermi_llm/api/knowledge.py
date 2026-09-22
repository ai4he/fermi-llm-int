"""Documentation search and upload (RAG)."""

from __future__ import annotations

import os
import re

from fastapi import APIRouter, HTTPException, Request

from ..core import kinds
from ..core.runtime import RUNTIME

router = APIRouter()


def _knowledge_sources():
    return RUNTIME.ctx.build_stack(kinds.KNOWLEDGE_SOURCE)


def _writable_source():
    """The source that accepts uploads (the local docs folder)."""
    for source in _knowledge_sources():
        if hasattr(source, 'rag_dir'):
            return source
    return None


@router.post("/api/rag/search")
async def rag_search(request: Request):
    body = await request.json()
    query = body.get("query", "")
    results = []
    for source in _knowledge_sources():
        results.extend(source.search(query))
    return {"results": [{"title": r["title"], "content": r["content"][:500]} for r in results]}


@router.post("/api/rag/add_doc")
async def add_rag_doc(request: Request):
    body = await request.json()
    title = body.get("title", "untitled")
    content = body.get("content", "")
    fname = re.sub(r'[^a-zA-Z0-9_-]', '_', title)[:50] + '.txt'
    source = _writable_source()
    if source is None:
        raise HTTPException(400, 'No writable knowledge source is configured')
    os.makedirs(source.rag_dir, exist_ok=True)
    fpath = os.path.join(source.rag_dir, fname)
    with open(fpath, 'w') as f:
        f.write(content)
    source.docs.append({
        'title': title, 'content': content,
        'keywords': title.lower().split()
    })
    return {"status": "ok", "filename": fname}