"""路由：/qa/stream（知识库 QA 精准模式）"""
from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm.provider import LLMProvider
from qa_agent import prepare_context, stream_response

router = APIRouter()

_provider: LLMProvider | None = None


def set_provider(p: LLMProvider | None) -> None:
    global _provider
    _provider = p


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QARequest(BaseModel):
    message: str
    history: list[HistoryItem] = []


@router.post("/qa/stream")
async def qa_stream(req: QARequest):
    """基于知识库的问答：先发 sources 事件，再流式发 LLM 回复。"""
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    ctx = await prepare_context(req.message, history_dicts, _provider)

    async def _generate():
        first_event: dict = {"sources": ctx.sources}
        if ctx.retrieval_query != ctx.original_query:
            first_event["rewritten_query"] = ctx.retrieval_query
        yield f"data: {json.dumps(first_event, ensure_ascii=False)}\n\n"

        if ctx.graph_viz and ctx.graph_viz.get("nodes"):
            yield f"data: {json.dumps({'graph': ctx.graph_viz}, ensure_ascii=False)}\n\n"

        try:
            async for chunk in stream_response(ctx, _provider):
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
