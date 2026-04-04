"""路由：/chat  /chat/stream（普通面试聊天模式）"""
from __future__ import annotations

import json
import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import rag
from llm.provider import LLMProvider

router = APIRouter()

# 注入 provider（由 main.py 在启动时调用 set_provider）
_provider: LLMProvider | None = None


def set_provider(p: LLMProvider | None) -> None:
    global _provider
    _provider = p


SYSTEM_PROMPT = """你是一位经验丰富的技术面试官，专注于软件工程领域。你的风格是：
- 问题由浅入深，先考察基础再深挖原理
- 对候选人的回答给出简洁点评，再追问下一个问题
- 语气专业但不冷漠，适当鼓励
- 每次回复控制在 2-4 句话，保持对话节奏

现在开始面试，先简单自我介绍并提出第一个问题。"""

MAX_HISTORY_MESSAGES = 12


def trim_history(history: list) -> list:
    return history[-MAX_HISTORY_MESSAGES:]


def build_prompt(session_id: str | None, query: str) -> str:
    ctx = rag.retrieve(query, session_id)
    prompt = SYSTEM_PROMPT
    if ctx["resume"]:
        prompt += f"\n\n## 候选人简历（相关片段）\n" + "\n\n".join(ctx["resume"])
    if ctx["knowledge"]:
        prompt += f"\n\n## 相关技术参考\n" + "\n\n---\n\n".join(ctx["knowledge"])
    if ctx.get("notes"):
        prompt += f"\n\n## 候选人知识笔记\n" + "\n\n---\n\n".join(ctx["notes"])
    return prompt


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryItem] = []
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _provider is None:
        return ChatResponse(
            response=(
                f"【占位回复】收到消息：「{req.message}」。"
                "请设置 LLM_PROVIDER 和 LLM_API_KEY 环境变量以启用 AI 面试官。"
            )
        )
    messages = [{"role": m.role, "content": m.content} for m in trim_history(req.history)]
    messages.append({"role": "user", "content": req.message})
    try:
        text = await _provider.chat(messages, build_prompt(req.session_id, req.message))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM 调用失败: {e}")
    return ChatResponse(response=text)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in trim_history(req.history)]
    messages.append({"role": "user", "content": req.message})
    system = build_prompt(req.session_id, req.message)

    if _provider is None:
        async def _placeholder():
            text = f"【占位回复】收到消息：「{req.message}」。请设置 LLM_PROVIDER 和 LLM_API_KEY 环境变量以启用 AI 面试官。"
            yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_placeholder(), media_type="text/event-stream")

    async def _generate():
        try:
            async for chunk in _provider.stream_chat(messages, system):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
