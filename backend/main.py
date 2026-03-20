"""
模拟面试后端 - FastAPI

环境变量:
    LLM_PROVIDER   = anthropic | openai | openai-compatible  (默认: anthropic)
    LLM_MODEL      = 模型名称 (各 provider 有默认值，见下方)
    LLM_API_KEY    = API Key
    LLM_BASE_URL   = 自定义 base URL（openai-compatible 必填，如 http://localhost:11434/v1）

启动示例:
    # Anthropic
    LLM_PROVIDER=anthropic LLM_API_KEY=sk-ant-xxx uvicorn main:app --reload

    # OpenAI
    LLM_PROVIDER=openai LLM_API_KEY=sk-xxx uvicorn main:app --reload

    # DeepSeek (openai-compatible)
    LLM_PROVIDER=openai-compatible LLM_API_KEY=xxx LLM_BASE_URL=https://api.deepseek.com/v1 LLM_MODEL=deepseek-chat uvicorn main:app --reload

    # Ollama (openai-compatible, 无需 key)
    LLM_PROVIDER=openai-compatible LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3 uvicorn main:app --reload
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位经验丰富的技术面试官，专注于软件工程领域。你的风格是：
- 问题由浅入深，先考察基础再深挖原理
- 对候选人的回答给出简洁点评，再追问下一个问题
- 语气专业但不冷漠，适当鼓励
- 每次回复控制在 2-4 句话，保持对话节奏

现在开始面试，先简单自我介绍并提出第一个问题。"""


# ── Provider 抽象 ─────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], system: str) -> str: ...

    @abstractmethod
    def stream_chat(self, messages: list[dict], system: str) -> AsyncIterator[str]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class AnthropicProvider(LLMProvider):
    DEFAULT_MODEL = "claude-opus-4-6"

    def __init__(self, api_key: str, model: str):
        import anthropic  # type: ignore
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"

    async def chat(self, messages: list[dict], system: str) -> str:
        result = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return result.content[0].text  # type: ignore[index]

    async def stream_chat(self, messages: list[dict], system: str) -> AsyncIterator[str]:  # type: ignore[override]
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=messages,
        ) as stream:
            async for chunk in stream.text_stream:
                yield chunk


class OpenAIProvider(LLMProvider):
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        from openai import AsyncOpenAI  # type: ignore
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._label = base_url or "openai"

    @property
    def name(self) -> str:
        return f"{self._label}/{self._model}"

    async def chat(self, messages: list[dict], system: str) -> str:
        all_messages = [{"role": "system", "content": system}] + messages
        result = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            messages=all_messages,  # type: ignore[arg-type]
        )
        return result.choices[0].message.content or ""

    async def stream_chat(self, messages: list[dict], system: str) -> AsyncIterator[str]:  # type: ignore[override]
        all_messages = [{"role": "system", "content": system}] + messages
        stream = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            messages=all_messages,  # type: ignore[arg-type]
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Provider 工厂 ─────────────────────────────────────────────────────────────

def _build_provider() -> LLMProvider | None:
    provider_name = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")
    base_url = os.environ.get("LLM_BASE_URL") or None

    try:
        if provider_name == "anthropic":
            if not api_key:
                return None
            return AnthropicProvider(
                api_key=api_key,
                model=model or AnthropicProvider.DEFAULT_MODEL,
            )

        if provider_name in ("openai", "openai-compatible"):
            # openai-compatible 可以没有 key（如本地 Ollama）
            if not api_key and provider_name == "openai":
                return None
            return OpenAIProvider(
                api_key=api_key or "ollama",  # openai SDK 要求非空，填占位值
                model=model or OpenAIProvider.DEFAULT_MODEL,
                base_url=base_url,
            )

        print(f"[warn] 未知 LLM_PROVIDER={provider_name!r}，AI 回复已禁用")
        return None

    except ImportError as e:
        print(f"[warn] 依赖缺失: {e}，AI 回复已禁用")
        return None
    except Exception as e:
        print(f"[warn] Provider 初始化失败: {e}，AI 回复已禁用")
        return None


_provider: LLMProvider | None = _build_provider()


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(title="Interview Me", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求 / 响应模型 ───────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryItem] = []


class ChatResponse(BaseModel):
    response: str


# ── 路由 ──────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ai_enabled": _provider is not None,
        "provider": _provider.name if _provider else None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _provider is None:
        return ChatResponse(
            response=(
                f"【占位回复】收到消息：「{req.message}」。"
                "请设置 LLM_PROVIDER 和 LLM_API_KEY 环境变量以启用 AI 面试官。"
            )
        )

    messages = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})

    try:
        text = await _provider.chat(messages, SYSTEM_PROMPT)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM 调用失败: {e}")

    return ChatResponse(response=text)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})

    if _provider is None:
        async def _placeholder():
            text = f"【占位回复】收到消息：「{req.message}」。请设置 LLM_PROVIDER 和 LLM_API_KEY 环境变量以启用 AI 面试官。"
            yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_placeholder(), media_type="text/event-stream")

    async def _generate():
        try:
            async for chunk in _provider.stream_chat(messages, SYSTEM_PROMPT):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁止 nginx 缓冲
        },
    )
