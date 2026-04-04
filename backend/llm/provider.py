"""
LLM Provider 抽象层

支持 Anthropic、OpenAI、OpenAI-compatible（DeepSeek、Ollama 等）。
通过环境变量配置：
    LLM_PROVIDER   = anthropic | openai | openai-compatible
    LLM_MODEL      = 模型名称
    LLM_API_KEY    = API Key
    LLM_BASE_URL   = 自定义 base URL（openai-compatible 必填）
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import AsyncIterator


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
            max_tokens=4096,
            system=system,
            messages=messages,
        )
        return result.content[0].text  # type: ignore[index]

    async def stream_chat(self, messages: list[dict], system: str) -> AsyncIterator[str]:  # type: ignore[override]
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=4096,
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
            max_tokens=4096,
            messages=all_messages,  # type: ignore[arg-type]
        )
        return result.choices[0].message.content or ""

    async def stream_chat(self, messages: list[dict], system: str) -> AsyncIterator[str]:  # type: ignore[override]
        all_messages = [{"role": "system", "content": system}] + messages
        stream = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=4096,
            messages=all_messages,  # type: ignore[arg-type]
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def build_provider() -> LLMProvider | None:
    """根据环境变量构建 LLM Provider，失败时返回 None。"""
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
            if not api_key and provider_name == "openai":
                return None
            return OpenAIProvider(
                api_key=api_key or "ollama",
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
