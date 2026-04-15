"""
LLM 工厂（读取与原系统完全相同的环境变量）
"""
from __future__ import annotations

import os


# ══════════════════════════════════════════════════════════════════════════════
# LLM 工厂（读取与原系统完全相同的环境变量）
# ══════════════════════════════════════════════════════════════════════════════

def _build_llm():
    """
    根据环境变量返回 LangChain LLM 对象。

    LangChain LLM 与原来自定义 LLMProvider 的区别：
      - 支持 .ainvoke([messages])   ← 直接调用
      - 支持 .bind_tools(tools)     ← 绑定工具（create_react_agent 用这个）
      - 支持 .astream([messages])   ← 流式输出
    """
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")
    base_url = os.environ.get("LLM_BASE_URL")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-opus-4-6",
            api_key=api_key,
            max_tokens=4096,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=api_key or "ollama",
            base_url=base_url,
            max_tokens=4096,
        )
