"""
模拟面试后端 - FastAPI

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

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import rag
from llm.provider import build_provider
from routes import (
    chat_router,
    qa_router,
    knowledge_router,
    graph_router,
    notes_router,
    qa_sessions_router,
    interview_router,
)
import routes.chat as _chat
import routes.qa as _qa
import routes.knowledge as _knowledge
import routes.graph as _graph
import routes.notes as _notes
import routes.interview as _interview

# ── Provider ──────────────────────────────────────────────────────────────────

_provider = build_provider()

# 注入 provider 到所有需要它的路由模块
_chat.set_provider(_provider)
_qa.set_provider(_provider)
_knowledge.set_provider(_provider)
_graph.set_provider(_provider)
_notes.set_provider(_provider)
_interview.set_provider(_provider)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Interview Me", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(qa_router)
app.include_router(knowledge_router)
app.include_router(graph_router)
app.include_router(notes_router)
app.include_router(qa_sessions_router)
app.include_router(interview_router)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """启动时在后台索引知识库（不阻塞服务启动）。"""
    async def _run():
        try:
            await rag.index_knowledge_with_qa(_provider)
        except Exception as e:
            print(f"[startup] 向量索引任务异常: {e}")

        try:
            filled = rag.backfill_qa_cache()
            if filled:
                print(f"[startup] QA 缓存回填完成: {filled}")
        except Exception as e:
            print(f"[startup] QA 缓存回填失败: {e}")

        if _provider is not None:
            try:
                import graph_rag as _gr
                await _gr.index_knowledge_graph(_provider)
            except Exception as e:
                print(f"[startup] 图索引任务异常: {e}")

    asyncio.create_task(_run())
