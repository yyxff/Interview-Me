"""路由：/notes/*  /qa/summarize（知识笔记管理）"""
from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import rag
from llm.provider import LLMProvider

router = APIRouter()

_provider: LLMProvider | None = None


def set_provider(p: LLMProvider | None) -> None:
    global _provider
    _provider = p


SUMMARIZE_PROMPT = """请将以下问答对话整理为结构化的知识点笔记，输出 JSON（只输出 JSON，不要加代码块或其他文字）。

格式：
{
  "title": "笔记标题",
  "questions": ["针对此知识点的典型面试问题1", "问题2", "问题3", "问题4"],
  "content": "Markdown 格式正文，分条列出核心知识点"
}

要求：
- questions 生成 4-6 个，覆盖不同角度和难度，用于后续向量化检索
- content 去除对话冗余，只保留知识本身，语言简洁准确"""


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SummarizeRequest(BaseModel):
    messages: list[HistoryItem]


class SaveNoteRequest(BaseModel):
    title:     str
    content:   str
    questions: list[str] = []


@router.post("/qa/summarize")
async def qa_summarize(req: SummarizeRequest):
    """将对话总结为知识点笔记（非流式）。"""
    import json as _json, re as _re
    if _provider is None:
        raise HTTPException(status_code=503, detail="LLM 未配置")

    conv = "\n".join(
        f"{'用户' if m.role == 'user' else 'AI'}: {m.content}"
        for m in req.messages
    )
    try:
        text = await _provider.chat(
            messages=[{"role": "user", "content": f"对话内容：\n\n{conv}"}],
            system=SUMMARIZE_PROMPT,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        cleaned = _re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=_re.IGNORECASE)
        cleaned = _re.sub(r'\s*```\s*$', '', cleaned)
        data      = _json.loads(cleaned)
        title     = data.get("title", "笔记").strip()
        questions = data.get("questions", [])
        content   = data.get("content", "").strip()
    except Exception:
        lines     = text.strip().split("\n", 1)
        title     = lines[0].strip()
        questions = []
        content   = lines[1].strip() if len(lines) > 1 else ""
    return {"title": title, "content": content, "questions": questions}


@router.post("/notes/save")
async def notes_save(req: SaveNoteRequest):
    note_id, text = rag.save_note_file(req.title, req.content, req.questions)
    return {
        "note_id":    note_id,
        "title":      req.title,
        "created_at": note_id[5:],
        "size":       len(text.encode()),
    }


@router.post("/notes/{note_id}/index")
async def notes_index(note_id: str):
    content = rag.get_note(note_id)
    if content is None:
        raise HTTPException(status_code=404, detail="笔记不存在")
    notes = rag.list_notes()
    note  = next((n for n in notes if n["note_id"] == note_id), None)
    title = note["title"] if note else note_id
    asyncio.create_task(asyncio.to_thread(rag.index_note, note_id, title, content))
    return {"ok": True}


@router.get("/notes/list")
async def notes_list():
    return {"notes": rag.list_notes()}


@router.get("/notes/{note_id}")
async def notes_get(note_id: str):
    content = rag.get_note(note_id)
    if content is None:
        raise HTTPException(status_code=404, detail="笔记不存在")
    questions = rag.get_note_questions(note_id)
    return {"note_id": note_id, "content": content, "questions": questions}


@router.delete("/notes/{note_id}")
async def notes_delete(note_id: str):
    ok = rag.delete_note(note_id)
    if not ok:
        raise HTTPException(status_code=404, detail="笔记不存在")
    return {"ok": True}
