"""Tool: search_knowledge — 搜索技术知识库"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interview_agent import InterviewSession


DESC = (
    "search_knowledge: 搜索技术知识库，了解某个技术点有哪些具体内容可以考察\n"
    '  用法：search_knowledge("MVCC binlog 原理")'
)


def make(session: "InterviewSession") -> dict:
    async def fn(query: str) -> str:
        import rag as _rag
        r = _rag.retrieve_rich(query)
        chunks = r.get("knowledge", [])[:3]
        if not chunks:
            return "知识库中未找到相关内容"
        return "\n".join(
            f"[{c['source']} §{c.get('chapter', '')}] {c['text'][:300]}"
            for c in chunks
        )

    return {"desc": DESC, "fn": fn}
