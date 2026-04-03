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
        import rag as _rag, graph_rag as _gr
        gr = _rag.retrieve_graph(query)
        extra = _gr.get_chunks_by_ids(gr.get("source_chunk_ids", []))
        path_map = {
            rel["source_chunk_id"]: (
                f"[图谱路径] {rel['subject']} --{rel['predicate']}--> {rel['object']}。"
                f"{rel.get('description', '')}"
            )
            for rel in gr.get("relations", []) if rel.get("source_chunk_id")
        }
        r = _rag.retrieve_rich(query, extra_chunks=extra or None, path_map=path_map or None)
        chunks = r.get("knowledge", [])[:3]
        if not chunks:
            return "知识库中未找到相关内容"
        return "\n".join(
            f"[{c['source']} §{c.get('chapter', '')}] {c['text'][:300]}"
            for c in chunks
        )

    return {"desc": DESC, "fn": fn}
