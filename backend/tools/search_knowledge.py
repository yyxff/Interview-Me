"""Tool: search_knowledge — 搜索技术知识库"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interview_agent import InterviewSession

logger = logging.getLogger(__name__)

DESC = (
    "search_knowledge: 搜索技术知识库，了解某个技术点有哪些具体内容可以考察\n"
    '  用法：search_knowledge("MVCC binlog 原理")'
)


def search_knowledge(query: str) -> str:
    """向量 + 图谱混合检索，返回最相关的知识库片段。"""
    import rag as _rag
    import graph_rag as _gr
    logger.info("search_knowledge query=%r", query)
    gr = _rag.retrieve_graph(query)
    extra = _gr.get_chunks_by_ids(gr.get("source_chunk_ids", []))
    path_map = {
        rel["source_chunk_id"]: (
            f"[图谱路径] {rel['subject']} --{rel['predicate']}--> {rel['object']}。"
            f"{rel.get('description', '')}"
        )
        for rel in gr.get("relations", []) if rel.get("source_chunk_id")
    }
    result = _rag.retrieve_rich(query, extra_chunks=extra or None, path_map=path_map or None)
    chunks = result.get("knowledge", [])
    log = result.get("retrieval_log", [])
    summary = next((e for e in reversed(log) if e.get("_summary")), None)
    if summary:
        logger.debug(
            "search_knowledge summary: be_candidates=%s graph_extra=%s "
            "rerank_input=%s final_output=%s",
            summary.get("be_candidates"), summary.get("graph_extra"),
            summary.get("rerank_input"), summary.get("final_output"),
        )
    for entry in log:
        if not entry.get("_summary"):
            logger.debug(
                "search_knowledge   chunk=%s source=%s bi_dist=%s rerank=%s",
                entry.get("chunk_id", "?")[:40], entry.get("source", "?"),
                entry.get("bi_dist"), entry.get("rerank_score"),
            )
    if not chunks:
        logger.info("search_knowledge → 知识库无相关内容")
        return "知识库无相关内容"
    logger.info("search_knowledge → %d chunks returned", len(chunks))
    return "\n---\n".join(
        f"[{c['source']} §{c.get('chapter', '')}] {c['text'][:300]}" for c in chunks
    )


# ── 老式接口（供旧版 interview_agent 使用） ───────────────────────────────────
def make(session: "InterviewSession") -> dict:
    async def fn(query: str) -> str:
        return search_knowledge(query)
    return {"desc": DESC, "fn": fn}
