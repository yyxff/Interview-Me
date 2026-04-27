"""基础检索原语 + 简单单路召回（retrieve）。

双路召回（向量 + 图谱 RRF + rerank）见 retrieval_rich.py。
"""
from __future__ import annotations

from .client import (
    is_available, KNOWLEDGE_TOP_K, KNOWLEDGE_RETURN_K, RESUME_TOP_K, QA_PER_CHUNK,
    _get_knowledge_col, _get_resume_col, _get_notes_col,
    rerank,
)

# cosine distance 阈值（bge 归一化向量下 ≈ cosine_sim > 0.755）
SIMILARITY_DISTANCE_THRESHOLD = 0.70


# ── 底层工具 ──────────────────────────────────────────────────────────────────

def _safe_query(
    col, query: str, n: int, where: dict | None = None, return_distances: bool = False
) -> list:
    """查询向量数据库，过滤相似度不足的结果（距离 > 阈值则丢弃）。"""
    try:
        kwargs: dict = {
            "query_texts": [query],
            "n_results":   n,
            "include":     ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results   = col.query(**kwargs)
        docs      = results.get("documents",  [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]
        if return_distances:
            return [
                (d, m, dist) for d, m, dist in zip(docs, metas, distances)
                if d and dist < SIMILARITY_DISTANCE_THRESHOLD
            ]
        return [
            (d, m) for d, m, dist in zip(docs, metas, distances)
            if d and dist < SIMILARITY_DISTANCE_THRESHOLD
        ]
    except Exception:
        return []


def _rrf_merge(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """标准 RRF：融合多路排名列表，返回按综合分数降序的 chunk_id 列表。"""
    scores: dict[str, float] = {}
    for ranked_list in rank_lists:
        for rank, cid in enumerate(ranked_list, start=1):
            if cid:
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def _dedupe_chunks(raw: list[tuple[str, dict]], limit: int) -> list[tuple[str, dict]]:
    """按 chunk_id 去重，保留前 limit 条。"""
    seen: set[str] = set()
    result: list[tuple[str, dict]] = []
    for doc, meta in raw:
        chunk_id = meta.get("chunk_id", doc[:30])
        if chunk_id not in seen:
            seen.add(chunk_id)
            result.append((doc, meta))
            if len(result) >= limit:
                break
    return result


# ── 共享子检索（resume / notes） ──────────────────────────────────────────────

def _fetch_resume(query: str, session_id: str) -> list[dict]:
    """从简历集合检索与问题相关的片段。"""
    try:
        col = _get_resume_col()
        existing = col.get(where={"session_id": session_id})
        count = len(existing["ids"])
        if count == 0:
            return []
        raw = _safe_query(col, query, min(RESUME_TOP_K, count),
                          where={"session_id": session_id})
        return [{"text": doc, "source": "resume", "chapter": "",
                 "chunk_id": meta.get("chunk_id", ""), "question": ""}
                for doc, meta in raw]
    except Exception:
        return []


def _fetch_notes(query: str) -> list[dict]:
    """从笔记集合检索相关笔记片段。"""
    try:
        col = _get_notes_col()
        if col.count() == 0:
            return []
        raw = _safe_query(col, query, min(2, col.count()))
        seen: set[str] = set()
        notes = []
        for doc, meta in raw:
            note_id = meta.get("note_id", doc[:30])
            if note_id in seen:
                continue
            seen.add(note_id)
            notes.append({
                "text":     meta.get("text", doc),
                "source":   "笔记",
                "chapter":  meta.get("title", ""),
                "chunk_id": note_id,
                "question": doc if doc != meta.get("text") else "",
            })
        return notes
    except Exception:
        return []


# ── 简单单路检索 ──────────────────────────────────────────────────────────────

def retrieve(query: str, session_id: str | None = None) -> dict:
    """简单模式：向量召回 → 去重 → rerank，不含图谱融合。"""
    if not is_available():
        return {"knowledge": [], "resume": [], "notes": []}

    knowledge_chunks: list[str] = []
    try:
        col = _get_knowledge_col()
        if col.count() > 0:
            n_candidates = min(KNOWLEDGE_TOP_K * QA_PER_CHUNK * 2, col.count())
            raw          = _safe_query(col, query, n_candidates)
            candidates   = _dedupe_chunks(raw, limit=KNOWLEDGE_TOP_K * 3)
            ranked       = rerank(query, candidates)
            knowledge_chunks = [doc for doc, _, _score in ranked[:KNOWLEDGE_TOP_K]]
    except Exception:
        pass

    return {
        "knowledge": knowledge_chunks,
        "resume":    _fetch_resume(query, session_id) if session_id else [],
        "notes":     _fetch_notes(query),
    }


# ── 图谱检索包装 ──────────────────────────────────────────────────────────────

def retrieve_graph(query: str) -> dict:
    """Graph RAG 查询包装 — 优雅降级，import 失败返回空结果。"""
    try:
        import graph_rag
        return graph_rag.retrieve_graph(query)
    except Exception:
        return {"entities": [], "relations": [], "source_chunk_ids": [], "graph_summary": ""}


# ── 辅助查询 ──────────────────────────────────────────────────────────────────

def has_resume(session_id: str) -> bool:
    if not is_available():
        return False
    try:
        col = _get_resume_col()
        existing = col.get(where={"session_id": session_id}, limit=1)
        return len(existing["ids"]) > 0
    except Exception:
        return False


def knowledge_count() -> int:
    if not is_available():
        return 0
    try:
        return _get_knowledge_col().count()
    except Exception:
        return 0
