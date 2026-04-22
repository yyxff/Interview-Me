"""检索：retrieve、retrieve_rich（bi-encoder + Graph RRF + rerank）"""
from __future__ import annotations

from .client import (
    is_available, KNOWLEDGE_TOP_K, KNOWLEDGE_RETURN_K, RESUME_TOP_K, QA_PER_CHUNK,
    _get_knowledge_col, _get_resume_col, _get_notes_col,
    rerank,
)

# cosine distance 阈值（bge 归一化向量下 ≈ cosine_sim > 0.755）
SIMILARITY_DISTANCE_THRESHOLD = 0.70


def _safe_query(
    col, query: str, n: int, where: dict | None = None, return_distances: bool = False
) -> list:
    """查询 ChromaDB，过滤距离超阈值的结果。"""
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
    """标准 RRF 融合多路排名列表，返回按综合分数降序的 chunk_id 列表。"""
    scores: dict[str, float] = {}
    for ranked_list in rank_lists:
        for rank, cid in enumerate(ranked_list, start=1):
            if cid:
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def _dedupe_chunks(raw: list[tuple[str, dict]], limit: int) -> list[tuple[str, dict]]:
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


def retrieve_rich(
    query: str,
    session_id: str | None = None,
    extra_chunks: list[dict] | None = None,
    top_k: int | None = None,
    return_k: int | None = None,
    path_map: dict[str, str] | None = None,
) -> dict:
    """
    精准模式检索：bi-encoder → 阈值过滤 → 去重 + 图谱 extra_chunks
    → RRF 融合 → cross-encoder rerank → top-K

    top_k:    候选池深度（影响 n_candidates / dedupe / rerank 窗口）
    return_k: 最终返回给 LLM 的 chunk 数，默认 KNOWLEDGE_RETURN_K
    """
    if not is_available():
        return {"knowledge": [], "resume": [], "notes": [], "retrieval_log": []}

    k  = top_k    if top_k    is not None else KNOWLEDGE_TOP_K
    rk = return_k if return_k is not None else KNOWLEDGE_RETURN_K
    knowledge: list[dict] = []
    retrieval_log: list[dict] = []

    try:
        col = _get_knowledge_col()
        if col.count() > 0:
            n_candidates    = min(k * QA_PER_CHUNK * 2, col.count())
            raw_with_dist   = _safe_query(col, query, n_candidates, return_distances=True)
            raw             = [(d, m) for d, m, _ in raw_with_dist]
            candidates      = _dedupe_chunks(raw, limit=k * 3)
            dist_map        = {m.get("chunk_id", ""): dist for d, m, dist in raw_with_dist}

            cand_map: dict[str, tuple[str, dict]] = {
                m.get("chunk_id", ""): (m.get("text", doc), m)
                for doc, m in candidates if m.get("chunk_id")
            }
            be_rank = list(cand_map.keys())

            extra_map: dict[str, dict] = {}
            graph_rank: list[str] = []
            if extra_chunks:
                for c in extra_chunks:
                    cid = c.get("chunk_id", "")
                    if cid:
                        extra_map[cid] = c
                        graph_rank.append(cid)
                        if cid not in cand_map:
                            cand_map[cid] = (c["text"], {
                                "chunk_id": cid, "text": c["text"],
                                "source": c.get("source", ""), "path": c.get("path", ""),
                                "chapter": c.get("chapter", ""), "question": "",
                            })

            graph_rank = graph_rank[:k * 3]   # 与 be_rank 上限对等，避免图路压制向量路
            be_only  = len(be_rank)
            gph_only = sum(1 for cid in graph_rank if cid not in set(be_rank))
            overlap  = sum(1 for cid in graph_rank if cid in set(be_rank))

            retrieval_log.append({
                "_be_stage": True,
                "candidates": [(cid, round(dist_map.get(cid, -1), 4)) for cid in be_rank],
            })

            if graph_rank:
                rrf_scores: dict[str, float] = {}
                for ranked_list in [be_rank, graph_rank]:
                    for rank, cid in enumerate(ranked_list, start=1):
                        if cid:
                            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank)
                merged_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)
            else:
                rrf_scores = {}
                merged_ids = be_rank

            RERANK_LIMIT  = max(k * 4, 15)
            rerank_cids   = [cid for cid in merged_ids if cid in cand_map][:RERANK_LIMIT]

            retrieval_log.append({
                "_rrf_stage": True,
                "candidates": [(cid, round(rrf_scores.get(cid, 0), 6)) for cid in rerank_cids],
            })
            rerank_inputs = []
            for cid in rerank_cids:
                raw_text, meta = cand_map[cid]
                enriched = path_map[cid] + "\n\n" + raw_text if (path_map and cid in path_map) else raw_text
                rerank_inputs.append((enriched, meta))
            ranked = rerank(query, rerank_inputs)

            be_cid_set = set(be_rank)
            for doc, meta, score in ranked[:rk]:
                cid        = meta.get("chunk_id", "")
                graph_only = cid in extra_map and cid not in be_cid_set
                knowledge.append({
                    "text":      meta.get("text", doc),
                    "source":    meta.get("source", ""),
                    "path":      meta.get("path", ""),
                    "chapter":   meta.get("chapter", meta.get("path", "")),
                    "chunk_id":  cid,
                    "question":  meta.get("question", ""),
                    "via_graph": cid in extra_map,
                })
                retrieval_log.append({
                    "chunk_id":     cid,
                    "source":       meta.get("source", ""),
                    "bi_dist":      round(dist_map.get(cid, -1), 4),
                    "rrf_score":    round(rrf_scores.get(cid, 0.0), 6),
                    "rerank_score": round(score, 4),
                    "via_graph":    cid in extra_map,
                    "graph_only":   graph_only,
                    "question":     meta.get("question", "")[:60],
                })

            retrieval_log.append({
                "_summary": True,
                "be_candidates": be_only, "graph_extra": len(graph_rank),
                "graph_new": gph_only, "graph_overlap": overlap,
                "rerank_input": len(rerank_cids), "final_output": len(knowledge),
            })
    except Exception:
        pass

    resume: list[dict] = []
    if session_id:
        try:
            col = _get_resume_col()
            existing = col.get(where={"session_id": session_id})
            count = len(existing["ids"])
            if count > 0:
                raw = _safe_query(col, query, min(RESUME_TOP_K, count),
                                  where={"session_id": session_id})
                resume = [{"text": doc, "source": "resume", "chapter": "",
                           "chunk_id": meta.get("chunk_id", ""), "question": ""}
                          for doc, meta in raw]
        except Exception:
            pass

    notes: list[dict] = []
    try:
        col = _get_notes_col()
        if col.count() > 0:
            raw = _safe_query(col, query, min(2, col.count()))
            seen_notes: set[str] = set()
            for doc, meta in raw:
                note_id = meta.get("note_id", doc[:30])
                if note_id in seen_notes:
                    continue
                seen_notes.add(note_id)
                notes.append({
                    "text":     meta.get("text", doc),
                    "source":   "笔记",
                    "chapter":  meta.get("title", ""),
                    "chunk_id": note_id,
                    "question": doc if doc != meta.get("text") else "",
                })
    except Exception:
        pass

    return {"knowledge": knowledge, "resume": resume, "notes": notes, "retrieval_log": retrieval_log}


def retrieve(query: str, session_id: str | None = None) -> dict:
    """简单检索模式（无 graph）：召回 → 去重 → rerank。"""
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

    resume_chunks: list[str] = []
    if session_id:
        try:
            col = _get_resume_col()
            existing = col.get(where={"session_id": session_id})
            count = len(existing["ids"])
            if count > 0:
                raw = _safe_query(col, query, min(RESUME_TOP_K, count),
                                  where={"session_id": session_id})
                resume_chunks = [doc for doc, _ in raw]
        except Exception:
            pass

    note_chunks: list[str] = []
    try:
        col = _get_notes_col()
        if col.count() > 0:
            raw = _safe_query(col, query, min(2, col.count()))
            seen: set[str] = set()
            for doc, meta in raw:
                note_id = meta.get("note_id", doc[:30])
                if note_id not in seen:
                    seen.add(note_id)
                    note_chunks.append(meta.get("text", doc))
    except Exception:
        pass

    return {"knowledge": knowledge_chunks, "resume": resume_chunks, "notes": note_chunks}


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


def retrieve_graph(query: str) -> dict:
    """Graph RAG 查询包装 — 优雅降级，import 失败返回空结果。"""
    try:
        import graph_rag
        return graph_rag.retrieve_graph(query)
    except Exception:
        return {"entities": [], "relations": [], "source_chunk_ids": [], "graph_summary": ""}
