"""双路召回：向量（bi-encoder）+ 图谱 → RRF 融合 → cross-encoder rerank。

流水线：
  ① _be_search      bi-encoder 向量搜索 → 去重候选池
  ② _inject_graph   图谱 extra_chunks 注入候选池
  ③ _fuse_rrf       RRF 融合两路排名（无图路时直接透传）
  ④ _run_rerank     图谱路径信息丰富文本 → cross-encoder rerank
  ⑤ _collect_results 按分数阈值过滤 → 构建知识条目 + 检索日志
"""
from __future__ import annotations

from .client import (
    is_available, KNOWLEDGE_TOP_K, KNOWLEDGE_RETURN_K, QA_PER_CHUNK,
    _get_knowledge_col, rerank,
)
from .retrieval import _safe_query, _dedupe_chunks, _fetch_resume, _fetch_notes

RERANK_SCORE_THRESHOLD = 0.5


# ── 流水线各步骤 ──────────────────────────────────────────────────────────────

def _be_search(
    col, query: str, k: int
) -> tuple[dict[str, tuple], dict[str, float], list[str]]:
    """① Bi-encoder 向量搜索 → 去重 → 候选池。

    Returns:
        cand_map  {chunk_id: (text, meta)}  所有候选
        dist_map  {chunk_id: distance}      用于日志
        be_rank   [chunk_id, ...]           向量路排名
    """
    n_candidates  = min(k * QA_PER_CHUNK * 2, col.count())
    raw_with_dist = _safe_query(col, query, n_candidates, return_distances=True)
    raw           = [(d, m) for d, m, _ in raw_with_dist]
    candidates    = _dedupe_chunks(raw, limit=k * 3)
    dist_map      = {m.get("chunk_id", ""): dist for d, m, dist in raw_with_dist}
    cand_map      = {
        m.get("chunk_id", ""): (m.get("text", doc), m)
        for doc, m in candidates if m.get("chunk_id")
    }
    return cand_map, dist_map, list(cand_map.keys())


def _inject_graph(
    cand_map: dict, extra_chunks: list[dict], k: int
) -> tuple[dict[str, dict], list[str]]:
    """② 将图谱检索结果注入候选池，补充向量路未命中的 chunk。

    Returns:
        extra_map  {chunk_id: chunk}  图谱来源的 chunk（用于标记 via_graph）
        graph_rank [chunk_id, ...]    图路排名（限制在 k*3 以内，避免图路压制向量路）
    """
    extra_map:  dict[str, dict] = {}
    graph_rank: list[str]       = []
    for c in extra_chunks:
        cid = c.get("chunk_id", "")
        if not cid:
            continue
        extra_map[cid] = c
        graph_rank.append(cid)
        if cid not in cand_map:  # 图路独有，补充进候选池
            cand_map[cid] = (c["text"], {
                "chunk_id": cid, "text": c["text"],
                "source":   c.get("source", ""), "path":    c.get("path", ""),
                "chapter":  c.get("chapter", ""), "question": "",
            })
    return extra_map, graph_rank[:k * 3]


def _fuse_rrf(
    be_rank: list[str], graph_rank: list[str]
) -> tuple[list[str], dict[str, float]]:
    """③ RRF 融合向量路和图路排名；无图路时直接透传向量路。

    Returns:
        merged_ids  融合后的 chunk_id 列表（降序）
        rrf_scores  {chunk_id: rrf_score}（无图路时为空 dict）
    """
    if not graph_rank:
        return be_rank, {}
    rrf_scores: dict[str, float] = {}
    for ranked_list in [be_rank, graph_rank]:
        for rank, cid in enumerate(ranked_list, start=1):
            if cid:
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank)
    merged_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)
    return merged_ids, rrf_scores


def _run_rerank(
    query: str,
    merged_ids: list[str],
    cand_map: dict,
    path_map: dict[str, str] | None,
    k: int,
) -> tuple[list, list[str]]:
    """④ Cross-encoder rerank。

    图谱提供的 path_map 会前置拼接到 chunk 文本，给 reranker 更多上下文。

    Returns:
        ranked      rerank 结果 [(doc, meta, score), ...]
        rerank_cids 进入 rerank 窗口的 chunk_id（用于日志）
    """
    rerank_limit = max(k * 4, 15)
    rerank_cids  = [cid for cid in merged_ids if cid in cand_map][:rerank_limit]
    rerank_inputs = []
    for cid in rerank_cids:
        raw_text, meta = cand_map[cid]
        # 如果图谱提供了路径摘要（subject→predicate→object），前置作为语义提示
        enriched = path_map[cid] + "\n\n" + raw_text if (path_map and cid in path_map) else raw_text
        rerank_inputs.append((enriched, meta))
    return rerank(query, rerank_inputs), rerank_cids


def _collect_results(
    ranked:     list,
    rk:         int,
    dist_map:   dict[str, float],
    rrf_scores: dict[str, float],
    extra_map:  dict[str, dict],
    be_rank:    list[str],
) -> tuple[list[dict], list[dict]]:
    """⑤ 按 rerank 分数阈值过滤，构建最终知识条目和检索日志。"""
    be_cid_set = set(be_rank)
    knowledge: list[dict] = []
    log:       list[dict] = []
    for doc, meta, score in ranked[:rk]:
        if score < RERANK_SCORE_THRESHOLD:
            break
        cid = meta.get("chunk_id", "")
        knowledge.append({
            "text":      meta.get("text", doc),
            "source":    meta.get("source", ""),
            "path":      meta.get("path", ""),
            "chapter":   meta.get("chapter", meta.get("path", "")),
            "chunk_id":  cid,
            "question":  meta.get("question", ""),
            "via_graph": cid in extra_map,
        })
        log.append({
            "chunk_id":     cid,
            "source":       meta.get("source", ""),
            "bi_dist":      round(dist_map.get(cid, -1), 4),
            "rrf_score":    round(rrf_scores.get(cid, 0.0), 6),
            "rerank_score": round(score, 4),
            "via_graph":    cid in extra_map,
            "graph_only":   cid in extra_map and cid not in be_cid_set,
            "question":     meta.get("question", "")[:60],
        })
    return knowledge, log


# ── 主入口 ────────────────────────────────────────────────────────────────────

def retrieve_rich(
    query: str,
    session_id: str | None = None,
    extra_chunks: list[dict] | None = None,
    top_k: int | None = None,
    return_k: int | None = None,
    path_map: dict[str, str] | None = None,
) -> dict:
    """精准模式检索：bi-encoder + 图谱 → RRF → rerank → top-K。

    Args:
        query:        检索问题
        session_id:   用于召回简历片段
        extra_chunks: 图谱检索已取出的 chunk，直接注入候选池
        top_k:        候选池深度，影响各阶段窗口大小
        return_k:     最终返回给 LLM 的 chunk 数
        path_map:     {chunk_id: 图谱路径摘要}，拼接到 chunk 文本供 reranker 参考
    """
    if not is_available():
        return {"knowledge": [], "resume": [], "notes": [], "retrieval_log": []}

    k  = top_k    if top_k    is not None else KNOWLEDGE_TOP_K
    rk = return_k if return_k is not None else KNOWLEDGE_RETURN_K

    knowledge:    list[dict] = []
    retrieval_log: list[dict] = []

    try:
        col = _get_knowledge_col()
        if col.count() > 0:
            # ① 向量搜索，建候选池
            cand_map, dist_map, be_rank = _be_search(col, query, k)
            retrieval_log.append({
                "_be_stage":  True,
                "candidates": [(cid, round(dist_map.get(cid, -1), 4)) for cid in be_rank],
            })

            # ② 注入图谱 extra_chunks
            extra_map, graph_rank = _inject_graph(cand_map, extra_chunks or [], k)
            be_only  = len(be_rank)
            gph_only = sum(1 for cid in graph_rank if cid not in set(be_rank))
            overlap  = sum(1 for cid in graph_rank if cid in set(be_rank))

            # ③ RRF 融合（有图路则融合，无图路则透传）
            merged_ids, rrf_scores = _fuse_rrf(be_rank, graph_rank)
            retrieval_log.append({
                "_rrf_stage": True,
                "candidates": [(cid, round(rrf_scores.get(cid, 0), 6)) for cid in merged_ids],
            })

            # ④ Cross-encoder rerank
            ranked, rerank_cids = _run_rerank(query, merged_ids, cand_map, path_map, k)

            # ⑤ 阈值过滤，构建结果
            knowledge, result_log = _collect_results(
                ranked, rk, dist_map, rrf_scores, extra_map, be_rank
            )
            retrieval_log.extend(result_log)
            retrieval_log.append({
                "_summary":      True,
                "be_candidates": be_only,      "graph_extra":  len(graph_rank),
                "graph_new":     gph_only,     "graph_overlap": overlap,
                "rerank_input":  len(rerank_cids), "final_output": len(knowledge),
            })
    except Exception:
        pass

    return {
        "knowledge":    knowledge,
        "resume":       _fetch_resume(query, session_id) if session_id else [],
        "notes":        _fetch_notes(query),
        "retrieval_log": retrieval_log,
    }
