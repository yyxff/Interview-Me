"""
methods.py — 所有 RAG 检索方法实现

每个 retrieve_* 函数统一签名：(query: str, top_k: int) -> list[tuple[str, str]]
返回值：[(chunk_id, text), ...]，按相关性降序排列

特殊方法（返回 (result, decision)，由 eval.py 处理）：
  retrieve_routed(query, top_k)
  retrieve_plan_a(query, top_k, G, threshold)

方法注册表 METHODS 供 eval.py 遍历使用。
"""

import json, os, sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
RAG_DIR     = SCRIPTS_DIR.parent
BACKEND_DIR = RAG_DIR.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# 加载 .env
env_file = BACKEND_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import rag
import graph_rag as _gr

MATCH_PREFIX_LEN = 60


# ── 内部工具 ──────────────────────────────────────────────────────────────────

def _norm(t):
    return "".join(t.split())

def build_chunk_index() -> dict[str, str]:
    """chunk_id → 原文，用于评分时做文本回退匹配。"""
    idx = {}
    for f in sorted(rag.KNOWLEDGE_DIR.glob("*.chunks.json")):
        src = f.stem.replace(".chunks", "")
        for i, c in enumerate(json.loads(f.read_text("utf-8"))):
            idx[f"{src}_{i}"] = c.get("text", "")
    return idx

def _safe_bi(query, n):
    """返回 (bi_ids, cand_map)，cand_map: chunk_id → text。"""
    col    = rag._get_knowledge_col()
    bi_raw = rag._safe_query(col, query, min(n, col.count()))
    ids, cmap = [], {}
    for doc, meta in bi_raw:
        cid = meta.get("chunk_id", "")
        if cid and cid not in cmap:
            ids.append(cid); cmap[cid] = doc
    return ids, cmap

def _get_graph_result(query):
    """
    调用 graph 检索，返回 (graph_ids, text_map, path_map)。
      graph_ids: 按图谱相关性排序的 chunk_id 列表
      text_map:  chunk_id → 原文
      path_map:  chunk_id → 路径文本（"[图谱路径] A --p--> B。desc"）
    """
    gr = rag.retrieve_graph(query)
    path_map: dict[str, str] = {}
    for r in gr.get("relations", []):
        cid = r.get("source_chunk_id", "")
        if cid and cid not in path_map:
            path_map[cid] = (
                f"[图谱路径] {r['subject']} --{r['predicate']}--> {r['object']}。"
                f"{r.get('description', '')}"
            )
    raw_ids = gr.get("source_chunk_ids", [])
    fetched  = _gr.get_chunks_by_ids(raw_ids)
    text_map = {c["chunk_id"]: c.get("text", "") for c in fetched if c.get("chunk_id")}
    graph_ids = [cid for cid in raw_ids if cid in text_map]
    return graph_ids, text_map, path_map

def _weighted_rrf(bi_ids, graph_ids, k_bi=60, k_graph=20):
    """加权 RRF：k 越小权重越高，k_graph < k_bi 表示 graph 权重更高。"""
    scores: dict[str, float] = {}
    for rank, cid in enumerate(bi_ids, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_bi + rank)
    for rank, cid in enumerate(graph_ids, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_graph + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


# ── LLM 工具（HyDE / 路由） ───────────────────────────────────────────────────

_HYDE_PROMPT = """\
你是一名计算机技术专家。请根据以下问题，写一段简短的技术回答（100字以内），\
内容要像教材或技术文档里的原文，直接给出答案，不要重复问题本身。

问题：{query}

回答："""

_ROUTE_PROMPT = """\
判断这道计算机面试题的检索策略。按以下两步思考：

第一步：问题里是否有关系动词？
关系动词包括：依赖、通过、使用、由...组成、包含、导致、触发、支持、绕过、\
创建、定义、发送、避免、影响、选择、启用、基于、利用、调用、产生、需要、实现

第二步：如果有关系动词，问题问的是「A [关系动词] 的是哪个具体事物/选项/协议/字段？」\
（即答案是一个问题里未出现的具体实体）→ relational
否则（问题在解释A本身如何工作、有什么特点、是什么原理）→ factual

示例：
  "TCP Keepalive 需要设置哪个 socket 选项？"  → relational
  "QUIC 底层基于哪种传输层协议？"             → relational
  "HTTP/2 如何解决队头阻塞？"                → factual
  "TCP 三次握手的过程是什么？"               → factual

只输出一个单词：factual 或 relational

问题：{query}"""

_llm_client = None

def _get_llm():
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI
        _llm_client = OpenAI(
            api_key  = os.environ.get("LLM_API_KEY", "x"),
            base_url = os.environ.get("LLM_BASE_URL") or None,
        )
    return _llm_client

def gen_hyde(query: str) -> str:
    try:
        model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        resp  = _get_llm().chat.completions.create(
            model=model, max_tokens=200, temperature=0.3,
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        )
        return (resp.choices[0].message.content or "").strip() or query
    except Exception:
        return query

def llm_route(query: str) -> str:
    """返回 'factual' 或 'relational'，出错时 fallback 到 'factual'。"""
    try:
        model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        resp  = _get_llm().chat.completions.create(
            model=model, max_tokens=10, temperature=0.0,
            messages=[{"role": "user", "content": _ROUTE_PROMPT.format(query=query)}],
        )
        result = (resp.choices[0].message.content or "").strip().lower()
        return "relational" if "relational" in result else "factual"
    except Exception as e:
        print(f"  [route err] {e}")
        return "factual"


# ── 检索方法 ──────────────────────────────────────────────────────────────────

def retrieve_bi(query, top_k):
    """纯 bi-encoder 向量检索。"""
    try:
        ids, cmap = _safe_bi(query, top_k * 8)
        return [(cid, cmap[cid]) for cid in ids[:top_k]]
    except Exception as e:
        print(f"  [bi err] {e}"); return []

def retrieve_rrf(query, top_k):
    """标准 bi+graph RRF → cross-encoder rerank（生产基线）。"""
    try:
        gr    = rag.retrieve_graph(query)
        ids   = gr.get("source_chunk_ids", [])
        extra = _gr.get_chunks_by_ids(ids) if ids else []
        result = rag.retrieve_rich(query, extra_chunks=extra or None, top_k=top_k)
        return [(c["chunk_id"], c["text"])
                for c in result.get("knowledge", []) if c.get("chunk_id")][:top_k]
    except Exception as e:
        print(f"  [rrf err] {e}"); return []

def retrieve_graph(query, top_k):
    """纯 graph BFS 检索，不做 rerank。"""
    try:
        gr  = rag.retrieve_graph(query)
        ids = gr.get("source_chunk_ids", [])[:top_k * 2]
        cks = _gr.get_chunks_by_ids(ids)
        return [(c["chunk_id"], c.get("text", ""))
                for c in cks if c.get("chunk_id")][:top_k]
    except Exception as e:
        print(f"  [graph err] {e}"); return []

def retrieve_rrf_no_rerank(query, top_k):
    """bi + graph → 标准 RRF 融合，跳过 cross-encoder rerank。"""
    try:
        bi_ids, cand_map = _safe_bi(query, top_k * 8)
        graph_ids, text_map, _ = _get_graph_result(query)
        for cid in graph_ids:
            cand_map.setdefault(cid, text_map[cid])
        merged = rag._rrf_merge([bi_ids, graph_ids])
        return [(cid, cand_map[cid]) for cid in merged if cid in cand_map][:top_k]
    except Exception as e:
        print(f"  [rrf_nr err] {e}"); return []

def retrieve_rrf_weighted(query, top_k, k_bi=60, k_graph=20):
    """bi + graph → 加权 RRF（graph 权重更高），不做 rerank。"""
    try:
        bi_ids, cand_map = _safe_bi(query, top_k * 8)
        graph_ids, text_map, _ = _get_graph_result(query)
        for cid in graph_ids:
            cand_map.setdefault(cid, text_map[cid])
        merged = _weighted_rrf(bi_ids, graph_ids, k_bi=k_bi, k_graph=k_graph)
        return [(cid, cand_map[cid]) for cid in merged if cid in cand_map][:top_k]
    except Exception as e:
        print(f"  [rrf_w err] {e}"); return []

def retrieve_graph_rerank(query, top_k, rerank_cands=20):
    """graph BFS → cross-encoder rerank（无 bi-encoder）。"""
    try:
        gr = rag.retrieve_graph(query)
        ids = gr.get("source_chunk_ids", [])[:rerank_cands]
        chunks = _gr.get_chunks_by_ids(ids)
        rerank_input = [
            (c["text"], {"chunk_id": c["chunk_id"], "text": c["text"]})
            for c in chunks if c.get("text") and c.get("chunk_id")
        ]
        if not rerank_input:
            return []
        ranked = rag.rerank(query, rerank_input)
        return [(meta["chunk_id"], doc) for doc, meta, _ in ranked[:top_k]]
    except Exception as e:
        print(f"  [graph_rr err] {e}"); return []

def retrieve_hyde(query, top_k):
    """HyDE：LLM 先生成假设答案，再做 bi-encoder 检索。"""
    return retrieve_bi(gen_hyde(query), top_k)

def retrieve_hyde_rrf_weighted(query, top_k, k_bi=60, k_graph=20):
    """HyDE bi-encoder（假设答案）+ graph（原始 query）→ 加权 RRF，不做 rerank。"""
    try:
        bi_ids, cand_map = _safe_bi(gen_hyde(query), top_k * 8)
        graph_ids, text_map, _ = _get_graph_result(query)
        for cid in graph_ids:
            cand_map.setdefault(cid, text_map[cid])
        merged = _weighted_rrf(bi_ids, graph_ids, k_bi=k_bi, k_graph=k_graph)
        return [(cid, cand_map[cid]) for cid in merged if cid in cand_map][:top_k]
    except Exception as e:
        print(f"  [hyde_rrf_w err] {e}"); return []

def retrieve_graph_path_rerank(query, top_k, rerank_cands=20):
    """graph BFS → 路径文本前置 → cross-encoder rerank（无 bi-encoder）。"""
    try:
        graph_ids, text_map, path_map = _get_graph_result(query)
        rerank_input = []
        for cid in graph_ids[:rerank_cands]:
            raw      = text_map.get(cid, "")
            enriched = (path_map[cid] + "\n\n" + raw) if cid in path_map else raw
            rerank_input.append((enriched, {"chunk_id": cid, "text": raw}))
        if not rerank_input:
            return []
        ranked = rag.rerank(query, rerank_input)
        return [(meta["chunk_id"], meta["text"]) for _, meta, _ in ranked[:top_k]]
    except Exception as e:
        print(f"  [graph_path_rr err] {e}"); return []

def retrieve_rrf_path_rerank(query, top_k, rerank_cands=30):
    """
    ★ 生产采纳方案
    bi + graph RRF → 路径文本前置 → cross-encoder rerank。
    O Score@5=0.833 MRR=0.700 / GE Score@5=0.933 MRR=0.833 / ALL MRR=0.767
    """
    try:
        bi_ids, cand_map = _safe_bi(query, rerank_cands)
        graph_ids, text_map, path_map = _get_graph_result(query)
        for cid in graph_ids:
            cand_map.setdefault(cid, text_map[cid])
        merged = rag._rrf_merge([bi_ids, graph_ids])
        rerank_input = []
        for cid in merged[:rerank_cands]:
            raw      = cand_map.get(cid, "")
            enriched = (path_map[cid] + "\n\n" + raw) if cid in path_map else raw
            rerank_input.append((enriched, {"chunk_id": cid, "text": raw}))
        if not rerank_input:
            return []
        ranked = rag.rerank(query, rerank_input)
        return [(meta["chunk_id"], meta["text"]) for _, meta, _ in ranked[:top_k]]
    except Exception as e:
        print(f"  [rrf_path_rr err] {e}"); return []

def retrieve_routed(query, top_k):
    """LLM 路由：factual → rrf，relational → graph。返回 (结果, 路由决策)。"""
    decision = llm_route(query)
    if decision == "relational":
        return retrieve_graph(query, top_k), "graph"
    return retrieve_rrf(query, top_k), "rrf"

def _query_top_entities(query: str, top_k: int = 3) -> list[tuple[float, str]]:
    ent_col = _gr._get_entities_col()
    if ent_col.count() == 0:
        return []
    try:
        r = ent_col.query(query_texts=[query],
                          n_results=min(top_k, ent_col.count()),
                          include=["metadatas", "distances"])
        return [(d, m["name"])
                for d, m in zip(r["distances"][0], r["metadatas"][0])]
    except Exception:
        return []

def route_plan_a(query: str, G, threshold: float) -> str:
    """图中有近邻实体（距离 < threshold 且有出边）→ graph；否则 → rrf。"""
    for dist, name in _query_top_entities(query, top_k=3):
        if dist < threshold and G.has_node(name) and G.degree(name) > 0:
            return "graph"
    return "rrf"

def retrieve_plan_a(query: str, top_k: int, G, threshold: float):
    """实体距离路由。返回 (结果, 路由决策)。"""
    decision = route_plan_a(query, G, threshold)
    if decision == "graph":
        return retrieve_graph(query, top_k), "graph"
    return retrieve_rrf(query, top_k), "rrf"


# ── 方法注册表 ────────────────────────────────────────────────────────────────
# 标准方法：签名 (query, top_k) → list[tuple[str, str]]
# 路由方法：签名 (query, top_k) → (list, str)，由 eval.py 特殊处理
# plan_a：需要额外参数 G 和 threshold，由 eval.py 特殊处理

METHODS: dict = {
    "bi":             retrieve_bi,
    "rrf":            retrieve_rrf,             # baseline
    "graph":          retrieve_graph,
    "rrf_nr":         retrieve_rrf_no_rerank,
    "rrf_w":          retrieve_rrf_weighted,
    "graph_rr":       retrieve_graph_rerank,
    "hyde":           retrieve_hyde,
    "hyde_rrf_w":     retrieve_hyde_rrf_weighted,
    "graph_path_rr":  retrieve_graph_path_rerank,
    "rrf_path_rr":    retrieve_rrf_path_rerank,  # ★ production
    "routed":         retrieve_routed,           # returns (result, decision)
    # "plan_a" 不在此表，由 eval.py 单独处理（需要 G 和 threshold）
}

ROUTING_METHODS = {"routed", "plan_a"}  # 这些方法返回 (result, decision)
