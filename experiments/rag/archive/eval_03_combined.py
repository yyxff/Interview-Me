"""
eval_combined.py — O + GE 1:1 混合评测

对比方法：bi / hyde / rrf_w / hyde+rrf_w / graph
按策略分组展示，最后汇总综合指标。

运行：
  /opt/homebrew/Caskroom/miniconda/base/envs/interview-me/bin/python eval_combined.py
"""

import json, os, random, sys, time
from pathlib import Path
from datetime import datetime

SCRIPTS_DIR  = Path(__file__).parent
RAG_DIR      = SCRIPTS_DIR.parent
BACKEND_DIR  = RAG_DIR.parent.parent / "backend"
TESTSETS_DIR = RAG_DIR / "testsets"
EVAL_LOG_DIR = RAG_DIR / "logs"
sys.path.insert(0, str(BACKEND_DIR))

# ── 加载 .env ─────────────────────────────────────────────────────────────────
env_file = BACKEND_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import rag, graph_rag as _gr
import networkx as nx

MATCH_PREFIX_LEN = 60

# ── 归一化测试集 ──────────────────────────────────────────────────────────────

def load_o_testset(path: Path, n: int, seed: int) -> list[dict]:
    data = json.loads(path.read_text("utf-8"))
    qs   = data.get("questions", data) if isinstance(data, dict) else data
    random.seed(seed)
    sample = random.sample(qs, min(n, len(qs)))
    return [{"question": q["question"],
             "gt_id":    q["chunk_id"],
             "strategy": "O"} for q in sample]

def load_ge_testset(path: Path, n: int, seed: int) -> list[dict]:
    data = json.loads(path.read_text("utf-8"))
    qs   = data.get("questions", [])
    random.seed(seed + 1)
    sample = random.sample(qs, min(n, len(qs)))
    return [{"question": q["question"],
             "gt_id":    q["ground_truth_chunk_ids"][0],
             "strategy": "GE"} for q in sample]

# ── chunk 原文索引 ────────────────────────────────────────────────────────────

def build_chunk_index() -> dict[str, str]:
    idx = {}
    for f in sorted(rag.KNOWLEDGE_DIR.glob("*.chunks.json")):
        src = f.stem.replace(".chunks", "")
        for i, c in enumerate(json.loads(f.read_text("utf-8"))):
            idx[f"{src}_{i}"] = c.get("text", "")
    return idx

# ── 检索函数 ──────────────────────────────────────────────────────────────────

def _norm(t): return "".join(t.split())

def _safe_bi(query, top_k):
    col    = rag._get_knowledge_col()
    bi_raw = rag._safe_query(col, query, min(top_k * 8, col.count()))
    ids, cmap = [], {}
    for doc, meta in bi_raw:
        cid = meta.get("chunk_id", "")
        if cid and cid not in cmap:
            ids.append(cid); cmap[cid] = doc
    return ids, cmap

def retrieve_bi(query, top_k):
    try:
        ids, cmap = _safe_bi(query, top_k)
        return [(cid, cmap[cid]) for cid in ids[:top_k]]
    except Exception as e:
        print(f"  [bi err] {e}"); return []

# HyDE
_HYDE_PROMPT = """\
你是一名计算机技术专家。请根据以下问题，写一段简短的技术回答（100字以内），\
内容要像教材或技术文档里的原文，直接给出答案，不要重复问题本身。

问题：{query}

回答："""
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
            messages=[{"role": "user",
                        "content": _HYDE_PROMPT.format(query=query)}],
        )
        return (resp.choices[0].message.content or "").strip() or query
    except Exception:
        return query

def retrieve_hyde(query, top_k):
    return retrieve_bi(gen_hyde(query), top_k)

# ── LLM 路由 ──────────────────────────────────────────────────────────────────

_ROUTE_PROMPT = """\
判断这道计算机面试题的检索策略。按以下两步思考：

第一步：问题里是否有关系动词？
关系动词包括：依赖、通过、使用、由...组成、包含、导致、触发、支持、绕过、\
创建、定义、发送、避免、影响、选择、启用、基于、利用、调用、产生、需要、实现

第二步：如果有关系动词，问题问的是「A [关系动词] 的是哪个具体事物/选项/协议/字段？」\
（即答案是一个问题里未出现的具体实体）→ relational
否则（问题在解释A本身如何工作、有什么特点、是什么原理）→ factual

示例：
  "TCP Keepalive 需要设置哪个 socket 选项？"   → 有关系词"需要"，答案是具体选项SO_KEEPALIVE → relational
  "协商缓存触发时依赖哪个响应头字段？"          → 有关系词"依赖"，答案是具体字段ETag → relational
  "QUIC 底层基于哪种传输层协议？"              → 有关系词"基于"，答案是具体协议UDP → relational
  "逻辑块由什么基本单元组成？"                 → 有关系词"由...组成"，答案是具体单元扇区 → relational
  "Nagle算法会导致什么综合症？"               → 有关系词"导致"，答案是具体综合症 → relational
  "HTTP/2 如何解决队头阻塞？"                 → 有关系词"解决"，但答案是解释HTTP/2自身的机制，不是另一实体 → factual
  "TCP 三次握手的过程是什么？"                → 无明显关系动词，答案解释握手本身 → factual
  "什么是 TLS 握手？"                        → 无关系动词，答案解释TLS本身 → factual
  "进程和线程有什么区别？"                    → 答案在解释两者本身的属性 → factual
  "InnoDB 引擎有哪些特点？"                  → 答案描述InnoDB本身 → factual

只输出一个单词：factual 或 relational

问题：{query}"""

def llm_route(query: str) -> str:
    """返回 'factual' 或 'relational'，出错时 fallback 到 'factual'。"""
    try:
        model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        resp  = _get_llm().chat.completions.create(
            model=model, max_tokens=10, temperature=0.0,
            messages=[{"role": "user",
                       "content": _ROUTE_PROMPT.format(query=query)}],
        )
        result = (resp.choices[0].message.content or "").strip().lower()
        return "relational" if "relational" in result else "factual"
    except Exception as e:
        print(f"  [route err] {e}")
        return "factual"

def retrieve_routed(query, top_k):
    """LLM 路由：factual → rrf，relational → graph。返回 (结果, 路由决策)。"""
    decision = llm_route(query)
    if decision == "relational":
        return retrieve_graph(query, top_k), "graph"
    else:
        return retrieve_rrf(query, top_k), "rrf"

# ── Plan A：实体距离路由 ──────────────────────────────────────────────────────

def _query_top_entities(query: str, top_k: int = 3) -> list[tuple[float, str]]:
    ent_col = _gr._get_entities_col()
    if ent_col.count() == 0:
        return []
    try:
        r = ent_col.query(query_texts=[query],
                          n_results=min(top_k, ent_col.count()),
                          include=["metadatas", "distances"])
        return [(d, m["name"]) for d, m in zip(r["distances"][0], r["metadatas"][0])]
    except Exception:
        return []

def route_plan_a(query: str, G, threshold: float) -> str:
    """图里有近邻实体（低距离 + 有出边）→ graph；否则 → rrf。"""
    ents = _query_top_entities(query, top_k=3)
    if not ents:
        return "rrf"
    # 取 top-3 中距离 < threshold 且在图中有边的实体
    for dist, name in ents:
        if dist < threshold and G.has_node(name) and G.degree(name) > 0:
            return "graph"
    return "rrf"

def retrieve_plan_a(query: str, top_k: int, G, threshold: float):
    decision = route_plan_a(query, G, threshold)
    if decision == "graph":
        return retrieve_graph(query, top_k), "graph"
    return retrieve_rrf(query, top_k), "rrf"


def retrieve_rrf(query, top_k):
    """标准 RRF+rerank：bi + graph → RRF → cross-encoder rerank。"""
    try:
        gr    = rag.retrieve_graph(query)
        ids   = gr.get("source_chunk_ids", [])
        extra = _gr.get_chunks_by_ids(ids) if ids else []
        result = rag.retrieve_rich(query, extra_chunks=extra or None, top_k=top_k)
        return [(c["chunk_id"], c["text"]) for c in result.get("knowledge", []) if c.get("chunk_id")][:top_k]
    except Exception as e:
        print(f"  [rrf err] {e}"); return []

def retrieve_rrf_path_rerank(query, top_k, rerank_cands=30):
    """
    方案一（完整版）：bi-encoder + graph RRF → 证据链文本前置 → cross-encoder rerank。
    bi 和 graph 都参与 RRF 融合，有路径的 chunk 在 rerank 时额外看到推理链。
    """
    try:
        # 1. bi-encoder 候选
        col    = rag._get_knowledge_col()
        bi_raw = rag._safe_query(col, query, min(rerank_cands, col.count()))
        bi_ids, cand_map = [], {}
        for doc, meta in bi_raw:
            cid = meta.get("chunk_id", "")
            if cid and cid not in cand_map:
                bi_ids.append(cid); cand_map[cid] = doc

        # 2. graph 候选 + 路径映射
        gr = rag.retrieve_graph(query)
        path_map: dict[str, str] = {}
        for r in gr.get("relations", []):
            cid = r.get("source_chunk_id", "")
            if cid and cid not in path_map:
                path_map[cid] = (
                    f"[图谱路径] {r['subject']} --{r['predicate']}--> {r['object']}。"
                    f"{r.get('description', '')}"
                )
        graph_raw_ids = gr.get("source_chunk_ids", [])
        fetched   = _gr.get_chunks_by_ids(graph_raw_ids)
        text_by_id = {c["chunk_id"]: c.get("text", "") for c in fetched if c.get("chunk_id")}
        graph_ids = []
        for cid in graph_raw_ids:
            if cid in text_by_id:
                graph_ids.append(cid)
                cand_map.setdefault(cid, text_by_id[cid])

        # 3. RRF 融合
        merged = rag._rrf_merge([bi_ids, graph_ids])

        # 4. 取 top rerank_cands，有路径的 chunk 前置路径文本
        rerank_input = []
        for cid in merged[:rerank_cands]:
            if cid not in cand_map:
                continue
            raw_text = cand_map[cid]
            enriched = (path_map[cid] + "\n\n" + raw_text) if cid in path_map else raw_text
            rerank_input.append((enriched, {"chunk_id": cid, "text": raw_text}))

        if not rerank_input:
            return []

        ranked = rag.rerank(query, rerank_input)
        return [(meta["chunk_id"], meta["text"]) for _, meta, _ in ranked[:top_k]]
    except Exception as e:
        print(f"  [rrf_path_rr err] {e}"); return []


def retrieve_graph_path_rerank(query, top_k, rerank_cands=20):
    """
    方案一：证据链 Reranking。
    Graph 检索 → 把边的路径文本（A --predicate--> B + description）前置到 chunk 文本
    → cross-encoder rerank。Reranker 能看到完整推理链，不再只看 B 的孤立内容。
    """
    try:
        gr = rag.retrieve_graph(query)

        # 建立 chunk_id → 路径文本 的映射（一个 chunk 可能对应多条关系，取最相关的第一条）
        path_map: dict[str, str] = {}
        for r in gr.get("relations", []):
            cid = r.get("source_chunk_id", "")
            if cid and cid not in path_map:
                path_map[cid] = (
                    f"[图谱路径] {r['subject']} --{r['predicate']}--> {r['object']}。"
                    f"{r.get('description', '')}"
                )

        # 按原始 graph 顺序取 top-N 候选
        raw_ids  = gr.get("source_chunk_ids", [])
        fetched  = _gr.get_chunks_by_ids(raw_ids[:rerank_cands])
        text_map = {c["chunk_id"]: c.get("text", "") for c in fetched if c.get("chunk_id")}

        # 构造 rerank 输入：有路径的 chunk 前置路径文本
        rerank_input = []
        for cid in raw_ids[:rerank_cands]:
            if cid not in text_map:
                continue
            raw_text = text_map[cid]
            enriched = (path_map[cid] + "\n\n" + raw_text) if cid in path_map else raw_text
            rerank_input.append((enriched, {"chunk_id": cid, "text": raw_text}))

        if not rerank_input:
            return []

        ranked = rag.rerank(query, rerank_input)
        return [(meta["chunk_id"], meta["text"]) for _, meta, _ in ranked[:top_k]]
    except Exception as e:
        print(f"  [path_rr err] {e}"); return []


def retrieve_graph(query, top_k):
    try:
        gr  = rag.retrieve_graph(query)
        ids = gr.get("source_chunk_ids", [])[:top_k * 2]
        cks = _gr.get_chunks_by_ids(ids)
        return [(c["chunk_id"], c.get("text","")) for c in cks if c.get("chunk_id")][:top_k]
    except Exception as e:
        print(f"  [gr err] {e}"); return []

# ── 评分 ──────────────────────────────────────────────────────────────────────

def metrics(retrieved, gt_id, chunk_idx):
    rank = None
    gt_norm = _norm(chunk_idx.get(gt_id, ""))[:MATCH_PREFIX_LEN]
    for i, (cid, txt) in enumerate(retrieved, 1):
        if cid == gt_id or (gt_norm and gt_norm in _norm(txt)):
            rank = i; break
    return {"hit1": int(rank==1), "hit3": int(rank is not None and rank<=3),
            "hit5": int(rank is not None and rank<=5),
            "mrr":  (1/rank if rank else 0.0)}

def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0.0

# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n",    type=int, default=30, help="每个策略的题数")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--o-testset",  default=str(TESTSETS_DIR/"testset_O_20260331_224445.json"))
    p.add_argument("--ge-testset", default=str(TESTSETS_DIR/"testset_GE_20260402_185222.json"))
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.18, 0.22, 0.26, 0.30])
    args = p.parse_args()

    print("[eval] 加载模型与数据...")
    chunk_idx = build_chunk_index()
    print(f"  chunks: {len(chunk_idx)}")

    print("[eval] 加载知识图谱...")
    G = _gr._load_all_graphs_into_nx()
    print(f"  图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    o_qs  = load_o_testset(Path(args.o_testset),  args.n, args.seed)
    ge_qs = load_ge_testset(Path(args.ge_testset), args.n, args.seed)
    questions = o_qs + ge_qs
    random.seed(args.seed)
    random.shuffle(questions)
    print(f"  O={len(o_qs)}  GE={len(ge_qs)}  total={len(questions)}\n")

    BASE_METHODS = ("bi", "rrf", "graph", "graph_path_rr", "rrf_path_rr")

    # 先跑 bi/rrf/graph 基线（每题只跑一次）
    base_cache: list[dict] = []   # {query, gt_id, strat, bi, rrf, graph}

    all_m   = {m: {k: [] for k in ("hit1","hit3","hit5","mrr")} for m in BASE_METHODS}
    strat_m = {s: {m: {k: [] for k in ("hit1","hit3","hit5","mrr")}
                   for m in BASE_METHODS} for s in ("O","GE")}

    t0 = time.time()
    print(f"\n[eval] 基线检索 ({len(questions)} 题)...")
    for i, q in enumerate(questions):
        query, gt_id, strat = q["question"], q["gt_id"], q["strategy"]
        bi_res       = retrieve_bi(query, args.topk)
        rrf_res      = retrieve_rrf(query, args.topk)
        graph_res    = retrieve_graph(query, args.topk)
        gpr_res      = retrieve_graph_path_rerank(query, args.topk)
        rpr_res      = retrieve_rrf_path_rerank(query, args.topk)
        base_cache.append({"query": query, "gt_id": gt_id, "strat": strat,
                            "bi": bi_res, "rrf": rrf_res, "graph": graph_res,
                            "graph_path_rr": gpr_res, "rrf_path_rr": rpr_res})
        res = {"bi": bi_res, "rrf": rrf_res, "graph": graph_res,
               "graph_path_rr": gpr_res, "rrf_path_rr": rpr_res}
        m_  = {m: metrics(res[m], gt_id, chunk_idx) for m in BASE_METHODS}
        tag = lambda m: "✓" if m_[m]["hit5"] else "✗"
        print(f"  [{i+1:2d}/{len(questions)}][{strat}] bi:{tag('bi')} rrf:{tag('rrf')} "
              f"graph:{tag('graph')} g_pr:{tag('graph_path_rr')} r_pr:{tag('rrf_path_rr')}")
        for m in BASE_METHODS:
            for k in ("hit1","hit3","hit5","mrr"):
                all_m[m][k].append(m_[m][k])
                strat_m[strat][m][k].append(m_[m][k])

    # Plan A：多阈值扫描（复用 base_cache，只新增路由逻辑）
    print(f"\n[eval] Plan A 阈值扫描: {args.thresholds}")
    plan_a_results = {}   # threshold → {O, GE, ALL metrics + routing acc}

    for thr in args.thresholds:
        a_all_m   = {k: [] for k in ("hit1","hit3","hit5","mrr")}
        a_strat_m = {"O": {k: [] for k in ("hit1","hit3","hit5","mrr")},
                     "GE": {k: [] for k in ("hit1","hit3","hit5","mrr")}}
        a_route   = {"O": [], "GE": []}   # 1=correct, 0=wrong

        for item in base_cache:
            query, gt_id, strat = item["query"], item["gt_id"], item["strat"]
            decision = route_plan_a(query, G, thr)
            rt_res   = item["rrf"] if decision == "rrf" else item["graph"]
            m = metrics(rt_res, gt_id, chunk_idx)
            expected = "rrf" if strat == "O" else "graph"
            a_route[strat].append(int(decision == expected))
            for k in ("hit1","hit3","hit5","mrr"):
                a_all_m[k].append(m[k])
                a_strat_m[strat][k].append(m[k])

        plan_a_results[thr] = {
            "O":  {k: avg(a_strat_m["O"][k])  for k in ("hit1","hit3","hit5","mrr")},
            "GE": {k: avg(a_strat_m["GE"][k]) for k in ("hit1","hit3","hit5","mrr")},
            "ALL":{k: avg(a_all_m[k])          for k in ("hit1","hit3","hit5","mrr")},
            "route_O":  avg(a_route["O"]),
            "route_GE": avg(a_route["GE"]),
            "route_all":avg(a_route["O"] + a_route["GE"]),
        }

    elapsed = round(time.time()-t0, 1)

    # 基线表格
    def print_table(title, mdict, methods, n):
        W = 9
        print(f"\n{'='*72}")
        print(f"  {title}  (n={n}, top_k={args.topk})")
        print(f"{'─'*72}")
        print(f"  {'指标':<10}" + "".join(f" {m:>{W}}" for m in methods))
        print(f"{'─'*72}")
        for metric in ("hit1","hit3","hit5","mrr"):
            label = {"hit1":"Hit@1","hit3":"Hit@3","hit5":"Score@5","mrr":"MRR"}[metric]
            vals  = {m: avg(mdict[m][metric]) for m in methods}
            best  = max(vals.values())
            marks = {m: "◀" if abs(vals[m]-best)<1e-6 else " " for m in vals}
            print(f"  {label:<10}" +
                  "".join(f" {vals[m]:>{W}.3f}{marks[m]}" for m in methods))
        print(f"{'='*72}")

    print_table("O 策略",  strat_m["O"],  BASE_METHODS, len(o_qs))
    print_table("GE 策略", strat_m["GE"], BASE_METHODS, len(ge_qs))
    print_table("综合 O+GE 1:1（基线）", all_m, BASE_METHODS, len(questions))

    # Plan A 阈值扫描汇总
    print(f"\n{'='*72}")
    print(f"  Plan A 实体距离路由  阈值扫描  (bi→rrf, graph→graph)")
    print(f"{'─'*72}")
    print(f"  {'thr':>6}  {'rAcc-O':>8} {'rAcc-GE':>9} {'rAcc':>6}"
          f"  {'Score@5-O':>10} {'Score@5-GE':>11} {'MRR-ALL':>8}")
    print(f"{'─'*72}")
    for thr, r in plan_a_results.items():
        print(f"  {thr:>6.2f}  {r['route_O']:>8.1%} {r['route_GE']:>9.1%} {r['route_all']:>6.1%}"
              f"  {r['O']['hit5']:>10.3f} {r['GE']['hit5']:>11.3f} {r['ALL']['mrr']:>8.3f}")
    print(f"{'─'*72}")
    # 参考基线
    def b(s, k): return avg(strat_m[s]["rrf"][k]) if s != "ALL" else avg(all_m["rrf"][k])
    print(f"  {'rrf':>6}  {'---':>8} {'---':>9} {'---':>6}"
          f"  {b('O','hit5'):>10.3f} {b('GE','hit5'):>11.3f} {avg(all_m['rrf']['mrr']):>8.3f}  ← 基线rrf")
    print(f"{'='*72}")
    print(f"\n  耗时: {elapsed}s")

    # 写日志
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config":    {"n_each": args.n, "top_k": args.topk, "seed": args.seed,
                      "thresholds": args.thresholds},
        "results": {
            "O":   {m: {k: avg(strat_m["O"][m][k])  for k in ("hit1","hit3","hit5","mrr")} for m in BASE_METHODS},
            "GE":  {m: {k: avg(strat_m["GE"][m][k]) for k in ("hit1","hit3","hit5","mrr")} for m in BASE_METHODS},
            "ALL": {m: {k: avg(all_m[m][k])          for k in ("hit1","hit3","hit5","mrr")} for m in BASE_METHODS},
        },
        "plan_a": plan_a_results,
    }
    out = EVAL_LOG_DIR / f"combined_{ts}.json"
    out.write_text(json.dumps(log, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n[eval] 日志: {out.name}")

if __name__ == "__main__":
    main()
