"""
eval_graph_edge.py — 图谱边关系测试集（GE 策略）

测试集设计：
  - 从图谱中选 A --[predicate]--> B 的边
  - 用 LLM 生成自然语言问题："A [predicate] 什么？"（问题只出现 A，不出现 B）
  - Ground truth = B 的 chunk（answer 在 B 的 chunk 里）
  - 此时：
      bi-encoder：只能靠 A 的语义检索，大概率找不到 B 的 chunk
      graph-only：BFS 从 A 出发一跳找到 B 的 chunk
      graph+rerank：BFS 候选 + cross-encoder 按 query↔chunk 重排

运行：
  /opt/homebrew/Caskroom/miniconda/base/envs/interview-me/bin/python eval_graph_edge.py --n 30 --top-k 5

  强制重新生成问题（忽略缓存）：
  --regen
"""

import argparse, asyncio, json, os, random, sys, time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR  = Path(__file__).parent
RAG_DIR      = SCRIPTS_DIR.parent
BACKEND_DIR  = RAG_DIR.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

TESTSETS_DIR = RAG_DIR / "testsets"
EVAL_LOG_DIR = RAG_DIR / "logs"
CACHE_DIR    = EVAL_LOG_DIR / "routing_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GE_CACHE = CACHE_DIR / "ge_questions.json"

import rag
import graph_rag as _gr

MATCH_PREFIX_LEN = 60

# ── chunk 原文索引 ────────────────────────────────────────────────────────────

def build_chunk_index() -> dict[str, str]:
    idx = {}
    for f in sorted(rag.KNOWLEDGE_DIR.glob("*.chunks.json")):
        src = f.stem.replace(".chunks", "")
        for i, c in enumerate(json.loads(f.read_text("utf-8"))):
            idx[f"{src}_{i}"] = c.get("text", "")
    return idx

# ── 图谱边过滤 ────────────────────────────────────────────────────────────────

BAD_CHUNK_KWS = ["一、前言", "适合什么群体", "要怎么阅读", "质量如何"]

def collect_candidate_edges(chunk_index: dict) -> list[dict]:
    """从所有 .graph.json 中收集高质量跨-chunk 边。"""
    edges = []
    for gf in sorted((_gr.GRAPH_DIR).glob("*.graph.json")):
        g = json.loads(gf.read_text("utf-8"))
        nodes = g.get("nodes", {})
        for e in g.get("edges", []):
            subj, obj, pred = e["subject"], e["object"], e["predicate"]
            rel_desc = e.get("description", "")
            if not rel_desc or len(rel_desc) < 15 or subj == obj:
                continue
            # GT = 边自身的 source_chunk_id（提取这条关系的 chunk）
            # 这个 chunk 才是 A→B 关系的实际来源，包含 B 的真实内容
            # 而不是 obj_cids[0]（B 实体的第一个出现 chunk，可能只是路人甲）
            src_cid = e.get("source_chunk_id", "")
            if not src_cid:
                continue
            src_text = chunk_index.get(src_cid, "")
            if len(src_text) < 150:
                continue
            if any(kw in src_text[:200] for kw in BAD_CHUNK_KWS):
                continue
            obj_node  = nodes.get(obj,  {})
            subj_node = nodes.get(subj, {})
            edges.append({
                "subject":   subj,
                "predicate": pred,
                "object":    obj,
                "obj_chunk": src_cid,   # GT = 边的来源 chunk
                "subj_desc": subj_node.get("description", "")[:80],
                "obj_desc":  obj_node.get("description",  "")[:80],
                "rel_desc":  rel_desc,
            })
    return edges

# ── LLM 问题生成 ──────────────────────────────────────────────────────────────

_GEN_PROMPT = """\
你是一个面试题出题助手。根据以下知识图谱中的一条关系，出一道面试问题。

关系信息：
  主体（A）：{subject}（{subj_desc}）
  谓词：{predicate}
  客体（B）：{object}（{obj_desc}）
  关系描述：{rel_desc}

要求：
1. 问题必须提到 A（主体），并体现"[predicate]"这个关系或动作
2. 问题**绝对不能**出现 B（客体）的名称或明显暗示 B 的词语
3. 问题要自然，像面试官会问的技术问题
4. 20-50 字，以"？"结尾
5. 只输出问题本身，不要任何前缀或解释

示例（仅格式参考，不要照抄）：
  A=HTTP/2, predicate=解决, B=队头阻塞 → "HTTP/2 引入了哪些机制来解决 HTTP/1.1 中严重影响性能的传输问题？"
  A=QUIC,   predicate=依赖, B=UDP      → "QUIC 协议在传输层选择了哪种底层协议作为基础？"

现在请出题："""


async def _gen_one(edge: dict, client, model: str) -> str:
    prompt = _GEN_PROMPT.format(
        subject   = edge["subject"],
        subj_desc = edge["subj_desc"],
        predicate = edge["predicate"],
        object    = edge["object"],
        obj_desc  = edge["obj_desc"],
        rel_desc  = edge["rel_desc"],
    )
    try:
        from openai import AsyncOpenAI
        resp = await client.chat.completions.create(
            model=model, max_tokens=800, temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        # Gemini 2.5 Flash 思考模型：content 可能为 None（token 耗在思考上）
        content = resp.choices[0].message.content
        if content is None:
            raw = str(resp.choices[0].message)
            for field in ("reasoning_content",):
                val = getattr(resp.choices[0].message, field, None)
                if val:
                    raw = val
                    break
            content = raw
        text = content.strip()
        # 简单验证：不能包含 object 名称
        if edge["object"] in text:
            return ""
        return text
    except Exception as e:
        print(f"\n  [gen error] {e}")
        return ""


async def generate_questions(
    edges: list[dict],
    concurrency: int = 6,
) -> list[dict]:
    """批量生成问题，写入缓存。"""
    cache: dict = {}
    if GE_CACHE.exists():
        try:
            cache = json.loads(GE_CACHE.read_text("utf-8"))
        except Exception:
            pass

    base_url = os.environ.get("LLM_BASE_URL") or None
    api_key  = os.environ.get("LLM_API_KEY", "")
    model    = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key or "x", base_url=base_url)
    sem = asyncio.Semaphore(concurrency)

    results = []
    need = [e for e in edges if e["obj_chunk"] not in cache]
    cached = [e for e in edges if e["obj_chunk"] in cache]

    print(f"[gen] 缓存命中 {len(cached)}，需生成 {len(need)} 条")

    async def gen(e):
        async with sem:
            q = await _gen_one(e, client, model)
            return e, q

    new_results = await asyncio.gather(*[gen(e) for e in need])
    for e, q in new_results:
        if q:
            cache[e["obj_chunk"]] = {"question": q, "edge": e}

    GE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")

    for e in edges:
        cached_item = cache.get(e["obj_chunk"])
        if cached_item and cached_item.get("question"):
            results.append({
                "id":       f"GE::{e['obj_chunk']}",
                "strategy": "GE",
                "question": cached_item["question"],
                "ground_truth_chunk_ids": [e["obj_chunk"]],
                "subject":  e["subject"],
                "predicate":e["predicate"],
                "object":   e["object"],
                "source":   e["obj_chunk"].split("_")[0],
            })

    return results

# ── 三路检索 ──────────────────────────────────────────────────────────────────

_HYDE_PROMPT = """\
你是一名计算机技术专家。请根据以下问题，写一段简短的技术回答（100字以内），\
内容要像一段教材或技术文档里的原文，直接给出答案，不要重复问题本身。

问题：{query}

回答："""

_llm_sync_client = None

def _get_llm_client():
    global _llm_sync_client
    if _llm_sync_client is None:
        from openai import OpenAI
        _llm_sync_client = OpenAI(
            api_key  = os.environ.get("LLM_API_KEY", "x"),
            base_url = os.environ.get("LLM_BASE_URL") or None,
        )
    return _llm_sync_client

def generate_hyde_answer(query: str) -> str:
    """用 LLM 生成假设性答案，用于 HyDE 检索。"""
    try:
        model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        resp = _get_llm_client().chat.completions.create(
            model    = model,
            max_tokens = 200,
            temperature = 0.3,
            messages = [{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  [hyde gen error] {e}")
        return query  # fallback: 用原始 query


def _norm(t): return "".join(t.split())

def _contains(r, c):
    k = _norm(c)[:MATCH_PREFIX_LEN]
    return bool(k) and k in _norm(r)

def retrieve_bi(query, top_k):
    try:
        col = rag._get_knowledge_col()
        raw = rag._safe_query(col, query, min(top_k * 4, col.count()))
        seen, res = set(), []
        for doc, meta in raw:
            cid = meta.get("chunk_id", ""); txt = meta.get("text", doc)
            key = cid if cid else _norm(txt)[:MATCH_PREFIX_LEN]
            if key and key not in seen:
                seen.add(key); res.append((cid, txt))
            if len(res) >= top_k: break
        return res
    except Exception as e:
        print(f"  [bi error] {e}"); return []

def retrieve_hyde(query, top_k):
    """HyDE：用 LLM 生成假设答案，再做 bi-encoder 检索。"""
    hypo = generate_hyde_answer(query)
    return retrieve_bi(hypo, top_k)


def retrieve_hyde_rrf_w(query, top_k, k_bi=60, k_graph=20):
    """HyDE bi-encoder + graph → 加权 RRF，不做 rerank。"""
    try:
        hypo = generate_hyde_answer(query)
        col  = rag._get_knowledge_col()
        bi_raw = rag._safe_query(col, hypo, min(top_k * 8, col.count()))
        bi_ids, cand_map = [], {}
        for doc, meta in bi_raw:
            cid = meta.get("chunk_id", "")
            if cid and cid not in cand_map:
                bi_ids.append(cid)
                cand_map[cid] = doc

        gr = rag.retrieve_graph(query)          # graph 还是用原始 query
        graph_raw_ids = gr.get("source_chunk_ids", [])
        fetched    = _gr.get_chunks_by_ids(graph_raw_ids)
        text_by_id = {c["chunk_id"]: c.get("text", "") for c in fetched if c.get("chunk_id")}
        graph_ids  = []
        for cid in graph_raw_ids:
            if cid in text_by_id:
                graph_ids.append(cid)
                if cid not in cand_map:
                    cand_map[cid] = text_by_id[cid]

        merged = _weighted_rrf(bi_ids, graph_ids, k_bi=k_bi, k_graph=k_graph)
        return [(cid, cand_map[cid]) for cid in merged if cid in cand_map][:top_k]
    except Exception as e:
        print(f"  [hyde_rrf error] {e}"); return []


def retrieve_rrf(query, top_k):
    try:
        gr = rag.retrieve_graph(query)
        ids = gr.get("source_chunk_ids", [])
        extra = _gr.get_chunks_by_ids(ids) if ids else []
        result = rag.retrieve_rich(query, extra_chunks=extra or None, top_k=top_k)
        return [(c["chunk_id"], c["text"]) for c in result.get("knowledge", []) if c.get("chunk_id")][:top_k]
    except Exception as e:
        print(f"  [rrf error] {e}"); return []

def retrieve_graph_only(query, top_k):
    try:
        gr = rag.retrieve_graph(query)
        ids = gr.get("source_chunk_ids", [])[:top_k * 2]
        chunks = _gr.get_chunks_by_ids(ids)
        return [(c["chunk_id"], c["text"]) for c in chunks if c.get("chunk_id")][:top_k]
    except Exception as e:
        print(f"  [graph error] {e}"); return []

def retrieve_rrf_no_rerank(query, top_k):
    """bi-encoder + graph → RRF 融合，不经过 cross-encoder rerank。"""
    try:
        # 1. bi-encoder 候选
        col = rag._get_knowledge_col()
        bi_raw = rag._safe_query(col, query, min(top_k * 8, col.count()))
        bi_ids, cand_map = [], {}
        for doc, meta in bi_raw:
            cid = meta.get("chunk_id", "")
            if cid and cid not in cand_map:
                bi_ids.append(cid)
                cand_map[cid] = doc

        # 2. graph 候选（保留原始排名顺序）
        gr = rag.retrieve_graph(query)
        graph_raw_ids = gr.get("source_chunk_ids", [])
        # get_chunks_by_ids 按 source 分组，会破坏跨 source 的排名顺序
        # 先用它建 text 索引，再按原始 ID 顺序构造 graph_ids
        fetched = _gr.get_chunks_by_ids(graph_raw_ids)
        text_by_id = {c["chunk_id"]: c.get("text", "") for c in fetched if c.get("chunk_id")}
        graph_ids = []
        for cid in graph_raw_ids:
            if cid in text_by_id:
                graph_ids.append(cid)
                if cid not in cand_map:
                    cand_map[cid] = text_by_id[cid]

        # 3. RRF 融合（无 rerank）
        merged = rag._rrf_merge([bi_ids, graph_ids])
        return [(cid, cand_map[cid]) for cid in merged if cid in cand_map][:top_k]
    except Exception as e:
        print(f"  [rrf_nr error] {e}"); return []


def _weighted_rrf(bi_ids, graph_ids, k_bi=60, k_graph=20):
    """加权 RRF：graph 用更小的 k（等效更高权重）。"""
    scores: dict[str, float] = {}
    for rank, cid in enumerate(bi_ids, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_bi + rank)
    for rank, cid in enumerate(graph_ids, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_graph + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def retrieve_rrf_weighted(query, top_k, k_bi=60, k_graph=20):
    """bi-encoder + graph → 加权 RRF（graph 权重更高），不做 rerank。"""
    try:
        col = rag._get_knowledge_col()
        bi_raw = rag._safe_query(col, query, min(top_k * 8, col.count()))
        bi_ids, cand_map = [], {}
        for doc, meta in bi_raw:
            cid = meta.get("chunk_id", "")
            if cid and cid not in cand_map:
                bi_ids.append(cid)
                cand_map[cid] = doc

        gr = rag.retrieve_graph(query)
        graph_raw_ids = gr.get("source_chunk_ids", [])
        fetched = _gr.get_chunks_by_ids(graph_raw_ids)
        text_by_id = {c["chunk_id"]: c.get("text", "") for c in fetched if c.get("chunk_id")}
        graph_ids = []
        for cid in graph_raw_ids:
            if cid in text_by_id:
                graph_ids.append(cid)
                if cid not in cand_map:
                    cand_map[cid] = text_by_id[cid]

        merged = _weighted_rrf(bi_ids, graph_ids, k_bi=k_bi, k_graph=k_graph)
        return [(cid, cand_map[cid]) for cid in merged if cid in cand_map][:top_k]
    except Exception as e:
        print(f"  [rrf_w error] {e}"); return []
    except Exception as e:
        print(f"  [rrf_nr error] {e}"); return []


def retrieve_graph_rerank(query, top_k, rerank_cands=20):
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
        print(f"  [graph+rr error] {e}"); return []

# ── 多指标评分 ────────────────────────────────────────────────────────────────

def compute_metrics(retrieved, gt_id, chunk_idx):
    rank = None
    for i, (cid, txt) in enumerate(retrieved, 1):
        if cid == gt_id or (chunk_idx.get(gt_id) and _contains(txt, chunk_idx[gt_id])):
            rank = i; break
    return {
        "hit1": int(rank == 1),
        "hit3": int(rank is not None and rank <= 3),
        "hit5": int(rank is not None and rank <= 5),
        "mrr":  (1.0 / rank) if rank else 0.0,
    }

# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",            type=int, default=30, help="题目数")
    parser.add_argument("--top-k",        type=int, default=5)
    parser.add_argument("--rerank-cands", type=int, default=20)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--regen", action="store_true", help="强制重新生成（清除缓存）")
    args = parser.parse_args()

    if args.regen and GE_CACHE.exists():
        GE_CACHE.unlink()
        print("[gen] 已清除问题缓存")

    # 加载 .env
    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    print("[eval] 初始化...")
    chunk_index = build_chunk_index()
    print(f"  {len(chunk_index)} chunks")

    # 收集候选边并随机选 N 条
    edges = collect_candidate_edges(chunk_index)
    random.seed(args.seed)
    random.shuffle(edges)
    # 优先选谓词多样的边（避免全是"包含"）
    selected = []
    pred_cnt: dict = {}
    for e in edges:
        p = e["predicate"]
        if pred_cnt.get(p, 0) < 5:  # 每个谓词最多5条
            selected.append(e)
            pred_cnt[p] = pred_cnt.get(p, 0) + 1
        if len(selected) >= args.n * 2:
            break
    selected = selected[:args.n * 2]  # 多选一些，生成后再截取

    print(f"  候选边: {len(edges)}  选取: {len(selected)}")
    print(f"  谓词分布: {pred_cnt}")

    # 生成问题
    print("\n[gen] 生成问题...")
    questions = asyncio.run(generate_questions(selected))
    questions = questions[:args.n]
    print(f"  生成 {len(questions)} 题\n")

    if not questions:
        print("错误：没有生成任何问题，请检查 LLM 配置")
        return

    # 保存测试集
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    testset_path = TESTSETS_DIR / f"testset_GE_{ts}.json"
    TESTSETS_DIR.mkdir(parents=True, exist_ok=True)
    testset_path.write_text(json.dumps({
        "description": "GE 策略：图谱边关系题（问 A predicate 什么，答案在 B 的 chunk）",
        "n": len(questions),
        "questions": questions,
    }, ensure_ascii=False, indent=2), "utf-8")
    print(f"[testset] 已保存: {testset_path.name}\n")

    # 打印几道样题
    print("── 样题预览 ────────────────────────────────────────────")
    for q in questions[:5]:
        print(f"  [{q['subject']} --{q['predicate']}--> {q['object']}]")
        print(f"  Q: {q['question']}")
        print(f"  GT: {q['ground_truth_chunk_ids'][0]}")
        print()

    # ── 评测 ─────────────────────────────────────────────────────────────────

    print(f"── 开始评测 ({len(questions)} 题, top_k={args.top_k}) ─────────────────")
    metrics = {m: {"hit1":[], "hit3":[], "hit5":[], "mrr":[]}
               for m in ("bi", "hyde", "rrf_w", "hyde_rrf_w", "graph")}

    t0 = time.time()
    for i, q_item in enumerate(questions):
        query  = q_item["question"]
        gt_id  = q_item["ground_truth_chunk_ids"][0]
        subj   = q_item["subject"]
        pred   = q_item["predicate"]
        obj    = q_item["object"]

        bi_res          = retrieve_bi(query, args.top_k)
        hyde_res        = retrieve_hyde(query, args.top_k)
        rrf_w_res       = retrieve_rrf_weighted(query, args.top_k)
        hyde_rrf_w_res  = retrieve_hyde_rrf_w(query, args.top_k)
        graph_res       = retrieve_graph_only(query, args.top_k)

        for label, res in [("bi", bi_res), ("hyde", hyde_res),
                            ("rrf_w", rrf_w_res), ("hyde_rrf_w", hyde_rrf_w_res),
                            ("graph", graph_res)]:
            m = compute_metrics(res, gt_id, chunk_index)
            for k, v in m.items():
                metrics[label][k].append(v)

        tag = lambda res: ("✓" if compute_metrics(res, gt_id, chunk_index)["hit5"] else "✗")
        print(
            f"  [{i+1:2d}/{len(questions)}] [{pred}] {subj}→{obj}  "
            f"bi:{tag(bi_res)} hyde:{tag(hyde_res)} "
            f"rrf_w:{tag(rrf_w_res)} hyde+w:{tag(hyde_rrf_w_res)} gr:{tag(graph_res)}"
        )

    elapsed = round(time.time() - t0, 1)

    def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0.0

    print(f"\n{'='*65}")
    print(f"  GE 策略多指标评测  (n={len(questions)}, top_k={args.top_k})")
    print(f"{'─'*65}")
    COLS = ("bi", "hyde", "rrf_w", "hyde_rrf_w", "graph")
    print(f"  {'指标':<10}" + "".join(f" {c:>9}" for c in COLS))
    print(f"{'─'*70}")
    for metric in ("hit1", "hit3", "hit5", "mrr"):
        label = {"hit1":"Hit@1","hit3":"Hit@3","hit5":"Score@5","mrr":"MRR"}[metric]
        vals  = {m: avg(metrics[m][metric]) for m in COLS}
        best  = max(vals.values())
        markers = {m: "◀" if abs(vals[m]-best)<1e-6 else " " for m in vals}
        print(
            f"  {label:<10}"
            + "".join(f" {vals[m]:>7.3f}{markers[m]} " for m in COLS)
        )
    print(f"\n  耗时: {elapsed}s")
    print(f"{'='*70}")

    # 写日志
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"n": len(questions), "top_k": args.top_k,
                   "rerank_cands": args.rerank_cands, "seed": args.seed},
        "results": {m: {k: avg(metrics[m][k]) for k in ("hit1","hit3","hit5","mrr")}
                    for m in ("bi","hyde","rrf_w","hyde_rrf_w","graph")},
    }
    log_path = EVAL_LOG_DIR / f"graph_edge_{ts}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n[eval] 日志已写入 {log_path.name}")


if __name__ == "__main__":
    main()
