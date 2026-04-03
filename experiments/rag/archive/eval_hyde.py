"""
eval_hyde.py — HyDE（Hypothetical Document Embeddings）增强实验

对比三路检索在 graph 侧的 HyDE 效果：
  graph         — 原始 query 直接做实体/关系向量检索（基线）
  graph-hyde    — LLM 先生成假想答案，用假想答案做实体/关系向量检索

HyDE 原理：
  query（抽象问题）→ LLM → 假想答案（含技术术语）→ embed → 实体匹配更准
  例：
    原始: "面对复杂多变的网络环境，两端系统如何确保数据流的稳定性？"
    假想: "TCP通过重传机制、滑动窗口、流量控制和拥塞控制确保可靠传输..."
    → embed 假想答案 → 匹配到实体"TCP重传"、"拥塞控制"→ BFS 找到正确 chunk

测试集：混合集（O 94 + GA 50 + GB 44 = 188 题）

运行：
  LLM_PROVIDER=openai-compatible \\
  LLM_API_KEY=xxx \\
  LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \\
  LLM_MODEL=gemini-2.5-flash \\
  /opt/homebrew/Caskroom/miniconda/base/envs/interview-me/bin/python3 eval_hyde.py

假想答案缓存写入 eval_logs/routing_cache/hyde_answers.json
日志写入 eval_logs/hyde_YYYYMMDD_HHMMSS.json
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

TESTSETS_DIR = BACKEND_DIR / "eval_logs" / "testsets"
EVAL_LOG_DIR = BACKEND_DIR / "eval_logs"
CACHE_DIR    = EVAL_LOG_DIR / "routing_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HYDE_CACHE_FILE = CACHE_DIR / "hyde_answers.json"

import rag

MATCH_PREFIX_LEN = 60


# ── chunk 原文索引 ────────────────────────────────────────────────────────────

def build_chunk_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for f in sorted(rag.KNOWLEDGE_DIR.glob("*.chunks.json")):
        source = f.stem.replace(".chunks", "")
        try:
            chunks = json.loads(f.read_text(encoding="utf-8"))
            for i, c in enumerate(chunks):
                index[f"{source}_{i}"] = c.get("text", "")
        except Exception:
            pass
    return index


# ── 测试集加载 ────────────────────────────────────────────────────────────────

def load_mixed_testset(seed: int) -> list[dict]:
    ga_file = max(TESTSETS_DIR.glob("testset_GA_*.json"), key=lambda f: f.stat().st_mtime)
    gb_file = max(TESTSETS_DIR.glob("testset_GB_*.json"), key=lambda f: f.stat().st_mtime)
    ga_questions = json.loads(ga_file.read_text(encoding="utf-8"))["questions"]
    gb_questions = json.loads(gb_file.read_text(encoding="utf-8"))["questions"]
    n_graph = len(ga_questions) + len(gb_questions)

    o_file = TESTSETS_DIR / "merged_O.json"
    o_raw = json.loads(o_file.read_text(encoding="utf-8"))["questions"]
    o_questions = [
        {
            "id": f"O::{item['chunk_id']}::{i}",
            "strategy": "O",
            "question": item["question"],
            "ground_truth_chunk_ids": [item["chunk_id"]],
            "source": item.get("source", ""),
        }
        for i, item in enumerate(o_raw)
        if item.get("question") and item.get("chunk_id")
    ]
    random.seed(seed)
    o_sample = random.sample(o_questions, min(n_graph, len(o_questions)))
    mixed = o_sample + ga_questions + gb_questions
    random.shuffle(mixed)
    print(f"[混合集] O={len(o_sample)} | GA={len(ga_questions)} | GB={len(gb_questions)} | 总计={len(mixed)}")
    return mixed


# ── HyDE：批量生成假想答案，带缓存 ────────────────────────────────────────────

_HYDE_PROMPT = """\
请用2-4句话简洁回答以下计算机技术问题。
要求：
- 直接给出技术性描述，包含关键技术术语
- 不需要完整准确，能体现核心概念即可
- 用中文回答

问题：{query}"""


async def _gen_hyde_one(query: str, client, model: str) -> str:
    """生成单条假想答案，失败时返回原 query。"""
    try:
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        )
        content = resp.choices[0].message.content
        return content.strip() if content else query
    except Exception as e:
        print(f"\n  [hyde error] {e}")
        return query  # fallback：用原始 query


async def generate_hyde_answers(
    queries: list[str],
    concurrency: int = 8,
) -> dict[str, str]:
    """
    批量生成假想答案，带文件缓存。
    返回 {query: hyde_answer} 字典。
    """
    # 加载已有缓存
    cache: dict[str, str] = {}
    if HYDE_CACHE_FILE.exists():
        try:
            cache = json.loads(HYDE_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    need = [q for q in queries if q not in cache]
    if not need:
        print(f"[hyde] 全部 {len(queries)} 条来自缓存")
        return cache

    # 初始化 LLM client
    provider_name = os.environ.get("LLM_PROVIDER", "openai-compatible").lower()
    api_key   = os.environ.get("LLM_API_KEY", "")
    base_url  = os.environ.get("LLM_BASE_URL") or None
    model     = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

    if not api_key:
        raise RuntimeError("LLM_API_KEY 未设置，无法生成 HyDE 假想答案。")

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    print(f"[hyde] 生成假想答案 {len(need)} 条（缓存命中 {len(queries)-len(need)}）")
    print(f"[hyde] model={model}  concurrency={concurrency}")

    sem = asyncio.Semaphore(concurrency)

    async def gen_one(q: str) -> tuple[str, str]:
        async with sem:
            ans = await _gen_hyde_one(q, client, model)
            return q, ans

    t0 = time.time()
    results = await asyncio.gather(*[gen_one(q) for q in need])
    elapsed = round(time.time() - t0, 1)

    for q, ans in results:
        cache[q] = ans

    # 写缓存
    HYDE_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[hyde] 完成 {len(need)} 条  {elapsed}s")
    print(f"[hyde] 缓存: {HYDE_CACHE_FILE.name}  (共 {len(cache)} 条)")
    return cache


# ── 检索函数 ──────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return "".join(text.split())


def _contains(result_text: str, chunk_text: str) -> bool:
    key = _norm(chunk_text)[:MATCH_PREFIX_LEN]
    return bool(key) and key in _norm(result_text)


def retrieve_graph_only(query: str, top_k: int) -> list[tuple[str, str]]:
    """标准 graph 检索：用 query 做实体/关系向量匹配。"""
    try:
        import graph_rag as _gr
        graph_result = rag.retrieve_graph(query)
        chunk_ids = graph_result.get("source_chunk_ids", [])[:top_k * 2]
        if not chunk_ids:
            return []
        chunks = _gr.get_chunks_by_ids(chunk_ids)
        return [(c["chunk_id"], c["text"]) for c in chunks if c.get("chunk_id")][:top_k]
    except Exception as e:
        print(f"  [graph error] {e}")
        return []


def retrieve_graph_hyde(hyde_answer: str, top_k: int) -> list[tuple[str, str]]:
    """HyDE graph 检索：用假想答案做实体/关系向量匹配，再 BFS 扩展。"""
    try:
        import graph_rag as _gr
        # 关键区别：把 hyde_answer 而不是 query 传给 retrieve_graph
        graph_result = rag.retrieve_graph(hyde_answer)
        chunk_ids = graph_result.get("source_chunk_ids", [])[:top_k * 2]
        if not chunk_ids:
            return []
        chunks = _gr.get_chunks_by_ids(chunk_ids)
        return [(c["chunk_id"], c["text"]) for c in chunks if c.get("chunk_id")][:top_k]
    except Exception as e:
        print(f"  [graph-hyde error] {e}")
        return []


def retrieve_bi(query: str, top_k: int) -> list[tuple[str, str]]:
    """bi-encoder 基线。"""
    try:
        col = rag._get_knowledge_col()
        if col.count() == 0:
            return []
        raw = rag._safe_query(col, query, min(top_k * 4, col.count()))
        seen, results = set(), []
        for doc, meta in raw:
            cid = meta.get("chunk_id", "")
            txt = meta.get("text", doc)
            dedup_key = cid if cid else _norm(txt)[:MATCH_PREFIX_LEN]
            if dedup_key and dedup_key not in seen:
                seen.add(dedup_key)
                results.append((cid, txt))
            if len(results) >= top_k:
                break
        return results
    except Exception as e:
        print(f"  [bi error] {e}")
        return []


def retrieve_rrf(query: str, top_k: int) -> list[tuple[str, str]]:
    """完整 RRF 基线。"""
    try:
        import graph_rag as _gr
        graph_result = rag.retrieve_graph(query)
        graph_chunk_ids = graph_result.get("source_chunk_ids", [])
        graph_extra = _gr.get_chunks_by_ids(graph_chunk_ids) if graph_chunk_ids else []
        result = rag.retrieve_rich(query, extra_chunks=graph_extra or None, top_k=top_k)
        return [(c["chunk_id"], c["text"]) for c in result.get("knowledge", []) if c.get("chunk_id")][:top_k]
    except Exception as e:
        print(f"  [rrf error] {e}")
        return []


# ── 命中评分 ──────────────────────────────────────────────────────────────────

def score_result(
    retrieved: list[tuple[str, str]],
    gt_chunk_ids: list[str],
    chunk_index: dict[str, str],
) -> float:
    hit_count = 0
    for cid, txt in retrieved:
        for gt_id in gt_chunk_ids:
            if cid == gt_id:
                hit_count += 1
                break
            gt_text = chunk_index.get(gt_id, "")
            if gt_text and _contains(txt, gt_text):
                hit_count += 1
                break
    n_gt = len(gt_chunk_ids)
    if hit_count == 0:
        return 0.0
    if hit_count >= n_gt:
        return 1.0
    return 0.5


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HyDE 增强 Graph 检索实验")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    print("[eval] 构建 chunk 索引 ...")
    chunk_index = build_chunk_index()
    print(f"  {len(chunk_index)} chunks\n")

    questions = load_mixed_testset(args.seed)
    n = len(questions)
    queries = [q["question"] for q in questions]

    # ── 批量预生成 HyDE 假想答案（带缓存）────────────────────────────────────
    hyde_cache = asyncio.run(generate_hyde_answers(queries))
    print()

    # ── 主评估循环 ────────────────────────────────────────────────────────────
    modes = ["bi", "rrf", "graph", "graph-hyde"]
    scores:      dict[str, list[float]] = {m: [] for m in modes}
    by_strategy: dict[str, dict[str, list[float]]] = {}

    t0 = time.time()
    for i, q_item in enumerate(questions):
        query    = q_item["question"]
        gt_ids   = q_item["ground_truth_chunk_ids"]
        strategy = q_item.get("strategy", "?")
        hyde_ans = hyde_cache.get(query, query)  # fallback 到原始 query

        elapsed = time.time() - t0
        eta = (elapsed / i * (n - i)) if i > 0 else 0
        bar = "█" * (i * 20 // n) + "░" * (20 - i * 20 // n)
        print(f"\r[{bar}] {i+1}/{n}  ETA {eta/60:.1f}min", end="", flush=True)

        # 检索
        bi_res    = retrieve_bi(query, args.top_k)
        rrf_res   = retrieve_rrf(query, args.top_k)
        graph_res = retrieve_graph_only(query, args.top_k)
        hyde_res  = retrieve_graph_hyde(hyde_ans, args.top_k)

        # 评分
        bi_s    = score_result(bi_res,    gt_ids, chunk_index)
        rrf_s   = score_result(rrf_res,   gt_ids, chunk_index)
        graph_s = score_result(graph_res, gt_ids, chunk_index)
        hyde_s  = score_result(hyde_res,  gt_ids, chunk_index)

        for m, s in [("bi", bi_s), ("rrf", rrf_s), ("graph", graph_s), ("graph-hyde", hyde_s)]:
            scores[m].append(s)
            if strategy not in by_strategy:
                by_strategy[strategy] = {m: [] for m in modes}
            by_strategy[strategy][m].append(s)

        tag = lambda s: ("✓" if s == 1.0 else ("½" if s == 0.5 else "✗"))
        # 标注 hyde 是否比 graph 更好
        delta = "↑" if hyde_s > graph_s else ("↓" if hyde_s < graph_s else "=")
        print(
            f"\r  [{i+1:3d}/{n}] [{strategy}] "
            f"{tag(bi_s)}bi {tag(rrf_s)}rrf "
            f"{tag(graph_s)}graph {tag(hyde_s)}hyde{delta}"
            f"  {query[:38].replace(chr(10),' ')!r}"
        )

    elapsed_total = round(time.time() - t0, 1)
    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    # ── 打印汇总 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  HyDE 实验结果  ({n} 题, top_k={args.top_k})")
    print(f"{'─'*65}")
    print(f"  {'策略':<20} {'Score@'+str(args.top_k):>10}  {'vs graph':>10}")
    print(f"{'─'*65}")

    graph_overall = avg(scores["graph"])
    for label, sc in [
        ("bi",         avg(scores["bi"])),
        ("rrf",        avg(scores["rrf"])),
        ("graph",      graph_overall),
        ("graph-hyde", avg(scores["graph-hyde"])),
    ]:
        diff = sc - graph_overall
        diff_str = f"{diff:+.3f}" if label != "graph" else "  (base)"
        print(f"  {label:<20} {sc:>10.3f}  {diff_str:>10}")

    print(f"\n  按策略细分：")
    print(f"  {'策略':<6} {'题数':>4}  {'bi':>7} {'rrf':>7} {'graph':>7} {'hyde':>7}  hyde提升")
    print(f"{'─'*65}")
    for strategy in ["O", "GA", "GB"]:
        sg = by_strategy.get(strategy, {})
        if not sg:
            continue
        n_s   = len(sg.get("bi", []))
        bi_v  = avg(sg.get("bi",         []))
        rrf_v = avg(sg.get("rrf",        []))
        gr_v  = avg(sg.get("graph",      []))
        hy_v  = avg(sg.get("graph-hyde", []))
        delta = f"{hy_v - gr_v:+.3f}"
        print(f"  {strategy:<6} {n_s:>4}  {bi_v:>7.3f} {rrf_v:>7.3f} {gr_v:>7.3f} {hy_v:>7.3f}  {delta}")

    # hyde 提升/下降题数统计
    graph_arr = scores["graph"]
    hyde_arr  = scores["graph-hyde"]
    improved  = sum(1 for g, h in zip(graph_arr, hyde_arr) if h > g)
    degraded  = sum(1 for g, h in zip(graph_arr, hyde_arr) if h < g)
    same      = sum(1 for g, h in zip(graph_arr, hyde_arr) if h == g)
    print(f"\n  hyde vs graph 逐题：↑改善={improved} ↓退步={degraded} =持平={same}")
    print(f"  总耗时: {elapsed_total}s")
    print(f"{'='*65}")

    # ── 写日志 ────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "top_k": args.top_k,
            "seed":  args.seed,
            "hyde_cache_file": str(HYDE_CACHE_FILE),
            "hyde_model": os.environ.get("LLM_MODEL", ""),
            "n_total": n,
        },
        "overall": {m: avg(scores[m]) for m in modes},
        "by_strategy": {
            s: {m: avg(v.get(m, [])) for m in modes}
            for s, v in by_strategy.items()
        },
        "n_by_strategy": {s: len(v.get("bi", [])) for s, v in by_strategy.items()},
        "hyde_delta": {
            "improved": improved,
            "degraded": degraded,
            "same":     same,
        },
        "elapsed_s": elapsed_total,
    }
    log_path = EVAL_LOG_DIR / f"hyde_{ts}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 日志已写入 {log_path.name}")
    print(f"[eval] HyDE 缓存已写入 {HYDE_CACHE_FILE.name}  ({len(hyde_cache)} 条)")


if __name__ == "__main__":
    main()
