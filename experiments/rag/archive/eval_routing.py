"""
eval_routing.py — Query Routing 实验

对比五种策略：
  bi            — 所有题目用 bi-encoder（基线）
  rrf           — 所有题目用 RRF+rerank（基线）
  graph         — 所有题目用 graph-only（基线）
  routing-rule  — 规则路由：关键词判断 → bi/graph（无需LLM）
  routing-llm   — LLM 路由：单次调用分类 → rrf/graph（需配置 LLM 环境变量）

路由逻辑：
  事实题（可直接查单段文字）→ RRF
  关系/推理题（需多概念联系/因果/多跳）→ graph

理论上限（oracle routing）≈ 0.739
  O  (94题) × rrf=0.851 + GA (50题) × graph=0.600 + GB (44题) × graph=0.659

运行：
  # 仅跑规则路由（无需 LLM）
  conda run -n interview-me python3 eval_routing.py --mode rule

  # 跑全部（含 LLM 路由，需设置环境变量）
  LLM_PROVIDER=anthropic LLM_API_KEY=sk-ant-xxx LLM_MODEL=claude-haiku-4-5-20251001 \\
    conda run -n interview-me python3 eval_routing.py --mode all

  # 仅 LLM 路由（跳过规则）
  conda run -n interview-me python3 eval_routing.py --mode llm

日志写入 eval_logs/routing_YYYYMMDD_HHMMSS.json
LLM 路由决策缓存写入 eval_logs/routing_cache/
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
ROUTING_CACHE_DIR = EVAL_LOG_DIR / "routing_cache"
ROUTING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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


# ── 测试集加载（复用 eval_mixed_og 逻辑）────────────────────────────────────

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
    print(f"[混合集] 策略O={len(o_sample)} | GA={len(ga_questions)} | GB={len(gb_questions)} | 总计={len(mixed)}")
    return mixed


# ── 三路检索（与 eval_mixed_og 相同）────────────────────────────────────────

def _norm(text: str) -> str:
    return "".join(text.split())


def _contains(result_text: str, chunk_text: str) -> bool:
    key = _norm(chunk_text)[:MATCH_PREFIX_LEN]
    return bool(key) and key in _norm(result_text)


def retrieve_bi(query: str, top_k: int) -> list[tuple[str, str]]:
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


def retrieve_graph_only(query: str, top_k: int) -> list[tuple[str, str]]:
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
    return 0.5  # partial（GB 专属）


# ── 规则路由 ──────────────────────────────────────────────────────────────────

# 关系/推理类关键词 → graph
_RELATIONAL_KWS = [
    # 因果
    "影响", "导致", "造成", "引发", "触发", "为什么", "原因", "为何",
    # 关系
    "区别", "不同", "对比", "比较", "优缺点", "优势", "劣势",
    "关系", "联系", "依赖", "依靠",
    # 多跳/传播
    "如何导致", "如何影响", "通过.*实现", "经过",
    # 场景推理
    "发生了什么", "什么情况下", "什么时候会",
    "如果.*会", "当.*时", "一旦",
    # 过程追踪
    "流程", "步骤", "过程",  # (注意：这些也可能是事实题，阈值靠其他词共现)
]

# 强事实类关键词 → rrf（override 上面的关系词）
_FACTUAL_OVERRIDE_KWS = [
    "是什么", "定义", "概念", "原理", "介绍", "特点", "特性",
    "有哪些", "几种", "分类", "类型",
]


def route_rule(query: str) -> str:
    """
    规则路由。返回 'rrf' 或 'graph'。

    策略：
      1. 若命中强事实关键词 → rrf（覆盖关系词）
      2. 若命中关系/推理关键词 → graph
      3. 否则 → rrf（事实题默认）
    """
    for kw in _FACTUAL_OVERRIDE_KWS:
        if kw in query:
            return "rrf"
    for kw in _RELATIONAL_KWS:
        if kw in query:
            return "graph"
    return "rrf"


# ── LLM 路由 ──────────────────────────────────────────────────────────────────

_ROUTE_PROMPT = """\
你是一个 RAG 检索路由器。对于给定的计算机技术问题，判断应该用哪种检索策略。

【策略 A：rrf】向量检索 + 图谱 + 重排序（综合策略，覆盖面广）
  适用：绝大多数问题，包括——
  - 定义、原理、步骤、特性类（什么是X，X的原理，X的过程）
  - 表面上看起来像关系题，但答案集中在一两段文字里的问题
  - "为什么"、"如何"开头，但本质是描述某个机制/特性的问题
  - 例：TCP三次握手的过程是什么？→ rrf
  - 例：为什么需要四次挥手？→ rrf（一段文字可以解释清楚）
  - 例：LRU算法的实现原理？→ rrf
  - 例：HTTPS建立连接的完整流程？→ rrf（流程题，单文档可答）
  - 例：为什么 SYN_SENT 状态会超时？→ rrf

【策略 B：graph】纯图谱检索（仅用于跨实体多跳推理）
  适用：必须同时找到两个不同概念的知识、再推导出联系，才能回答的问题
  - 严格要求：答案需要"A的知识" + "B的知识"组合推理
  - "A 如何影响 B"类，且 A 和 B 是完全独立的两个系统/模块
  - 例：SYN泛洪攻击如何导致死锁？→ graph（需要同时理解 SYN 攻击机制 + 死锁条件）
  - 例：TCP拥塞控制与进程调度的关系？→ graph（跨领域多跳）
  - 注意：如果问题虽然问"关系/影响"，但一段文字就够了，选 rrf

【决策原则】
  有疑问时选 rrf，因为 rrf 已经包含了图谱检索。
  只有当你确信"这道题必须同时从两个独立知识点推理"时才选 graph。

只输出「factual」或「relational」，不要任何其他文字。
  factual = 选策略 rrf
  relational = 选策略 graph

问题：{query}"""


async def _classify_one_anthropic(query: str, client, model: str) -> str:
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": _ROUTE_PROMPT.format(query=query)}],
        )
        text = resp.content[0].text.strip().lower()
        return "graph" if "relational" in text else "rrf"
    except Exception as e:
        print(f"\n  [llm route error] {e}")
        return "rrf"  # fallback


async def _classify_one_openai(query: str, client, model: str) -> str:
    try:
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=200,  # thinking 模型需要更多 token
            messages=[{"role": "user", "content": _ROUTE_PROMPT.format(query=query)}],
        )
        content = resp.choices[0].message.content
        # thinking 模型（如 gemini-2.5-flash）可能返回 None content + thinking 字段
        if content is None:
            # 尝试从 thinking 中提取（OpenAI compat 格式有时会放在 reasoning_content）
            msg = resp.choices[0].message
            content = getattr(msg, "reasoning_content", None) or ""
            # 或直接从所有字段找 factual/relational
            raw = str(msg)
            if "relational" in raw.lower():
                content = "relational"
            elif "factual" in raw.lower():
                content = "factual"
            else:
                content = ""
        text = content.strip().lower()
        return "graph" if "relational" in text else "rrf"
    except Exception as e:
        print(f"\n  [llm route error] {e}")
        return "rrf"


async def route_llm_batch(
    queries: list[str],
    cache_file: str = "llm_routing.json",
    concurrency: int = 8,
) -> list[str]:
    """
    批量 LLM 路由，带本地缓存（避免重复调用）。
    返回与 queries 等长的列表，每个元素为 'rrf' 或 'graph'。
    """
    cache_path = ROUTING_CACHE_DIR / cache_file
    cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    need = [q for q in queries if q not in cache]
    if not need:
        print(f"[routing-llm] 全部 {len(queries)} 条来自缓存")
        return [cache[q] for q in queries]

    # 初始化 LLM client
    provider_name = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL") or None
    model = os.environ.get("LLM_MODEL", "")

    if not api_key and provider_name == "anthropic":
        raise RuntimeError("LLM_API_KEY 未设置，无法运行 LLM 路由。请先设置环境变量或改用 --mode rule。")

    print(f"[routing-llm] 调用 LLM 分类 {len(need)} 条（缓存命中 {len(queries)-len(need)}）")
    print(f"[routing-llm] provider={provider_name}, model={model or '(default)'}")

    sem = asyncio.Semaphore(concurrency)

    if provider_name == "anthropic":
        from anthropic import AsyncAnthropic
        _model = model or "claude-haiku-4-5-20251001"
        client = AsyncAnthropic(api_key=api_key)

        async def classify(q: str) -> tuple[str, str]:
            async with sem:
                return q, await _classify_one_anthropic(q, client, _model)
    else:
        from openai import AsyncOpenAI
        _model = model or "gpt-4o-mini"
        client = AsyncOpenAI(api_key=api_key or "ollama", base_url=base_url)

        async def classify(q: str) -> tuple[str, str]:
            async with sem:
                return q, await _classify_one_openai(q, client, _model)

    t0 = time.time()
    results = await asyncio.gather(*[classify(q) for q in need])
    elapsed = round(time.time() - t0, 1)

    # 更新缓存
    for q, r in results:
        cache[q] = r
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计分类分布
    rrf_cnt = sum(1 for _, r in results if r == "rrf")
    graph_cnt = sum(1 for _, r in results if r == "graph")
    print(f"[routing-llm] 完成 {len(need)} 条  {elapsed}s  → rrf={rrf_cnt} graph={graph_cnt}")
    print(f"[routing-llm] 缓存: {cache_path.name}")

    return [cache[q] for q in queries]


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query Routing 实验")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["rule", "llm", "all"],
        default="rule",
        help="路由模式：rule=规则路由, llm=LLM路由, all=两者都跑（默认: rule）",
    )
    parser.add_argument(
        "--llm-cache",
        type=str,
        default="llm_routing.json",
        help="LLM 路由决策缓存文件名（eval_logs/routing_cache/ 下，默认 llm_routing.json）",
    )
    args = parser.parse_args()

    print("[eval] 构建 chunk 索引 ...")
    chunk_index = build_chunk_index()
    print(f"  {len(chunk_index)} chunks\n")

    questions = load_mixed_testset(args.seed)
    n = len(questions)
    queries = [q["question"] for q in questions]

    # ── LLM 路由预计算（批量，带缓存）──────────────────────────────────────
    llm_routes: list[str] = []
    if args.mode in ("llm", "all"):
        llm_routes = asyncio.run(route_llm_batch(queries, cache_file=args.llm_cache))

    # ── 规则路由预计算 ───────────────────────────────────────────────────────
    rule_routes = [route_rule(q) for q in queries]
    if args.mode in ("rule", "all"):
        rrf_cnt = rule_routes.count("rrf")
        graph_cnt = rule_routes.count("graph")
        print(f"[routing-rule] rrf={rrf_cnt} graph={graph_cnt} (total={n})")

    # ── 主评估循环 ───────────────────────────────────────────────────────────
    modes_baseline = ["bi", "rrf", "graph"]
    scores: dict[str, list[float]] = {m: [] for m in modes_baseline}
    by_strategy: dict[str, dict[str, list[float]]] = {}

    # routing 得分
    rule_scores: list[float] = []
    llm_scores:  list[float] = []

    # 路由决策追踪（用于统计准确率）
    rule_correct = 0   # 路由和 oracle 一致
    llm_correct  = 0

    t0 = time.time()
    for i, q_item in enumerate(questions):
        query    = q_item["question"]
        gt_ids   = q_item["ground_truth_chunk_ids"]
        strategy = q_item.get("strategy", "?")

        elapsed = time.time() - t0
        eta = (elapsed / i * (n - i)) if i > 0 else 0
        bar = "█" * (i * 20 // n) + "░" * (20 - i * 20 // n)
        print(f"\r[{bar}] {i+1}/{n}  ETA {eta/60:.1f}min", end="", flush=True)

        # 三路检索
        bi_res    = retrieve_bi(query, args.top_k)
        rrf_res   = retrieve_rrf(query, args.top_k)
        graph_res = retrieve_graph_only(query, args.top_k)

        # 基线得分
        bi_s    = score_result(bi_res,    gt_ids, chunk_index)
        rrf_s   = score_result(rrf_res,   gt_ids, chunk_index)
        graph_s = score_result(graph_res, gt_ids, chunk_index)

        for m, s in [("bi", bi_s), ("rrf", rrf_s), ("graph", graph_s)]:
            scores[m].append(s)
            by_strategy.setdefault(strategy, {m: [] for m in modes_baseline})[m].append(s)

        # oracle（理论最优）
        oracle = "rrf" if strategy == "O" else "graph"

        # 规则路由得分
        rule_route_i = rule_routes[i]
        rule_s = rrf_s if rule_route_i == "rrf" else graph_s
        rule_scores.append(rule_s)
        if rule_route_i == oracle:
            rule_correct += 1

        # LLM 路由得分
        if llm_routes:
            llm_route_i = llm_routes[i]
            llm_s = rrf_s if llm_route_i == "rrf" else graph_s
            llm_scores.append(llm_s)
            if llm_route_i == oracle:
                llm_correct += 1

        tag = lambda s: ("✓" if s == 1.0 else ("½" if s == 0.5 else "✗"))
        route_tag = f"rule→{rule_route_i[0]}"
        llm_tag   = f"llm→{llm_routes[i][0]}" if llm_routes else ""
        print(
            f"\r  [{i+1:3d}/{n}] [{strategy}] "
            f"{tag(bi_s)}bi {tag(rrf_s)}rrf {tag(graph_s)}gr | "
            f"{tag(rule_s)}{route_tag}"
            + (f" {tag(llm_s)}{llm_tag}" if llm_routes else "")
            + f"  {query[:40].replace(chr(10),' ')!r}"
        )

    elapsed_total = round(time.time() - t0, 1)
    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    # ── 打印汇总 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Query Routing 实验结果  ({n} 题, top_k={args.top_k})")
    print(f"{'─'*65}")

    # 整体对比
    print(f"  {'策略':<22} {'Score@'+str(args.top_k):>10}  {'vs RRF':>8}")
    print(f"{'─'*65}")
    rrf_overall = avg(scores["rrf"])
    for label, sc in [
        ("bi (baseline)",    avg(scores["bi"])),
        ("rrf (baseline)",   rrf_overall),
        ("graph (baseline)", avg(scores["graph"])),
        ("routing-rule",     avg(rule_scores)),
        *([("routing-llm",  avg(llm_scores))] if llm_scores else []),
    ]:
        diff = sc - rrf_overall
        diff_str = f"{diff:+.3f}" if diff != 0 else "   —  "
        marker = " ◀ best" if sc == max(
            avg(scores["bi"]), rrf_overall, avg(scores["graph"]),
            avg(rule_scores), avg(llm_scores) if llm_scores else 0
        ) else ""
        print(f"  {label:<22} {sc:>10.3f}  {diff_str:>8}{marker}")

    # 按策略细分
    print(f"\n  按策略细分：")
    print(f"  {'策略':<6} {'题数':>4}  {'bi':>7} {'rrf':>7} {'graph':>7} {'rule':>7}" +
          ("  llm" if llm_scores else ""))
    print(f"{'─'*65}")
    for strategy in ["O", "GA", "GB"]:
        sg = by_strategy.get(strategy, {})
        if not sg:
            continue
        idxs = [j for j, q in enumerate(questions) if q.get("strategy") == strategy]
        n_s = len(idxs)
        bi_v    = avg(sg.get("bi",    []))
        rrf_v   = avg(sg.get("rrf",   []))
        graph_v = avg(sg.get("graph", []))
        rule_v  = avg([rule_scores[j] for j in idxs]) if rule_scores else 0
        llm_v   = avg([llm_scores[j]  for j in idxs]) if llm_scores  else 0
        llm_col = f" {llm_v:>7.3f}" if llm_scores else ""
        print(f"  {strategy:<6} {n_s:>4}  {bi_v:>7.3f} {rrf_v:>7.3f} {graph_v:>7.3f} {rule_v:>7.3f}{llm_col}")

    # 路由准确率
    oracle_count = n  # 每题都有 oracle
    print(f"\n  路由准确率（与 oracle 一致）：")
    if args.mode in ("rule", "all"):
        print(f"    routing-rule: {rule_correct}/{oracle_count} = {rule_correct/oracle_count:.1%}")
    if llm_routes and args.mode in ("llm", "all"):
        print(f"    routing-llm:  {llm_correct}/{oracle_count} = {llm_correct/oracle_count:.1%}")

    # 理论上限提示
    theory = avg(
        [scores["rrf"][j]   for j, q in enumerate(questions) if q.get("strategy") == "O"] +
        [scores["graph"][j] for j, q in enumerate(questions) if q.get("strategy") != "O"]
    )
    print(f"\n  Oracle 上限（O→rrf, GA/GB→graph）: {theory:.3f}")
    print(f"  总耗时: {elapsed_total}s")
    print(f"{'='*65}")

    # ── 写日志 ────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "top_k": args.top_k,
            "seed": args.seed,
            "mode": args.mode,
            "n_total": n,
        },
        "overall": {
            "bi":           avg(scores["bi"]),
            "rrf":          avg(scores["rrf"]),
            "graph":        avg(scores["graph"]),
            "routing-rule": avg(rule_scores),
            **({"routing-llm": avg(llm_scores)} if llm_scores else {}),
            "oracle_upper_bound": theory,
        },
        "by_strategy": {
            s: {
                "bi":    avg(by_strategy.get(s, {}).get("bi",    [])),
                "rrf":   avg(by_strategy.get(s, {}).get("rrf",   [])),
                "graph": avg(by_strategy.get(s, {}).get("graph", [])),
                "rule_routing": avg([rule_scores[j] for j, q in enumerate(questions) if q.get("strategy") == s]),
                **({"llm_routing": avg([llm_scores[j] for j, q in enumerate(questions) if q.get("strategy") == s])}
                   if llm_scores else {}),
                "n": len([q for q in questions if q.get("strategy") == s]),
            }
            for s in ["O", "GA", "GB"] if any(q.get("strategy") == s for q in questions)
        },
        "routing_accuracy": {
            "rule": round(rule_correct / oracle_count, 4),
            **({"llm": round(llm_correct / oracle_count, 4)} if llm_routes else {}),
            "oracle_count": oracle_count,
        },
        "routing_distribution": {
            "rule": {"rrf": rule_routes.count("rrf"), "graph": rule_routes.count("graph")},
            **({"llm": {
                "rrf":   llm_routes.count("rrf"),
                "graph": llm_routes.count("graph"),
            }} if llm_routes else {}),
        },
        "elapsed_s": elapsed_total,
    }
    log_path = EVAL_LOG_DIR / f"routing_{ts}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 日志已写入 {log_path.name}")


if __name__ == "__main__":
    main()
