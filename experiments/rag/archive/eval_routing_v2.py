"""
eval_routing_v2.py — 路由方案 A + C 实验

方案 A：实体距离路由（纯路由，0 额外检索）
  原理：query 与实体库的距离反映"是否直接点名实体"
  - 低距离（< DIST_LOW）→ 实体被直接点名 → 事实题 → rrf
  - 高距离（≥ DIST_LOW）+ 实体有边 → 抽象/场景查询 → graph
  - 高距离但实体无边 → rrf

方案 C：检索置信度自适应（先跑 graph，按置信度决定是否 fallback）
  原理：graph 的"可信度"由实体距离 + 图中节点度数综合决定
  - 置信度高（低距离 or 高距离 + 高度数） → 根据距离判断
    - 低距离 → rrf（直接命名实体 → 事实题）
    - 高距离 + 图中有边 → 跑 graph，若有结果则用 graph，否则 fallback
  - 图谱找到 ≥ 2 个 chunk → 信任 graph 结果
  - 图谱结果不足 → fallback rrf

对比基线：bi / rrf / graph
理论上限（oracle）：O → rrf, GA/GB → graph

运行：
  /opt/homebrew/Caskroom/miniconda/base/envs/interview-me/bin/python eval_routing_v2.py [--top-k 5] [--seed 42] [--dist-threshold 0.22]
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

TESTSETS_DIR = BACKEND_DIR / "eval_logs" / "testsets"
EVAL_LOG_DIR = BACKEND_DIR / "eval_logs"

import rag
import graph_rag as _gr

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
    print(f"[混合集] 策略O={len(o_sample)} | GA={len(ga_questions)} | GB={len(gb_questions)} | 总计={len(mixed)}")
    return mixed


# ── 三路检索 ──────────────────────────────────────────────────────────────────

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
        graph_result = rag.retrieve_graph(query)
        chunk_ids = graph_result.get("source_chunk_ids", [])[:top_k * 2]
        if not chunk_ids:
            return []
        chunks = _gr.get_chunks_by_ids(chunk_ids)
        return [(c["chunk_id"], c["text"]) for c in chunks if c.get("chunk_id")][:top_k]
    except Exception as e:
        print(f"  [graph error] {e}")
        return []


# ── 评分 ──────────────────────────────────────────────────────────────────────

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


# ── 实体距离路由工具 ──────────────────────────────────────────────────────────

def _query_top_entities(query: str, top_k: int = 3) -> list[tuple[float, str]]:
    """返回 [(distance, entity_name), ...] top-k 实体匹配结果。"""
    ent_col = _gr._get_entities_col()
    if ent_col.count() == 0:
        return []
    try:
        results = ent_col.query(
            query_texts=[query],
            n_results=min(top_k, ent_col.count()),
            include=["metadatas", "distances"],
        )
        return [
            (dist, meta["name"])
            for dist, meta in zip(results["distances"][0], results["metadatas"][0])
        ]
    except Exception:
        return []


# ── 方案 A：实体距离路由 ──────────────────────────────────────────────────────

def route_entity_distance(query: str, G, dist_threshold: float) -> str:
    """
    纯路由决策（0 额外检索开销）。
    - top 实体距离 < dist_threshold → 实体被直接点名 → 事实题 → rrf
    - top 实体距离 ≥ dist_threshold + 图中有边 → 抽象/场景查询 → graph
    - 其他 → rrf
    """
    ents = _query_top_entities(query, top_k=3)
    if not ents:
        return "rrf"

    top_dist, top_name = ents[0]

    if top_dist < dist_threshold:
        return "rrf"

    # 抽象查询：检查 top 实体是否在图中有出入边
    if G.has_node(top_name) and G.degree(top_name) > 0:
        return "graph"

    # top 实体无边，再检查 top-3 中是否有有边的
    for _, name in ents[1:]:
        if G.has_node(name) and G.degree(name) > 0:
            return "graph"

    return "rrf"


# ── 方案 C：检索置信度自适应 ──────────────────────────────────────────────────

def retrieve_adaptive(
    query: str,
    top_k: int,
    G,
    dist_threshold: float,
    min_graph_results: int = 2,
) -> tuple[list[tuple[str, str]], str]:
    """
    先跑 graph 检索，根据实体距离 + 结果数量决定是否信任 graph 结果。

    决策逻辑：
      1. 实体距离 < dist_threshold → 直接命名实体 → 事实题 → rrf（跳过 graph）
      2. 实体距离 ≥ dist_threshold + 有边节点 → 抽象查询：
         a. 运行 graph 检索
         b. graph 返回 ≥ min_graph_results 个 chunk → 信任 graph，返回 graph 结果
         c. graph 返回不足 → fallback rrf
      3. 其他（无边 or 无实体） → rrf

    与方案 A 的差异：
      A 是纯路由（不运行检索）；
      C 实际运行 graph 检索并用结果数量做二次验证（防止 graph 召回为空时返回差结果）。
    """
    ents = _query_top_entities(query, top_k=3)
    if not ents:
        return retrieve_rrf(query, top_k), "rrf"

    top_dist, top_name = ents[0]

    # 直接命名实体 → 事实题 → 直接走 rrf
    if top_dist < dist_threshold:
        return retrieve_rrf(query, top_k), "rrf"

    # 抽象查询：检查 top-3 中最大度数
    all_names = [name for _, name in ents]
    max_degree = max(
        (G.degree(n) if G.has_node(n) else 0)
        for n in all_names
    )

    if max_degree == 0:
        # 图中没有有边节点 → rrf
        return retrieve_rrf(query, top_k), "rrf"

    # 抽象查询 + 有边 → 运行 graph
    graph_res = retrieve_graph_only(query, top_k)
    if len(graph_res) >= min_graph_results:
        return graph_res, "graph"

    # graph 结果不足 → fallback rrf
    return retrieve_rrf(query, top_k), "rrf"


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="路由方案 A+C 实验")
    parser.add_argument("--top-k",         type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--dist-threshold", type=float, default=0.22,
                        help="实体距离阈值（低于此 → rrf，高于此 → graph 候选）")
    args = parser.parse_args()

    print("[eval] 初始化图谱 & chunk 索引 ...")
    G = _gr._get_nx_graph()
    chunk_index = build_chunk_index()
    print(f"  图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边  |  chunks: {len(chunk_index)}\n")

    questions = load_mixed_testset(args.seed)
    n = len(questions)

    # ── 收集结果 ─────────────────────────────────────────────────────────────

    scores: dict[str, list[float]] = {m: [] for m in ("bi", "rrf", "graph", "A", "C")}
    route_A: list[str] = []  # 路由决策记录
    route_C: list[str] = []

    # 分策略统计
    by_strategy: dict[str, dict[str, list[float]]] = {}

    t0 = time.time()
    for i, q_item in enumerate(questions):
        query    = q_item["question"]
        gt_ids   = q_item["ground_truth_chunk_ids"]
        strategy = q_item.get("strategy", "?")

        elapsed = time.time() - t0
        eta = (elapsed / i * (n - i)) if i > 0 else 0
        bar = "█" * (i * 20 // n) + "░" * (20 - i * 20 // n)
        print(f"\r[{bar}] {i+1}/{n}  ETA {eta/60:.1f}min", end="", flush=True)

        # 三路基线
        bi_res    = retrieve_bi(query, args.top_k)
        rrf_res   = retrieve_rrf(query, args.top_k)
        graph_res = retrieve_graph_only(query, args.top_k)

        bi_s    = score_result(bi_res,    gt_ids, chunk_index)
        rrf_s   = score_result(rrf_res,   gt_ids, chunk_index)
        graph_s = score_result(graph_res, gt_ids, chunk_index)

        # 方案 A（纯路由）
        r_A = route_entity_distance(query, G, args.dist_threshold)
        route_A.append(r_A)
        A_s = rrf_s if r_A == "rrf" else graph_s

        # 方案 C（自适应检索）
        C_res, r_C = retrieve_adaptive(query, args.top_k, G, args.dist_threshold)
        route_C.append(r_C)
        C_s = score_result(C_res, gt_ids, chunk_index)

        for m, s in [("bi", bi_s), ("rrf", rrf_s), ("graph", graph_s), ("A", A_s), ("C", C_s)]:
            scores[m].append(s)
            if strategy not in by_strategy:
                by_strategy[strategy] = {m: [] for m in ("bi", "rrf", "graph", "A", "C")}
            by_strategy[strategy][m].append(s)

        oracle = "rrf" if strategy == "O" else "graph"
        tag = lambda s: ("✓" if s == 1.0 else ("½" if s == 0.5 else "✗"))
        print(
            f"\r  [{i+1:3d}/{n}] [{strategy}] "
            f"{tag(bi_s)}bi {tag(rrf_s)}rrf {tag(graph_s)}gr "
            f"| A→{r_A[0]}:{tag(A_s)} C→{r_C[0]}:{tag(C_s)}"
            f"  {query[:38].replace(chr(10),' ')!r}"
        )

    elapsed_total = round(time.time() - t0, 1)
    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    # ── 汇总 ─────────────────────────────────────────────────────────────────

    print(f"\n{'='*65}")
    print(f"  路由方案 A + C 实验结果  ({n} 题, top_k={args.top_k}, dist_thr={args.dist_threshold})")
    print(f"{'─'*65}")

    rrf_overall = avg(scores["rrf"])
    labels = [
        ("bi (baseline)",    "bi"),
        ("rrf (baseline)",   "rrf"),
        ("graph (baseline)", "graph"),
        ("routing-A (dist)", "A"),
        ("routing-C (adap)", "C"),
    ]
    best_score = max(avg(scores[k]) for _, k in labels)
    print(f"  {'策略':<22} {'Score@'+str(args.top_k):>10}  {'vs RRF':>8}")
    print(f"{'─'*65}")
    for label, key in labels:
        sc = avg(scores[key])
        diff = sc - rrf_overall
        diff_str = f"{diff:+.3f}" if diff != 0 else "   —  "
        marker = " ◀ best" if abs(sc - best_score) < 1e-6 else ""
        print(f"  {label:<22} {sc:>10.3f}  {diff_str:>8}{marker}")

    print(f"\n  按策略细分：")
    print(f"  {'策略':<6} {'题数':>4}  {'bi':>7} {'rrf':>7} {'graph':>7} {'A':>7} {'C':>7}")
    print(f"{'─'*65}")
    for strategy in ["O", "GA", "GB"]:
        sg = by_strategy.get(strategy)
        if not sg:
            continue
        n_s = len(sg["bi"])
        row = [avg(sg[k]) for k in ("bi", "rrf", "graph", "A", "C")]
        print(f"  {strategy:<6} {n_s:>4}  " + "  ".join(f"{v:>7.3f}" for v in row))

    # 路由分布 & oracle 准确率
    oracle_map = {q["question"]: ("rrf" if q.get("strategy") == "O" else "graph") for q in questions}
    A_correct = sum(1 for q, r in zip(questions, route_A) if oracle_map[q["question"]] == r)
    C_correct = sum(1 for q, r in zip(questions, route_C) if oracle_map[q["question"]] == r)
    print(f"\n  路由准确率（与 oracle 一致）：")
    print(f"    A: {A_correct}/{n} = {A_correct/n:.1%}  (rrf={route_A.count('rrf')} graph={route_A.count('graph')})")
    print(f"    C: {C_correct}/{n} = {C_correct/n:.1%}  (rrf={route_C.count('rrf')} graph={route_C.count('graph')})")

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
            "seed":  args.seed,
            "dist_threshold": args.dist_threshold,
            "n_total": n,
        },
        "overall": {k: avg(scores[k]) for k in ("bi", "rrf", "graph", "A", "C")}
        | {"oracle_upper_bound": theory},
        "by_strategy": {
            s: {k: avg(by_strategy[s][k]) for k in ("bi", "rrf", "graph", "A", "C")}
            | {"n": len(by_strategy[s]["bi"])}
            for s in ["O", "GA", "GB"] if s in by_strategy
        },
        "routing_accuracy": {
            "A": round(A_correct / n, 4),
            "C": round(C_correct / n, 4),
            "oracle_count": n,
        },
        "routing_distribution": {
            "A": {"rrf": route_A.count("rrf"), "graph": route_A.count("graph")},
            "C": {"rrf": route_C.count("rrf"), "graph": route_C.count("graph")},
        },
        "elapsed_s": elapsed_total,
    }
    log_path = EVAL_LOG_DIR / f"routing_v2_{ts}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 日志已写入 {log_path.name}")


if __name__ == "__main__":
    main()
