"""
eval_graph_rerank.py — Graph + Rerank 实验

对比四路检索：
  rrf          — bi + graph 候选 → cross-encoder rerank（现有最强基线）
  graph        — graph BFS → 按图结构截断 top_k（无 rerank，当前 graph 路径）
  graph+rerank — graph BFS → cross-encoder rerank → top_k（新增）
  oracle       — O→rrf / GA/GB→graph+rerank（理论上限）

核心问题：给 graph 候选池加 reranker，能比 graph-only 提升多少？
  - graph 的 BFS 找到的候选顺序是"关系 > 实体 > BFS邻居"，非语义相关性
  - reranker 按 query↔chunk 语义相关度重排，能从候选池里准确选出正确 chunk

运行：
  /opt/homebrew/Caskroom/miniconda/base/envs/interview-me/bin/python eval_graph_rerank.py [--top-k 5] [--seed 42] [--rerank-cands 20]
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
    print(f"[混合集] O={len(o_sample)} | GA={len(ga_questions)} | GB={len(gb_questions)} | 总计={len(mixed)}")
    return mixed


# ── 检索函数 ──────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return "".join(text.split())


def _contains(result_text: str, chunk_text: str) -> bool:
    key = _norm(chunk_text)[:MATCH_PREFIX_LEN]
    return bool(key) and key in _norm(result_text)


def retrieve_rrf(query: str, top_k: int) -> list[tuple[str, str]]:
    """bi + graph 候选 → cross-encoder rerank（现有最强基线）。"""
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
    """Graph BFS → 按图结构顺序截断（无 rerank）。"""
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


def retrieve_graph_rerank(query: str, top_k: int, rerank_cands: int) -> list[tuple[str, str]]:
    """
    Graph BFS → cross-encoder rerank → top_k。

    关键改进：
      - graph BFS 返回的候选顺序是"关系 > 实体 > BFS邻居"（图结构顺序），
        与 query 语义相关性无关
      - cross-encoder 按 query↔chunk_text 语义重排，从候选池中选最相关的 top_k
      - rerank_cands 控制送入 reranker 的候选数（越多越准，但越慢）
    """
    try:
        graph_result = rag.retrieve_graph(query)
        # 取更多候选送 reranker（不再截断到 top_k * 2）
        chunk_ids = graph_result.get("source_chunk_ids", [])[:rerank_cands]
        if not chunk_ids:
            return []

        chunks = _gr.get_chunks_by_ids(chunk_ids)
        if not chunks:
            return []

        # 准备 rerank 输入：(text, meta_dict)
        rerank_input = [
            (c["text"], {"chunk_id": c["chunk_id"], "text": c["text"]})
            for c in chunks
            if c.get("text") and c.get("chunk_id")
        ]
        if not rerank_input:
            return []

        # cross-encoder rerank
        ranked = rag.rerank(query, rerank_input)

        return [(meta["chunk_id"], doc) for doc, meta, _score in ranked[:top_k]]
    except Exception as e:
        print(f"  [graph+rerank error] {e}")
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


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph + Rerank 实验")
    parser.add_argument("--top-k",        type=int, default=5)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--rerank-cands", type=int, default=20,
                        help="送入 reranker 的 graph 候选数（默认 20）")
    args = parser.parse_args()

    print(f"[eval] 初始化 chunk 索引 ...")
    chunk_index = build_chunk_index()
    print(f"  {len(chunk_index)} chunks\n")

    questions = load_mixed_testset(args.seed)
    n = len(questions)

    scores: dict[str, list[float]] = {m: [] for m in ("rrf", "graph", "graph_rr")}
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

        rrf_res     = retrieve_rrf(query, args.top_k)
        graph_res   = retrieve_graph_only(query, args.top_k)
        graph_rr    = retrieve_graph_rerank(query, args.top_k, args.rerank_cands)

        rrf_s    = score_result(rrf_res,   gt_ids, chunk_index)
        graph_s  = score_result(graph_res, gt_ids, chunk_index)
        graph_rr_s = score_result(graph_rr, gt_ids, chunk_index)

        for m, s in [("rrf", rrf_s), ("graph", graph_s), ("graph_rr", graph_rr_s)]:
            scores[m].append(s)
            if strategy not in by_strategy:
                by_strategy[strategy] = {m: [] for m in ("rrf", "graph", "graph_rr")}
            by_strategy[strategy][m].append(s)

        tag = lambda s: ("✓" if s == 1.0 else ("½" if s == 0.5 else "✗"))
        print(
            f"\r  [{i+1:3d}/{n}] [{strategy}] "
            f"{tag(rrf_s)}rrf {tag(graph_s)}gr {tag(graph_rr_s)}gr+rr"
            f"  {query[:42].replace(chr(10),' ')!r}"
        )

    elapsed_total = round(time.time() - t0, 1)
    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    # ── 汇总 ─────────────────────────────────────────────────────────────────

    print(f"\n{'='*65}")
    print(f"  Graph + Rerank 实验  ({n} 题, top_k={args.top_k}, rerank_cands={args.rerank_cands})")
    print(f"{'─'*65}")

    rrf_overall   = avg(scores["rrf"])
    graph_overall = avg(scores["graph"])
    gr_rr_overall = avg(scores["graph_rr"])

    labels = [
        ("rrf (bi+graph+rerank)",  rrf_overall),
        ("graph-only",             graph_overall),
        ("graph+rerank (新)",       gr_rr_overall),
    ]
    best = max(s for _, s in labels)
    print(f"  {'策略':<24} {'Score@'+str(args.top_k):>10}  {'vs graph-only':>14}")
    print(f"{'─'*65}")
    for label, sc in labels:
        diff = sc - graph_overall
        diff_str = f"{diff:+.3f}" if diff != 0 else "    —   "
        marker = " ◀ best" if abs(sc - best) < 1e-6 else ""
        print(f"  {label:<24} {sc:>10.3f}  {diff_str:>14}{marker}")

    print(f"\n  按策略细分：")
    print(f"  {'策略':<6} {'题数':>4}  {'rrf':>7} {'graph':>7} {'gr+rr':>7}  {'gr+rr提升':>10}")
    print(f"{'─'*65}")
    for strategy in ["O", "GA", "GB"]:
        sg = by_strategy.get(strategy)
        if not sg:
            continue
        n_s  = len(sg["rrf"])
        rrf_v  = avg(sg["rrf"])
        gr_v   = avg(sg["graph"])
        rr_v   = avg(sg["graph_rr"])
        delta  = rr_v - gr_v
        print(f"  {strategy:<6} {n_s:>4}  {rrf_v:>7.3f} {gr_v:>7.3f} {rr_v:>7.3f}  {delta:>+10.3f}")

    # oracle（O→rrf, GA/GB→graph+rerank）
    oracle = avg(
        [scores["rrf"][j]      for j, q in enumerate(questions) if q.get("strategy") == "O"] +
        [scores["graph_rr"][j] for j, q in enumerate(questions) if q.get("strategy") != "O"]
    )
    print(f"\n  Oracle（O→rrf, GA/GB→graph+rerank）: {oracle:.3f}")
    print(f"  总耗时: {elapsed_total}s")
    print(f"{'='*65}")

    # ── 写日志 ────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "top_k": args.top_k,
            "seed": args.seed,
            "rerank_cands": args.rerank_cands,
            "n_total": n,
        },
        "overall": {
            "rrf":      rrf_overall,
            "graph":    graph_overall,
            "graph_rr": gr_rr_overall,
            "oracle_rrf_plus_graph_rr": oracle,
        },
        "by_strategy": {
            s: {
                "rrf":      avg(by_strategy[s]["rrf"]),
                "graph":    avg(by_strategy[s]["graph"]),
                "graph_rr": avg(by_strategy[s]["graph_rr"]),
                "n": len(by_strategy[s]["rrf"]),
            }
            for s in ["O", "GA", "GB"] if s in by_strategy
        },
        "elapsed_s": elapsed_total,
    }
    log_path = EVAL_LOG_DIR / f"graph_rerank_{ts}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 日志已写入 {log_path.name}")


if __name__ == "__main__":
    main()
