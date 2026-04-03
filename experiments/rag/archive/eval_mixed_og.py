"""
eval_mixed_og.py — 混合测试集评测（策略 O + GA + GB，1:1 比例）

测试集构成：
  - 策略 O（单概念事实题）：从 merged_O.json 随机采样，数量 = len(GA) + len(GB)
  - GA（单边场景化题）：全部 50 题，GT=1 chunk
  - GB（二跳路径题）：全部 44 题，GT=2 chunks

评分规则（统一）：
  GA / O : Full=1.0 / Miss=0.0
  GB     : Full=1.0 / Partial=0.5（找到1/2 GT）/ Miss=0.0

运行：
  conda run -n interview-me python3 eval_mixed_og.py [--top-k 5] [--seed 42]
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


# ── 测试集构建 ────────────────────────────────────────────────────────────────

def load_mixed_testset(seed: int) -> list[dict]:
    # GA
    ga_file = max(TESTSETS_DIR.glob("testset_GA_*.json"), key=lambda f: f.stat().st_mtime)
    ga_questions = json.loads(ga_file.read_text(encoding="utf-8"))["questions"]

    # GB
    gb_file = max(TESTSETS_DIR.glob("testset_GB_*.json"), key=lambda f: f.stat().st_mtime)
    gb_questions = json.loads(gb_file.read_text(encoding="utf-8"))["questions"]

    n_graph = len(ga_questions) + len(gb_questions)  # 94

    # 策略 O：转换格式，采样 n_graph 题
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

    # 统一 GA / GB 格式（已经有 ground_truth_chunk_ids）
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


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    print("[eval] 构建 chunk 索引 ...")
    chunk_index = build_chunk_index()
    print(f"  {len(chunk_index)} chunks\n")

    questions = load_mixed_testset(args.seed)
    n = len(questions)

    modes = ["bi", "rrf", "graph"]
    scores: dict[str, list[float]] = {m: [] for m in modes}
    # 按策略分组
    by_strategy: dict[str, dict[str, list[float]]] = {}

    t0 = time.time()
    for i, q in enumerate(questions):
        query    = q["question"]
        gt_ids   = q["ground_truth_chunk_ids"]
        strategy = q.get("strategy", "?")

        elapsed = time.time() - t0
        eta = (elapsed / i * (n - i)) if i > 0 else 0
        bar = "█" * (i * 20 // n) + "░" * (20 - i * 20 // n)
        print(f"\r[{bar}] {i+1}/{n}  ETA {eta/60:.1f}min", end="", flush=True)

        bi_res    = retrieve_bi(query, args.top_k)
        rrf_res   = retrieve_rrf(query, args.top_k)
        graph_res = retrieve_graph_only(query, args.top_k)

        bi_s    = score_result(bi_res,    gt_ids, chunk_index)
        rrf_s   = score_result(rrf_res,   gt_ids, chunk_index)
        graph_s = score_result(graph_res, gt_ids, chunk_index)

        for m, s in [("bi", bi_s), ("rrf", rrf_s), ("graph", graph_s)]:
            scores[m].append(s)
            by_strategy.setdefault(strategy, {m: [] for m in modes})[m].append(s)

        tag = lambda s: ("✓" if s == 1.0 else ("½" if s == 0.5 else "✗"))
        print(f"\r  [{i+1:3d}/{n}] [{strategy}] {tag(bi_s)} bi | {tag(rrf_s)} rrf | {tag(graph_s)} graph  {query[:45].replace(chr(10),' ')!r}")

    elapsed_total = round(time.time() - t0, 1)

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    # ── 汇总打印 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  混合测试集总计  ({n} 题, top_k={args.top_k})")
    print(f"{'─'*60}")
    print(f"  {'模式':<12} {'Score@'+str(args.top_k):>10}  {'样本数':>6}")
    print(f"{'─'*60}")
    for m in modes:
        print(f"  {m:<12} {avg(scores[m]):>10.3f}  {n:>6}")

    print(f"\n  按策略细分：")
    print(f"  {'策略':<6} {'题数':>4}  {'bi':>8} {'rrf':>8} {'graph':>8}  胜者")
    print(f"{'─'*60}")
    for strategy in ["O", "GA", "GB"]:
        sg = by_strategy.get(strategy, {})
        if not sg:
            continue
        n_s = len(sg.get("bi", []))
        m_vals = {m: avg(sg.get(m, [])) for m in modes}
        winner = max(modes, key=lambda m: m_vals[m])
        print(f"  {strategy:<6} {n_s:>4}  {m_vals['bi']:>8.3f} {m_vals['rrf']:>8.3f} {m_vals['graph']:>8.3f}  ← {winner}")
    print(f"{'='*60}")
    print(f"  总耗时 {elapsed_total}s")

    # ── 写日志 ────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"top_k": args.top_k, "seed": args.seed, "n_total": n},
        "overall": {m: avg(scores[m]) for m in modes},
        "by_strategy": {
            s: {m: avg(v.get(m, [])) for m in modes}
            for s, v in by_strategy.items()
        },
        "n_by_strategy": {s: len(v.get("bi", [])) for s, v in by_strategy.items()},
        "elapsed_s": elapsed_total,
    }
    log_path = EVAL_LOG_DIR / f"mixed_og_{ts}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 日志已写入 {log_path.name}")


if __name__ == "__main__":
    main()
