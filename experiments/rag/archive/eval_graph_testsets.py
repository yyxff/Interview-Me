"""
图谱 RAG 专项评测脚本

测试集：
  GA — 单边场景化题（ground_truth = 1 个 chunk）
       Hit@K：top-K 结果中含该 chunk_id 即命中
  GB — 二跳路径题（ground_truth = 2 个 chunk）
       Full@K：top-K 中含全部 GT chunk → 1.0
       Partial@K：top-K 中含至少 1 个 GT chunk → 0.5
       Miss：0 个 → 0.0
       平均分即 Score@K

三路对比：bi-encoder / RRF+rerank / graph-only

运行：
  conda run -n interview-me python3 eval_graph_testsets.py [--top-k 5]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

TESTSETS_DIR = BACKEND_DIR / "eval_logs" / "testsets"
EVAL_LOG_DIR = BACKEND_DIR / "eval_logs"

import rag

# ── chunk 原文索引（chunk_id → text）────────────────────────────────────────

def build_chunk_index() -> dict[str, str]:
    """从 .chunks.json 建 chunk_id → text 索引，供命中判断用。"""
    index: dict[str, str] = {}
    for f in sorted(rag.KNOWLEDGE_DIR.glob("*.chunks.json")):
        source = f.stem.replace(".chunks", "")
        try:
            chunks = json.loads(f.read_text(encoding="utf-8"))
            for i, c in enumerate(chunks):
                cid = f"{source}_{i}"
                index[cid] = c.get("text", "")
        except Exception:
            pass
    return index


# ── 三路检索（复用 eval_retrieval 逻辑）────────────────────────────────────

MATCH_PREFIX_LEN = 60

def _norm(text: str) -> str:
    return "".join(text.split())


def _text_contains_chunk(result_text: str, chunk_text: str) -> bool:
    key = _norm(chunk_text)[:MATCH_PREFIX_LEN]
    return key in _norm(result_text)


def retrieve_bi(query: str, top_k: int) -> list[tuple[str, str]]:
    """返回 [(chunk_id, text), ...] top-k 结果。"""
    try:
        col = rag._get_knowledge_col()
        if col.count() == 0:
            return []
        n = min(top_k * 4, col.count())
        raw = rag._safe_query(col, query, n)
        seen_ids: set[str] = set()
        results: list[tuple[str, str]] = []
        for doc, meta in raw:
            cid = meta.get("chunk_id", "")
            txt = meta.get("text", doc)
            dedup_key = cid if cid else "".join(txt.split())[:MATCH_PREFIX_LEN]
            if dedup_key and dedup_key not in seen_ids:
                seen_ids.add(dedup_key)
                results.append((cid, txt))
            if len(results) >= top_k:
                break
        return results
    except Exception as e:
        print(f"  [bi error] {e}")
        return []


def retrieve_rrf(query: str, top_k: int) -> list[tuple[str, str]]:
    """完整精准模式：bi + graph → RRF → rerank。"""
    try:
        graph_result = rag.retrieve_graph(query)
        graph_chunk_ids = graph_result.get("source_chunk_ids", [])
        graph_extra: list[dict] = []
        if graph_chunk_ids:
            try:
                import graph_rag as _gr
                graph_extra = _gr.get_chunks_by_ids(graph_chunk_ids)
            except Exception:
                pass
        result = rag.retrieve_rich(query, extra_chunks=graph_extra or None, top_k=top_k)
        return [(c["chunk_id"], c["text"]) for c in result.get("knowledge", []) if c.get("chunk_id")][:top_k]
    except Exception as e:
        print(f"  [rrf error] {e}")
        return []


def retrieve_graph_only(query: str, top_k: int) -> list[tuple[str, str]]:
    """纯图谱召回（含 BFS 展开）。"""
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


# ── 命中判断 ─────────────────────────────────────────────────────────────────

def score_result(
    retrieved: list[tuple[str, str]],
    gt_chunk_ids: list[str],
    chunk_index: dict[str, str],
) -> float:
    """
    返回 0.0 / 0.5 / 1.0。
    命中条件：retrieved 的 chunk_id 在 gt_chunk_ids 中，
              或者 retrieved 的文本包含 GT chunk 文本的前缀（兼容旧索引无 chunk_id）。
    """
    hit_count = 0
    for cid, txt in retrieved:
        for gt_id in gt_chunk_ids:
            if cid == gt_id:
                hit_count += 1
                break
            gt_text = chunk_index.get(gt_id, "")
            if gt_text and _text_contains_chunk(txt, gt_text):
                hit_count += 1
                break

    n_gt = len(gt_chunk_ids)
    if hit_count == 0:
        return 0.0
    if hit_count >= n_gt:
        return 1.0
    return 0.5  # partial（仅 GB 会出现）


# ── 主流程 ────────────────────────────────────────────────────────────────────

def eval_testset(
    questions: list[dict],
    top_k: int,
    chunk_index: dict[str, str],
    label: str,
) -> dict:
    """跑单个 testset，返回三路指标。"""
    modes = ["bi", "rrf", "graph"]
    scores: dict[str, list[float]] = {m: [] for m in modes}
    latencies: list[float] = []

    n = len(questions)
    for i, q in enumerate(questions):
        query   = q["question"]
        gt_ids  = q["ground_truth_chunk_ids"]
        strategy = q.get("strategy", "?")

        t0 = time.time()
        bi_res    = retrieve_bi(query, top_k)
        rrf_res   = retrieve_rrf(query, top_k)
        graph_res = retrieve_graph_only(query, top_k)
        latencies.append(time.time() - t0)

        bi_s    = score_result(bi_res,    gt_ids, chunk_index)
        rrf_s   = score_result(rrf_res,   gt_ids, chunk_index)
        graph_s = score_result(graph_res, gt_ids, chunk_index)

        scores["bi"].append(bi_s)
        scores["rrf"].append(rrf_s)
        scores["graph"].append(graph_s)

        tag = lambda s: ("✓" if s == 1.0 else ("½" if s == 0.5 else "✗"))
        print(
            f"  [{i+1:3d}/{n}] {tag(bi_s)} bi | {tag(rrf_s)} rrf | {tag(graph_s)} graph"
            f" | {latencies[-1]:.1f}s | {query[:45].replace(chr(10),' ')!r}"
        )

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    metrics = {m: avg(scores[m]) for m in modes}
    return {
        "label":    label,
        "n":        n,
        "top_k":    top_k,
        "metrics":  metrics,
        "latency_p50": round(sorted(latencies)[len(latencies)//2], 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("[eval] 构建 chunk 索引 ...")
    chunk_index = build_chunk_index()
    print(f"  {len(chunk_index)} chunks 已索引\n")

    ga_file = max(TESTSETS_DIR.glob("testset_GA_*.json"), key=lambda f: f.stat().st_mtime)
    gb_file = max(TESTSETS_DIR.glob("testset_GB_*.json"), key=lambda f: f.stat().st_mtime)
    gc_files = sorted(TESTSETS_DIR.glob("testset_GC_*.json"), key=lambda f: f.stat().st_mtime)
    gc_file  = gc_files[-1] if gc_files else None

    all_results = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    testset_list = [(ga_file, "GA"), (gb_file, "GB")]
    if gc_file:
        testset_list.append((gc_file, "GC"))

    for path, label in testset_list:
        data = json.loads(path.read_text(encoding="utf-8"))
        questions = data["questions"]
        print(f"=== {label}  ({len(questions)} 题, top_k={args.top_k}) ===")
        print(f"  文件: {path.name}\n")
        result = eval_testset(questions, args.top_k, chunk_index, label)
        all_results.append(result)

        m = result["metrics"]
        print(f"\n  {'':8} {'bi':>8} {'rrf':>8} {'graph':>8}")
        print(f"  {'Score@'+str(args.top_k):<8} {m['bi']:>8.3f} {m['rrf']:>8.3f} {m['graph']:>8.3f}")
        if label == "GA":
            note = "（GA: Hit@K；0 或 1）"
        else:
            note = "（Full=1.0 / Partial=0.5 / Miss=0）"
        print(f"  {note}\n")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print("=" * 55)
    print(f"{'':12} {'bi':>10} {'rrf':>10} {'graph':>10}")
    print("-" * 55)
    for r in all_results:
        m = r["metrics"]
        winner = max(["bi", "rrf", "graph"], key=lambda k: m[k])
        print(
            f"  {r['label']:<10} {m['bi']:>10.3f} {m['rrf']:>10.3f} {m['graph']:>10.3f}"
            f"  ← {winner}"
        )
    print("=" * 55)

    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"top_k": args.top_k},
        "results": all_results,
        "note": (
            "GA: 单边场景化题，Score=Hit@K（0或1）。"
            "GB: 二跳路径题，Score=Full(1.0)/Partial(0.5)/Miss(0)，衡量图谱多跳召回能力。"
            "GC: 双模态互补题，C1 bi 可命中，C2 只能图谱找到，理论上 RRF 应当胜出。"
        ),
    }
    log_path = EVAL_LOG_DIR / f"graph_eval_{ts}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 日志已写入 {log_path.name}")


if __name__ == "__main__":
    main()
