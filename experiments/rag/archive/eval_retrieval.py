"""
检索质量评估脚本（精准模式基线）

查询来源：从 .chunks.json 抽取 30% chunk，用原文前 200 字作为查询。
说明：因知识库集合为旧版索引（无 chunk_id/question 元数据），暂用文本匹配判断命中。
      后续重新索引后可改用 chunk_id 精确匹配，并补充真实用户问题集。

评测三路：
  - bi     : 纯向量召回（_safe_query，无 rerank）
  - rrf    : bi-encoder + graph → RRF → cross-encoder rerank（完整精准模式）
  - graph  : 纯图谱召回（retrieve_graph，取 source_chunk_ids 对应原文）

指标：Hit@1, Hit@3, MRR@3

运行方式：
  conda run -n interview-me python3 eval_retrieval.py [--sample-rate 0.3] [--seed 42] [--top-k 3]

结果写入 eval_logs/YYYYMMDD_HHMMSS.json，方便多次对比。
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 路径设置 ─────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))
EVAL_LOG_DIR = BACKEND_DIR / "eval_logs"
EVAL_LOG_DIR.mkdir(exist_ok=True)

import rag

KNOWLEDGE_DIR = rag.KNOWLEDGE_DIR

# ── 命中判断 ──────────────────────────────────────────────────────────────────
MATCH_PREFIX_LEN = 80  # 用原文前 N 字判断命中（去空白）

def _normalize(text: str) -> str:
    return "".join(text.split())

def _is_hit(query_chunk_text: str, result_text: str) -> bool:
    """判断 result_text 是否命中 query_chunk_text（前缀匹配）。"""
    key = _normalize(query_chunk_text)[:MATCH_PREFIX_LEN]
    return key in _normalize(result_text)

# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_all_chunks() -> list[dict]:
    chunks = []
    for f in sorted(KNOWLEDGE_DIR.glob("*.chunks.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            source = f.stem
            for i, c in enumerate(data):
                text = c.get("text", "")
                if len(text.strip()) < 50:   # 跳过太短的 chunk
                    continue
                chunks.append({
                    "chunk_idx":  i,
                    "source":     c.get("source", source),
                    "chapter":    c.get("chapter", c.get("h2", "")),
                    "text":       text,
                    "query":      text[:200].strip(),  # 用原文前200字作查询
                })
        except Exception as e:
            print(f"[load] 跳过 {f.name}: {e}")
    return chunks

# ── 三路检索 ──────────────────────────────────────────────────────────────────

def retrieve_bi(query: str, top_k: int) -> list[str]:
    """纯向量召回，返回 top-k 文本列表。"""
    try:
        col = rag._get_knowledge_col()
        if col.count() == 0:
            return []
        n = min(top_k * 4, col.count())
        raw = rag._safe_query(col, query, n)
        seen, texts = set(), []
        for doc, meta in raw:
            txt = meta.get("text", doc)
            key = _normalize(txt)[:MATCH_PREFIX_LEN]
            if key not in seen:
                seen.add(key)
                texts.append(txt)
            if len(texts) >= top_k:
                break
        return texts
    except Exception as e:
        print(f"  [bi error] {e}")
        return []


def retrieve_rrf(query: str, top_k: int) -> list[str]:
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
        result = rag.retrieve_rich(query, extra_chunks=graph_extra or None)
        return [c["text"] for c in result.get("knowledge", [])][:top_k]
    except Exception as e:
        print(f"  [rrf error] {e}")
        return []


def retrieve_graph_only(query: str, top_k: int) -> list[str]:
    """纯图谱召回，用 chunk_ids 取原文。"""
    try:
        import graph_rag as _gr
        graph_result = rag.retrieve_graph(query)
        chunk_ids = graph_result.get("source_chunk_ids", [])[:top_k * 2]
        if not chunk_ids:
            return []
        chunks = _gr.get_chunks_by_ids(chunk_ids)
        return [c["text"] for c in chunks[:top_k]]
    except Exception as e:
        print(f"  [graph error] {e}")
        return []

# ── 指标计算 ──────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], top_k: int) -> dict:
    """
    results: [{"hit_ranks": [1, 3, ...] or [] for each mode}, ...]
    返回每路的 Hit@1, Hit@{top_k}, MRR@{top_k}
    """
    modes = ["bi", "rrf", "graph"]
    metrics: dict[str, dict] = {m: {"hit1": 0, "hitk": 0, "mrr": 0.0} for m in modes}
    n = len(results)
    if n == 0:
        return metrics
    for r in results:
        for m in modes:
            rank = r.get(f"{m}_rank")  # 1-indexed, None = miss
            if rank is not None:
                if rank == 1:
                    metrics[m]["hit1"] += 1
                if rank <= top_k:
                    metrics[m]["hitk"] += 1
                    metrics[m]["mrr"] += 1.0 / rank
    for m in modes:
        metrics[m]["hit1"]  = round(metrics[m]["hit1"] / n, 4)
        metrics[m]["hitk"]  = round(metrics[m]["hitk"] / n, 4)
        metrics[m]["mrr"]   = round(metrics[m]["mrr"]  / n, 4)
    return metrics

# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=float, default=0.3)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--top-k",       type=int,   default=3)
    args = parser.parse_args()

    print(f"[eval] 加载 chunks ...")
    all_chunks = load_all_chunks()
    print(f"[eval] 共 {len(all_chunks)} 个有效 chunk")

    random.seed(args.seed)
    sample = random.sample(all_chunks, max(1, int(len(all_chunks) * args.sample_rate)))
    print(f"[eval] 抽样 {len(sample)} 个 (rate={args.sample_rate}, seed={args.seed})")
    print(f"[eval] top_k={args.top_k}  匹配方式: 文本前 {MATCH_PREFIX_LEN} 字\n")

    results = []
    t0 = time.time()
    n_total = len(sample)

    for i, chunk in enumerate(sample):
        query = chunk["query"]
        src   = chunk["source"]

        # ── 进度行 ───────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        avg_s   = elapsed / i if i > 0 else 0
        eta_s   = avg_s * (n_total - i)
        pct     = (i / n_total) * 100
        bar     = "█" * (i * 20 // n_total) + "░" * (20 - i * 20 // n_total)
        eta_str = f"ETA {eta_s/60:.1f}min" if i > 0 else "..."
        print(f"\r[{bar}] {pct:4.0f}% {i+1}/{n_total}  {eta_str}  | {src[:20]} {query[:30].replace(chr(10),' ')!r}", end="", flush=True)

        t_item = time.time()
        bi_texts    = retrieve_bi(query, args.top_k)
        rrf_texts   = retrieve_rrf(query, args.top_k)
        graph_texts = retrieve_graph_only(query, args.top_k)
        item_s = round(time.time() - t_item, 1)

        def find_rank(texts):
            for j, t in enumerate(texts, 1):
                if _is_hit(chunk["text"], t):
                    return j
            return None

        bi_rank    = find_rank(bi_texts)
        rrf_rank   = find_rank(rrf_texts)
        graph_rank = find_rank(graph_texts)

        tag = lambda r: f"✓@{r}" if r else "✗"
        print(f"\r  [{i+1}/{n_total}] {tag(bi_rank):4s} bi | {tag(rrf_rank):4s} rrf | {tag(graph_rank):4s} graph | {item_s}s | {src[:20]} {query[:30].replace(chr(10),' ')!r}")

        results.append({
            "source":     src,
            "chapter":    chunk["chapter"],
            "query":      query,
            "bi_rank":    bi_rank,
            "rrf_rank":   rrf_rank,
            "graph_rank": graph_rank,
        })

    elapsed = round(time.time() - t0, 1)
    metrics = compute_metrics(results, args.top_k)

    # ── 打印汇总 ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"{'':12} {'Hit@1':>8} {'Hit@'+str(args.top_k):>8} {'MRR@'+str(args.top_k):>8}")
    print(f"{'-'*55}")
    for mode, label in [("bi","bi-encoder"), ("rrf","RRF+rerank"), ("graph","graph-only")]:
        m = metrics[mode]
        print(f"  {label:<12} {m['hit1']:>8.3f} {m['hitk']:>8.3f} {m['mrr']:>8.3f}")
    print(f"{'='*55}")
    print(f"样本数={len(sample)}  耗时={elapsed}s")
    print()
    print("⚠️  注意：查询来自 chunk 原文，对 bi-encoder 有利（自我检索），")
    print("    RRF/graph 数值偏低不代表真实效果差，需真实用户问题集才能公平对比。")

    # ── 写日志 ──────────────────────────────────────────────────────────────
    log = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "config": {
            "sample_rate": args.sample_rate,
            "seed":        args.seed,
            "top_k":       args.top_k,
            "match_prefix_len": MATCH_PREFIX_LEN,
            "query_source": "chunk_text[:200]",
            "note": "旧版索引无chunk_id/question，用文本前缀匹配命中。查询来自chunk原文，bi-encoder自我检索偏高，RRF/graph数值不可直接与bi比较，需真实问题集才能公平评测。",
        },
        "metrics":  metrics,
        "summary": {
            "total_chunks": len(all_chunks),
            "sampled":      len(sample),
            "elapsed_s":    elapsed,
        },
        "results": results,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = EVAL_LOG_DIR / f"{ts}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 日志已写入 {log_path}")


if __name__ == "__main__":
    main()
