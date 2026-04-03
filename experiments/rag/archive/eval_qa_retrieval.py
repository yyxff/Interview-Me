"""
eval_qa_retrieval.py — 基于合成 QA 问题的向量检索质量评估

查询来源：.qa.json 中 LLM 合成的问题（每 chunk 4 题），比原文自我检索更接近真实用户提问。
命中判断：文本前缀匹配（当前索引无 chunk_id 元数据，后续重建索引后可改为精确匹配）。

指标：
  Hit@1   — 第 1 条召回就命中目标 chunk 的比例
  Hit@3   — Top-3 中包含目标 chunk 的比例
  Hit@5   — Top-5 中包含目标 chunk 的比例
  MRR@5   — 目标 chunk 在 Top-5 中排名倒数的均值（体现排序质量）

运行方式：
  python3 eval_qa_retrieval.py                   # 默认：50 题，seed=42
  python3 eval_qa_retrieval.py --n 100           # 抽 100 题
  python3 eval_qa_retrieval.py --seed 0          # 换随机种子
  python3 eval_qa_retrieval.py --top-k 5         # 指定召回深度

结果写入 eval_logs/qa_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 路径：支持从 worktree 运行，通过 --backend-dir 指向真实项目 ─────────────────
_SCRIPT_BACKEND = Path(__file__).parent

def _resolve_backend_dir(cli_arg: str | None) -> Path:
    if cli_arg:
        return Path(cli_arg).resolve()
    # 自动探测：优先用 cwd（若含 chroma_db/），其次脚本同级目录
    cwd = Path.cwd()
    if (cwd / "chroma_db").exists():
        return cwd
    return _SCRIPT_BACKEND

# parse --backend-dir early, before importing rag
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--backend-dir", default=None)
_known, _ = _pre.parse_known_args()
BACKEND_DIR = _resolve_backend_dir(_known.backend_dir)

sys.path.insert(0, str(BACKEND_DIR))
EVAL_LOG_DIR = BACKEND_DIR / "eval_logs"
EVAL_LOG_DIR.mkdir(exist_ok=True)

import rag

KNOWLEDGE_DIR = rag.KNOWLEDGE_DIR
MATCH_PREFIX_LEN = 80   # 用原文前 N 字（去空白）判断命中


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return "".join(text.split())


def _is_hit(chunk_text: str, result_text: str) -> bool:
    """result_text 是否包含 chunk_text 的前 MATCH_PREFIX_LEN 字（去空白）。"""
    key = _normalize(chunk_text)[:MATCH_PREFIX_LEN]
    return key in _normalize(result_text)


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_test_cases_from_testset(testset_path: Path) -> list[dict]:
    """从 gen_testset.py 生成的 merged_*.json 加载测试用例。"""
    data = json.loads(testset_path.read_text(encoding="utf-8"))
    cases = []
    for item in data.get("questions", []):
        if item.get("question") and item.get("chunk_text") and item.get("chunk_id"):
            cases.append({
                "question":   item["question"],
                "chunk_text": item["chunk_text"],
                "chunk_id":   item["chunk_id"],
                "source":     item.get("source", ""),
            })
    return cases


def load_test_cases() -> list[dict]:
    """
    从 .qa.json + .chunks.json 构建测试用例列表。
    每条：{ question, chunk_text, chunk_id, source }
    """
    cases = []
    for qa_file in sorted(KNOWLEDGE_DIR.glob("*.qa.json")):
        stem = qa_file.name.replace(".qa.json", "")  # e.g. "图解系统-小林coding-v1.0"
        chunks_file = KNOWLEDGE_DIR / f"{stem}.chunks.json"
        if not chunks_file.exists():
            print(f"[load] 跳过 {qa_file.name}：找不到对应 chunks 文件")
            continue

        try:
            qa_data = json.loads(qa_file.read_text(encoding="utf-8"))
            chunks_data = json.loads(chunks_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[load] 跳过 {qa_file.name}: {e}")
            continue

        # .qa.json 格式: { "{stem}_{idx}": ["问题1", "问题2", ...], ... }
        for chunk_id, questions in qa_data.items():
            # chunk_id 形如 "图解系统-小林coding-v1.0_42"，取尾部数字
            try:
                idx = int(chunk_id.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if idx >= len(chunks_data):
                continue

            chunk = chunks_data[idx]
            chunk_text = chunk.get("text", "")
            if len(chunk_text.strip()) < 50:
                continue

            for q in questions:
                if q and len(q.strip()) > 5:
                    cases.append({
                        "question":   q.strip(),
                        "chunk_text": chunk_text,
                        "chunk_id":   chunk_id,
                        "source":     stem,
                    })

    return cases


# ── 三路检索 ──────────────────────────────────────────────────────────────────

def retrieve_bi(query: str, top_k: int) -> list[str]:
    """纯向量召回，返回 top_k 原文文本列表（去重）。"""
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


def retrieve_graph_only(query: str, top_k: int) -> list[str]:
    """纯图谱召回：实体/关系向量 → source_chunk_ids → 原文。"""
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


def retrieve_rrf(query: str, top_k: int) -> list[str]:
    """bi + graph → RRF → rerank（完整精准模式）。"""
    try:
        import graph_rag as _gr
        graph_result = rag.retrieve_graph(query)
        graph_chunk_ids = graph_result.get("source_chunk_ids", [])
        graph_extra = _gr.get_chunks_by_ids(graph_chunk_ids) if graph_chunk_ids else []
        result = rag.retrieve_rich(query, extra_chunks=graph_extra or None, top_k=top_k)
        return [c["text"] for c in result.get("knowledge", [])][:top_k]
    except Exception as e:
        print(f"  [rrf error] {e}")
        return []


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def compute_metrics(records: list[dict], top_k: int) -> dict:
    n = len(records)
    if n == 0:
        return {}

    hit1 = hit3 = hit5 = mrr = 0.0
    for r in records:
        rank = r["rank"]  # 1-indexed, None = miss
        if rank is not None:
            if rank == 1:
                hit1 += 1
            if rank <= 3:
                hit3 += 1
            if rank <= 5:
                hit5 += 1
            if rank <= top_k:
                mrr += 1.0 / rank

    return {
        "Hit@1":  round(hit1 / n, 4),
        "Hit@3":  round(hit3 / n, 4),
        "Hit@5":  round(hit5 / n, 4),
        f"MRR@{top_k}": round(mrr  / n, 4),
        "n":      n,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

RETRIEVE_FNS = {
    "bi":    retrieve_bi,
    "graph": retrieve_graph_only,
    "rrf":   retrieve_rrf,
}

MODE_LABELS = {
    "bi":    "bi-encoder (Vector only)",
    "graph": "graph-only (Graph RAG)",
    "rrf":   "rrf (Vector + Graph + Rerank)",
}


def run_mode(mode: str, sample: list[dict], top_k: int) -> tuple[list[dict], float]:
    retrieve_fn = RETRIEVE_FNS[mode]
    records = []
    t0 = time.time()
    n = len(sample)

    for i, case in enumerate(sample):
        q         = case["question"]
        chunk_txt = case["chunk_text"]
        source    = case["source"]

        elapsed = time.time() - t0
        eta = (elapsed / i * (n - i)) if i > 0 else 0
        eta_str = f"ETA {eta/60:.1f}min" if i > 0 else "..."
        bar = "█" * (i * 20 // n) + "░" * (20 - i * 20 // n)
        print(f"\r[{bar}] {i+1}/{n} {eta_str}", end="", flush=True)

        t_q = time.time()
        texts = retrieve_fn(q, top_k)
        latency_ms = round((time.time() - t_q) * 1000)

        rank = None
        for j, t in enumerate(texts, 1):
            if _is_hit(chunk_txt, t):
                rank = j
                break

        tag = f"✓@{rank}" if rank else "✗   "
        print(f"\r  [{i+1:>3}/{n}] {tag}  {latency_ms:>4}ms  {source[:18]}  {q[:50]!r}")

        records.append({
            "question":   q,
            "chunk_id":   case["chunk_id"],
            "source":     source,
            "rank":       rank,
            "latency_ms": latency_ms,
        })

    return records, round(time.time() - t0, 1)


def print_summary(mode: str, records: list[dict], elapsed: float, top_k: int):
    metrics = compute_metrics(records, top_k)
    n = len(records)
    hits = sum(1 for r in records if r["rank"] is not None)
    latencies = [r["latency_ms"] for r in records]
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"\n{'='*56}")
    print(f"  模式: {MODE_LABELS[mode]}")
    print(f"{'─'*56}")
    print(f"  Hit@1        {metrics['Hit@1']:>8.3f}  ({int(metrics['Hit@1']*n)}/{n})")
    print(f"  Hit@3        {metrics['Hit@3']:>8.3f}")
    print(f"  Hit@5        {metrics['Hit@5']:>8.3f}  ({hits}/{n} 命中)")
    print(f"  MRR@{top_k}       {metrics[f'MRR@{top_k}']:>8.3f}")
    print(f"{'─'*56}")
    print(f"  Latency P50  {p50:>6} ms")
    print(f"  Latency P95  {p95:>6} ms")
    print(f"  总耗时       {elapsed} s  ({n} 题)")
    print(f"{'='*56}")
    return metrics, {"p50_ms": p50, "p95_ms": p95, "total_s": elapsed}


def main():
    parser = argparse.ArgumentParser(description="基于合成 QA 的 RAG 检索评估（支持三路对比）")
    parser.add_argument("--mode",          choices=["bi", "graph", "rrf", "all"], default="bi",
                        help="检索模式（bi/graph/rrf/all，默认 bi）")
    parser.add_argument("--n",             type=int,  default=None, help="抽取题目数（默认全部）")
    parser.add_argument("--seed",          type=int,  default=42,  help="随机种子（默认 42）")
    parser.add_argument("--top-k",         type=int,  default=5,   help="召回深度（默认 5）")
    parser.add_argument("--testset",       type=Path, default=None, help="直接指定 merged_*.json 测试集文件")
    parser.add_argument("--knowledge-dir", type=Path, default=None, help="knowledge 目录路径")
    parser.add_argument("--backend-dir",   type=Path, default=None, help="后端目录（已在启动时解析）")
    args = parser.parse_args()

    global KNOWLEDGE_DIR
    if args.knowledge_dir:
        KNOWLEDGE_DIR = args.knowledge_dir

    print("[eval] 加载 QA 测试集 ...")
    if args.testset:
        all_cases = load_test_cases_from_testset(args.testset)
        print(f"[eval] 共 {len(all_cases)} 条（来自 {args.testset.name}）")
    else:
        all_cases = load_test_cases()
        print(f"[eval] 共 {len(all_cases)} 条（来自合成 QA 问题）")

    if len(all_cases) == 0:
        print("[eval] 无测试数据，退出。")
        return

    random.seed(args.seed)
    n = args.n if args.n is not None else len(all_cases)
    sample = random.sample(all_cases, min(n, len(all_cases)))
    print(f"[eval] 抽样 {len(sample)} 题  seed={args.seed}  top_k={args.top_k}")
    print(f"[eval] 命中判断：文本前 {MATCH_PREFIX_LEN} 字匹配\n")

    modes = ["bi", "graph", "rrf"] if args.mode == "all" else [args.mode]
    all_results = {}

    for mode in modes:
        print(f"\n[eval] === 运行模式: {MODE_LABELS[mode]} ===")
        records, elapsed = run_mode(mode, sample, args.top_k)
        metrics, latency = print_summary(mode, records, elapsed, args.top_k)
        all_results[mode] = {"metrics": metrics, "latency": latency, "records": records}

    # 多路对比汇总
    if len(modes) > 1:
        print(f"\n{'='*56}")
        print(f"  三路对比汇总  ({len(sample)} 题)")
        print(f"{'─'*56}")
        print(f"  {'模式':<10} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR@5':>7}")
        print(f"{'─'*56}")
        for mode in modes:
            m = all_results[mode]["metrics"]
            print(f"  {mode:<10} {m['Hit@1']:>7.3f} {m['Hit@3']:>7.3f} {m['Hit@5']:>7.3f} {m[f'MRR@{args.top_k}']:>7.3f}")
        print(f"{'='*56}")

    # 写日志
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "modes":        modes,
            "testset":      str(args.testset) if args.testset else "qa.json",
            "match_method": f"文本前 {MATCH_PREFIX_LEN} 字前缀匹配",
            "n":            len(sample),
            "seed":         args.seed,
            "top_k":        args.top_k,
        },
        "results": {
            mode: {"metrics": v["metrics"], "latency": v["latency"]}
            for mode, v in all_results.items()
        },
        "records": {mode: v["records"] for mode, v in all_results.items()},
    }
    log_path = EVAL_LOG_DIR / f"qa_{ts}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 结果已写入 {log_path}")


if __name__ == "__main__":
    main()
