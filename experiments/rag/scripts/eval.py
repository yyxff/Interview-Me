"""
eval.py — RAG 检索评测主脚本

用法示例：
  # O 类题，测 bi / rrf / rrf_path_rr
  python eval.py --dataset O --methods bi rrf rrf_path_rr

  # GE 类题，测所有方法
  python eval.py --dataset GE --methods all

  # O+GE 混合，测主要方法 + Plan A 路由
  python eval.py --dataset combined --methods bi rrf graph rrf_path_rr plan_a

  # debug 模式：少量题目，逐题展示各路召回的 chunk 内容和 GT 排名
  python eval.py --dataset GE --methods bi graph rrf_path_rr --debug
  python eval.py --dataset GE --methods bi graph rrf_path_rr --debug --n 1 --ids 3

可选方法（--methods）：
  bi, rrf, graph, rrf_nr, rrf_w, graph_rr, hyde, hyde_rrf_w,
  graph_path_rr, rrf_path_rr, routed, plan_a, all

日志写入：logs/eval_<dataset>_<timestamp>.json（debug 模式不写日志）
"""

import argparse, json, random, sys, time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR  = Path(__file__).parent
RAG_DIR      = SCRIPTS_DIR.parent
BACKEND_DIR  = RAG_DIR.parent.parent / "backend"
TESTSETS_DIR = RAG_DIR / "testsets"
EVAL_LOG_DIR = RAG_DIR / "logs"
sys.path.insert(0, str(BACKEND_DIR))

from methods import (
    METHODS, ROUTING_METHODS,
    build_chunk_index, _norm,
    retrieve_plan_a,
)

import rag
import graph_rag as _gr

METRICS_KEYS    = ("hit1", "hit3", "hit5", "mrr")
MATCH_PREFIX_LEN = 60
PREVIEW_LEN      = 200

ALL_METHOD_NAMES = list(METHODS.keys()) + ["plan_a"]


# ── 数据集加载 ────────────────────────────────────────────────────────────────

def load_o_testset(path: Path, n: int, seed: int, ids: list[int] = None) -> list[dict]:
    data = json.loads(path.read_text("utf-8"))
    qs   = data.get("questions", data) if isinstance(data, dict) else data
    if ids:
        qs = [qs[i - 1] for i in ids if 1 <= i <= len(qs)]
    else:
        random.seed(seed)
        qs = random.sample(qs, min(n, len(qs)))
    return [{"question": q["question"], "gt_id": q["chunk_id"], "strategy": "O"} for q in qs]

def load_ge_testset(path: Path, n: int, seed: int, ids: list[int] = None) -> list[dict]:
    data = json.loads(path.read_text("utf-8"))
    qs   = data.get("questions", [])
    if ids:
        qs = [qs[i - 1] for i in ids if 1 <= i <= len(qs)]
    else:
        random.seed(seed + 1)
        qs = random.sample(qs, min(n, len(qs)))
    return [{"question": q["question"],
             "gt_id":    q["ground_truth_chunk_ids"][0],
             "strategy": "GE",
             "_meta":    q} for q in qs]


# ── 评분 ──────────────────────────────────────────────────────────────────────

def compute_metrics(retrieved: list, gt_id: str, chunk_idx: dict) -> dict:
    rank = None
    gt_norm = _norm(chunk_idx.get(gt_id, ""))[:MATCH_PREFIX_LEN]
    for i, (cid, txt) in enumerate(retrieved, 1):
        if cid == gt_id or (gt_norm and gt_norm in _norm(txt)):
            rank = i; break
    return {
        "hit1": int(rank == 1),
        "hit3": int(rank is not None and rank <= 3),
        "hit5": int(rank is not None and rank <= 5),
        "mrr":  (1.0 / rank if rank else 0.0),
    }

def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

def is_hit(cid, txt, gt_id, chunk_idx):
    if cid == gt_id: return True
    gt_norm = _norm(chunk_idx.get(gt_id, ""))[:MATCH_PREFIX_LEN]
    return bool(gt_norm) and gt_norm in _norm(txt)


# ── debug 输出 ────────────────────────────────────────────────────────────────

def _print_results(label, results, gt_id, chunk_idx):
    print(f"\n  ── {label} ──")
    for i, (cid, txt) in enumerate(results, 1):
        hit    = is_hit(cid, txt, gt_id, chunk_idx)
        marker = "  ✓ HIT" if hit else ""
        print(f"  [{i}] {cid}{marker}")
        print(f"       {txt[:PREVIEW_LEN].replace(chr(10), ' ')}…")

def _print_graph_candidates(query, gt_id, chunk_idx, n_cands=30):
    """展示 graph BFS 完整候选列表，标出 GT 排名（rerank 前）。"""
    gr      = rag.retrieve_graph(query)
    raw_ids = gr.get("source_chunk_ids", [])[:n_cands]
    chunks  = _gr.get_chunks_by_ids(raw_ids)
    cands   = [(c["chunk_id"], c.get("text", "")) for c in chunks if c.get("chunk_id")]

    print(f"\n  ── graph 完整候选（共 {len(cands)} 条，rerank 前顺序）──")
    found_at = None
    for i, (cid, txt) in enumerate(cands, 1):
        hit = is_hit(cid, txt, gt_id, chunk_idx)
        if hit and found_at is None:
            found_at = i
        marker = f"  ← GT @ rank {i}" if hit else ""
        print(f"    [{i:2d}] {cid}{marker}")
    if found_at is None:
        print("    （GT 不在候选列表中）")

def show_debug(q, results_by_method, gt_id, chunk_idx, methods, top_k):
    query = q["question"]
    meta  = q.get("_meta", {})

    print("\n" + "═" * 70)
    print(f"  题目：{query}")
    if meta.get("subject"):
        print(f"  边：  {meta['subject']} --{meta['predicate']}--> {meta['object']}")
    print(f"  GT：  {gt_id}")
    gt_preview = chunk_idx.get(gt_id, "（未找到）")[:PREVIEW_LEN].replace("\n", " ")
    print(f"  GT预览：{gt_preview}…")

    for m in methods:
        result = results_by_method[m]
        met    = compute_metrics(result, gt_id, chunk_idx)
        hit_str = f"Hit@1={met['hit1']} Hit@3={met['hit3']} Score@5={met['hit5']} MRR={met['mrr']:.3f}"
        _print_results(f"{m}  [{hit_str}]", result, gt_id, chunk_idx)

    # 如果 graph 在方法里，额外展示完整候选排名
    if "graph" in methods or any(m in methods for m in ("graph_rr", "graph_path_rr", "rrf_path_rr")):
        _print_graph_candidates(query, gt_id, chunk_idx)


# ── 结果表格 ──────────────────────────────────────────────────────────────────

def print_table(title, mdict, methods, n, top_k):
    W = 10
    print(f"\n{'='*72}")
    print(f"  {title}  (n={n}, top_k={top_k})")
    print(f"{'─'*72}")
    print(f"  {'指标':<10}" + "".join(f" {m:>{W}}" for m in methods))
    print(f"{'─'*72}")
    for metric in METRICS_KEYS:
        label = {"hit1": "Hit@1", "hit3": "Hit@3", "hit5": "Score@5", "mrr": "MRR"}[metric]
        vals  = {m: avg(mdict[m][metric]) for m in methods}
        best  = max(vals.values()) if vals else 0
        marks = {m: "◀" if abs(vals[m] - best) < 1e-6 else " " for m in vals}
        print(f"  {label:<10}" +
              "".join(f" {vals[m]:>{W}.3f}{marks[m]}" for m in methods))
    print(f"{'='*72}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="RAG 检索评测")
    p.add_argument("--dataset",  choices=["O", "GE", "combined"], default="combined",
                   help="测试集：O（事实型）/ GE（关系型）/ combined（O+GE 混合）")
    p.add_argument("--methods",  nargs="+", default=["bi", "rrf", "graph", "rrf_path_rr"],
                   help=f"检索方法，可多选或填 all。可选: {ALL_METHOD_NAMES}")
    p.add_argument("--n",        type=int,   default=30,   help="每个数据集的题数")
    p.add_argument("--top-k",    type=int,   default=5)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--threshold",type=float, default=0.26, help="plan_a 实体距离阈值")
    p.add_argument("--o-testset", default=str(TESTSETS_DIR / "testset_O_20260331_224445.json"))
    p.add_argument("--ge-testset",default=str(TESTSETS_DIR / "testset_GE_20260402_185222.json"))
    # debug 模式
    p.add_argument("--debug",    action="store_true",
                   help="逐题展示各路召回的 chunk 内容和 GT 排名，不写日志")
    p.add_argument("--ids",      type=int, nargs="+",
                   help="debug 模式下指定题号（1-based），不指定则随机抽 --n 题")
    args = p.parse_args()

    # debug 模式默认只看 2 题
    if args.debug and args.n == 30 and not args.ids:
        args.n = 2

    # 解析方法列表
    methods = ALL_METHOD_NAMES if args.methods == ["all"] else args.methods
    invalid = [m for m in methods if m not in ALL_METHOD_NAMES]
    if invalid:
        print(f"未知方法: {invalid}。可选: {ALL_METHOD_NAMES}")
        sys.exit(1)

    print("[eval] 初始化...")
    chunk_idx = build_chunk_index()
    print(f"  chunks: {len(chunk_idx)}")

    G = None
    if "plan_a" in methods:
        G = _gr._load_all_graphs_into_nx()
        print(f"  图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # 加载测试集
    questions = []
    if args.dataset in ("O", "combined"):
        o_qs = load_o_testset(Path(args.o_testset), args.n, args.seed,
                              ids=args.ids if args.dataset == "O" else None)
        questions += o_qs
        print(f"  O: {len(o_qs)} 题")
    if args.dataset in ("GE", "combined"):
        ge_qs = load_ge_testset(Path(args.ge_testset), args.n, args.seed,
                                ids=args.ids if args.dataset == "GE" else None)
        questions += ge_qs
        print(f"  GE: {len(ge_qs)} 题")

    if not args.ids:
        random.seed(args.seed)
        random.shuffle(questions)
    print(f"  总计: {len(questions)} 题，方法: {methods}\n")

    # 累计器
    strats    = sorted({q["strategy"] for q in questions})
    strat_m   = {s: {m: {k: [] for k in METRICS_KEYS} for m in methods} for s in strats}
    all_m     = {m: {k: [] for k in METRICS_KEYS} for m in methods}
    route_acc = {m: {"O": [], "GE": []} for m in methods if m in ROUTING_METHODS}

    t0 = time.time()
    for i, q in enumerate(questions):
        query, gt_id, strat = q["question"], q["gt_id"], q["strategy"]
        results_by_method = {}
        row = []

        for m in methods:
            decision = None
            if m == "plan_a":
                result, decision = retrieve_plan_a(query, args.top_k, G, args.threshold)
            elif m in ROUTING_METHODS:
                result, decision = METHODS[m](query, args.top_k)
            else:
                result = METHODS[m](query, args.top_k)

            results_by_method[m] = result
            met = compute_metrics(result, gt_id, chunk_idx)
            for k in METRICS_KEYS:
                all_m[m][k].append(met[k])
                strat_m[strat][m][k].append(met[k])

            if decision and m in route_acc and strat in route_acc[m]:
                expected = "rrf" if strat == "O" else "graph"
                route_acc[m][strat].append(int(decision == expected))

            row.append(f"{m}:{'✓' if met['hit5'] else '✗'}")

        print(f"  [{i+1:2d}/{len(questions)}][{strat}] " + "  ".join(row))

        if args.debug:
            show_debug(q, results_by_method, gt_id, chunk_idx, methods, args.top_k)

    elapsed = round(time.time() - t0, 1)

    if args.debug:
        print("\n" + "═" * 70)
        print("  [debug 模式，不写日志]")
        return

    # 打印汇总表格
    for s in strats:
        n_s = sum(1 for q in questions if q["strategy"] == s)
        print_table(f"{s} 策略", strat_m[s], methods, n_s, args.top_k)
    if len(strats) > 1:
        print_table("综合", all_m, methods, len(questions), args.top_k)

    for m, acc in route_acc.items():
        all_acc = acc.get("O", []) + acc.get("GE", [])
        if all_acc:
            print(f"\n  [{m}] 路由准确率  O={avg(acc.get('O',[])):.1%}"
                  f"  GE={avg(acc.get('GE',[])):.1%}  综合={avg(all_acc):.1%}")

    print(f"\n  耗时: {elapsed}s")

    # 写日志
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "dataset": args.dataset, "methods": methods,
            "n": args.n, "top_k": args.top_k, "seed": args.seed,
            "threshold": args.threshold,
            "o_testset":  args.o_testset,
            "ge_testset": args.ge_testset,
        },
        "results": {
            s: {m: {k: avg(strat_m[s][m][k]) for k in METRICS_KEYS} for m in methods}
            for s in strats
        },
    }
    if len(strats) > 1:
        log["results"]["ALL"] = {
            m: {k: avg(all_m[m][k]) for k in METRICS_KEYS} for m in methods
        }
    if route_acc:
        log["routing"] = {
            m: {"O_acc": avg(v.get("O", [])), "GE_acc": avg(v.get("GE", [])),
                "all_acc": avg(v.get("O", []) + v.get("GE", []))}
            for m, v in route_acc.items() if v.get("O") or v.get("GE")
        }

    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_LOG_DIR / f"eval_{args.dataset.lower()}_{ts}.json"
    out.write_text(json.dumps(log, ensure_ascii=False, indent=2), "utf-8")
    print(f"[eval] 日志: {out.name}")


if __name__ == "__main__":
    main()
