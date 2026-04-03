"""
eval_metrics_o.py — 策略 O 多指标评测

指标：Hit@1 / Hit@3 / Score@5 / MRR
三路：bi-encoder / RRF+rerank / graph-only
"""
import json, random, sys, time
from pathlib import Path

SCRIPTS_DIR  = Path(__file__).parent
RAG_DIR      = SCRIPTS_DIR.parent
BACKEND_DIR  = RAG_DIR.parent.parent / "backend"
TESTSETS_DIR = RAG_DIR / "testsets"
EVAL_LOG_DIR = RAG_DIR / "logs"
sys.path.insert(0, str(BACKEND_DIR))

import rag, graph_rag as _gr

MATCH_PREFIX_LEN = 60
TOP_K = 5
SEED  = 42

def _norm(t): return "".join(t.split())
def _contains(r, c):
    k = _norm(c)[:MATCH_PREFIX_LEN]
    return bool(k) and k in _norm(r)

def build_chunk_index():
    idx = {}
    for f in sorted(rag.KNOWLEDGE_DIR.glob("*.chunks.json")):
        src = f.stem.replace(".chunks","")
        for i, c in enumerate(json.loads(f.read_text("utf-8"))):
            idx[f"{src}_{i}"] = c.get("text","")
    return idx

def load_o_questions(seed):
    o_raw = json.loads((TESTSETS_DIR/"testset_O_20260331_224445.json").read_text("utf-8"))["questions"]
    qs = [{"question": x["question"], "gt": x["chunk_id"]}
          for x in o_raw if x.get("question") and x.get("chunk_id")]
    random.seed(seed)
    return random.sample(qs, 94)  # 与混合集一致

# ── 三路检索（返回有序列表，保留 rank 信息）────────────────────────────────

def retrieve_bi(query):
    col = rag._get_knowledge_col()
    raw = rag._safe_query(col, query, TOP_K * 4)
    seen, res = set(), []
    for doc, meta in raw:
        cid = meta.get("chunk_id",""); txt = meta.get("text", doc)
        key = cid if cid else _norm(txt)[:MATCH_PREFIX_LEN]
        if key and key not in seen:
            seen.add(key); res.append((cid, txt))
        if len(res) >= TOP_K: break
    return res

def retrieve_rrf(query):
    gr = rag.retrieve_graph(query)
    ids = gr.get("source_chunk_ids",[])
    extra = _gr.get_chunks_by_ids(ids) if ids else []
    result = rag.retrieve_rich(query, extra_chunks=extra or None, top_k=TOP_K)
    return [(c["chunk_id"], c["text"]) for c in result.get("knowledge",[]) if c.get("chunk_id")][:TOP_K]

def retrieve_graph(query):
    gr = rag.retrieve_graph(query)
    ids = gr.get("source_chunk_ids",[])[:TOP_K*2]
    chunks = _gr.get_chunks_by_ids(ids)
    return [(c["chunk_id"], c["text"]) for c in chunks if c.get("chunk_id")][:TOP_K]

# ── 指标计算 ──────────────────────────────────────────────────────────────────

def compute_metrics(retrieved: list[tuple[str,str]], gt_id: str, chunk_idx: dict) -> dict:
    """
    返回 {hit1, hit3, hit5, mrr}
    hit@k = 1 if GT 在 top-k 内，else 0
    mrr   = 1/rank（rank从1开始），未命中=0
    """
    rank = None
    for i, (cid, txt) in enumerate(retrieved, 1):
        if cid == gt_id or (chunk_idx.get(gt_id) and _contains(txt, chunk_idx[gt_id])):
            rank = i
            break
    return {
        "hit1":  int(rank == 1),
        "hit3":  int(rank is not None and rank <= 3),
        "hit5":  int(rank is not None and rank <= 5),
        "mrr":   (1.0 / rank) if rank else 0.0,
    }

# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    print("[eval] 初始化...")
    chunk_idx = build_chunk_index()
    questions = load_o_questions(SEED)
    n = len(questions)
    print(f"  策略 O: {n} 题\n")

    results = {m: {"hit1":[], "hit3":[], "hit5":[], "mrr":[]}
               for m in ("bi","rrf","graph")}

    t0 = time.time()
    for i, q in enumerate(questions):
        query, gt = q["question"], q["gt"]
        print(f"\r  [{i+1:3d}/{n}]", end="", flush=True)

        for label, fn in [("bi", retrieve_bi), ("rrf", retrieve_rrf), ("graph", retrieve_graph)]:
            res = fn(query)
            m = compute_metrics(res, gt, chunk_idx)
            for k, v in m.items():
                results[label][k].append(v)

    elapsed = round(time.time() - t0, 1)

    def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0.0

    print(f"\n\n{'='*60}")
    print(f"  策略 O 多指标评测  (n={n}, top_k={TOP_K})")
    print(f"{'─'*60}")
    print(f"  {'指标':<12} {'bi':>8} {'rrf':>8} {'graph':>8}  {'rrf提升':>10}")
    print(f"{'─'*60}")
    for metric in ("hit1","hit3","hit5","mrr"):
        bi_v  = avg(results["bi"][metric])
        rrf_v = avg(results["rrf"][metric])
        gr_v  = avg(results["graph"][metric])
        delta = rrf_v - bi_v
        label = {"hit1":"Hit@1","hit3":"Hit@3","hit5":"Score@5","mrr":"MRR"}[metric]
        best  = max(bi_v, rrf_v, gr_v)
        markers = {k: (" ◀" if abs(avg(results[k][metric])-best)<1e-6 else "")
                   for k in ("bi","rrf","graph")}
        print(f"  {label:<12} {bi_v:>8.3f}{markers['bi']:<2} {rrf_v:>8.3f}{markers['rrf']:<2} {gr_v:>8.3f}{markers['graph']:<2}  {delta:>+10.3f}")
    print(f"\n  耗时: {elapsed}s")
    print(f"{'='*60}")

    # 写日志
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategy": "O", "n": n, "top_k": TOP_K,
        "results": {m: {k: avg(results[m][k]) for k in ("hit1","hit3","hit5","mrr")}
                    for m in ("bi","rrf","graph")}
    }
    p = EVAL_LOG_DIR/f"metrics_o_{ts}.json"
    p.write_text(json.dumps(log, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n[eval] 日志已写入 {p.name}")

if __name__ == "__main__":
    main()
