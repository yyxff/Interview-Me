"""
debug_ge.py — 查看 GE 测试集单题的各路召回详情

用法：
  # 随机抽 2 题
  python debug_ge.py

  # 指定题号（从 1 开始）
  python debug_ge.py --ids 3 11

  # 指定随机种子
  python debug_ge.py --seed 7

  # 修改每路展示 top-k
  python debug_ge.py --top-k 5
"""

import argparse, json, random, sys
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

import rag
import graph_rag as _gr

MATCH_PREFIX_LEN = 60
PREVIEW_LEN = 200  # chunk 预览字数

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

def is_hit(cid, txt, gt_id, chunk_idx):
    if cid == gt_id: return True
    gt_text = chunk_idx.get(gt_id,"")
    return bool(gt_text) and _contains(txt, gt_text)

def retrieve_bi(query, top_k):
    col = rag._get_knowledge_col()
    raw = rag._safe_query(col, query, min(top_k*4, col.count()))
    seen, res = set(), []
    for doc, meta in raw:
        cid = meta.get("chunk_id",""); txt = meta.get("text", doc)
        key = cid if cid else _norm(txt)[:MATCH_PREFIX_LEN]
        if key and key not in seen:
            seen.add(key); res.append((cid, txt))
        if len(res) >= top_k: break
    return res

def retrieve_rrf(query, top_k):
    gr = rag.retrieve_graph(query)
    ids = gr.get("source_chunk_ids",[])
    extra = _gr.get_chunks_by_ids(ids) if ids else []
    result = rag.retrieve_rich(query, extra_chunks=extra or None, top_k=top_k)
    return [(c["chunk_id"], c["text"]) for c in result.get("knowledge",[]) if c.get("chunk_id")][:top_k]

def retrieve_graph_raw(query, top_k, n_cands):
    """返回 graph BFS 候选（rerank 前）和 rerank 后各 top_k 条。"""
    gr = rag.retrieve_graph(query)
    ids = gr.get("source_chunk_ids",[])[:n_cands]
    chunks = _gr.get_chunks_by_ids(ids)
    before = [(c["chunk_id"], c["text"]) for c in chunks if c.get("chunk_id")]

    # rerank
    rerank_input = [(c["text"], {"chunk_id": c["chunk_id"]}) for c in chunks if c.get("text") and c.get("chunk_id")]
    if rerank_input:
        ranked = rag.rerank(query, rerank_input)
        after = [(meta["chunk_id"], doc) for doc, meta, _ in ranked[:top_k]]
    else:
        after = before[:top_k]

    return before[:top_k], after, before  # (graph-only top_k, graph+rr top_k, full candidate list)

def print_chunks(label, results, gt_id, chunk_idx, full_preview=False):
    print(f"\n  ── {label} ──")
    for i, (cid, txt) in enumerate(results, 1):
        hit = is_hit(cid, txt, gt_id, chunk_idx)
        marker = "  ✓ HIT" if hit else ""
        preview = txt[:PREVIEW_LEN].replace("\n"," ")
        print(f"  [{i}] {cid}{marker}")
        print(f"       {preview}…")

def show_question(q_item, chunk_idx, top_k, n_cands):
    query  = q_item["question"]
    gt_id  = q_item["ground_truth_chunk_ids"][0]
    subj   = q_item.get("subject","?")
    pred   = q_item.get("predicate","?")
    obj    = q_item.get("object","?")

    print("\n" + "═"*70)
    print(f"  题目：{query}")
    print(f"  边：  {subj} --{pred}--> {obj}")
    print(f"  GT：  {gt_id}")
    gt_text = chunk_idx.get(gt_id,"（未找到）")
    print(f"  GT预览：{gt_text[:200].replace(chr(10),' ')}…")

    print_chunks("bi-encoder", retrieve_bi(query, top_k), gt_id, chunk_idx)
    print_chunks("RRF+rerank", retrieve_rrf(query, top_k), gt_id, chunk_idx)

    before_top, after_top, all_cands = retrieve_graph_raw(query, top_k, n_cands)
    print_chunks(f"graph BFS-only (top{top_k})", before_top, gt_id, chunk_idx)
    print_chunks(f"graph+rerank  (top{top_k})", after_top, gt_id, chunk_idx)

    # 显示完整候选列表，标出 GT 在第几位
    print(f"\n  ── graph 完整候选（共 {len(all_cands)} 条，rerank 前顺序）──")
    found_at = None
    for i, (cid, txt) in enumerate(all_cands, 1):
        hit = is_hit(cid, txt, gt_id, chunk_idx)
        if hit and found_at is None:
            found_at = i
        marker = f"  ← GT @ rank {i}" if hit else ""
        print(f"    [{i:2d}] {cid}{marker}")
    if found_at is None:
        print(f"    （GT 不在候选列表中）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids",   type=int, nargs="+", help="指定题号（1-based）")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--n",     type=int, default=2, help="随机抽几题（--ids 优先）")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cands", type=int, default=30, help="graph 候选池大小（rerank 前）")
    args = parser.parse_args()

    # 加载测试集
    testsets_dir = BACKEND_DIR / "eval_logs" / "testsets"
    ge_file = max(testsets_dir.glob("testset_GE_*.json"), key=lambda f: f.stat().st_mtime)
    questions = json.loads(ge_file.read_text("utf-8"))["questions"]
    print(f"[load] {ge_file.name}  ({len(questions)} 题)")

    # 选题
    if args.ids:
        selected = [questions[i-1] for i in args.ids if 1 <= i <= len(questions)]
    else:
        random.seed(args.seed)
        selected = random.sample(questions, min(args.n, len(questions)))

    print("[init] 加载模型...")
    chunk_idx = build_chunk_index()

    for q in selected:
        show_question(q, chunk_idx, args.top_k, args.cands)

    print("\n" + "═"*70)


if __name__ == "__main__":
    main()
