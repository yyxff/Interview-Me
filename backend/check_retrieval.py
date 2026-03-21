#!/usr/bin/env python3
"""
检索召回测试工具

用法:
    python check_retrieval.py "进程间通信"           # 全库（知识库+笔记），有重排
    python check_retrieval.py --knowledge "..."      # 仅知识库
    python check_retrieval.py --notes "..."          # 仅笔记库
    python check_retrieval.py --norerank "..."       # 跳过 cross-encoder 重排
    python check_retrieval.py --top 8 "..."          # 指定返回条数
    python check_retrieval.py                        # 交互模式（全库+重排）
"""

import sys
import argparse
from typing import Literal

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
import rag

# ── 原始向量查询（不带阈值过滤） ────────────────────────────────────────────

RawHit = tuple[str, dict, float]   # (document, metadata, distance)


def _raw_query(col, query: str, n: int) -> list[RawHit]:
    """直接查 ChromaDB，返回带距离的原始结果，不做阈值过滤。"""
    res   = col.query(query_texts=[query], n_results=n,
                      include=["documents", "metadatas", "distances"])
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return [(d, m, dist) for d, m, dist in zip(docs, metas, dists) if d]


def _dedupe_knowledge(hits: list[RawHit]) -> list[RawHit]:
    """按 chunk_id 去重（每个文本片段保留最低距离的命中）。"""
    seen: dict[str, RawHit] = {}
    for doc, meta, dist in hits:
        cid = meta.get("chunk_id", doc[:30])
        if cid not in seen or dist < seen[cid][2]:
            seen[cid] = (doc, meta, dist)
    return sorted(seen.values(), key=lambda x: x[2])


def _dedupe_notes(hits: list[RawHit]) -> list[RawHit]:
    """按 note_id 去重，每条笔记保留最低距离的问题命中。"""
    seen: dict[str, RawHit] = {}
    for doc, meta, dist in hits:
        nid = meta.get("note_id", doc[:20])
        if nid not in seen or dist < seen[nid][2]:
            seen[nid] = (doc, meta, dist)
    return sorted(seen.values(), key=lambda x: x[2])


# ── 展示 ─────────────────────────────────────────────────────────────────────

def _print_header(query: str, scope: str, do_rerank: bool) -> None:
    tag = f"{scope}" + (" + rerank" if do_rerank else " 无重排")
    print(f"\n{'─'*64}")
    print(f"  查询: {query}   [{tag}]")
    print(f"  阈值: distance < {rag.SIMILARITY_DISTANCE_THRESHOLD}")
    print(f"{'─'*64}")


def _print_knowledge_hits(
    hits: list[RawHit],
    ranked: list[tuple[str, dict, float]] | None,
    top_k: int,
) -> None:
    """
    hits   — 去重后的原始结果（含阈值内外）
    ranked — rerank 后顺序（None 表示跳过重排，按 bi-encoder 距离展示）
    """
    threshold = rag.SIMILARITY_DISTANCE_THRESHOLD
    passed   = [h for h in hits if h[2] < threshold]
    filtered = [h for h in hits if h[2] >= threshold]
    print(f"  bi-encoder 通过阈值: {len(passed)}  过滤: {len(filtered)}\n")

    if ranked is not None:
        display = ranked[:top_k]
        label   = "rerank_score"
    else:
        display = [(doc, meta, dist) for doc, meta, dist in passed[:top_k]]
        label   = "bi-encoder_dist"

    if not display:
        print("  （无通过阈值的结果）\n")
        return

    for i, (doc, meta, score) in enumerate(display, 1):
        source   = meta.get("source", "?")
        chapter  = meta.get("chapter", "")
        question = meta.get("question", "")
        chunk_id = meta.get("chunk_id", "?")
        # 找回原始 bi-encoder 距离
        orig_dist = next(
            (d for dd, mm, d in hits
             if mm.get("chunk_id") == chunk_id and dd == doc),
            None,
        )
        loc = source + (f" > {chapter}" if chapter else "")
        print(f"[{i}] {loc}")
        if ranked is not None:
            print(f"    {label}={score:.4f}", end="")
            if orig_dist is not None:
                print(f"  bi-encoder_dist={orig_dist:.4f}", end="")
        else:
            print(f"    {label}={score:.4f}", end="")
        print()
        print(f"    chunk_id : {chunk_id}")
        if question:
            print(f"    命中问题 : {question}")
        preview = doc.replace("\n", " ").strip()[:300]
        print(f"    原文预览 : {preview}{'…' if len(doc) > 300 else ''}")
        print()

    # 展示被过滤的条目（距离太大）
    if filtered:
        print(f"  ── 以下 {len(filtered)} 条超出阈值（供参考）──")
        for doc, meta, dist in filtered:
            chunk_id = meta.get("chunk_id", "?")
            chapter  = meta.get("chapter", "")
            source   = meta.get("source", "?")
            loc      = source + (f" > {chapter}" if chapter else "")
            preview  = doc.replace("\n", " ").strip()[:120]
            print(f"  ✗  dist={dist:.4f}  {loc}")
            print(f"     {preview}{'…' if len(doc) > 120 else ''}")
        print()


def _print_note_hits(
    hits: list[RawHit],
    ranked: list[tuple[str, dict, float]] | None,
    top_k: int,
) -> None:
    threshold = rag.SIMILARITY_DISTANCE_THRESHOLD
    passed   = [h for h in hits if h[2] < threshold]
    filtered = [h for h in hits if h[2] >= threshold]
    print(f"  bi-encoder 通过阈值: {len(passed)}  过滤: {len(filtered)}\n")

    display = ranked[:top_k] if ranked is not None else [(d, m, dist) for d, m, dist in passed[:top_k]]

    if not display:
        print("  （无通过阈值的笔记）\n")
    else:
        for i, (doc, meta, score) in enumerate(display, 1):
            nid   = meta.get("note_id", "?")
            title = meta.get("title", nid)
            orig  = meta.get("text", "")
            orig_dist = next((d for dd, mm, d in hits if mm.get("note_id") == nid and dd == doc), None)

            print(f"[{i}] ✓ {title}")
            print(f"    note_id  : {nid}")
            if ranked is not None:
                print(f"    rerank_score={score:.4f}", end="")
                if orig_dist is not None:
                    print(f"  bi-encoder_dist={orig_dist:.4f}", end="")
            else:
                print(f"    bi-encoder_dist={score:.4f}", end="")
            print()
            print(f"    命中向量 : {doc[:120].replace(chr(10), ' ')}{'…' if len(doc) > 120 else ''}")
            if orig and orig != doc:
                print(f"    原文预览 : {orig[:200].replace(chr(10), ' ')}{'…' if len(orig) > 200 else ''}")
            print()

    if filtered:
        print(f"  ── 以下 {len(filtered)} 条超出阈值（供参考）──")
        for doc, meta, dist in filtered:
            title = meta.get("title", meta.get("note_id", "?"))
            print(f"  ✗  dist={dist:.4f}  {title}")
            print(f"     {doc[:100].replace(chr(10), ' ')}{'…' if len(doc) > 100 else ''}")
        print()


# ── 主查询入口 ────────────────────────────────────────────────────────────────

Scope = Literal["all", "knowledge", "notes"]


def run_query(query: str, scope: Scope, do_rerank: bool, top_k: int) -> None:
    if not rag.is_available():
        print("  RAG 不可用（依赖未安装）")
        return

    _print_header(query, scope, do_rerank)

    # ── 知识库 ──
    if scope in ("all", "knowledge"):
        k_col   = rag._get_knowledge_col()
        k_total = k_col.count()
        print(f"  [知识库] 向量总数: {k_total}")
        if k_total == 0:
            print("  知识库为空，请先上传并索引 EPUB。\n")
        else:
            n_raw  = min(top_k * rag.QA_PER_CHUNK * 2, k_total)
            k_hits = _dedupe_knowledge(_raw_query(k_col, query, n_raw))
            threshold = rag.SIMILARITY_DISTANCE_THRESHOLD
            if do_rerank:
                candidates = [(d, m) for d, m, dist in k_hits if dist < threshold]
                ranked     = rag.rerank(query, candidates)
                print(f"  Reranker 重排完成")
            else:
                ranked = None
            _print_knowledge_hits(k_hits, ranked, top_k)

    # ── 笔记库 ──
    if scope in ("all", "notes"):
        n_col   = rag._get_notes_col()
        n_total = n_col.count()
        print(f"  [笔记库] 向量总数: {n_total}")
        if n_total == 0:
            print("  笔记库为空，请先点击「加入知识库」。\n")
        else:
            n_raw  = min(top_k * 4, n_total)
            n_hits = _dedupe_notes(_raw_query(n_col, query, n_raw))
            threshold = rag.SIMILARITY_DISTANCE_THRESHOLD
            if do_rerank:
                candidates = [(d, m) for d, m, dist in n_hits if dist < threshold]
                ranked     = rag.rerank(query, candidates)
                print(f"  Reranker 重排完成")
            else:
                ranked = None
            _print_note_hits(n_hits, ranked, top_k)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="检索召回测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python check_retrieval.py "进程间通信"          全库+重排
  python check_retrieval.py --knowledge "..."     仅知识库
  python check_retrieval.py --notes "..."         仅笔记库
  python check_retrieval.py --norerank "..."      跳过 cross-encoder
  python check_retrieval.py --top 8 "..."         指定返回条数
  python check_retrieval.py                       交互模式""",
    )
    parser.add_argument("query",      nargs="*",         help="查询内容（省略则进入交互模式）")
    parser.add_argument("--knowledge", action="store_true", help="仅查知识库 collection")
    parser.add_argument("--notes",     action="store_true", help="仅查笔记 collection")
    parser.add_argument("--norerank",  action="store_true", help="跳过 cross-encoder 重排")
    parser.add_argument("--top",       type=int, default=rag.KNOWLEDGE_TOP_K, help="返回条数")
    args = parser.parse_args()

    if args.knowledge and args.notes:
        parser.error("--knowledge 和 --notes 不能同时使用")

    scope: Scope = "knowledge" if args.knowledge else "notes" if args.notes else "all"
    do_rerank    = not args.norerank

    if args.query:
        run_query(" ".join(args.query), scope, do_rerank, args.top)
    else:
        scope_label = {"all": "全库", "knowledge": "知识库", "notes": "笔记库"}[scope]
        rerank_label = "有重排" if do_rerank else "无重排"
        print(f"交互模式 [{scope_label} · {rerank_label}] — 输入查询，回车执行，q 退出")
        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in ("q", "quit", "exit", ""):
                break
            run_query(query, scope, do_rerank, args.top)


if __name__ == "__main__":
    main()
