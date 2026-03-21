#!/usr/bin/env python3
"""
检索召回测试工具

用法:
    python check_retrieval.py "什么是 Redis 持久化"     # 查知识库
    python check_retrieval.py --notes "进程间通信"       # 查笔记 collection
    python check_retrieval.py                            # 交互模式
    python check_retrieval.py --top 5 "..."              # 指定返回条数
"""

import sys
import argparse

# 加载 rag 模块
sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
import rag


def show_notes_results(query: str, top_k: int = 5) -> None:
    """查询笔记 collection，显示距离（不走 reranker）。"""
    print(f"\n{'─'*60}")
    print(f"  查询 [笔记库]: {query}")
    print(f"{'─'*60}")

    if not rag.is_available():
        print("  RAG 不可用（依赖未安装）")
        return

    col   = rag._get_notes_col()
    total = col.count()
    if total == 0:
        print("  笔记库为空，请先加入知识库。")
        return

    print(f"  笔记向量总数: {total}")
    print(f"  相似度阈值(distance): < {rag.SIMILARITY_DISTANCE_THRESHOLD}\n")

    n_raw = min(top_k * 2, total)
    raw   = col.query(
        query_texts=[query],
        n_results=n_raw,
        include=["documents", "metadatas", "distances"],
    )
    docs      = raw.get("documents",  [[]])[0]
    metas     = raw.get("metadatas",  [[]])[0]
    distances = raw.get("distances",  [[]])[0]

    # 按 note_id 去重，展示每个笔记最佳命中
    seen: dict[str, tuple[str, dict, float]] = {}
    for doc, meta, dist in zip(docs, metas, distances):
        nid = meta.get("note_id", doc[:20])
        if nid not in seen or dist < seen[nid][2]:
            seen[nid] = (doc, meta, dist)

    threshold = rag.SIMILARITY_DISTANCE_THRESHOLD
    for i, (nid, (doc, meta, dist)) in enumerate(
        sorted(seen.items(), key=lambda x: x[1][2]), 1
    ):
        passed = dist < threshold
        mark   = "✓" if passed else "✗"
        title  = meta.get("title", nid)
        print(f"[{i}] {mark} {title}")
        print(f"    note_id  : {nid}")
        print(f"    distance : {dist:.4f}  {'← 通过阈值' if passed else '← 被过滤'}")
        print(f"    命中向量 : {doc[:120].replace(chr(10), ' ')}{'…' if len(doc) > 120 else ''}")
        orig = meta.get("text", "")
        if orig and orig != doc:
            print(f"    原文预览 : {orig[:200].replace(chr(10), ' ')}{'…' if len(orig) > 200 else ''}")
        print()


def show_results(query: str, top_k: int = rag.KNOWLEDGE_TOP_K) -> None:
    print(f"\n{'─'*60}")
    print(f"  查询 [知识库]: {query}")
    print(f"{'─'*60}")

    if not rag.is_available():
        print("  RAG 不可用（依赖未安装）")
        return

    col = rag._get_knowledge_col()
    total_vectors = col.count()
    if total_vectors == 0:
        print("  知识库为空，请先上传并索引 EPUB。")
        return

    print(f"  知识库向量总数: {total_vectors}")
    print(f"  相似度阈值(distance): < {rag.SIMILARITY_DISTANCE_THRESHOLD}")

    threshold = rag.SIMILARITY_DISTANCE_THRESHOLD
    n_raw     = min(top_k * rag.QA_PER_CHUNK, total_vectors)
    results_raw = col.query(
        query_texts=[query],
        n_results=n_raw,
        include=["documents", "metadatas", "distances"],
    )
    docs      = results_raw.get("documents",  [[]])[0]
    metas     = results_raw.get("metadatas",  [[]])[0]
    distances = results_raw.get("distances",  [[]])[0]

    seen: set[str] = set()
    results: list[tuple[str, dict, float]] = []
    for doc, meta, dist in zip(docs, metas, distances):
        if not doc:
            continue
        chunk_id = meta.get("chunk_id", doc[:30])
        if chunk_id not in seen:
            seen.add(chunk_id)
            results.append((doc, meta, dist))
            if len(results) >= top_k * 2:
                break

    passed   = [(d, m, dist) for d, m, dist in results if dist < threshold]
    filtered = [(d, m, dist) for d, m, dist in results if dist >= threshold]
    print(f"  通过阈值: {len(passed)} 条  过滤掉: {len(filtered)} 条")

    candidates = [(doc, meta) for doc, meta, dist in results if dist < threshold]
    ranked = rag.rerank(query, candidates)
    print(f"  Reranker 重排完成\n")

    for i, (doc, meta, score) in enumerate(ranked[:top_k * 2], 1):
        source   = meta.get("source", "?")
        chapter  = meta.get("chapter", "")
        question = meta.get("question", "")
        chunk_id = meta.get("chunk_id", "?")
        orig_dist = next(
            (d for dd, _, d in results
             if meta.get("chunk_id") == _.get("chunk_id") and dd == doc),
            None,
        )
        loc = source + (f" > {chapter}" if chapter else "")
        print(f"[{i}] {loc}")
        print(f"    rerank_score={score:.4f}", end="")
        if orig_dist is not None:
            print(f"  bi-encoder_dist={orig_dist:.4f}", end="")
        print()
        print(f"    chunk_id : {chunk_id}")
        if question:
            print(f"    命中问题 : {question}")
        preview = doc.replace("\n", " ").strip()[:300]
        print(f"    原文预览 : {preview}{'…' if len(doc) > 300 else ''}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="检索召回测试")
    parser.add_argument("query", nargs="*", help="查询内容（省略则进入交互模式）")
    parser.add_argument("--top",   type=int,  default=rag.KNOWLEDGE_TOP_K, help="返回条数")
    parser.add_argument("--notes", action="store_true", help="查询笔记 collection")
    args = parser.parse_args()

    fn = show_notes_results if args.notes else show_results

    if args.query:
        fn(" ".join(args.query), top_k=args.top)
    else:
        mode = "笔记库" if args.notes else "知识库"
        print(f"交互模式 [{mode}] — 输入查询，回车执行，输入 q 退出")
        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in ("q", "quit", "exit", ""):
                break
            fn(query, top_k=args.top)


if __name__ == "__main__":
    main()
