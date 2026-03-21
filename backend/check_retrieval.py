#!/usr/bin/env python3
"""
检索召回测试工具

用法:
    python check_retrieval.py "什么是 Redis 持久化"   # 单次查询
    python check_retrieval.py                          # 交互模式
    python check_retrieval.py --top 5 "..."            # 指定返回条数
"""

import sys
import argparse

# 加载 rag 模块
sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
import rag


def show_results(query: str, top_k: int = rag.KNOWLEDGE_TOP_K, show_raw: bool = False) -> None:
    print(f"\n{'─'*60}")
    print(f"  查询: {query}")
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

    # 直接调 ChromaDB 获取距离，方便调阈值
    n_raw = min(top_k * rag.QA_PER_CHUNK, total_vectors)
    results_raw = col.query(
        query_texts=[query],
        n_results=n_raw,
        include=["documents", "metadatas", "distances"],
    )
    docs      = results_raw.get("documents",  [[]])[0]
    metas     = results_raw.get("metadatas",  [[]])[0]
    distances = results_raw.get("distances",  [[]])[0]

    # 去重 + 标记是否超阈值
    threshold = rag.SIMILARITY_DISTANCE_THRESHOLD
    threshold = 0.35
    seen: set[str] = set()
    results: list[tuple[str, dict, float]] = []
    for doc, meta, dist in zip(docs, metas, distances):
        if not doc:
            continue
        chunk_id = meta.get("chunk_id", doc[:30])
        if chunk_id not in seen:
            seen.add(chunk_id)
            results.append((doc, meta, dist))
            if len(results) >= top_k * 2:  # 多拿一些，显示被过滤的
                break

    passed   = [(d, m, dist) for d, m, dist in results if dist < threshold]
    filtered = [(d, m, dist) for d, m, dist in results if dist >= threshold]

    print(f"  通过阈值: {len(passed)} 条  过滤掉: {len(filtered)} 条")

    # Cross-encoder 重排
    candidates = [(doc, meta) for doc, meta, dist in results if dist < threshold]
    ranked = rag.rerank(query, candidates)  # [(doc, meta, score), ...]
    print(f"  Reranker 重排完成\n")

    # 显示重排后的顺序
    for i, (doc, meta, score) in enumerate(ranked[:top_k * 2], 1):
        source   = meta.get("source", "?")
        chapter  = meta.get("chapter", "")
        question = meta.get("question", "")
        chunk_id = meta.get("chunk_id", "?")
        # 找回原始 distance
        orig_dist = next((d for dd, _, d in results if meta.get("chunk_id") == _.get("chunk_id") if dd == doc), None)

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
    parser.add_argument("--top", type=int, default=rag.KNOWLEDGE_TOP_K, help="返回条数")
    args = parser.parse_args()

    if args.query:
        show_results(" ".join(args.query), top_k=args.top)
    else:
        print("交互模式 — 输入查询，回车执行，输入 q 退出")
        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in ("q", "quit", "exit", ""):
                break
            show_results(query, top_k=args.top)


if __name__ == "__main__":
    main()
