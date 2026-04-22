#!/usr/bin/env python3
"""
一次性脚本：从 ChromaDB 回填 knowledge/{source}.qa.json 缓存文件。

已有 .qa.json 的 source 自动跳过。
不调用任何 LLM，纯读取 ChromaDB metadata 中的 question 字段。

用法：
    cd backend
    python scripts/backfill_qa.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import rag

def main() -> None:
    if not rag.is_available():
        print("RAG 不可用（依赖未安装）")
        return
    if not rag.KNOWLEDGE_DIR.exists():
        print(f"knowledge/ 目录不存在: {rag.KNOWLEDGE_DIR}")
        return

    col = rag._get_knowledge_col()
    if col.count() == 0:
        print("ChromaDB 知识库为空，无数据可回填")
        return

    md_files = sorted(rag.KNOWLEDGE_DIR.glob("*.md"))
    if not md_files:
        print("knowledge/ 目录没有 .md 文件")
        return

    for md_file in md_files:
        source  = md_file.stem
        qa_path = md_file.with_suffix(".qa.json")

        if qa_path.exists():
            data = json.loads(qa_path.read_text(encoding="utf-8"))
            print(f"[skip] {source}  ({len(data)} chunks，已有缓存)")
            continue

        # 取出该 source 在 ChromaDB 中的全部 metadata
        try:
            result = col.get(where={"source": source}, include=["metadatas"])
        except Exception as e:
            print(f"[error] {source}: {e}")
            continue

        metas = result.get("metadatas") or []
        if not metas:
            print(f"[skip] {source}  (ChromaDB 中无数据)")
            continue

        # 按 chunk_id 聚合 question（保持原顺序）
        qa_cache: dict[str, list[str]] = {}
        for meta in metas:
            cid = meta.get("chunk_id", "")
            q   = meta.get("question", "")
            if not cid or not q:
                continue
            if cid not in qa_cache:
                qa_cache[cid] = []
            if q not in qa_cache[cid]:
                qa_cache[cid].append(q)

        if not qa_cache:
            print(f"[skip] {source}  (metadata 中无 question 字段，可能是无 LLM 模式索引的)")
            continue

        qa_path.write_text(
            json.dumps(qa_cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[done] {source}  → {qa_path.name}  ({len(qa_cache)} chunks, "
              f"{sum(len(v) for v in qa_cache.values())} questions)")

    print("\n回填完成")


if __name__ == "__main__":
    main()
