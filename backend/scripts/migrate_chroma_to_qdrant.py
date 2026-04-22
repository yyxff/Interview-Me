"""
一次性迁移脚本：Chroma → Qdrant
================================

将 chroma_db/ 中所有集合的向量直接搬到 qdrant_db/，无需重新运行 Embedding 模型。

迁移策略：
  Dense  向量 → 直接从 Chroma 读取原始向量，原样写入 Qdrant
  Sparse 向量 → 从 payload 中的原文本实时计算（纯字符 hash，毫秒级）
  Point ID   → UUID5(原始字符串 ID)，与 client.py 中 _str_id() 保持一致

用法：
  cd backend/
  python scripts/migrate_chroma_to_qdrant.py

前提：
  - conda activate interview-me
  - pip install qdrant-client chromadb
"""
from __future__ import annotations

import re
import sys
import uuid
from collections import Counter
from pathlib import Path

# ── 路径 ────────────────────────────────────────────────────────────────────
CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
QDRANT_PATH = Path(__file__).parent.parent / "qdrant_db"

# client.py 中使用的同一个 namespace
_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

DENSE_DIM = 512  # BAAI/bge-small-zh-v1.5 输出维度
BATCH     = 128  # 每批写入 Qdrant 的 point 数


# ── helpers ──────────────────────────────────────────────────────────────────

def _str_id(s: str) -> str:
    """与 client.py._str_id() 完全一致，确保 ID 映射相同。"""
    return str(uuid.uuid5(_NS, s))


def _text_to_sparse(text: str):
    """与 client.py._text_to_sparse() 完全一致。"""
    from qdrant_client.models import SparseVector

    tokens: list[str] = []
    for w in re.findall(r"[A-Za-z][A-Za-z0-9_]*|[0-9]+", text):
        tokens.append(w.lower())
    for i, ch in enumerate(text):
        if "\u4e00" <= ch <= "\u9fff":
            tokens.append(ch)
            if i + 1 < len(text) and "\u4e00" <= text[i + 1] <= "\u9fff":
                tokens.append(text[i : i + 2])

    idx_map: dict[int, float] = {}
    for token, cnt in Counter(tokens).items():
        idx = abs(hash(token)) % 32768
        idx_map[idx] = idx_map.get(idx, 0.0) + float(cnt)

    return SparseVector(indices=list(idx_map), values=[idx_map[i] for i in idx_map])


def _ensure_collection(qdrant, name: str) -> None:
    from qdrant_client.models import (
        Distance, VectorParams,
        SparseVectorParams, SparseIndexParams,
    )
    existing = {c.name for c in qdrant.get_collections().collections}
    if name in existing:
        print(f"  [skip] collection '{name}' 已存在，跳过创建")
        return
    qdrant.create_collection(
        collection_name       = name,
        vectors_config        = {
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config = {
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    print(f"  [created] collection '{name}'")


# ── 核心迁移逻辑 ──────────────────────────────────────────────────────────────

def migrate_collection(chroma, qdrant, col_name: str) -> int:
    """把一个 Chroma collection 完整迁移到 Qdrant，返回迁移的 point 数。"""
    from qdrant_client.models import PointStruct

    print(f"\n{'='*55}")
    print(f"  迁移: {col_name}")

    chroma_col = chroma.get_collection(col_name)
    total      = chroma_col.count()
    print(f"  Chroma 向量数: {total}")

    if total == 0:
        print(f"  [skip] 空集合，跳过")
        return 0

    _ensure_collection(qdrant, col_name)

    # 检查 Qdrant 是否已有数据（幂等：可重复执行）
    existing = qdrant.count(col_name).count
    if existing >= total:
        print(f"  [skip] Qdrant 已有 {existing} 条，无需重迁移")
        return 0

    # 分批从 Chroma 读取并写入 Qdrant
    offset    = 0
    migrated  = 0

    while offset < total:
        batch_size = min(BATCH, total - offset)

        # Chroma 按 offset 分页（用 limit + offset）
        raw = chroma_col.get(
            limit   = batch_size,
            offset  = offset,
            include = ["embeddings", "documents", "metadatas"],
        )

        ids        = raw["ids"]
        embeddings = raw["embeddings"]   # list[list[float]]，已是 512d
        documents  = raw["documents"]
        metadatas  = raw["metadatas"]

        points = []
        for sid, dvec, doc, meta in zip(ids, embeddings, documents, metadatas):
            sparse = _text_to_sparse(doc)
            points.append(PointStruct(
                id      = _str_id(sid),
                vector  = {"dense": dvec, "sparse": sparse},
                payload = {**meta, "_document": doc, "_original_id": sid},
            ))

        qdrant.upsert(col_name, points=points)
        migrated += len(points)
        offset   += batch_size

        pct = migrated / total * 100
        print(f"  {migrated:>5}/{total}  ({pct:.0f}%)", end="\r", flush=True)

    print(f"  {migrated:>5}/{total}  (100%) ✓                    ")
    return migrated


# ── 主入口 ───────────────────────────────────────────────────────────────────

def main() -> None:
    import chromadb
    from qdrant_client import QdrantClient

    if not CHROMA_PATH.exists():
        print(f"[error] chroma_db/ 不存在: {CHROMA_PATH}")
        sys.exit(1)

    print(f"Chroma  → {CHROMA_PATH}")
    print(f"Qdrant  → {QDRANT_PATH}")
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    qdrant = QdrantClient(path=str(QDRANT_PATH))

    # 迁移 Chroma 中存在的所有集合
    chroma_cols = [c.name for c in chroma.list_collections()]
    print(f"\nChroma 集合: {chroma_cols}")

    total_migrated = 0
    for col_name in chroma_cols:
        n = migrate_collection(chroma, qdrant, col_name)
        total_migrated += n

    # 验证
    print(f"\n{'='*55}")
    print("验证结果：")
    all_ok = True
    for col_name in chroma_cols:
        c_count = chroma.get_collection(col_name).count()
        q_count = qdrant.count(col_name).count
        status  = "✓" if q_count >= c_count else "✗"
        print(f"  {status}  {col_name:<22} Chroma={c_count}  Qdrant={q_count}")
        if q_count < c_count:
            all_ok = False

    print()
    if all_ok:
        print(f"迁移完成，共 {total_migrated} 条 point。Dense 向量直接复用，Sparse 向量实时生成。")
    else:
        print("部分集合迁移不完整，请检查错误后重新运行（脚本支持幂等重试）。")
        sys.exit(1)


if __name__ == "__main__":
    main()
