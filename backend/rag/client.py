"""Qdrant 客户端、Hybrid Embedding（Dense + Sparse BM25-like）、Reranker — 懒加载单例

检索策略：
  Dense  : BAAI/bge-small-zh-v1.5（sentence_transformers，语义相似）
  Sparse : 字符 n-gram + ASCII 词 hash → SparseVector（关键词精确匹配）
  融合   : Qdrant 内置 RRF（Reciprocal Rank Fusion），dense + sparse 两路召回
  重排   : cross-encoder BAAI/bge-reranker-base

为什么 Sparse 有价值：
  Dense 擅长同义词/换说法，但 "TIME_WAIT"、"epoll"、"三次握手" 等精确术语
  在向量空间中可能被映射到语义相近但不精确的位置；Sparse 直接做关键词匹配，
  两路 RRF 融合后覆盖率明显提升。

对外接口与 Chroma 版完全一致，上层代码零改动。
"""
from __future__ import annotations

import re
import uuid
from collections import Counter
from pathlib import Path

# ── 路径常量 ───────────────────────────────────────────────────────────────────

QDRANT_PATH   = Path(__file__).parent.parent / "qdrant_db"
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
NOTES_DIR     = Path(__file__).parent.parent / "notes"

# 集合名（与 Chroma 版保持一致，便于 A/B 对比说明）
_KNOWLEDGE_COL = "knowledge"
_RESUME_COL    = "resumes"
_NOTES_COL     = "notes"

# 检索参数
KNOWLEDGE_TOP_K = 3
RESUME_TOP_K    = 2
QA_PER_CHUNK    = 4

# UUID namespace（固定，保证同一 string id 始终映射到同一 UUID）
_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# ── 懒加载单例 ─────────────────────────────────────────────────────────────────

_qdrant_client  = None
_embed_fn       = None
_reranker       = None
_dense_dim: int | None = None
_knowledge_col  = None
_resume_col     = None
_notes_col      = None
_rag_available  = None


# ── 可用性检测 ─────────────────────────────────────────────────────────────────

def is_available() -> bool:
    global _rag_available
    if _rag_available is not None:
        return _rag_available
    try:
        import qdrant_client          # noqa: F401
        import sentence_transformers  # noqa: F401
        import pdfplumber             # noqa: F401
        _rag_available = True
    except ImportError:
        _rag_available = False
    return _rag_available


# ── Embedding（Dense） ─────────────────────────────────────────────────────────

def _get_ef():
    """返回 callable: list[str] -> list[list[float]]，与 Chroma EF 接口兼容。"""
    global _embed_fn
    if _embed_fn is None:
        from sentence_transformers import SentenceTransformer
        print("[rag] 加载 Embedding 模型 BAAI/bge-small-zh-v1.5 ...")
        _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        print("[rag] Embedding 模型加载完成")

        class _EF:
            def __call__(self, texts: list[str]) -> list[list[float]]:
                return _model.encode(texts, normalize_embeddings=True).tolist()

        _embed_fn = _EF()
    return _embed_fn


def _get_dense_dim() -> int:
    global _dense_dim
    if _dense_dim is None:
        _dense_dim = len(_get_ef()(["dim"])[0])
    return _dense_dim


# ── Sparse 向量（BM25-like，无需语料库） ──────────────────────────────────────

def _text_to_sparse(text: str):
    """
    将文本转为稀疏向量，用于关键词匹配。

    策略：
      - 英文/数字词：小写 unigram（"tcp"、"time_wait"）
      - 中文：字符 unigram + bigram（"握手"、"三次"、"握"、"手"）
      - 全部 hash 到 32768 维，用词频作为权重

    优势：不需要预建词表，不需要 IDF 全局统计，任意文本即时生成。
    """
    from qdrant_client.models import SparseVector

    tokens: list[str] = []

    # 英文/数字词（保留大小写原词用于精确匹配缩写，如 TCP / TIME_WAIT）
    for w in re.findall(r"[A-Za-z][A-Za-z0-9_]*|[0-9]+", text):
        tokens.append(w.lower())

    # 中文字符 unigram + bigram
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


# ── where 过滤转换（Chroma → Qdrant） ────────────────────────────────────────

def _to_qdrant_filter(where: dict):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    must = []
    for key, val in where.items():
        if isinstance(val, str):
            must.append(FieldCondition(key=key, match=MatchValue(value=val)))
        elif isinstance(val, dict):
            v = val.get("$eq", val.get("$ne"))
            if v is not None:
                must.append(FieldCondition(key=key, match=MatchValue(value=v)))
    return Filter(must=must) if must else None


# ── QdrantCollectionAdapter：模拟 Chroma Collection 接口 ─────────────────────

class _QdrantCol:
    """
    用 Qdrant collection 模拟 ChromaDB Collection，对外接口完全兼容。

    方法对照表：
      .count()                     → qdrant client.count()
      .add(ids, documents, metas)  → upsert PointStruct（dense + sparse 双向量）
      .get(where, include, limit)  → scroll with filter
      .query(texts, n, include)    → hybrid search（dense prefetch + sparse prefetch → RRF）
      .delete(ids / where)         → delete by point IDs or filter
    """

    def __init__(self, qdrant, name: str) -> None:
        self._q    = qdrant
        self._name = name

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _uid(s: str) -> str:
        return str(uuid.uuid5(_NS, s))

    # ── Chroma-compatible API ─────────────────────────────────────────────────

    def count(self) -> int:
        return self._q.count(self._name).count

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        ef         = _get_ef()
        dense_vecs = ef(documents)
        points = []
        for sid, doc, meta, dvec in zip(ids, documents, metadatas, dense_vecs):
            points.append(PointStruct(
                id      = self._uid(sid),
                vector  = {"dense": dvec, "sparse": _text_to_sparse(doc)},
                payload = {**meta, "_document": doc, "_original_id": sid},
            ))
        BATCH = 64
        for i in range(0, len(points), BATCH):
            self._q.upsert(self._name, points=points[i : i + BATCH])

    def get(
        self,
        where: dict | None = None,
        include: list | None = None,
        limit: int | None = None,
    ) -> dict:
        qf       = _to_qdrant_filter(where) if where else None
        page     = limit or 500
        offset   = None
        all_pts: list = []
        while True:
            pts, nxt = self._q.scroll(
                self._name,
                scroll_filter = qf,
                with_payload  = True,
                with_vectors  = False,
                limit         = page,
                offset        = offset,
            )
            all_pts.extend(pts)
            if nxt is None or (limit and len(all_pts) >= limit):
                break
            offset = nxt

        ids   = [p.payload.get("_original_id", "") for p in all_pts]
        metas = [{k: v for k, v in p.payload.items() if not k.startswith("_")} for p in all_pts]
        docs  = [p.payload.get("_document", "") for p in all_pts]

        result: dict = {"ids": ids}
        if include is None or "metadatas" in include:
            result["metadatas"] = metas
        if include is None or "documents" in include:
            result["documents"] = docs
        return result

    def query(
        self,
        query_texts: list[str],
        n_results: int,
        include: list,
        where: dict | None = None,
    ) -> dict:
        """
        Hybrid 检索：
          1. Dense prefetch  →  语义召回 top n*4
          2. Sparse prefetch →  关键词召回 top n*4
          3. Qdrant RRF 融合 →  按综合排名取 top n_results
          4. 额外做一次 dense-only search 获取 cosine distance，
             供上层 SIMILARITY_DISTANCE_THRESHOLD 过滤（与 Chroma 版行为兼容）
        """
        from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

        qt      = query_texts[0]
        ef      = _get_ef()
        dvec    = ef([qt])[0]
        svec    = _text_to_sparse(qt)
        qf      = _to_qdrant_filter(where) if where else None
        over    = n_results * 4

        # ── 混合召回 ──────────────────────────────────────────────────────────
        hybrid_pts = self._q.query_points(
            collection_name = self._name,
            prefetch = [
                Prefetch(query=dvec, using="dense", limit=over),
                Prefetch(
                    query  = SparseVector(indices=svec.indices, values=svec.values),
                    using  = "sparse",
                    limit  = over,
                ),
            ],
            query        = FusionQuery(fusion=Fusion.RRF),
            with_payload = True,
            limit        = n_results,
            query_filter = qf,
        ).points

        # ── dense-only 分数（用于 cosine distance 阈值兼容） ─────────────────
        dense_hits  = self._q.search(
            collection_name = self._name,
            query_vector    = ("dense", dvec),
            limit           = over,
            with_payload    = False,
            query_filter    = qf,
        )
        dense_score = {p.id: p.score for p in dense_hits}

        docs, metas, dists = [], [], []
        for p in hybrid_pts:
            docs.append(p.payload.get("_document", ""))
            metas.append({k: v for k, v in p.payload.items() if not k.startswith("_")})
            # sparse-only 命中时无 dense 分数，给 0.40（cosine dist = 0.60 < 阈值 0.70）
            cos_sim = dense_score.get(p.id, 0.40)
            dists.append(1.0 - cos_sim)

        return {"documents": [docs], "metadatas": [metas], "distances": [dists]}

    def delete(self, ids: list[str] | None = None, where: dict | None = None) -> None:
        from qdrant_client.models import PointIdsList, FilterSelector

        if ids:
            self._q.delete(
                self._name,
                points_selector = PointIdsList(points=[self._uid(s) for s in ids]),
            )
        elif where:
            qf = _to_qdrant_filter(where)
            if qf:
                self._q.delete(self._name, points_selector=FilterSelector(filter=qf))


# ── QdrantClientAdapter：模拟 Chroma PersistentClient ────────────────────────

class _QdrantClientAdapter:
    """让 QdrantClient 对外表现得像 chromadb.PersistentClient（仅实现项目用到的方法）。"""

    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams,
            SparseVectorParams, SparseIndexParams,
        )
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        self._q   = QdrantClient(path=str(QDRANT_PATH))
        self._VectorParams       = VectorParams
        self._Distance           = Distance
        self._SparseVectorParams = SparseVectorParams
        self._SparseIndexParams  = SparseIndexParams

    def get_or_create_collection(self, name: str, embedding_function=None) -> _QdrantCol:
        existing = {c.name for c in self._q.get_collections().collections}
        if name not in existing:
            dim = _get_dense_dim()
            self._q.create_collection(
                collection_name = name,
                vectors_config  = {
                    "dense": self._VectorParams(
                        size     = dim,
                        distance = self._Distance.COSINE,
                    ),
                },
                sparse_vectors_config = {
                    "sparse": self._SparseVectorParams(
                        index = self._SparseIndexParams(on_disk=False),
                    ),
                },
            )
            print(f"[rag] Qdrant collection 已创建: {name}（dense={dim}d + sparse）")
        return _QdrantCol(self._q, name)


# ── 顶层单例工厂 ───────────────────────────────────────────────────────────────

def _get_client() -> _QdrantClientAdapter:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = _QdrantClientAdapter()
    return _qdrant_client


def _get_knowledge_col() -> _QdrantCol:
    global _knowledge_col
    if _knowledge_col is None:
        _knowledge_col = _get_client().get_or_create_collection(_KNOWLEDGE_COL)
    return _knowledge_col


def _get_resume_col() -> _QdrantCol:
    global _resume_col
    if _resume_col is None:
        _resume_col = _get_client().get_or_create_collection(_RESUME_COL)
    return _resume_col


def _get_notes_col() -> _QdrantCol:
    global _notes_col
    if _notes_col is None:
        _notes_col = _get_client().get_or_create_collection(_NOTES_COL)
    return _notes_col


# ── Reranker ───────────────────────────────────────────────────────────────────

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("[rag] 加载 Reranker 模型 BAAI/bge-reranker-base ...")
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
        print("[rag] Reranker 加载完成")
    return _reranker


def rerank(query: str, chunks: list[tuple[str, dict]]) -> list[tuple[str, dict, float]]:
    """用 cross-encoder 对候选 chunks 重排序，返回 [(doc, meta, score), ...] 按 score 降序。"""
    if not chunks:
        return []
    try:
        reranker = _get_reranker()
        pairs    = [(query, doc) for doc, _ in chunks]
        scores   = reranker.predict(pairs).tolist()
        ranked   = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [(doc, meta, score) for (doc, meta), score in ranked]
    except Exception as e:
        print(f"[rag] Reranker 失败，退化为原顺序: {e}")
        return [(doc, meta, 0.0) for doc, meta in chunks]
