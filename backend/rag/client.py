"""ChromaDB 客户端、Embedding 函数、Reranker — 懒加载单例"""
from __future__ import annotations

from pathlib import Path

# ── 路径常量 ───────────────────────────────────────────────────────────────────

CHROMA_PATH   = Path(__file__).parent.parent / "chroma_db"
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
NOTES_DIR     = Path(__file__).parent.parent / "notes"

# ChromaDB 集合名
_KNOWLEDGE_COL = "knowledge"
_RESUME_COL    = "resumes"
_NOTES_COL     = "notes"

# 检索参数
KNOWLEDGE_TOP_K = 3
RESUME_TOP_K    = 2
QA_PER_CHUNK    = 4

# ── 懒加载单例 ─────────────────────────────────────────────────────────────────

_chroma_client = None
_embed_fn      = None
_reranker      = None
_knowledge_col = None
_resume_col    = None
_notes_col     = None
_rag_available = None


def is_available() -> bool:
    global _rag_available
    if _rag_available is not None:
        return _rag_available
    try:
        import chromadb             # noqa: F401
        import sentence_transformers  # noqa: F401
        import pdfplumber           # noqa: F401
        _rag_available = True
    except ImportError:
        _rag_available = False
    return _rag_available


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _chroma_client


def _get_ef():
    global _embed_fn
    if _embed_fn is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        print("[rag] 加载 Embedding 模型 BAAI/bge-small-zh-v1.5 ...")
        _embed_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
        print("[rag] Embedding 模型加载完成")
    return _embed_fn


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
        pairs  = [(query, doc) for doc, _ in chunks]
        scores = reranker.predict(pairs).tolist()
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [(doc, meta, score) for (doc, meta), score in ranked]
    except Exception as e:
        print(f"[rag] Reranker 失败，退化为原顺序: {e}")
        return [(doc, meta, 0.0) for doc, meta in chunks]


def _get_knowledge_col():
    global _knowledge_col
    if _knowledge_col is None:
        _knowledge_col = _get_client().get_or_create_collection(
            _KNOWLEDGE_COL, embedding_function=_get_ef()
        )
    return _knowledge_col


def _get_resume_col():
    global _resume_col
    if _resume_col is None:
        _resume_col = _get_client().get_or_create_collection(
            _RESUME_COL, embedding_function=_get_ef()
        )
    return _resume_col


def _get_notes_col():
    global _notes_col
    if _notes_col is None:
        _notes_col = _get_client().get_or_create_collection(
            _NOTES_COL, embedding_function=_get_ef()
        )
    return _notes_col
