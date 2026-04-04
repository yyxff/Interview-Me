"""rag 包 — 向量检索增强生成"""
from .client import (
    CHROMA_PATH, KNOWLEDGE_DIR, NOTES_DIR,
    _KNOWLEDGE_COL, _RESUME_COL, _NOTES_COL,
    KNOWLEDGE_TOP_K, RESUME_TOP_K, QA_PER_CHUNK,
    is_available,
    _get_client, _get_ef, _get_reranker,
    _get_knowledge_col, _get_resume_col, _get_notes_col,
    rerank,
)
from .chunking import _chunk_markdown, _chunk_pdf, _load_or_build_chunks
from .documents import epub_to_markdown, ingest_epub, index_resume
from .indexing import (
    get_index_progress, index_knowledge_with_qa,
    backfill_qa_cache, index_knowledge, AdaptiveSemaphore,
    LLM_TIMEOUT, _is_ratelimit,
)
from .retrieval import (
    SIMILARITY_DISTANCE_THRESHOLD,
    _safe_query, _dedupe_chunks,
    retrieve, retrieve_rich, retrieve_graph,
    has_resume, knowledge_count,
)
from .notes import (
    save_note_file, index_note,
    list_notes, get_note, get_note_questions, delete_note, notes_count,
)
from .profile import save_profile, get_profile_text, profile_status

__all__ = [
    # client
    "CHROMA_PATH", "KNOWLEDGE_DIR", "NOTES_DIR",
    "KNOWLEDGE_TOP_K", "RESUME_TOP_K", "QA_PER_CHUNK",
    "is_available",
    "_get_client", "_get_ef", "_get_knowledge_col", "_get_resume_col", "_get_notes_col",
    "rerank",
    # chunking
    "_chunk_markdown", "_chunk_pdf", "_load_or_build_chunks",
    # documents
    "epub_to_markdown", "ingest_epub", "index_resume",
    # indexing
    "get_index_progress", "index_knowledge_with_qa",
    "backfill_qa_cache", "index_knowledge", "AdaptiveSemaphore",
    "LLM_TIMEOUT", "_is_ratelimit",
    # retrieval
    "SIMILARITY_DISTANCE_THRESHOLD", "_safe_query",
    "retrieve", "retrieve_rich", "retrieve_graph",
    "has_resume", "knowledge_count",
    # notes
    "save_note_file", "index_note",
    "list_notes", "get_note", "get_note_questions", "delete_note", "notes_count",
    # profile
    "save_profile", "get_profile_text", "profile_status",
]
