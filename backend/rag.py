"""
RAG 模块 - 向量检索增强生成

知识库:  backend/knowledge/*.md  (全局，启动时自动索引)
简历库:  用户上传 PDF，按 session_id 隔离
向量库:  ChromaDB (backend/chroma_db/, 本地持久化)
Embedding: BAAI/bge-small-zh-v1.5 (本地，首次使用自动下载 ~130MB)

索引策略:
  - 若传入 LLM provider，对每个 chunk 生成 N 个问题，每个问题单独向量化
    但 document 仍存原文 → 检索时返回原文给 LLM，同时去重
  - 若无 provider，直接用原文向量化（退化模式）
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import LLMProvider

# ── 路径常量 ───────────────────────────────────────────────────────────────────

CHROMA_PATH   = Path(__file__).parent / "chroma_db"
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
NOTES_DIR     = Path(__file__).parent / "notes"

# ChromaDB 集合名
_KNOWLEDGE_COL = "knowledge"
_RESUME_COL    = "resumes"
_NOTES_COL     = "notes"

# 检索参数
KNOWLEDGE_TOP_K = 3   # 去重后最多返回几条原文
RESUME_TOP_K    = 2

# 每个 chunk 生成的问题数量
QA_PER_CHUNK = 4

# ── 懒加载单例 ─────────────────────────────────────────────────────────────────

_chroma_client  = None
_embed_fn       = None
_reranker       = None
_knowledge_col  = None
_resume_col     = None
_notes_col      = None
_rag_available  = None

# ── 索引进度 ───────────────────────────────────────────────────────────────────

import time as _time

_index_progress: dict = {
    "status":        "idle",   # idle | running | done | error
    "file":          "",
    "chunks_done":   0,
    "chunks_total":  0,
    "vectors_added": 0,
    "elapsed_s":     0.0,
    "eta_s":         None,     # None = 未知
    "error":         None,
}


def get_index_progress() -> dict:
    return dict(_index_progress)


def is_available() -> bool:
    global _rag_available
    if _rag_available is not None:
        return _rag_available
    try:
        import chromadb                    # noqa: F401
        import sentence_transformers       # noqa: F401
        import pdfplumber                  # noqa: F401
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
        _embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5"
        )
        print("[rag] Embedding 模型加载完成")
    return _embed_fn


def _get_reranker():
    """获取 Cross-encoder Reranker（首次调用会下载模型 ~280MB）。"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("[rag] 加载 Reranker 模型 BAAI/bge-reranker-base ...")
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
        print("[rag] Reranker 加载完成")
    return _reranker


def rerank(query: str, chunks: list[tuple[str, dict]]) -> list[tuple[str, dict, float]]:
    """
    用 cross-encoder 对候选 chunks 重排序。
    输入：[(doc, meta), ...]
    返回：[(doc, meta, score), ...] 按 score 降序
    """
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


# ── 文本切块 ───────────────────────────────────────────────────────────────────

def _chunk_markdown(text: str, source: str) -> list[dict]:
    """
    按 ## 标题切割 Markdown，每块 ≤800 字符。
    返回 [{"text": ..., "source": ..., "chapter": ...}, ...]
    """
    chunks: list[dict] = []
    sections = re.split(r'\n(?=## )', text.strip())

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # 提取章节标题（## 行）
        first_line = section.split('\n', 1)[0]
        chapter = first_line.lstrip('#').strip() if first_line.startswith('#') else ''

        if len(section) <= 800:
            chunks.append({"text": section, "source": source, "chapter": chapter})
        else:
            lines = section.split('\n', 1)
            header = lines[0] if len(lines) > 1 else ''
            body   = lines[1] if len(lines) > 1 else lines[0]
            paragraphs = re.split(r'\n{2,}', body)

            current = header
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) + 2 > 800 and current != header:
                    chunks.append({"text": current.strip(), "source": source, "chapter": chapter})
                    current = header + '\n\n' + para
                else:
                    current += '\n\n' + para
            if current.strip() and current.strip() != header.strip():
                chunks.append({"text": current.strip(), "source": source, "chapter": chapter})

    return chunks


def _chunk_pdf(path: str) -> list[str]:
    """解析简历 PDF，合并段落到 ~600 字符。"""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
    full_text = "\n\n".join(pages_text)

    raw_paras = [p.strip() for p in re.split(r'\n{2,}', full_text) if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in raw_paras:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= 600:
            current += "\n\n" + para
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)

    return chunks


# ── EPUB → Markdown ───────────────────────────────────────────────────────────

def epub_to_markdown(file_bytes: bytes) -> str:
    """将 EPUB 字节流转换为 Markdown 字符串。"""
    import ebooklib
    from ebooklib import epub
    import html2text
    from bs4 import BeautifulSoup

    converter = html2text.HTML2Text()
    converter.ignore_links  = True
    converter.ignore_images = True
    converter.body_width    = 0
    converter.unicode_snob  = True

    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        book = epub.read_epub(tmp_path, options={"ignore_ncx": True})
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    sections: list[str] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_bytes = item.get_content()
        if not html_bytes:
            continue
        soup = BeautifulSoup(html_bytes, "html.parser")
        if len(soup.get_text().strip()) < 100:
            continue
        md = converter.handle(html_bytes.decode("utf-8", errors="replace")).strip()
        if md:
            sections.append(md)

    return "\n\n---\n\n".join(sections)


def ingest_epub(file_bytes: bytes, name: str) -> Path:
    """将 EPUB 转为 Markdown 保存到 knowledge/<name>.md，清除旧的向量索引。"""
    md_path = KNOWLEDGE_DIR / f"{name}.md"
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    if md_path.exists() and is_available():
        col = _get_knowledge_col()
        existing = col.get(where={"source": name})
        if existing["ids"]:
            col.delete(ids=existing["ids"])

    markdown = epub_to_markdown(file_bytes)
    md_path.write_text(markdown, encoding="utf-8")
    return md_path


# ── LLM 问题生成 ───────────────────────────────────────────────────────────────

async def _generate_questions(chunk_text: str, provider: "LLMProvider", n: int = QA_PER_CHUNK) -> list[str]:
    """调用 LLM 对 chunk 生成 n 个不同角度的面试问题。"""
    prompt = (
        f"根据以下技术内容，生成{n}个不同角度的面试问题。\n"
        "只输出问题本身，每行一个，不加序号和任何前缀符号。\n\n"
        f"内容：\n{chunk_text[:800]}"
    )
    try:
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            system="你是一位技术面试题生成助手，根据知识点生成有代表性、不同问法的面试问题。",
        )
        questions = [line.strip() for line in response.strip().split('\n') if line.strip()]
        return questions[:n]
    except Exception as e:
        print(f"[rag] 问题生成失败: {e}")
        return []


# ── 索引操作 ───────────────────────────────────────────────────────────────────

async def index_knowledge_with_qa(provider: "LLMProvider | None" = None) -> int:
    """
    扫描 knowledge/，对每个 chunk 生成多问题向量（若有 provider）。
    - 向量用问题 embed，document 存原文，metadata 含 chunk_id/chapter/question
    - 无 provider 时直接用原文向量化（退化模式）
    返回本次新增的 chunk 数。
    """
    global _index_progress

    if not is_available():
        return 0
    if not KNOWLEDGE_DIR.exists():
        return 0

    # 统计需要处理的总 chunk 数
    col = _get_knowledge_col()
    pending: list[tuple[Path, list[dict]]] = []
    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        existing = col.get(where={"source": md_file.stem}, limit=1)
        if existing["ids"]:
            continue
        text = md_file.read_text(encoding="utf-8")
        chunks = _chunk_markdown(text, md_file.stem)
        if chunks:
            pending.append((md_file, chunks))

    total_chunks = sum(len(c) for _, c in pending)
    if total_chunks == 0:
        _index_progress.update({"status": "idle"})
        return 0

    mode = f"Q&A×{QA_PER_CHUNK}" if provider else "直接"
    _index_progress.update({
        "status":        "running",
        "chunks_done":   0,
        "chunks_total":  total_chunks,
        "vectors_added": 0,
        "elapsed_s":     0.0,
        "eta_s":         None,
        "error":         None,
    })
    t_start = _time.monotonic()
    total_new = 0

    try:
        for md_file, chunks in pending:
            source = md_file.stem
            _index_progress["file"] = md_file.name
            print(f"[rag] 开始索引({mode}): {md_file.name}  ({len(chunks)} chunks)")

            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict] = []

            for i, chunk in enumerate(chunks):
                chunk_id   = f"{source}_{i}"
                chunk_text = chunk["text"]
                chapter    = chunk.get("chapter", "")

                if provider:
                    questions = await _generate_questions(chunk_text, provider)
                else:
                    questions = []

                if questions:
                    for j, q in enumerate(questions):
                        ids.append(f"{chunk_id}_q{j}")
                        documents.append(chunk_text)
                        metadatas.append({
                            "source":   source,
                            "chapter":  chapter,
                            "chunk_id": chunk_id,
                            "question": q,
                        })
                else:
                    ids.append(chunk_id)
                    documents.append(chunk_text)
                    metadatas.append({
                        "source":   source,
                        "chapter":  chapter,
                        "chunk_id": chunk_id,
                    })

                # 更新进度
                done = _index_progress["chunks_done"] + 1
                elapsed = _time.monotonic() - t_start
                eta = (elapsed / done * (total_chunks - done)) if done > 0 else None
                _index_progress.update({
                    "chunks_done": done,
                    "elapsed_s":   round(elapsed, 1),
                    "eta_s":       round(eta, 0) if eta is not None else None,
                })
                print(f"[rag]   chunk {done}/{total_chunks}  {source} > {chapter[:30]}")

            if ids:
                col.add(ids=ids, documents=documents, metadatas=metadatas)
                _index_progress["vectors_added"] += len(ids)
                total_new += len(chunks)
                print(f"[rag] 完成: {md_file.name} → {len(chunks)} chunks, {len(ids)} vectors")

        _index_progress["status"] = "done"
    except Exception as e:
        print(f"[rag] 索引出错: {e}")
        _index_progress.update({"status": "error", "error": str(e)})

    return total_new


def index_knowledge() -> int:
    """同步退化版本（无 LLM），供外部同步调用。"""
    import asyncio
    return asyncio.run(index_knowledge_with_qa(provider=None))


# ── 检索操作 ───────────────────────────────────────────────────────────────────

# cosine distance 阈值：ChromaDB 默认用 L2，bge 系列用余弦距离
# 余弦距离 ∈ [0, 2]，0=完全相同，越大越不相关
# 0.9 约对应余弦相似度 ~0.55，经验值，可按实际效果调整
SIMILARITY_DISTANCE_THRESHOLD = 0.35


def _safe_query(col, query: str, n: int, where: dict | None = None) -> list[tuple[str, dict]]:
    """查询，返回 [(document, metadata), ...] 列表，距离超过阈值的结果会被过滤。"""
    try:
        kwargs: dict = {
            "query_texts": [query],
            "n_results":   n,
            "include":     ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = col.query(**kwargs)
        docs      = results.get("documents",  [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]
        return [
            (d, m) for d, m, dist in zip(docs, metas, distances)
            if d and dist < SIMILARITY_DISTANCE_THRESHOLD
        ]
    except Exception:
        return []


def _dedupe_chunks(raw: list[tuple[str, dict]], limit: int) -> list[tuple[str, dict]]:
    """按 chunk_id 去重，最多返回 limit 条。"""
    seen: set[str] = set()
    result: list[tuple[str, dict]] = []
    for doc, meta in raw:
        chunk_id = meta.get("chunk_id", doc[:30])
        if chunk_id not in seen:
            seen.add(chunk_id)
            result.append((doc, meta))
            if len(result) >= limit:
                break
    return result


def retrieve_rich(query: str, session_id: str | None = None) -> dict:
    """
    检索流程：bi-encoder 召回 → 阈值过滤 → 去重 → cross-encoder 重排 → 取 top-K

    返回:
        {
            "knowledge": [{"text":..., "source":..., "chapter":..., "chunk_id":..., "question":...}],
            "resume":    [...],
        }
    """
    if not is_available():
        return {"knowledge": [], "resume": []}

    knowledge: list[dict] = []
    try:
        col = _get_knowledge_col()
        if col.count() > 0:
            # 扩大候选池再去重，给 reranker 更多选择
            n_candidates = min(KNOWLEDGE_TOP_K * QA_PER_CHUNK * 2, col.count())
            raw = _safe_query(col, query, n_candidates)
            candidates = _dedupe_chunks(raw, limit=KNOWLEDGE_TOP_K * 3)

            # Cross-encoder 重排
            ranked = rerank(query, candidates)

            for doc, meta, _score in ranked[:KNOWLEDGE_TOP_K]:
                knowledge.append({
                    "text":     doc,
                    "source":   meta.get("source", ""),
                    "chapter":  meta.get("chapter", ""),
                    "chunk_id": meta.get("chunk_id", doc[:30]),
                    "question": meta.get("question", ""),
                })
    except Exception:
        pass

    resume: list[dict] = []
    if session_id:
        try:
            col = _get_resume_col()
            existing = col.get(where={"session_id": session_id})
            count = len(existing["ids"])
            if count > 0:
                raw = _safe_query(col, query, min(RESUME_TOP_K, count),
                                  where={"session_id": session_id})
                resume = [{"text": doc, "source": "resume", "chapter": "",
                           "chunk_id": meta.get("chunk_id", ""), "question": ""}
                          for doc, meta in raw]
        except Exception:
            pass

    # ── 笔记检索 ──────────────────────────────────
    notes: list[dict] = []
    try:
        col = _get_notes_col()
        if col.count() > 0:
            raw = _safe_query(col, query, min(2, col.count()))
            seen_notes: set[str] = set()
            for doc, meta in raw:
                note_id = meta.get("note_id", doc[:30])
                if note_id in seen_notes:
                    continue
                seen_notes.add(note_id)
                notes.append({
                    "text":     meta.get("text", doc),  # 返回原文，非问题字符串
                    "source":   "笔记",
                    "chapter":  meta.get("title", ""),
                    "chunk_id": note_id,
                    "question": doc if doc != meta.get("text") else "",
                })
    except Exception:
        pass

    return {"knowledge": knowledge, "resume": resume, "notes": notes}


def retrieve(query: str, session_id: str | None = None) -> dict:
    """
    根据 query 检索知识库和（可选的）简历，返回去重后的原文片段。

    返回:
        {
            "knowledge": ["原文片段1", ...],
            "resume":    ["原文片段1", ...],
        }
    """
    if not is_available():
        return {"knowledge": [], "resume": []}

    # ── 知识库检索：召回 → 去重 → 重排 ──────────────
    knowledge_chunks: list[str] = []
    try:
        col = _get_knowledge_col()
        if col.count() > 0:
            n_candidates = min(KNOWLEDGE_TOP_K * QA_PER_CHUNK * 2, col.count())
            raw        = _safe_query(col, query, n_candidates)
            candidates = _dedupe_chunks(raw, limit=KNOWLEDGE_TOP_K * 3)
            ranked     = rerank(query, candidates)
            knowledge_chunks = [doc for doc, _, _score in ranked[:KNOWLEDGE_TOP_K]]
    except Exception:
        pass

    # ── 简历检索 ──────────────────────────────────
    resume_chunks: list[str] = []
    if session_id:
        try:
            col = _get_resume_col()
            existing = col.get(where={"session_id": session_id})
            count = len(existing["ids"])
            if count > 0:
                raw = _safe_query(
                    col, query, min(RESUME_TOP_K, count),
                    where={"session_id": session_id},
                )
                resume_chunks = [doc for doc, _ in raw]
        except Exception:
            pass

    # ── 笔记检索 ──────────────────────────────────
    note_chunks: list[str] = []
    try:
        col = _get_notes_col()
        if col.count() > 0:
            raw = _safe_query(col, query, min(2, col.count()))
            seen_notes: set[str] = set()
            for doc, meta in raw:
                note_id = meta.get("note_id", doc[:30])
                if note_id not in seen_notes:
                    seen_notes.add(note_id)
                    note_chunks.append(meta.get("text", doc))
    except Exception:
        pass

    return {"knowledge": knowledge_chunks, "resume": resume_chunks, "notes": note_chunks}


def has_resume(session_id: str) -> bool:
    if not is_available():
        return False
    try:
        col = _get_resume_col()
        existing = col.get(where={"session_id": session_id}, limit=1)
        return len(existing["ids"]) > 0
    except Exception:
        return False


def knowledge_count() -> int:
    """返回知识库中的向量总数（含多问题扩展后的数量）。"""
    if not is_available():
        return 0
    try:
        return _get_knowledge_col().count()
    except Exception:
        return 0


# ── 笔记 CRUD ──────────────────────────────────────────────────────────────────

import datetime as _dt


def save_note_file(title: str, content: str, questions: list[str] | None = None) -> tuple[str, str]:
    """
    把笔记写到磁盘，立即返回 (note_id, full_text)。
    同时写 note_id.meta.json（包含 questions 和 indexed 状态）。
    不做 ChromaDB 索引——由调用方在后台线程执行 index_note()。
    """
    import json as _json
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    note_id = "note_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    text    = f"# {title}\n\n{content}"
    (NOTES_DIR / f"{note_id}.md").write_text(text, encoding="utf-8")
    (NOTES_DIR / f"{note_id}.meta.json").write_text(
        _json.dumps({"questions": questions or [], "indexed": False}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return note_id, text


def _read_meta(note_id: str) -> dict:
    """读取 .meta.json，兼容旧版 .qa.json + .indexed 格式。"""
    import json as _json
    meta_path = NOTES_DIR / f"{note_id}.meta.json"
    if meta_path.exists():
        try:
            return _json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {"questions": [], "indexed": False}
    # 旧格式兼容
    questions: list[str] = []
    qa_path = NOTES_DIR / f"{note_id}.qa.json"
    if qa_path.exists():
        try:
            questions = _json.loads(qa_path.read_text(encoding="utf-8")).get("questions", [])
        except Exception:
            pass
    indexed = (NOTES_DIR / f"{note_id}.indexed").exists()
    return {"questions": questions, "indexed": indexed}


def _write_meta(note_id: str, meta: dict) -> None:
    import json as _json
    (NOTES_DIR / f"{note_id}.meta.json").write_text(
        _json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def index_note(note_id: str, title: str, text: str) -> None:
    """
    在 ChromaDB 中索引一条笔记（同步，供后台线程调用）。
    若 meta.json 中有 questions，则以每个问题为 document 向量化；
    否则降级为直接向量化原文。
    """
    if not is_available():
        return
    try:
        col = _get_notes_col()
        # 先删除该笔记已有的所有向量
        try:
            existing = col.get(where={"note_id": note_id})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception:
            pass

        meta = _read_meta(note_id)
        questions = meta.get("questions", [])

        if questions:
            col.add(
                ids=[f"{note_id}_q{i}" for i in range(len(questions))],
                documents=questions,  # 嵌入问题向量
                metadatas=[{
                    "note_id": note_id,
                    "title":   title,
                    "text":    text,  # 检索时返回原文
                } for _ in questions],
            )
            print(f"[rag] 笔记已索引 {len(questions)} 个问题: {note_id}")
        else:
            col.add(
                ids=[note_id],
                documents=[text],
                metadatas=[{"note_id": note_id, "title": title, "text": text}],
            )
            print(f"[rag] 笔记已索引(原文): {note_id}")
        # 更新 meta.json 中的 indexed 状态
        meta["indexed"] = True
        _write_meta(note_id, meta)
    except Exception as e:
        print(f"[rag] 笔记索引失败: {e}")


def list_notes() -> list[dict]:
    """列出所有笔记，按时间倒序。"""
    if not NOTES_DIR.exists():
        return []
    notes = []
    for f in sorted(NOTES_DIR.glob("note_*.md"), reverse=True):
        lines = f.read_text(encoding="utf-8").split("\n", 2)
        title = lines[0].lstrip("# ").strip() if lines else f.stem
        notes.append({
            "note_id":    f.stem,
            "title":      title,
            "size":       f.stat().st_size,
            "created_at": f.stem[5:],  # "20240115_103045"
            "indexed":    _read_meta(f.stem).get("indexed", False),
        })
    return notes


def get_note(note_id: str) -> str | None:
    """返回笔记内容，不存在时返回 None。"""
    path = NOTES_DIR / f"{note_id}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def get_note_questions(note_id: str) -> list[str]:
    """返回笔记对应的问题列表，不存在则返回空列表。"""
    return _read_meta(note_id).get("questions", [])


def delete_note(note_id: str) -> bool:
    """删除笔记文件、meta.json 和向量索引，返回是否成功。"""
    path = NOTES_DIR / f"{note_id}.md"
    if not path.exists():
        return False
    path.unlink()
    for suffix in (".meta.json", ".qa.json", ".indexed"):
        p = NOTES_DIR / f"{note_id}{suffix}"
        if p.exists():
            p.unlink()
    if is_available():
        try:
            col = _get_notes_col()
            existing = col.get(where={"note_id": note_id})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception:
            pass
    return True


def notes_count() -> int:
    if not is_available():
        return 0
    try:
        return _get_notes_col().count()
    except Exception:
        return 0
