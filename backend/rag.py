"""
RAG 模块 - 向量检索增强生成

知识库:  backend/knowledge/*.md  (全局，启动时自动索引)
简历库:  用户上传 PDF，按 session_id 隔离
向量库:  ChromaDB (backend/chroma_db/, 本地持久化)
Embedding: BAAI/bge-small-zh-v1.5 (本地，首次使用自动下载 ~130MB)
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

# ── 路径常量 ───────────────────────────────────────────────────────────────────

CHROMA_PATH   = Path(__file__).parent / "chroma_db"
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# ChromaDB 集合名
_KNOWLEDGE_COL = "knowledge"
_RESUME_COL    = "resumes"

# 检索参数
KNOWLEDGE_TOP_K = 3   # 每次从知识库取几条
RESUME_TOP_K    = 2   # 每次从简历取几条

# ── 懒加载单例 ─────────────────────────────────────────────────────────────────
# 避免后端启动时就加载大模型，首次用到时再初始化

_chroma_client  = None
_embed_fn       = None
_knowledge_col  = None
_resume_col     = None
_rag_available  = None   # None = 未检测，True/False = 已检测


def is_available() -> bool:
    """检查 RAG 依赖是否已安装。"""
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
    """获取 Embedding 函数（首次调用会下载模型）。"""
    global _embed_fn
    if _embed_fn is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        print("[rag] 加载 Embedding 模型 BAAI/bge-small-zh-v1.5 ...")
        _embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5"
        )
        print("[rag] Embedding 模型加载完成")
    return _embed_fn


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


# ── 文本切块 ───────────────────────────────────────────────────────────────────

def _chunk_markdown(text: str, source: str) -> list[dict]:
    """
    按 ## 标题切割 Markdown。
    每个切块保留标题上下文，长度控制在 ~800 字符以内。
    返回 [{"text": ..., "source": ...}, ...]
    """
    chunks: list[dict] = []
    # 按 ## 分割（保留标题行）
    sections = re.split(r'\n(?=## )', text.strip())

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= 800:
            chunks.append({"text": section, "source": source})
        else:
            # 段落太长：按空行再切，首行（标题）带入每个子块
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
                    chunks.append({"text": current.strip(), "source": source})
                    current = header + '\n\n' + para
                else:
                    current += '\n\n' + para
            if current.strip() and current.strip() != header.strip():
                chunks.append({"text": current.strip(), "source": source})

    return chunks


def _chunk_pdf(path: str) -> list[str]:
    """
    解析简历 PDF，按段落切块，合并过短的段落到 ~600 字符。
    """
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
    full_text = "\n\n".join(pages_text)

    # 按连续空行分段
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


# ── 索引操作 ───────────────────────────────────────────────────────────────────

def index_knowledge() -> int:
    """
    扫描 knowledge/ 目录，索引所有 .md 文件。
    已索引的文件（按文件名判断）跳过，只增量更新。
    返回本次新增的 chunk 数。
    """
    if not is_available():
        return 0
    if not KNOWLEDGE_DIR.exists():
        return 0

    col = _get_knowledge_col()
    total_new = 0

    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        source = md_file.stem  # 以文件名（无后缀）作为 source 标识

        # 检查是否已索引（取一条看看）
        existing = col.get(where={"source": source}, limit=1)
        if existing["ids"]:
            continue  # 已有，跳过

        text = md_file.read_text(encoding="utf-8")
        chunks = _chunk_markdown(text, source)
        if not chunks:
            continue

        ids       = [f"{source}_{i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"]} for c in chunks]

        col.add(ids=ids, documents=documents, metadatas=metadatas)
        total_new += len(chunks)
        print(f"[rag] 索引知识库: {md_file.name} → {len(chunks)} chunks")

    return total_new


def index_resume(file_bytes: bytes, session_id: str) -> int:
    """
    索引用户简历（PDF 字节流）。
    同一 session 的旧简历会先删除再重建。
    返回入库的 chunk 数。
    """
    if not is_available():
        return 0

    col = _get_resume_col()

    # 删除该 session 的旧数据
    existing = col.get(where={"session_id": session_id})
    if existing["ids"]:
        col.delete(ids=existing["ids"])

    # 写临时文件供 pdfplumber 解析
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        chunks = _chunk_pdf(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not chunks:
        return 0

    ids       = [f"{session_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"session_id": session_id} for _ in chunks]

    col.add(ids=ids, documents=chunks, metadatas=metadatas)
    print(f"[rag] 简历已索引: session={session_id[:8]}… → {len(chunks)} chunks")
    return len(chunks)


# ── 检索操作 ───────────────────────────────────────────────────────────────────

def _safe_query(col, query: str, n: int, where: dict | None = None) -> list[str]:
    """安全查询：自动处理结果数不足的边界情况。"""
    try:
        kwargs: dict = {"query_texts": [query], "n_results": n}
        if where:
            kwargs["where"] = where
        results = col.query(**kwargs)
        docs = results.get("documents", [[]])[0]
        return [d for d in docs if d]
    except Exception:
        return []


def retrieve(query: str, session_id: str | None = None) -> dict:
    """
    根据 query 检索知识库和（可选的）简历，返回相关文本片段。

    返回:
        {
            "knowledge": ["片段1", "片段2", ...],
            "resume":    ["片段1", ...],
        }
    """
    if not is_available():
        return {"knowledge": [], "resume": []}

    # ── 知识库检索 ────────────────────────────────
    knowledge_chunks: list[str] = []
    try:
        col = _get_knowledge_col()
        if col.count() > 0:
            knowledge_chunks = _safe_query(
                col, query, min(KNOWLEDGE_TOP_K, col.count())
            )
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
                resume_chunks = _safe_query(
                    col, query, min(RESUME_TOP_K, count),
                    where={"session_id": session_id}
                )
        except Exception:
            pass

    return {"knowledge": knowledge_chunks, "resume": resume_chunks}


def has_resume(session_id: str) -> bool:
    """检查该 session 是否已上传简历。"""
    if not is_available():
        return False
    try:
        col = _get_resume_col()
        existing = col.get(where={"session_id": session_id}, limit=1)
        return len(existing["ids"]) > 0
    except Exception:
        return False


def knowledge_count() -> int:
    """返回知识库中的 chunk 总数。"""
    if not is_available():
        return 0
    try:
        return _get_knowledge_col().count()
    except Exception:
        return 0
