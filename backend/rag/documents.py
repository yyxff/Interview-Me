"""文档处理：EPUB → Markdown，索引简历 PDF"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .client import (
    KNOWLEDGE_DIR, is_available,
    _get_knowledge_col, _get_resume_col,
)
from .chunking import _chunk_pdf


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


def index_resume(file_bytes: bytes, session_id: str) -> int:
    """将简历 PDF 字节流切块并索引到 ChromaDB（按 session_id 隔离）。返回 chunk 数。"""
    if not is_available():
        return 0
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            chunks = _chunk_pdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not chunks:
            return 0

        col = _get_resume_col()
        # 先删除该 session 旧的向量
        try:
            existing = col.get(where={"session_id": session_id})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception:
            pass

        ids       = [f"{session_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"session_id": session_id, "chunk_id": f"{session_id}_{i}"}
                     for i in range(len(chunks))]
        col.add(ids=ids, documents=chunks, metadatas=metadatas)
        print(f"[rag] 简历索引完成 session={session_id} chunks={len(chunks)}")
        return len(chunks)
    except Exception as e:
        print(f"[rag] 简历索引失败: {e}")
        return 0
