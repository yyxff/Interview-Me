"""路由：上传、知识库管理、RAG 状态"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import rag
from llm.provider import LLMProvider

router = APIRouter()

_provider: LLMProvider | None = None


def set_provider(p: LLMProvider | None) -> None:
    global _provider
    _provider = p


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "ai_enabled": _provider is not None,
        "provider": _provider.name if _provider else None,
        "rag_enabled": rag.is_available(),
        "knowledge_chunks": rag.knowledge_count(),
    }


@router.post("/upload/resume")
async def upload_resume(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")
    if not rag.is_available():
        raise HTTPException(status_code=503, detail="RAG 模块未安装")
    file_bytes = await file.read()
    chunk_count = rag.index_resume(file_bytes, session_id)
    return {"ok": True, "chunks": chunk_count}


@router.post("/profile/upload")
async def profile_upload(file: UploadFile = File(...)):
    """上传 MD 格式的简历，保存为全文（不建向量索引，直接塞上下文）。"""
    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="只支持 Markdown (.md) 文件")
    content = (await file.read()).decode("utf-8")
    try:
        rag.save_profile(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}


@router.get("/profile/status")
def profile_status():
    return rag.profile_status()


@router.post("/upload/knowledge")
async def upload_knowledge(file: UploadFile = File(...)):
    """接受 EPUB 文件，转为 Markdown 保存到 knowledge/ 并立即索引。"""
    import asyncio as _asyncio
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not file.filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="目前只支持 EPUB 格式")
    if not rag.is_available():
        raise HTTPException(status_code=503, detail="RAG 模块未安装")

    name = Path(file.filename).stem
    file_bytes = await file.read()
    md_path = rag.ingest_epub(file_bytes, name)
    chunk_count = await rag.index_knowledge_with_qa(_provider)

    if _provider is not None:
        async def _run_graph():
            try:
                import graph_rag as _gr
                await _gr.index_knowledge_graph(_provider)
            except Exception as e:
                print(f"[upload/knowledge] 图索引失败: {e}")
        _asyncio.create_task(_run_graph())

    return {"ok": True, "name": name, "md_path": str(md_path), "new_chunks": chunk_count}


@router.get("/knowledge/list")
async def knowledge_list():
    """列出所有已转换的知识库文件。"""
    files = []
    if rag.KNOWLEDGE_DIR.exists():
        for md_file in sorted(rag.KNOWLEDGE_DIR.glob("*.md")):
            col = rag._get_knowledge_col() if rag.is_available() else None
            indexed = False
            if col:
                existing = col.get(where={"source": md_file.stem}, limit=1)
                indexed = bool(existing["ids"])
            files.append({
                "name":     md_file.stem,
                "filename": md_file.name,
                "size":     md_file.stat().st_size,
                "indexed":  indexed,
            })
    return {"files": files}


@router.get("/knowledge/{name}")
async def knowledge_content(name: str):
    """返回某个知识库文件的 Markdown 内容。"""
    md_path = rag.KNOWLEDGE_DIR / f"{name}.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return {"name": name, "content": md_path.read_text(encoding="utf-8")}


@router.get("/rag/status")
async def rag_status(session_id: str | None = None):
    return {
        "rag_enabled":      rag.is_available(),
        "knowledge_chunks": rag.knowledge_count(),
        "has_resume":       rag.has_resume(session_id) if session_id else False,
    }


@router.get("/rag/index-progress")
async def rag_index_progress():
    return rag.get_index_progress()
