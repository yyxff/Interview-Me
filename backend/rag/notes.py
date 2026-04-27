"""笔记 CRUD + ChromaDB 索引"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import TYPE_CHECKING

from .client import is_available, NOTES_DIR, _get_notes_col, _get_knowledge_col

if TYPE_CHECKING:
    from llm.provider import LLMProvider


def save_note_file(title: str, content: str, questions: list[str] | None = None) -> tuple[str, str]:
    """把笔记写到磁盘，立即返回 (note_id, full_text)。同时写 meta.json。"""
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


def _index_note_sync(note_id: str, title: str, text: str) -> None:
    """同步写入 notes collection 和 knowledge collection。"""
    if not is_available():
        return

    # notes collection
    try:
        col  = _get_notes_col()
        meta = _read_meta(note_id)
        try:
            existing = col.get(where={"note_id": note_id})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception:
            pass

        questions = meta.get("questions", [])
        if questions:
            col.add(
                ids=[f"{note_id}_q{i}" for i in range(len(questions))],
                documents=questions,
                metadatas=[{"note_id": note_id, "title": title, "text": text} for _ in questions],
            )
            print(f"[rag] 笔记已索引 {len(questions)} 个问题: {note_id}")
        else:
            col.add(
                ids=[note_id],
                documents=[text],
                metadatas=[{"note_id": note_id, "title": title, "text": text}],
            )
            print(f"[rag] 笔记已索引(原文): {note_id}")
        meta["indexed"] = True
        _write_meta(note_id, meta)
    except Exception as e:
        print(f"[rag] 笔记索引失败(notes): {e}")

    # knowledge collection — 整篇笔记作为单 chunk
    chunk_id = f"note:{note_id}"
    try:
        kcol = _get_knowledge_col()
        try:
            existing = kcol.get(ids=[chunk_id])
            if existing["ids"]:
                kcol.delete(ids=[chunk_id])
        except Exception:
            pass
        kcol.add(
            ids=[chunk_id],
            documents=[text],
            metadatas=[{
                "chunk_id": chunk_id,
                "text":     text,
                "source":   "笔记",
                "path":     "",
                "chapter":  title,
                "question": "",
            }],
        )
        print(f"[rag] 笔记已写入 knowledge collection: {chunk_id}")
    except Exception as e:
        print(f"[rag] 笔记写入 knowledge 失败: {e}")


async def index_note(note_id: str, title: str, text: str,
                     provider: "LLMProvider | None" = None) -> None:
    """索引笔记（async）：sync DB 操作在线程池执行，graph RAG 抽取在 event loop 执行。"""
    import asyncio
    await asyncio.to_thread(_index_note_sync, note_id, title, text)

    if provider is None:
        return

    chunk_id = f"note:{note_id}"
    try:
        import graph_rag as _gr
        from graph_rag.extractor import _extract_entities_relations

        entities, relations = await _extract_entities_relations(text, provider)
        if not entities:
            print(f"[graph_rag] 笔记无可抽取实体，跳过图谱索引: {note_id}")
            return

        source = f"note:{note_id}"
        graph  = _gr._build_graph_for_source(source, [(entities, relations, chunk_id)])
        _gr._save_graph(source, graph)
        _gr._index_graph_to_qdrant(graph)
        print(f"[graph_rag] 笔记图谱已索引: {source} ({len(entities)} 实体, {len(relations)} 关系)")
    except Exception as e:
        print(f"[graph_rag] 笔记图谱索引失败: {e}")


def list_notes() -> list[dict]:
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
            "created_at": f.stem[5:],
            "indexed":    _read_meta(f.stem).get("indexed", False),
        })
    return notes


def get_note(note_id: str) -> str | None:
    path = NOTES_DIR / f"{note_id}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def get_note_questions(note_id: str) -> list[str]:
    return _read_meta(note_id).get("questions", [])


def delete_note(note_id: str) -> bool:
    path = NOTES_DIR / f"{note_id}.md"
    if not path.exists():
        return False
    path.unlink()
    for suffix in (".meta.json", ".qa.json", ".indexed"):
        p = NOTES_DIR / f"{note_id}{suffix}"
        if p.exists():
            p.unlink()

    if not is_available():
        return True

    # notes collection
    try:
        col      = _get_notes_col()
        existing = col.get(where={"note_id": note_id})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except Exception:
        pass

    # knowledge collection
    try:
        kcol     = _get_knowledge_col()
        chunk_id = f"note:{note_id}"
        existing = kcol.get(ids=[chunk_id])
        if existing["ids"]:
            kcol.delete(ids=[chunk_id])
    except Exception:
        pass

    # graph RAG — Qdrant entity/relation collections
    try:
        import graph_rag as _gr
        source = f"note:{note_id}"
        _gr._clear_vectors_for_source(_gr._get_entities_col(), source)
        _gr._clear_vectors_for_source(_gr._get_relations_col(), source)
    except Exception:
        pass

    # graph RAG — graph.json file
    try:
        import graph_rag as _gr
        graph_file = _gr.GRAPH_DIR / f"note:{note_id}.graph.json"
        if graph_file.exists():
            graph_file.unlink()
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
