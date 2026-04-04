"""笔记 CRUD + ChromaDB 索引"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

from .client import is_available, NOTES_DIR, _get_notes_col


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


def index_note(note_id: str, title: str, text: str) -> None:
    """在 ChromaDB 中索引一条笔记（同步，供后台线程调用）。"""
    if not is_available():
        return
    try:
        col = _get_notes_col()
        try:
            existing = col.get(where={"note_id": note_id})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception:
            pass

        meta      = _read_meta(note_id)
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
        print(f"[rag] 笔记索引失败: {e}")


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
