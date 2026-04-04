"""路由：/qa-sessions/*（QA 会话持久化）"""
from __future__ import annotations

import json
import time as _time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_QA_SESSIONS_DIR = Path(__file__).parent.parent / "qa_sessions"
_QA_SESSIONS_DIR.mkdir(exist_ok=True)


def _qa_session_path(session_id: str) -> Path | None:
    for p in _QA_SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("session_id") == session_id:
                return p
        except Exception:
            pass
    return None


def _write_qa_session(data: dict) -> None:
    path = _qa_session_path(data["session_id"])
    if path is None:
        ts   = _time.strftime("%Y%m%d_%H%M%S")
        path = _QA_SESSIONS_DIR / f"{ts}_{data['session_id'][:8]}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_qa_sessions_raw() -> list[dict]:
    results = []
    for p in sorted(_QA_SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            results.append({
                "session_id": data["session_id"],
                "title":      data.get("title", ""),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "node_count": len(data.get("nodes", {})),
            })
        except Exception:
            pass
    return results


class QASessionSaveRequest(BaseModel):
    session_id: str
    title:      str
    nodes:      dict
    root_ids:   list[str]
    tabs:       list[dict]


@router.get("/qa-sessions")
def qa_sessions_list():
    return {"sessions": _list_qa_sessions_raw()}


@router.post("/qa-sessions/save")
def qa_sessions_save(req: QASessionSaveRequest):
    now = _time.strftime("%Y%m%d_%H%M%S")
    existing_path = _qa_session_path(req.session_id)
    created_at = now
    if existing_path:
        try:
            old = json.loads(existing_path.read_text(encoding="utf-8"))
            created_at = old.get("created_at", now)
        except Exception:
            pass
    data = {
        "session_id": req.session_id,
        "title":      req.title,
        "created_at": created_at,
        "updated_at": now,
        "nodes":      req.nodes,
        "root_ids":   req.root_ids,
        "tabs":       req.tabs,
    }
    _write_qa_session(data)
    return {"ok": True}


@router.get("/qa-sessions/{session_id}")
def qa_sessions_get(session_id: str):
    path = _qa_session_path(session_id)
    if path is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return json.loads(path.read_text(encoding="utf-8"))


@router.delete("/qa-sessions/{session_id}")
def qa_sessions_delete(session_id: str):
    path = _qa_session_path(session_id)
    if path is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    path.unlink()
    return {"ok": True}
