"""路由：/interview/*（模拟面试状态机驱动）"""
from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import rag
import interview_agent as _ia
from llm.provider import LLMProvider

router = APIRouter()

_provider: LLMProvider | None = None


def set_provider(p: LLMProvider | None) -> None:
    global _provider
    _provider = p
    _ia.set_provider(p)


class StartInterviewRequest(BaseModel):
    jd: str = ""
    direction: str = ""


class InterviewChatRequest(BaseModel):
    session_id: str
    message:    str = ""


@router.post("/interview/start")
async def interview_start(req: StartInterviewRequest):
    if _provider is None:
        raise HTTPException(status_code=503, detail="LLM 未配置")

    _ia.set_provider(_provider)
    profile_text = rag.get_profile_text() or ""

    session = _ia.InterviewSession(
        session_id=str(uuid.uuid4()),
        jd=req.jd,
        direction=req.direction,
        profile_text=profile_text,
    )
    _ia._sessions[session.session_id] = session

    try:
        opening_message = await _ia.start_interview(session)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"面试规划失败: {e}")

    print(
        f"[interview/start] session={session.session_id} "
        f"tasks={len(session.roots)} sm={session.sm.state}"
    )
    return {
        "session_id":      session.session_id,
        "opening_message": opening_message,
        "tree":            _ia.tree_to_dict(session.roots),
        "sm":              session.sm.to_dict(),
    }


@router.post("/interview/session/{session_id}/save")
def interview_session_save(session_id: str):
    s = _ia.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    path = _ia.save_session(s)
    return {"filename": path.name}


@router.get("/interview/results")
def interview_results_list():
    files = sorted(_ia.SESSIONS_DIR.glob("*.json"), reverse=True)
    results = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            tree = d.get("tree", [])
            scores = [n["score"] for n in _ia._flat_dict(tree) if n.get("score") is not None]
            results.append({
                "filename":   f.name,
                "saved_at":   d.get("saved_at", ""),
                "direction":  d.get("direction", ""),
                "jd_snippet": d.get("jd", "")[:60],
                "sm_state":   d.get("sm_final", {}).get("state", ""),
                "task_count": sum(1 for n in tree if n.get("node_type") == "task"),
                "avg_score":  round(sum(scores) / len(scores), 1) if scores else None,
            })
        except Exception:
            results.append({
                "filename": f.name, "saved_at": "", "direction": "",
                "jd_snippet": "", "sm_state": "", "task_count": 0, "avg_score": None,
            })
    return {"results": results}


@router.get("/interview/results/{filename}")
def interview_result_get(filename: str):
    path = _ia.SESSIONS_DIR / filename
    if not path.exists() or path.suffix != ".json":
        raise HTTPException(status_code=404, detail="结果文件不存在")
    return json.loads(path.read_text(encoding="utf-8"))


@router.delete("/interview/results/{filename}")
def interview_result_delete(filename: str):
    path = _ia.SESSIONS_DIR / filename
    if not path.exists() or path.suffix != ".json":
        raise HTTPException(status_code=404, detail="结果文件不存在")
    path.unlink()
    return {"ok": True}


@router.get("/interview/session/{session_id}")
def interview_session_get(session_id: str):
    s = _ia.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": s.session_id,
        "sm":         s.sm.to_dict(),
        "tree":       _ia.tree_to_dict(s.roots),
        "sm_log":     s.sm.event_log[-20:],
    }


@router.post("/interview/chat")
async def interview_chat(req: InterviewChatRequest):
    _ia.set_provider(_provider)
    s = _ia.get_session(req.session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if _provider is None:
        async def _no_llm():
            yield f"data: {json.dumps({'text': '请配置 LLM_PROVIDER 和 LLM_API_KEY'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_no_llm(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if s.sm.state == "DONE":
        async def _done():
            yield f"data: {json.dumps({'text': '面试已全部结束，感谢您的参与！'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'sm': s.sm.to_dict(), 'tree': _ia.tree_to_dict(s.roots)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_done(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if s.sm.state != "ANSWERING":
        raise HTTPException(
            status_code=409,
            detail=f"状态机当前为 {s.sm.state}，只有 ANSWERING 状态才能接收消息",
        )

    async def _generate():
        try:
            result = await _ia.run_turn(session=s, user_answer=req.message.strip())
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        words = result["response"].split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.015)

        yield f"data: {json.dumps({'sm': result['sm'], 'tree': result['tree']}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
