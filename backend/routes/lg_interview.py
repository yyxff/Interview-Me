"""
路由：/interview/*  （LangGraph 版）
======================================

取代原来的 routes/interview.py，状态管理完全交给 LangGraph。

核心变化对比：
┌──────────────────────────────────────────────────────────────────┐
│  旧版                          │  LangGraph 版                   │
│  ─────────────────────────     │  ─────────────────────────────  │
│  _sessions = {}  手动维护      │  MemorySaver 自动管理            │
│  InterviewSM 手写状态机        │  图结构即状态机                  │
│  orchestrator.py 大段 if-else  │  条件边路由                      │
│  两个接口各自调度 agent        │  ainvoke + Command(resume=...)   │
└──────────────────────────────────────────────────────────────────┘

LangGraph interrupt 机制与 HTTP 的对应关系：

  POST /interview/start
    → graph.ainvoke(initial_state, config)
    → 图跑到 ask_node 里的 interrupt(question)
    → 返回 question 给前端

  POST /interview/chat  { session_id, message }
    → graph.ainvoke(Command(resume=message), config)
    → 图从 interrupt() 处继续：score → decide → ask → interrupt(next_q)
    → 返回 next_q 给前端（或 done=True）
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import rag as _rag

# 导入 LangGraph 图 + 必要的类型
from agents.state import InterviewState, _dict_to_node
from agents.graph import interview_graph
from agents.models import tree_to_dict
from langgraph.types import Command

router = APIRouter()

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response 模型
# ══════════════════════════════════════════════════════════════════════════════

class StartRequest(BaseModel):
    jd: str = ""
    direction: str = ""


class ChatRequest(BaseModel):
    session_id: str
    message: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数：从图中读取 interrupt 值
# ══════════════════════════════════════════════════════════════════════════════

async def _run_until_interrupt(input_, config: dict) -> tuple[str | None, dict]:
    """
    驱动图运行，直到遇到 interrupt 或结束。

    返回 (question, final_state_snapshot)：
      - question = interrupt() 的值（面试问题文本），None 表示面试结束
      - final_state_snapshot = 最后一次 state 快照（用于返回树结构）

    关键：用 astream(..., stream_mode="updates") 监听每一步的输出。
    当图执行到 interrupt() 时，框架会在 updates 中注入 "__interrupt__" key。
    """
    question: str | None = None
    last_state: dict = {}

    async for chunk in interview_graph.astream(input_, config, stream_mode="updates"):
        # 每个 chunk 是 {node_name: {更新的字段}} 或 {"__interrupt__": [...]}
        if "__interrupt__" in chunk:
            # interrupt() 的值就在这里
            interrupts = chunk["__interrupt__"]
            question = interrupts[0].value if interrupts else None
        else:
            # 普通节点输出，记录最新 state 片段
            for node_output in chunk.values():
                if isinstance(node_output, dict):
                    last_state.update(node_output)

    return question, last_state


async def _get_current_state(config: dict) -> InterviewState:
    """从 checkpointer 读取当前完整 state。"""
    snapshot = await interview_graph.aget_state(config)
    return snapshot.values  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
# 路由
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/interview/start")
async def lg_interview_start(req: StartRequest):
    """
    开始面试。

    流程：
      1. 生成 session_id（作为 LangGraph 的 thread_id）
      2. 构造初始 state
      3. 调用 graph.astream() → 跑到第一个 interrupt → 返回第一道题
    """
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    # 构建初始 state（所有字段必须存在）
    initial_state: InterviewState = {
        "session_id": session_id,
        "jd": req.jd,
        "direction": req.direction,
        "profile_text": _rag.get_profile_text() or "",
        "roots_data": [],
        "current_task_id": None,
        "current_question_id": None,
        "last_score": None,
        "last_verdict": None,
        "last_sub_questions": [],
        "last_director_reasoning": "",
        "messages": [],
    }

    try:
        question, _ = await _run_until_interrupt(initial_state, config)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"面试规划失败: {e}")

    # 读取完整 state（含思维树）
    current_state = await _get_current_state(config)

    return {
        "session_id": session_id,
        "opening_message": question or "（面试规划中，请稍候）",
        "done": question is None,
        "tree": tree_to_dict([_dict_to_node(d) for d in current_state.get("roots_data", [])]),
    }


@router.post("/interview/chat")
async def lg_interview_chat(req: ChatRequest):
    """
    提交候选人的回答，获取下一道题。

    流程：
      1. 用 Command(resume=message) 恢复被 interrupt 暂停的图
      2. 图从 ask_node 的 interrupt() 处继续：记录答案 → score → decide → ask → interrupt
      3. 返回下一道题（或 done=True）
    """
    config = {"configurable": {"thread_id": req.session_id}}

    try:
        next_question, _ = await _run_until_interrupt(
            Command(resume=req.message),
            config,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"处理回答失败: {e}")

    current_state = await _get_current_state(config)
    roots_data = current_state.get("roots_data", [])
    roots = [_dict_to_node(d) for d in roots_data]

    if next_question is None:
        # 图执行到 END（所有任务完成）
        _save_session(current_state)
        return {
            "response": "非常感谢你的参与，今天的面试到此结束！请稍等查看评分详情。",
            "done": True,
            "tree": tree_to_dict(roots),
        }

    return {
        "response": next_question,
        "done": False,
        "tree": tree_to_dict(roots),
        "last_score": current_state.get("last_score"),
        "last_verdict": current_state.get("last_verdict"),
    }


@router.post("/interview/session/{session_id}/save")
async def lg_interview_session_save(session_id: str):
    """
    保存当前 session 到 JSON 文件。
    LangGraph 版：从 checkpointer 读取当前 state 并写入磁盘。
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = await _get_current_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session 不存在: {e}")

    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    path = _save_session(state)
    return {"filename": path.name}


@router.get("/interview/results")
def lg_interview_results_list():
    """列出所有已保存的面试结果文件。"""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
    results = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            tree = d.get("tree", [])
            scores = [
                n["score"]
                for n in _flat_dict(tree)
                if n.get("score") is not None
            ]
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
def lg_interview_result_get(filename: str):
    """返回指定结果文件的完整 JSON 内容。"""
    path = SESSIONS_DIR / filename
    if not path.exists() or path.suffix != ".json":
        raise HTTPException(status_code=404, detail="结果文件不存在")
    return json.loads(path.read_text(encoding="utf-8"))


@router.delete("/interview/results/{filename}")
def lg_interview_result_delete(filename: str):
    """删除指定结果文件。"""
    path = SESSIONS_DIR / filename
    if not path.exists() or path.suffix != ".json":
        raise HTTPException(status_code=404, detail="结果文件不存在")
    path.unlink()
    return {"ok": True}


@router.get("/interview/session/{session_id}")
async def lg_interview_session_get(session_id: str):
    """
    返回当前 session 的状态树（与旧版 GET /interview/session/{session_id} 响应格式兼容）。
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = await _get_current_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session 不存在: {e}")

    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    roots = [_dict_to_node(d) for d in state.get("roots_data", [])]
    return {
        "session_id": session_id,
        "sm": {
            "state": "DONE" if state.get("last_verdict") == "end" else "ANSWERING",
        },
        "tree": tree_to_dict(roots),
        "sm_log": [],
    }


@router.get("/v2/interview/session/{session_id}/state")
async def lg_get_state(session_id: str):
    """调试接口：查看当前 session 的完整 state。"""
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = await _get_current_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session 不存在: {e}")

    roots = [_dict_to_node(d) for d in state.get("roots_data", [])]
    return {
        "session_id": session_id,
        "current_task_id": state.get("current_task_id"),
        "last_score": state.get("last_score"),
        "last_verdict": state.get("last_verdict"),
        "tree": tree_to_dict(roots),
        "message_count": len(state.get("messages", [])),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数：递归展平树节点（供 results 列表接口计算分数用）
# ══════════════════════════════════════════════════════════════════════════════

def _flat_dict(nodes: list[dict]) -> list[dict]:
    """递归展平树节点列表（dict 格式）。"""
    result = []
    for n in nodes:
        result.append(n)
        result.extend(_flat_dict(n.get("children", [])))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 会话保存（面试结束时写入 JSON 文件，与原版格式兼容）
# ══════════════════════════════════════════════════════════════════════════════

def _save_session(state: dict) -> Path:
    """将面试结果保存为 JSON（格式与原版兼容）。返回保存路径。"""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_id = state.get("session_id", "unknown")
    path = SESSIONS_DIR / f"{ts}_{session_id[:8]}_lg.json"

    roots = [_dict_to_node(d) for d in state.get("roots_data", [])]
    messages = [
        {"role": "assistant" if m.__class__.__name__ == "AIMessage" else "user",
         "content": m.content}
        for m in state.get("messages", [])
    ]

    data = {
        "session_id": session_id,
        "saved_at": ts,
        "engine": "langgraph",
        "jd": state.get("jd", ""),
        "direction": state.get("direction", ""),
        "tree": tree_to_dict(roots),
        "history": messages,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[session saved] {path}")
    return path
