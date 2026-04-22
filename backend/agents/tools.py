"""
LangGraph 工具适配层

职责：把 backend/tools/ 里的普通函数包装成 LangGraph 可用的 @tool，
      不包含任何业务逻辑，描述也复用 backend/tools/ 里的 DESC。
"""
from __future__ import annotations

import uuid

from langchain_core.tools import tool

from tools.search_knowledge import search_knowledge as _search_knowledge, DESC as _SK_DESC
from tools.search_profile import search_profile as _search_profile, DESC as _SP_DESC
from tools.search_past_sessions import search_past_sessions as _search_past_sessions, DESC as _SPS_DESC
from tools.explore_concept import explore_concept as _explore_concept, DESC as _EC_DESC

from .models import ThoughtNode
from .state import InterviewState


def _search_knowledge_tool():
    @tool(description=_SK_DESC)
    async def search_knowledge(query: str) -> str:
        return _search_knowledge(query)
    return search_knowledge


def _search_profile_tool(profile: str):
    @tool(description=_SP_DESC)
    async def search_profile(query: str) -> str:
        return _search_profile(profile, query)
    return search_profile


def _search_past_sessions_tool(session_id: str = ""):
    @tool(description=_SPS_DESC)
    async def search_past_sessions(query: str) -> str:
        return _search_past_sessions(query, exclude_session_id=session_id)
    return search_past_sessions


def _explore_concept_tool():
    @tool(description=_EC_DESC)
    async def explore_concept(entity: str) -> str:
        return _explore_concept(entity)
    return explore_concept


def _add_task_tool(roots: list[ThoughtNode]):
    @tool(description=(
        "add_task: 在面试任务列表末尾追加一个新的考察任务。\n"
        "  task      : 任务描述，一句话说明考察方向\n"
        "  task_type : experience/knowledge/concept/design/debug/scenario\n"
        '  用法：add_task(task="Kafka 消息队列使用经验", task_type="experience")'
    ))
    async def add_task(task: str, task_type: str = "knowledge") -> str:
        node = ThoughtNode(
            id=str(uuid.uuid4()),
            node_type="task",
            text=task.strip(),
            task_type=task_type,
            status="pending",
        )
        roots.append(node)
        return f"已添加任务「{task}」(id={node.id})"
    return add_task


def _remove_task_tool(roots: list[ThoughtNode]):
    @tool(description=(
        "remove_task: 从任务列表中移除一个尚未开始的任务（仅限 pending 状态）。\n"
        "  task_id : 任务节点的 id\n"
        "  用法：remove_task(task_id=\"uuid-...\")"
    ))
    async def remove_task(task_id: str) -> str:
        for i, node in enumerate(roots):
            if node.id == task_id:
                if node.status != "pending":
                    return f"任务「{node.text}」状态为 {node.status}，无法移除"
                roots.pop(i)
                return f"已移除任务「{node.text}」"
        return f"未找到 task_id={task_id}"
    return remove_task


# ── Node 工具组合 ─────────────────────────────────────────────────────────────

def _make_plan_tools(state: InterviewState) -> list:
    """plan：知识库 + 简历（有则加）+ 历史薄弱点"""
    tools = [
        _search_knowledge_tool(),
        _search_past_sessions_tool(state.get("session_id", "")),
    ]
    if state.get("profile_text"):
        tools.insert(1, _search_profile_tool(state["profile_text"]))
    return tools


def _make_ask_tools(state: InterviewState) -> list:
    """ask：知识库 + 简历（有则加）+ 历史薄弱点 + 图谱联想"""
    tools = [
        _search_knowledge_tool(),
        _search_past_sessions_tool(state.get("session_id", "")),
        _explore_concept_tool(),
    ]
    if state.get("profile_text"):
        tools.insert(1, _search_profile_tool(state["profile_text"]))
    return tools


def _make_score_tools(state: InterviewState) -> list:
    """score：仅知识库"""
    return [_search_knowledge_tool()]


def _make_decide_tools(roots: list[ThoughtNode]) -> list:
    """decide：动态调整任务树"""
    return [_add_task_tool(roots), _remove_task_tool(roots)]
