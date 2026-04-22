"""
LangGraph 工具适配层

职责：把 backend/tools/ 里的普通函数包装成 LangGraph 可用的 @tool，
      不包含任何业务逻辑，描述也复用 backend/tools/ 里的 DESC。
"""
from __future__ import annotations

from langchain_core.tools import tool

from tools.search_knowledge import search_knowledge as _search_knowledge, DESC as _SK_DESC
from tools.search_profile import search_profile as _search_profile, DESC as _SP_DESC
from tools.search_past_sessions import search_past_sessions as _search_past_sessions, DESC as _SPS_DESC
from tools.explore_concept import explore_concept as _explore_concept, DESC as _EC_DESC

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
