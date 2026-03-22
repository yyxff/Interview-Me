"""
tools — Agent 工具注册中心
===========================

用法：
    from tools import build_toolset

    # 按名字组装工具集（desc + fn），传给 _react_loop
    tool_fns = build_toolset(session, ["search_knowledge", "search_profile"])

注册新工具：在本目录下新建 <tool_name>.py，实现 make(session) -> {"desc": str, "fn": async fn}，
然后在下面 REGISTRY 中加一行即可。
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interview_agent import InterviewSession

from tools import search_knowledge, search_profile, search_past_sessions

REGISTRY: dict[str, object] = {
    "search_knowledge":    search_knowledge,
    "search_profile":      search_profile,
    "search_past_sessions": search_past_sessions,
}


def build_toolset(
    session: "InterviewSession",
    names: list[str],
) -> dict[str, dict]:
    """
    返回 {tool_name: {"desc": str, "fn": async callable}} 的子集。
    names 决定该 agent 的工具权限。
    """
    result: dict[str, dict] = {}
    for name in names:
        mod = REGISTRY.get(name)
        if mod is None:
            raise KeyError(f"未知工具: {name!r}，已注册: {list(REGISTRY)}")
        result[name] = mod.make(session)  # type: ignore[attr-defined]
    return result
