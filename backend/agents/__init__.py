"""agents 包 — 多 Agent 模拟面试引擎"""
from __future__ import annotations
from typing import Any

from .models import (
    ThoughtNode, InterviewSM, InterviewSession,
    flat, find, flat_dict, tree_to_dict, add_planned_nodes,
)
from .orchestrator import start_interview, run_turn, save_session, SESSIONS_DIR
from .react import set_provider as _react_set_provider
from .director import set_provider as _director_set_provider
from .interviewer import set_provider as _interviewer_set_provider
from .scorer import set_provider as _scorer_set_provider

# 全局会话字典
_sessions: dict[str, InterviewSession] = {}

_provider: Any = None


def set_provider(p: Any) -> None:
    global _provider
    _provider = p
    _react_set_provider(p)
    _director_set_provider(p)
    _interviewer_set_provider(p)
    _scorer_set_provider(p)


def get_session(sid: str) -> InterviewSession | None:
    return _sessions.get(sid)


__all__ = [
    "ThoughtNode", "InterviewSM", "InterviewSession",
    "flat", "find", "flat_dict", "tree_to_dict", "add_planned_nodes",
    "start_interview", "run_turn", "save_session", "SESSIONS_DIR",
    "set_provider", "get_session",
    "_sessions",
]
