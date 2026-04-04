"""
interview_agent — 向后兼容 shim，实际逻辑已迁移至 agents/ 包。

原有的 `import interview_agent as _ia` 导入继续有效。
"""
from agents import (
    ThoughtNode, InterviewSM, InterviewSession,
    flat as _flat, find as _find, flat_dict as _flat_dict,
    tree_to_dict, add_planned_nodes,
    start_interview, run_turn, save_session, SESSIONS_DIR,
    set_provider, get_session,
    _sessions,
)

# 别名保持与原始 interview_agent 内部名称一致
_flat = _flat
_find = _find

__all__ = [
    "ThoughtNode", "InterviewSM", "InterviewSession",
    "_flat", "_find", "_flat_dict", "tree_to_dict",
    "start_interview", "run_turn", "save_session", "SESSIONS_DIR",
    "set_provider", "get_session", "_sessions",
]
