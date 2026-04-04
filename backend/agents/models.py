"""数据模型：ThoughtNode、InterviewSM、InterviewSession，以及树操作工具函数"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# 状态机合法转换表
_TRANSITIONS: dict[str, set[str]] = {
    "INIT":       {"PLANNING"},
    "PLANNING":   {"ASKING"},
    "ASKING":     {"ANSWERING"},
    "ANSWERING":  {"SCORING"},
    "SCORING":    {"DIRECTING"},
    "DIRECTING":  {"ASKING", "DONE"},
    "DONE":       set(),
}


class InterviewSM:
    def __init__(self):
        self.state: str = "INIT"
        self.current_node_id: str | None = None
        self.current_task_id: str | None = None
        self.last_score: dict | None = None
        self.event_log: list[dict] = []

    def transition(self, new_state: str, **meta) -> None:
        allowed = _TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(f"非法转换 {self.state!r} → {new_state!r}，允许: {allowed}")
        self.event_log.append({"from": self.state, "to": new_state, "ts": round(time.time(), 3), **meta})
        self.state = new_state

    def to_dict(self) -> dict:
        return {
            "state":           self.state,
            "current_node_id": self.current_node_id,
            "current_task_id": self.current_task_id,
            "last_score":      self.last_score,
        }


@dataclass
class ThoughtNode:
    id:              str
    node_type:       str           # 'task' | 'question'
    text:            str
    answer:          str  = ""
    depth:           int  = 0
    status:          str  = "pending"
    score:           int | None = None
    verdict:         str | None = None
    feedback:        str  = ""
    reasoning:       str  = ""
    director_note:   str  = ""
    question_intent: str  = ""
    task_type:       str  = ""
    summary:         str  = ""
    children:        list["ThoughtNode"] = field(default_factory=list)
    parent_id:       str | None = None


@dataclass
class InterviewSession:
    session_id:   str
    jd:           str
    direction:    str
    profile_text: str
    sm:           InterviewSM       = field(default_factory=InterviewSM)
    roots:        list[ThoughtNode] = field(default_factory=list)
    history:      list[dict]        = field(default_factory=list)


# ── 树操作 ─────────────────────────────────────────────────────────────────────

def flat(nodes: list[ThoughtNode]) -> list[ThoughtNode]:
    out: list[ThoughtNode] = []
    q = list(nodes)
    while q:
        n = q.pop(0)
        out.append(n)
        q.extend(n.children)
    return out


def find(roots: list[ThoughtNode], nid: str | None) -> ThoughtNode | None:
    if nid is None:
        return None
    for n in flat(roots):
        if n.id == nid:
            return n
    return None


def next_pending_task(roots: list[ThoughtNode]) -> ThoughtNode | None:
    for n in roots:
        if n.status == "pending":
            return n
    return None


def flat_dict(nodes: list[dict]) -> list[dict]:
    """将序列化后的树展平，供 routes/interview.py 统计使用。"""
    out: list[dict] = []
    q = list(nodes)
    while q:
        n = q.pop(0)
        out.append(n)
        q.extend(n.get("children", []))
    return out


def tree_to_dict(nodes: list[ThoughtNode]) -> list[dict]:
    return [
        {
            "id":              n.id,
            "node_type":       n.node_type,
            "text":            n.text,
            "answer":          n.answer[:120] if n.answer else "",
            "depth":           n.depth,
            "status":          n.status,
            "score":           n.score,
            "verdict":         n.verdict,
            "feedback":        n.feedback,
            "reasoning":       n.reasoning,
            "director_note":   n.director_note,
            "question_intent": n.question_intent,
            "task_type":       n.task_type,
            "summary":         n.summary,
            "children":        tree_to_dict(n.children),
        }
        for n in nodes
    ]


def add_planned_nodes(
    parent_node: ThoughtNode,
    focuses: list[str],
) -> list[ThoughtNode]:
    """在 parent_node 下批量添加 planned 子节点（只存意图，问题文本待面试官生成）。"""
    nodes = []
    for focus in focuses:
        n = ThoughtNode(
            id=str(uuid.uuid4()),
            node_type="question",
            text="",
            depth=parent_node.depth + 1,
            status="planned",
            parent_id=parent_node.id,
            question_intent=focus,
        )
        parent_node.children.append(n)
        nodes.append(n)
    return nodes
