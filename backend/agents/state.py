"""
State 定义 + 内部工具函数
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .models import ThoughtNode

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# ① State 定义
# ══════════════════════════════════════════════════════════════════════════════

class InterviewState(TypedDict):
    """
    LangGraph State = 节点间共享的"公告板"（TypedDict）。

    每个节点返回一个 dict，只包含"本轮要更新的字段"：
      - 普通字段：直接覆盖（最新值生效）
      - messages：Annotated + add_messages → 框架自动"追加"而非覆盖

    为什么用 TypedDict 而不是 dataclass？
      → LangGraph 需要能序列化/反序列化 state（持久化到 checkpointer），
        TypedDict 里只放 JSON 兼容的类型，ThoughtNode 对象存成 dict。
    """
    # ── 会话基本信息 ──────────────────────────────────────────────────
    session_id: str
    jd: str
    direction: str
    profile_text: str

    # ── 思维树（序列化为 dict，方便 JSON 持久化）─────────────────────
    roots_data: list[dict]           # ThoughtNode 的完整 dict 表示
    current_task_id: str | None      # 当前正在考察的任务节点 id
    current_question_id: str | None  # 当前最新问题节点 id

    # ── 上一轮结果（节点间传递数据）──────────────────────────────────
    last_score: int | None
    last_verdict: str | None         # deepen | pivot | back_up | pass
    last_sub_questions: list[str]    # 导演规划的子问题方向
    last_director_reasoning: str     # 导演的推理过程（供下一轮 ask 参考）

    # ── 对话历史 ─────────────────────────────────────────────────────
    # Annotated + add_messages：每次节点返回 messages 时框架自动追加
    # 而不是覆盖，这是 LangGraph 最常用的 Reducer 模式
    messages: Annotated[list[BaseMessage], add_messages]


# ══════════════════════════════════════════════════════════════════════════════
# 内部工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _node_to_dict(n: ThoughtNode) -> dict:
    """完整序列化 ThoughtNode（不截断 answer，供 state 存储）。"""
    return {
        "id": n.id, "node_type": n.node_type, "text": n.text,
        "answer": n.answer, "depth": n.depth, "status": n.status,
        "score": n.score, "verdict": n.verdict, "feedback": n.feedback,
        "reasoning": n.reasoning, "director_note": n.director_note,
        "question_intent": n.question_intent, "task_type": n.task_type,
        "summary": n.summary, "parent_id": n.parent_id,
        "children": [_node_to_dict(c) for c in n.children],
    }


def _dict_to_node(d: dict) -> ThoughtNode:
    """从 dict 还原 ThoughtNode（递归）。"""
    children = [_dict_to_node(c) for c in d.get("children", [])]
    return ThoughtNode(
        id=d["id"], node_type=d["node_type"], text=d["text"],
        answer=d.get("answer", ""), depth=d.get("depth", 0),
        status=d.get("status", "pending"), score=d.get("score"),
        verdict=d.get("verdict"), feedback=d.get("feedback", ""),
        reasoning=d.get("reasoning", ""), director_note=d.get("director_note", ""),
        question_intent=d.get("question_intent", ""), task_type=d.get("task_type", ""),
        summary=d.get("summary", ""), children=children,
        parent_id=d.get("parent_id"),
    )


def _flat_dict(nodes: list[dict]) -> list[dict]:
    """将 dict 格式的树展平。"""
    out: list[dict] = []
    q = list(nodes)
    while q:
        n = q.pop(0)
        out.append(n)
        q.extend(n.get("children", []))
    return out


def _parse_json(text: str, default: Any) -> Any:
    """宽松解析 LLM 输出的 JSON（去掉代码块标记）。"""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"[\[\{].*[\]\}]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return default
