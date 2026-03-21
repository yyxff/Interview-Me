"""多 Agent 模拟面试系统
======================
状态机驱动的三 Agent 协作：

    INIT
     │ Director.plan()
     ▼
    PLANNING
     │ 任务树写入完成
     ▼
    READY
     │ 开始面试
     ▼
    INTERVIEWING  ◄──────────────────────────────┐
     │ Interviewer 调用 end_topic                │
     ▼                                            │
    SCORING                                       │
     │ Scorer 评分完成                             │
     ▼                                            │
    DIRECTING                                     │
     │ Director 决策：expand → 插入子任务 ─────────┘
     │              next   → 下一 pending 任务 ──┘
     └─────────────── done → DONE

面试官工具：
  rag_search(query)  — 内部检索，不改变 SM 状态
  end_topic()        — 触发 SM INTERVIEWING → SCORING 转换
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ── 状态机 ────────────────────────────────────────────────────────────────────

_TRANSITIONS: dict[str, set[str]] = {
    "INIT":        {"PLANNING"},
    "PLANNING":    {"READY"},
    "READY":       {"INTERVIEWING"},
    "INTERVIEWING":{"SCORING"},
    "SCORING":     {"DIRECTING"},
    "DIRECTING":   {"INTERVIEWING", "DONE"},
    "DONE":        set(),
}


class InterviewSM:
    """面试状态机 —— 所有 Agent 通过它协调。"""

    def __init__(self):
        self.state: str = "INIT"
        self.current_task_id: str | None = None
        self.last_score: dict | None = None        # {score, feedback}
        self.event_log: list[dict] = []

    def transition(self, new_state: str, **meta) -> None:
        allowed = _TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"非法状态转换 {self.state!r} → {new_state!r}，"
                f"允许的目标状态: {allowed}"
            )
        self.event_log.append({
            "from": self.state, "to": new_state,
            "ts": round(time.time(), 3), **meta,
        })
        self.state = new_state

    def to_dict(self) -> dict:
        return {
            "state":           self.state,
            "current_task_id": self.current_task_id,
            "last_score":      self.last_score,
        }


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class TaskNode:
    id:          str
    task:        str            # 具体考察内容，e.g."解释 TCP 三次握手的原理"
    task_type:   str = "knowledge"  # experience/knowledge/concept/design/debug/scenario
    depth:       int = 0
    status:      str = "pending"   # pending/active/done/skipped
    score:       int | None = None
    feedback:    str = ""
    children:    list["TaskNode"] = field(default_factory=list)
    parent_id:   str | None = None


@dataclass
class InterviewSession:
    session_id:    str
    jd:            str
    direction:     str
    profile_text:  str
    sm:            InterviewSM = field(default_factory=InterviewSM)
    tasks:         list[TaskNode] = field(default_factory=list)
    history:       list[dict] = field(default_factory=list)   # 全程对话
    topic_history: list[dict] = field(default_factory=list)   # 当前话题对话（评分用）


# ── 全局状态 ──────────────────────────────────────────────────────────────────

_sessions: dict[str, InterviewSession] = {}
_provider: Any = None


def set_provider(p: Any) -> None:
    global _provider
    _provider = p


def get_session(session_id: str) -> InterviewSession | None:
    return _sessions.get(session_id)


# ── 树操作 ────────────────────────────────────────────────────────────────────

def _flat(nodes: list[TaskNode]) -> list[TaskNode]:
    out: list[TaskNode] = []
    q = list(nodes)
    while q:
        n = q.pop(0)
        out.append(n)
        q.extend(n.children)
    return out


def _find(nodes: list[TaskNode], task_id: str | None) -> TaskNode | None:
    if task_id is None:
        return None
    for n in _flat(nodes):
        if n.id == task_id:
            return n
    return None


def _next_pending(nodes: list[TaskNode]) -> TaskNode | None:
    for n in _flat(nodes):
        if n.status == "pending":
            return n
    return None


def tasks_to_dict(nodes: list[TaskNode]) -> list[dict]:
    return [
        {
            "id":        n.id,
            "task":      n.task,
            "task_type": n.task_type,
            "depth":     n.depth,
            "status":    n.status,
            "score":     n.score,
            "feedback":  n.feedback,
            "children":  tasks_to_dict(n.children),
        }
        for n in nodes
    ]


# ── LLM ───────────────────────────────────────────────────────────────────────

async def _llm(messages: list[dict], system: str = "", timeout: float = 50.0) -> str:
    if _provider is None:
        raise RuntimeError("LLM provider 未注入，请先调用 set_provider()")
    return await asyncio.wait_for(_provider.chat(messages, system), timeout=timeout)


def _parse_json(text: str, default: Any) -> Any:
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'[\[\{].*[\]\}]', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return default


# ── DirectorAgent ─────────────────────────────────────────────────────────────

_PLAN_SYSTEM = """\
你是技术面试导演，负责将整场面试拆分为具体的考察任务。

要求：
- 生成 5-8 个任务，每个是一道具体的面试题/场景，不是宽泛话题
- 任务覆盖候选人背景、岗位要求，由浅入深排列
- task_type 只能是: experience / knowledge / concept / design / debug / scenario

输出 JSON 数组（只输出数组，不加代码块）：
[
  {"task":"介绍你做过的最有挑战的项目，重点说说遇到了哪些技术难点","task_type":"experience"},
  {"task":"解释 TCP 三次握手的原理以及为什么需要三次","task_type":"concept"},
  ...
]"""

_PLAN_USER = """\
候选人 Profile：
{profile}

岗位 JD：
{jd}

考察方向（可选）：
{direction}"""

_DECIDE_SYSTEM = """\
你是技术面试导演，根据评分结果决定下一步。

规则：
- score ≥ 4：标记完成，进入下一任务
- score 2-3：拆解为 2-3 个更具体的跟进任务
- score 1：拆解为基础概念确认 + 简单应用题（至少 2 个子任务）

输出 JSON（二选一，只输出 JSON）：
{"action":"next"}
{"action":"expand","subtasks":[{"task":"...","task_type":"concept"},{"task":"...","task_type":"knowledge"}]}"""

_DECIDE_USER = """\
刚完成的考察任务：「{task}」
候选人评分：{score}/5
评分反馈：{feedback}

当前任务树（供参考）：
{tree}"""


async def director_plan(session: InterviewSession) -> None:
    """PLANNING 阶段：拆分整场面试为具体任务树，写入 SM。"""
    assert session.sm.state == "PLANNING"

    user_msg = _PLAN_USER.format(
        profile=(session.profile_text or "（未提供）")[:2000],
        jd=(session.jd or "（未指定，通用技术岗位）")[:1000],
        direction=(session.direction or "（未指定，根据候选人背景判断）"),
    )
    text = await _llm([{"role": "user", "content": user_msg}], system=_PLAN_SYSTEM)
    raw = _parse_json(text, default=[])
    if not isinstance(raw, list):
        raw = []

    tasks: list[TaskNode] = []
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        task_text = item.get("task", "").strip()
        task_type = item.get("task_type", "knowledge")
        if task_text:
            tasks.append(TaskNode(
                id=str(uuid.uuid4()),
                task=task_text,
                task_type=task_type,
                depth=0,
            ))

    if not tasks:  # 降级兜底
        tasks = [
            TaskNode(id=str(uuid.uuid4()), task="介绍一下你最近做的项目", task_type="experience"),
            TaskNode(id=str(uuid.uuid4()), task="描述一个你熟悉的核心技术概念", task_type="concept"),
        ]

    session.tasks = tasks
    session.sm.transition("READY")


async def director_decide(session: InterviewSession, task_id: str) -> None:
    """DIRECTING 阶段：读评分，决定 expand/next/done，转换 SM 状态。"""
    assert session.sm.state == "DIRECTING"

    score_info = session.sm.last_score or {"score": 3, "feedback": "无反馈"}
    task       = _find(session.tasks, task_id)

    if task:
        task.score    = score_info["score"]
        task.feedback = score_info["feedback"]
        task.status   = "done"

    user_msg = _DECIDE_USER.format(
        task=task.task if task else "（未知）",
        score=score_info["score"],
        feedback=score_info["feedback"],
        tree=json.dumps(tasks_to_dict(session.tasks), ensure_ascii=False, indent=2)[:3000],
    )
    text = await _llm([{"role": "user", "content": user_msg}], system=_DECIDE_SYSTEM)
    decision = _parse_json(text, default={"action": "next"})

    if decision.get("action") == "expand" and task:
        for st in decision.get("subtasks", [])[:3]:
            if not isinstance(st, dict):
                continue
            child = TaskNode(
                id=str(uuid.uuid4()),
                task=st.get("task", "").strip(),
                task_type=st.get("task_type", "knowledge"),
                depth=(task.depth + 1),
                status="pending",
                parent_id=task.id,
            )
            if child.task:
                task.children.append(child)

    nxt = _next_pending(session.tasks)
    if nxt:
        nxt.status = "active"
        session.sm.current_task_id = nxt.id
        session.sm.transition("INTERVIEWING", task_id=nxt.id)
    else:
        session.sm.current_task_id = None
        session.sm.transition("DONE")


# ── ScorerAgent ───────────────────────────────────────────────────────────────

_SCORER_SYSTEM = """\
你是技术面试评分员。对候选人在指定考察任务上的表现进行评分。

评分标准（1-5 分）：
5 = 准确完整，有深度，结合实际经验
4 = 正确，覆盖核心要点
3 = 基本正确，但有遗漏或不够深入
2 = 有概念但理解不清
1 = 基本不了解

只输出 JSON（不加代码块）：{"score":4,"feedback":"一句话总结表现"}"""


async def scorer_evaluate(task: TaskNode, topic_history: list[dict]) -> dict:
    """SCORING 阶段：对本话题对话评分，返回 {score, feedback}。"""
    conv = "\n".join(
        f"{'面试官' if m['role'] == 'assistant' else '候选人'}: {m['content'][:400]}"
        for m in topic_history[-14:]
    )
    user_msg = f"考察任务：「{task.task}」（类型：{task.task_type}）\n\n对话内容：\n{conv}"
    text = await _llm([{"role": "user", "content": user_msg}], system=_SCORER_SYSTEM)
    result = _parse_json(text, default={"score": 3, "feedback": "评分解析失败"})
    return {
        "score":    max(1, min(5, int(result.get("score", 3)))),
        "feedback": result.get("feedback", ""),
    }


# ── InterviewerAgent（ReAct） ──────────────────────────────────────────────────

_INTERVIEWER_SYSTEM = """\
你是一位专业技术面试官。当前考察任务如下：

【考察任务】{task}
【类型】{task_type}
{profile_section}
## 可用工具

rag_search — 检索技术知识库，为深度追问提供参考
格式：
Action: rag_search
Action Input: 查询关键词

end_topic — 当你判断该任务已考察完毕时调用（候选人已充分作答或拒绝回答）
格式：
Action: end_topic
Action Input: 无

## ReAct 输出格式

Thought: 分析当前情况
Action: rag_search 或 end_topic
Action Input: 输入内容
Observation: 工具返回（由系统填写）
... (可多轮)
Final Answer: 直接面向候选人的回复（只有这部分展示给用户）

## 面试官规则

- 首轮：围绕考察任务提出第一个具体问题，不要先自我介绍
- 追问：根据候选人上一条回答针对性展开，不重复同一问题
- 每次只问一个问题，等候选人回答
- 候选人已回答 2-3 次后，评估是否调用 end_topic
- Final Answer 语气自然，≤4 句话"""

_REACT_MAX_STEPS = 6


async def interviewer_respond(
    session: InterviewSession,
    task: TaskNode,
    user_message: str | None,
) -> tuple[str, bool]:
    """
    INTERVIEWING 阶段：ReAct 循环。
    user_message=None 表示面试官开场提问。
    返回 (response_text, topic_ended)。
    """
    import rag as _rag

    profile_section = ""
    if session.profile_text:
        profile_section = f"【候选人背景】\n{session.profile_text[:1000]}\n"

    # 父任务提示（深挖子任务时）
    parent_hint = ""
    if task.parent_id:
        parent = _find(session.tasks, task.parent_id)
        if parent:
            parent_hint = f"（是「{parent.task}」的跟进子任务）"

    system = _INTERVIEWER_SYSTEM.format(
        task=task.task + parent_hint,
        task_type=task.task_type,
        profile_section=profile_section,
    )

    # 工作消息 = 最近 8 条全程历史 + 本轮用户消息
    working: list[dict] = list(session.history[-8:])
    if user_message:
        working.append({"role": "user", "content": user_message})
    elif not working:
        working.append({"role": "user", "content": "请开始考察，向候选人提出第一个问题。"})

    topic_ended = False
    final_answer = ""

    for _step in range(_REACT_MAX_STEPS):
        response = await _llm(working, system=system)

        if "Final Answer:" in response:
            final_answer = response.split("Final Answer:", 1)[1].strip()
            break

        action_m  = re.search(r"Action:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        input_m   = re.search(r"Action Input:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)

        if not action_m:
            final_answer = response.strip()
            break

        action       = action_m.group(1).strip().lower()
        action_input = input_m.group(1).strip() if input_m else ""

        working.append({"role": "assistant", "content": response})

        if "end_topic" in action:
            topic_ended = True
            # 让面试官给出一句过渡语
            working.append({
                "role": "user",
                "content": (
                    "Observation: 话题已结束，系统将转入评分。\n"
                    "请在 Final Answer 中给候选人一句简短的过渡语（不要透露分数）。"
                ),
            })
            closing = await _llm(working, system=system)
            final_answer = (
                closing.split("Final Answer:", 1)[1].strip()
                if "Final Answer:" in closing
                else closing.strip()
            )
            break

        elif "rag_search" in action:
            try:
                result = _rag.retrieve_rich(action_input)
                chunks = result.get("knowledge", [])[:3]
                obs = (
                    "检索到相关内容：\n" + "\n---\n".join(
                        f"[{c['source']}] {c['text'][:300]}" for c in chunks
                    )
                    if chunks else "知识库中未找到相关内容。"
                )
            except Exception as e:
                obs = f"检索失败: {e}"
            working.append({"role": "user", "content": f"Observation: {obs}"})

        else:
            final_answer = response.strip()
            break

    if not final_answer:
        final_answer = "请继续，我在听您的回答。"

    return final_answer, topic_ended


# ── 完整话题轮次编排（供 main.py 调用） ──────────────────────────────────────

async def run_turn(session: InterviewSession, user_message: str) -> dict:
    """
    处理一轮用户消息，驱动状态机前进，返回：
    {
      "response": str,           # 面试官回复
      "topic_ended": bool,
      "sm": dict,                # 状态机快照
      "tasks": list,             # 最新任务树
    }
    如果 topic_ended，内部会自动完成 SCORING → DIRECTING，直到下一个 INTERVIEWING 或 DONE。
    """
    sm = session.sm
    assert sm.state == "INTERVIEWING", f"期望 INTERVIEWING，当前 {sm.state}"

    task = _find(session.tasks, sm.current_task_id)
    if task is None:
        raise RuntimeError("current_task_id 指向不存在的任务")

    # 记录用户消息
    if user_message:
        session.history.append({"role": "user", "content": user_message})
        session.topic_history.append({"role": "user", "content": user_message})

    # 面试官回复
    response_text, topic_ended = await interviewer_respond(session, task, user_message or None)

    # 记录面试官回复
    session.history.append({"role": "assistant", "content": response_text})
    session.topic_history.append({"role": "assistant", "content": response_text})

    if topic_ended:
        # SCORING
        sm.transition("SCORING", task_id=task.id)
        score_result = await scorer_evaluate(task, session.topic_history)
        sm.last_score = score_result
        print(
            f"[scorer] task='{task.task[:40]}' "
            f"score={score_result['score']} feedback={score_result['feedback']}"
        )

        # DIRECTING
        sm.transition("DIRECTING", **score_result)
        await director_decide(session, task.id)
        # ↑ director_decide 会将 SM 转换到 INTERVIEWING(next) 或 DONE

        # 重置本话题历史
        session.topic_history = []

        print(
            f"[director] SM → {sm.state} "
            f"current_task={sm.current_task_id}"
        )

    return {
        "response":    response_text,
        "topic_ended": topic_ended,
        "sm":          sm.to_dict(),
        "tasks":       tasks_to_dict(session.tasks),
    }
