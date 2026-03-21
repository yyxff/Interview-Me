"""多 Agent 模拟面试系统
======================
三个角色：
  - DirectorAgent  : 面试导演，维护 TaskTree，决定面试走向
  - ScorerAgent    : 评分员，对候选人的一段作答打分并给出反馈
  - InterviewerAgent: 面试官，ReAct 范式，工具：rag_search / topic_completed
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class TaskNode:
    id: str
    topic: str                          # 考察话题，如"进程调度算法"
    depth: int = 0                      # 树深度，0=一级话题
    status: str = "pending"             # pending / active / passed / deep_dive / done
    score: int | None = None            # 1-5
    feedback: str = ""
    children: list["TaskNode"] = field(default_factory=list)
    parent_id: str | None = None


@dataclass
class InterviewSession:
    session_id: str
    jd: str                             # 岗位 JD（可为空）
    direction: str                      # 考察方向（可为空）
    profile_text: str                   # 候选人 Profile 全文
    tasks: list[TaskNode] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # [{role, content}]
    current_task_id: str | None = None
    status: str = "active"             # active / done


# ── 全局状态 ──────────────────────────────────────────────────────────────────

_sessions: dict[str, InterviewSession] = {}
_provider: Any = None


def set_provider(p: Any) -> None:
    global _provider
    _provider = p


def get_session(session_id: str) -> InterviewSession | None:
    return _sessions.get(session_id)


# ── 树操作工具 ────────────────────────────────────────────────────────────────

def _flat_tasks(nodes: list[TaskNode]) -> list[TaskNode]:
    """BFS 展开所有任务节点。"""
    result: list[TaskNode] = []
    queue = list(nodes)
    while queue:
        n = queue.pop(0)
        result.append(n)
        queue.extend(n.children)
    return result


def _find_task(nodes: list[TaskNode], task_id: str | None) -> TaskNode | None:
    if task_id is None:
        return None
    for n in _flat_tasks(nodes):
        if n.id == task_id:
            return n
    return None


def _next_pending(nodes: list[TaskNode]) -> TaskNode | None:
    for n in _flat_tasks(nodes):
        if n.status == "pending":
            return n
    return None


def tasks_to_dict(nodes: list[TaskNode]) -> list[dict]:
    return [
        {
            "id":       n.id,
            "topic":    n.topic,
            "depth":    n.depth,
            "status":   n.status,
            "score":    n.score,
            "feedback": n.feedback,
            "children": tasks_to_dict(n.children),
        }
        for n in nodes
    ]


# ── LLM 工具 ─────────────────────────────────────────────────────────────────

async def _llm(messages: list[dict], system: str = "", timeout: float = 45.0) -> str:
    if _provider is None:
        raise RuntimeError("LLM provider not injected — call set_provider() first")
    return await asyncio.wait_for(_provider.chat(messages, system), timeout=timeout)


def _parse_json(text: str, default: Any) -> Any:
    """安全解析 LLM 返回的 JSON，失败返回 default。"""
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

_DIRECTOR_INIT = """\
你是一位资深技术面试导演，负责规划面试考察路径。

根据以下信息，生成面试考察任务列表（JSON 数组）：

**候选人 Profile：**
{profile}

**岗位 JD：**
{jd}

**考察方向（可选）：**
{direction}

请输出 3-5 个一级考察话题，每项只需 topic 字段（≤15字，技术话题为主）。
只输出 JSON 数组，不加代码块或注释。示例：
[{{"topic":"进程与线程"}},{{"topic":"内存管理"}},{{"topic":"HTTP 与网络"}}]"""

_DIRECTOR_UPDATE = """\
你是面试导演，当前话题「{topic}」考察结束，候选人评分 {score}/5，反馈：{feedback}。

当前任务树：
{task_tree}

请决定下一步（输出 JSON，只输出 JSON）：
- 若评分 ≥ 4：进入下一个 pending 任务
- 若评分 2-3：对当前话题拆出 1-2 个子话题深挖
- 若评分 1：拆出 2 个子话题（基础概念 + 应用场景）

输出格式（二选一）：
{{"action":"next"}}
{{"action":"deep_dive","subtopics":["子话题A","子话题B"]}}"""


async def director_initialize(session: InterviewSession) -> None:
    """生成初始任务树，写入 session.tasks，激活第一个任务。"""
    prompt = _DIRECTOR_INIT.format(
        profile=(session.profile_text or "（未提供）")[:2000],
        jd=(session.jd or "（未指定，通用技术岗位）")[:1000],
        direction=(session.direction or "（未指定，按候选人背景和 JD 判断）"),
    )
    text = await _llm([{"role": "user", "content": prompt}])
    raw = _parse_json(text, default=[])
    if not isinstance(raw, list):
        raw = []

    tasks: list[TaskNode] = []
    for item in raw[:6]:
        topic = item.get("topic", "") if isinstance(item, dict) else str(item)
        if topic:
            tasks.append(TaskNode(id=str(uuid.uuid4()), topic=topic, depth=0))

    if not tasks:
        tasks = [
            TaskNode(id=str(uuid.uuid4()), topic="基础技术概念", depth=0),
            TaskNode(id=str(uuid.uuid4()), topic="项目经验", depth=0),
        ]

    session.tasks = tasks
    tasks[0].status = "active"
    session.current_task_id = tasks[0].id


async def director_after_score(
    session: InterviewSession, task_id: str, score: int, feedback: str
) -> TaskNode | None:
    """话题结束后更新任务树，返回下一个任务（None 表示面试结束）。"""
    task = _find_task(session.tasks, task_id)
    if task is None:
        nxt = _next_pending(session.tasks)
        if nxt:
            nxt.status = "active"
            session.current_task_id = nxt.id
        else:
            session.status = "done"
            session.current_task_id = None
        return nxt

    task.score    = score
    task.feedback = feedback
    task.status   = "done"

    tree_json = json.dumps(tasks_to_dict(session.tasks), ensure_ascii=False, indent=2)
    prompt = _DIRECTOR_UPDATE.format(
        topic=task.topic, score=score, feedback=feedback,
        task_tree=tree_json[:3000],
    )
    text = await _llm([{"role": "user", "content": prompt}])
    decision = _parse_json(text, default={"action": "next"})

    if decision.get("action") == "deep_dive":
        for st in decision.get("subtopics", [])[:3]:
            child = TaskNode(
                id=str(uuid.uuid4()),
                topic=str(st),
                depth=task.depth + 1,
                status="pending",
                parent_id=task.id,
            )
            task.children.append(child)

    nxt = _next_pending(session.tasks)
    if nxt:
        nxt.status = "active"
        session.current_task_id = nxt.id
    else:
        session.status = "done"
        session.current_task_id = None

    return nxt


# ── ScorerAgent ───────────────────────────────────────────────────────────────

_SCORER_SYSTEM = """\
你是面试评分员，对候选人针对某话题的作答进行评分。
评分标准（1-5分）：
5 = 准确完整，有深度理解和实践经验
4 = 正确，覆盖核心知识点
3 = 基本正确，但有遗漏或不够深入
2 = 有一些概念但理解不清晰
1 = 基本不了解

只输出 JSON，不加代码块：{"score":3,"feedback":"一句话总结候选人表现"}"""


async def scorer_evaluate(topic: str, conversation: list[dict]) -> dict:
    """对候选人的作答评分，返回 {score, feedback}。"""
    conv = "\n".join(
        f"{'面试官' if m['role'] == 'assistant' else '候选人'}: {m['content'][:400]}"
        for m in conversation[-12:]
    )
    user_msg = f"考察话题：「{topic}」\n\n对话内容：\n{conv}"
    text = await _llm([{"role": "user", "content": user_msg}], system=_SCORER_SYSTEM)
    result = _parse_json(text, default={"score": 3, "feedback": "评分解析失败"})
    score = max(1, min(5, int(result.get("score", 3))))
    return {"score": score, "feedback": result.get("feedback", "")}


# ── InterviewerAgent（ReAct） ──────────────────────────────────────────────────

_INTERVIEWER_SYSTEM = """\
你是一位专业的技术面试官，正在考察话题「{topic}」。

{profile_section}
当前考察任务：{task_desc}

## 可用工具

**rag_search**
用途：检索技术知识库，获取相关参考内容（仅在需要深度知识时使用）
用法：
Action: rag_search
Action Input: 查询关键词

**topic_completed**
用途：当该话题已考察完毕时（候选人已回答过主要问题），使用此工具结束本话题
用法：
Action: topic_completed
Action Input: 无

## ReAct 格式

Thought: 分析当前情况，决定下一步
Action: rag_search 或 topic_completed
Action Input: 工具输入（如有）
Observation: 工具返回
...（可多次）
Final Answer: 直接面向候选人的回复（只有这部分会展示给用户）

## 面试规则

- 每次只问一个问题，问完等候选人回答
- 首轮必须提问（不要闲聊或自我介绍）
- 追问时根据候选人的上一条回答展开，不要重复同一问题
- 若候选人回答了 2-3 次，评估是否使用 topic_completed
- Final Answer 语气自然专业，≤3 句话"""

_REACT_MAX_STEPS = 5


async def interviewer_respond(
    session: InterviewSession,
    task: TaskNode,
    user_message: str | None,
) -> tuple[str, bool]:
    """
    面试官 ReAct 循环。
    user_message=None 时为开场（面试官主动提问）。
    返回 (response_text, topic_done)。
    """
    import rag as _rag

    profile_section = ""
    if session.profile_text:
        profile_section = f"候选人背景：\n{session.profile_text[:1000]}\n"

    parent_hint = ""
    if task.parent_id:
        parent = _find_task(session.tasks, task.parent_id)
        if parent:
            parent_hint = f"（是「{parent.topic}」的深挖子话题）"

    system = _INTERVIEWER_SYSTEM.format(
        topic=task.topic,
        profile_section=profile_section,
        task_desc=f"考察「{task.topic}」{parent_hint}",
    )

    # 构建工作消息（最近 8 条历史 + 当前用户消息）
    working: list[dict] = list(session.history[-8:])
    if user_message:
        working.append({"role": "user", "content": user_message})
    elif not working:
        # 开场：让面试官主动发问
        working.append({"role": "user", "content": "请开始面试，向候选人提出第一个问题。"})

    topic_done = False
    final_answer = ""

    for step in range(_REACT_MAX_STEPS):
        response = await _llm(working, system=system)

        # 提取 Final Answer
        if "Final Answer:" in response:
            final_answer = response.split("Final Answer:", 1)[1].strip()
            break

        # 提取 Action
        action_m       = re.search(r"Action:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        action_input_m = re.search(r"Action Input:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)

        if not action_m:
            # LLM 没遵守格式，当作直接回复
            final_answer = response.strip()
            break

        action      = action_m.group(1).strip().lower()
        action_input = action_input_m.group(1).strip() if action_input_m else ""

        # 把 LLM 的这轮输出追加到工作历史
        working.append({"role": "assistant", "content": response})

        if "topic_completed" in action:
            topic_done = True
            # 请 LLM 生成收尾话语
            working.append({
                "role": "user",
                "content": "Observation: 话题已标记完成。\n请在 Final Answer 中给候选人一句简短的过渡语，然后结束本话题。",
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
                if chunks:
                    obs = "检索到相关内容：\n" + "\n---\n".join(
                        f"[{c['source']}] {c['text'][:300]}" for c in chunks
                    )
                else:
                    obs = "知识库中未找到相关内容。"
            except Exception as e:
                obs = f"检索失败: {e}"
            working.append({"role": "user", "content": f"Observation: {obs}"})

        else:
            # 未知工具，当作直接回复
            final_answer = response.strip()
            break

    if not final_answer:
        final_answer = "请继续，我在听您的回答。"

    return final_answer, topic_done
