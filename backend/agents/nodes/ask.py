"""
ask_node：Interviewer 出题 + interrupt() 等待回答
"""
from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from agents.models import ThoughtNode, find, flat

from ..llm import _build_llm
from ..state import InterviewState, _dict_to_node, _node_to_dict, _parse_json
from ..tools import _make_ask_tools


_INTERVIEWER_SYSTEM_TEMPLATE = (
    "你是一位专业技术面试官。\n\n"
    "当前考察任务：{task_text}\n"
    "出题场景：{question_context}\n"
    "{profile_section}\n"
    "规则：\n"
    "- 问题要有针对性，不重复已问过的问题\n"
    "- 语气自然专业，问题长度 1-2 句话\n"
    '- 如需了解具体知识点细节，可先搜索知识库\n\n'
    '输出 JSON（不加代码块）：\n'
    '{"intent": "考察意图描述", "question": "具体的面试问题"}'
)

_REFLECT_SYSTEM = """\
你是面试质量审核员。检查下面这道面试题是否合格：
1. 是否真正考察了出题意图中的知识点？
2. 候选人能否明确理解这道题在问什么？
只输出最终问题文本，不加任何解释。"""


# ── 上下文函数 ────────────────────────────────────────────────────────────────

def _get_ask_context(state: InterviewState) -> tuple:
    """从 state 还原树，返回出题所需的全部上下文。"""
    roots = [_dict_to_node(d) for d in state["roots_data"]]
    task_node = find(roots, state["current_task_id"])
    if task_node is None:
        raise RuntimeError("current_task_id 指向不存在的节点")

    asked = [
        n for n in flat([task_node])
        if n.node_type == "question" and n.status not in ("planned", "skipped")
    ]
    is_first = len(asked) == 0
    cur_qnode = find(roots, state["current_question_id"]) if not is_first else None
    director_focus = state.get("last_director_reasoning", "") or ""
    profile_section = (
        f"候选人背景：\n{state['profile_text'][:800]}\n"
        if state["profile_text"] else ""
    )
    return roots, task_node, cur_qnode, is_first, director_focus, profile_section


def _build_system(task_node: ThoughtNode, is_first: bool,
                  director_focus: str, profile_section: str) -> str:
    task_type = task_node.task_type
    if is_first:
        question_context = f"首问，话题开场（task_type={task_type}）"
    else:
        question_context = f"追问/转向（导演指示：{director_focus or '无'}）"
    # 用字符串拼接而非 .format()，避免 task_node.text 中的花括号被误解析
    return (
        _INTERVIEWER_SYSTEM_TEMPLATE
        .replace("{task_text}", task_node.text)
        .replace("{question_context}", question_context)
        .replace("{profile_section}", profile_section)
    )


def _build_user_message(task_node: ThoughtNode, is_first: bool,
                        cur_qnode: ThoughtNode | None, director_focus: str) -> str:
    if is_first:
        return f"这是对话的开始，请提出关于「{task_node.text}」的第一个问题。"
    return (
        f"上一个问题：{cur_qnode.text if cur_qnode else ''}\n"
        f"候选人的回答：{cur_qnode.answer[:300] if cur_qnode else ''}\n"
        f"导演指示：{director_focus or '根据上下文自行判断'}"
    )


async def _reflect(llm, intent: str, draft_q: str) -> str:
    """Self-reflection：验证生成的问题是否符合考察意图。"""
    if not (intent and draft_q):
        return draft_q
    try:
        result = await llm.ainvoke([
            SystemMessage(content=_REFLECT_SYSTEM),
            HumanMessage(content=f"出题意图：{intent}\n生成的问题：{draft_q}"),
        ])
        return result.content.strip() or draft_q
    except Exception:
        return draft_q


# ── Node ──────────────────────────────────────────────────────────────────────

async def ask_node(state: InterviewState) -> dict:
    """
    ── Node②：Interviewer 出题 + ③ interrupt() 等待回答 ───────────────

    演示要点：interrupt() 实现 Human-in-the-Loop
    ┌─────────────────────────────────────────────────────────────────┐
    │  正常代码：                                                      │
    │    answer = some_function()   # 同步等待                        │
    │                                                                 │
    │  LangGraph interrupt：                                          │
    │    answer = interrupt(question)  # 暂停图、持久化状态            │
    │    # ↑ 这行之后的代码在下次 ainvoke(Command(resume=...)) 时执行  │
    │                                                                 │
    │  对应两个 HTTP 请求：                                            │
    │    POST /interview/start → 图运行到 interrupt → 返回问题         │
    │    POST /interview/chat  → Command(resume=answer) → 继续        │
    └─────────────────────────────────────────────────────────────────┘
    """
    roots, task_node, cur_qnode, is_first, director_focus, profile_section = _get_ask_context(state)
    print(f"[ask] ▶ start  task='{task_node.text[:40]}'  is_first={is_first}")

    llm = _build_llm()
    system = _build_system(task_node, is_first, director_focus, profile_section)
    user_msg = _build_user_message(task_node, is_first, cur_qnode, director_focus)

    tools = _make_ask_tools(state)
    react_agent = create_react_agent(llm, tools, prompt=SystemMessage(content=system))
    print(f"[ask] calling ReAct ...")
    result = await react_agent.ainvoke({"messages": [HumanMessage(content=user_msg)]})
    print(f"[ask] ReAct done")

    parsed = _parse_json(result["messages"][-1].content, default={"intent": "", "question": ""})
    intent  = parsed.get("intent", "")
    draft_q = parsed.get("question", "").strip() or result["messages"][-1].content.strip()
    final_q = await _reflect(llm, intent, draft_q)

    new_qnode = ThoughtNode(
        id=str(uuid.uuid4()), node_type="question",
        text=final_q, depth=task_node.depth + 1,
        status="asking", parent_id=task_node.id,
        question_intent=intent,
    )
    task_node.children.append(new_qnode)

    print(f"[ask] intent='{intent[:50]}' q='{final_q[:60]}'")

    # ③ interrupt() ────────────────────────────────────────────────────────────
    # 执行到这里：图暂停，final_q 作为"中断值"返回给 HTTP 调用者。
    # 框架把当前完整 state 存入 checkpointer（按 thread_id 索引）。
    #
    # 下次调用 graph.ainvoke(Command(resume=user_answer), config) 时：
    #   → 框架从 checkpointer 恢复 state
    #   → 从 interrupt() 这行继续执行
    #   → user_answer 就是 Command(resume=...) 里传入的值
    print(f"[ask] ✔ question ready, calling interrupt()  q='{final_q[:60]}'")
    user_answer: str = interrupt(final_q)
    # ──────────────────────────────────────────────────────────────────────────

    print(f"[ask] ▶ resumed from interrupt, answer='{user_answer[:40]}'")
    new_qnode.answer = user_answer
    new_qnode.status = "answered"

    return {
        "roots_data": [_node_to_dict(r) for r in roots],
        "current_question_id": new_qnode.id,
        "messages": [AIMessage(content=final_q), HumanMessage(content=user_answer)],
    }
