"""
plan_node：Director 规划任务
"""
from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agents.models import ThoughtNode, add_planned_nodes

from ..llm import _build_llm
from ..state import InterviewState, _node_to_dict, _parse_json
from ..tools import _make_plan_tools


_PLAN_SYSTEM = """\
你是技术面试导演。根据候选人 Profile 和 JD，将整场面试拆分为 4-6 个具体的考察任务。
每个任务是一道具体的面试题方向，不是宽泛话题。
task_type 只能是: experience/knowledge/concept/design/debug/scenario

对每个任务，可在 sub_questions 中预规划 1-3 个具体子问题方向（考察角度）。

你可以先搜索知识库/简历/历史记录，了解候选人背景，再制定计划。

Final Answer 必须是 JSON 数组（只输出数组，不加代码块）：
[{"task":"介绍你做过最有挑战的项目","task_type":"experience","sub_questions":["项目背景","你的职责","技术挑战"]},...]"""


# ── 上下文函数 ────────────────────────────────────────────────────────────────

def _build_user_content(state: InterviewState) -> str:
    return (
        f"候选人 Profile：\n{state['profile_text'][:2000]}\n\n"
        f"岗位 JD：\n{state['jd'][:1000]}\n\n"
        f"考察方向：\n{state['direction']}"
    )


def _parse_tasks(raw: list) -> list[ThoughtNode]:
    """将 LLM 输出的 JSON 列表解析为 ThoughtNode 根节点列表。"""
    roots: list[ThoughtNode] = []
    for item in (raw if isinstance(raw, list) else [])[:6]:
        t = item.get("task", "").strip() if isinstance(item, dict) else ""
        if not t:
            continue
        task_node = ThoughtNode(
            id=str(uuid.uuid4()), node_type="task",
            text=t, task_type=item.get("task_type", "knowledge"),
        )
        sub_qs = [q.strip() for q in item.get("sub_questions", []) if isinstance(q, str) and q.strip()]
        add_planned_nodes(task_node, sub_qs)
        roots.append(task_node)
    if not roots:
        roots = [ThoughtNode(id=str(uuid.uuid4()), node_type="task", text="介绍你的项目经历")]
    return roots


# ── Node ──────────────────────────────────────────────────────────────────────

async def plan_node(state: InterviewState) -> dict:
    """
    ── Node①：Director 规划任务 ────────────────────────────────────────

    演示要点：create_react_agent 作为"子图"调用
    ┌─────────────────────────────────────────────────────────────────┐
    │  create_react_agent(llm, tools, prompt=system_msg)             │
    │    ↓                                                            │
    │  内部自动循环：                                                  │
    │    LLM 输出 → 有工具调用？→ 执行工具 → 追加观测 → LLM 输出 → …  │
    │    直到 LLM 不再调用工具 → 返回最终消息                          │
    │                                                                 │
    │  对比原来的 react_loop：用文本解析 Action/Observation            │
    │  现在：用模型原生 function calling，更健壮、更准确               │
    └─────────────────────────────────────────────────────────────────┘
    """
    llm = _build_llm()
    tools = _make_plan_tools(state)
    react_agent = create_react_agent(llm, tools, prompt=SystemMessage(content=_PLAN_SYSTEM))

    result = await react_agent.ainvoke({
        "messages": [HumanMessage(content=_build_user_content(state))]
    })

    raw = _parse_json(result["messages"][-1].content, default=[])
    roots = _parse_tasks(raw)
    roots[0].status = "active"

    print(f"[plan] tasks={len(roots)} planned={[len(r.children) for r in roots]}")

    return {
        "roots_data": [_node_to_dict(r) for r in roots],
        "current_task_id": roots[0].id,
        "current_question_id": None,
        "last_verdict": None,
        "last_sub_questions": [],
        "last_director_reasoning": "",
    }
