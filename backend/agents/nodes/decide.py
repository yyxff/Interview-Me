"""
decide_node：Director 决策 + route_after_decide 条件边路由函数
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

from agents.models import ThoughtNode, add_planned_nodes, find, flat, next_pending_task

from ..llm import _build_llm
from ..state import InterviewState, _dict_to_node, _node_to_dict, _parse_json
from ..tools import _make_decide_tools


_DECIDE_SYSTEM = """\
你是面试导演。根据当前评分和全局进度决定下一步策略，必要时动态调整任务列表。

── 当前任务决策（必须输出）────────────────────────────────────────
决策选项（四选一）：
- "deepen"  : 候选人理解不够，从多个角度继续考察当前知识点（下钻）
- "pivot"   : 当前答题尚可，但任务还有其他重要角度未覆盖（同级）
- "back_up" : 追问已经太细太偏，退回上一层换角度
- "pass"    : 当前任务已充分考察，推进下一任务

决策依据：
- score≤2 且深度<3：优先 deepen，规划 2-3 个具体角度
- score≥4：优先 pass 或 pivot
- 深度≥3：优先 back_up 或 pass
- 本任务已问题数≥4：强制 pass

── 全局任务调整（可选，调用工具）────────────────────────────────
结合候选人表现和全局进度，判断是否需要：
- add_task   : 候选人简历/回答中暴露出值得深挖的新方向，且现有任务未覆盖
- remove_task: 某个 pending 任务与候选人能力明显不匹配，或时间不足

调用工具后，再输出下面的 JSON。

输出 JSON（不加代码块）：
{"decision": "deepen", "reasoning": "理由", "sub_questions": ["子问题方向1","子问题方向2"]}"""


# ── 上下文函数 ────────────────────────────────────────────────────────────────

def _build_tree_summary(roots: list[ThoughtNode], current_task_id: str | None) -> str:
    """把整棵任务树压缩成一段导演视角的摘要。"""
    lines = [f"任务概览（共 {len(roots)} 个）："]
    for node in roots:
        questions = [n for n in flat([node]) if n.node_type == "question"
                     and n.status not in ("planned", "skipped")]
        scores = [n.score for n in questions if n.score is not None]
        avg = f"均分{sum(scores)/len(scores):.1f}" if scores else "未评分"
        q_count = f"{len(questions)}题"
        marker = "▶ " if node.id == current_task_id else "  "
        status_tag = {"done": "✓", "active": "▶", "pending": "○"}.get(node.status, "?")
        lines.append(f"  {marker}[{status_tag}] {node.text}  {q_count} {avg}")
    return "\n".join(lines)


def _get_decide_context(state: InterviewState) -> tuple[list, ThoughtNode | None, ThoughtNode | None, int, str]:
    roots = [_dict_to_node(d) for d in state["roots_data"]]
    qnode = find(roots, state["current_question_id"])
    task_node = find(roots, state["current_task_id"])

    if qnode is None or task_node is None:
        return roots, qnode, task_node, 0, "无"

    question_count = sum(
        1 for n in flat([task_node])
        if n.node_type == "question" and n.status not in ("planned", "skipped")
    )
    parent = find(roots, qnode.parent_id)
    pending_plan = "、".join(
        n.question_intent for n in (parent.children if parent else [])
        if n.status == "planned" and n.id != qnode.id
    ) or "无"

    return roots, qnode, task_node, question_count, pending_plan


def _build_decide_prompt(state: InterviewState, roots: list[ThoughtNode],
                         qnode: ThoughtNode, task_node: ThoughtNode,
                         question_count: int, pending_plan: str) -> str:
    tree_summary = _build_tree_summary(roots, state.get("current_task_id"))
    return (
        f"{tree_summary}\n\n"
        f"当前考察任务：{task_node.text}（task_id={task_node.id}）\n"
        f"面试官的问题：{qnode.text}\n"
        f"候选人的回答：{qnode.answer[:400]}\n"
        f"评分：{state['last_score']}/5\n"
        f"评分分析：{qnode.reasoning[:400]}\n"
        f"当前问题深度：{qnode.depth}\n"
        f"本任务已问题数：{question_count}\n"
        f"已有待问计划：{pending_plan}"
    )


def _apply_decision(decision: str, sub_questions: list[str],
                    qnode: ThoughtNode, task_node: ThoughtNode,
                    roots: list, current_task_id: str | None) -> str | None:
    """根据决策更新思维树，返回更新后的 current_task_id。"""
    if decision == "pass":
        task_node.status = "done"
        next_task = next_pending_task(roots)
        if next_task:
            next_task.status = "active"
            return next_task.id
        return None

    if decision in ("deepen", "pivot", "back_up"):
        if decision == "deepen":
            plan_parent = qnode
        elif decision == "pivot":
            plan_parent = task_node
        else:  # back_up
            gp = find(roots, qnode.parent_id)
            plan_parent = (find(roots, gp.parent_id) if gp and gp.parent_id else gp) or task_node
        if sub_questions:
            add_planned_nodes(plan_parent, sub_questions)

    return current_task_id


# ── Node ──────────────────────────────────────────────────────────────────────

async def decide_node(state: InterviewState) -> dict:
    roots, qnode, task_node, question_count, pending_plan = _get_decide_context(state)

    if qnode is None or task_node is None:
        return {"last_verdict": "pass", "last_sub_questions": [], "last_director_reasoning": "节点不存在"}

    logger.info("[decide] ▶ start  score=%s  depth=%d  q_count=%d  tasks=%d",
                state['last_score'], qnode.depth, question_count, len(roots))

    llm   = _build_llm()
    tools = _make_decide_tools(roots)   # 闭包捕获 roots，工具直接原地修改树
    react_agent = create_react_agent(llm, tools, prompt=SystemMessage(content=_DECIDE_SYSTEM))

    result = await react_agent.ainvoke({
        "messages": [HumanMessage(content=_build_decide_prompt(
            state, roots, qnode, task_node, question_count, pending_plan
        ))]
    })

    d = _parse_json(result["messages"][-1].content,
                    default={"decision": "pass", "reasoning": "解析失败", "sub_questions": []})
    decision = d.get("decision", "pass")
    if decision not in ("deepen", "pivot", "back_up", "pass"):
        decision = "pass"
    sub_questions = [q.strip() for q in d.get("sub_questions", []) if q.strip()][:3]

    qnode.verdict = decision
    qnode.director_note = d.get("reasoning", "")
    qnode.status = "done"

    # roots 可能已被工具（add_task/remove_task）原地修改，直接在此基础上应用决策
    updated_task_id = _apply_decision(decision, sub_questions, qnode, task_node, roots, state["current_task_id"])

    logger.info("[decide] verdict=%s  score=%s  tasks_now=%d", decision, state['last_score'], len(roots))

    return {
        "roots_data": [_node_to_dict(r) for r in roots],
        "current_task_id": updated_task_id,
        "last_verdict": decision,
        "last_sub_questions": sub_questions,
        "last_director_reasoning": d.get("reasoning", ""),
    }


# ── 条件边路由函数 ────────────────────────────────────────────────────────────

def route_after_decide(state: InterviewState) -> Literal["ask", "__end__"]:
    verdict = state.get("last_verdict", "pass")
    current_task_id = state.get("current_task_id")

    if verdict == "pass" and current_task_id is None:
        return "__end__"
    return "ask"
