"""DirectorAgent：面试任务规划 + 决策推进"""
from __future__ import annotations

import asyncio
from typing import Any

from .models import ThoughtNode, InterviewSession, flat, find, next_pending_task, add_planned_nodes
from .react import _REACT_HEADER, react_loop

_provider: Any = None


def set_provider(p: Any) -> None:
    global _provider
    _provider = p


async def _llm(messages: list[dict], system: str = "", timeout: float = 50.0) -> str:
    if _provider is None:
        raise RuntimeError("LLM provider 未注入")
    return await asyncio.wait_for(_provider.chat(messages, system), timeout=timeout)


def _parse_json(text: str, default: Any) -> Any:
    import json, re
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


# ── PLANNING ──────────────────────────────────────────────────────────────────

_PLAN_SYS = """\
你是技术面试导演。根据候选人 Profile 和 JD，将整场面试拆分为 4-6 个具体的考察任务。
每个任务是一道具体的面试题方向，不是宽泛话题。
task_type 只能是: experience/knowledge/concept/design/debug/scenario

对每个任务，你可以在 sub_questions 中预规划 1-3 个具体子问题方向（考察角度）。
面试时面试官会依次展开这些角度，未回答完的子问题随时可修改。

你可以先搜索知识库/简历/历史记录，了解候选人背景和可考察内容，再制定计划。

{react_header}

Final Answer 必须是 JSON 数组（只输出数组，不加代码块）：
[{{"task":"介绍你做过最有挑战的项目","task_type":"experience","sub_questions":["项目背景和解决的问题","你的具体职责和贡献","遇到的最大技术挑战"]}},...]\
"""

_PLAN_USR = "候选人 Profile：\n{profile}\n\n岗位 JD：\n{jd}\n\n考察方向：\n{direction}"


async def director_plan(session: InterviewSession) -> None:
    """PLANNING 阶段：ReAct → 创建根任务节点。"""
    import uuid
    from tools import build_toolset

    assert session.sm.state == "PLANNING"
    tools      = build_toolset(session, ["search_knowledge", "search_profile", "search_past_sessions"])
    tool_fns   = {n: t["fn"] for n, t in tools.items()}
    tool_descs = "\n".join(t["desc"] for t in tools.values())
    react_header = _REACT_HEADER.format(max_steps=3, tool_descs=tool_descs)

    system = _PLAN_SYS.format(react_header=react_header)
    user   = _PLAN_USR.format(
        profile=(session.profile_text or "（未提供）")[:2000],
        jd=(session.jd or "（未指定）")[:1000],
        direction=(session.direction or "（未指定）"),
    )
    text = await react_loop(system, user, tool_fns, max_steps=3, timeout_per_step=50.0)

    raw = _parse_json(text, default=[])
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
    session.roots = roots
    print(f"[director_plan] tasks={len(roots)} planned_per_task={[len(r.children) for r in roots]}")


def _skip_planned_descendants(node: ThoughtNode) -> None:
    for n in flat([node]):
        if n.id != node.id and n.status == "planned":
            n.status = "skipped"
            print(f"[skip] planned node skipped: '{n.question_intent[:50]}'")


async def director_advance(session: InterviewSession) -> ThoughtNode | None:
    """DIRECTING 阶段：标记当前任务完成，返回下一个待处理任务。"""
    assert session.sm.state == "DIRECTING"
    cur = find(session.roots, session.sm.current_task_id)
    if cur:
        _skip_planned_descendants(cur)
        cur.status = "done"
    nxt = next_pending_task(session.roots)
    if nxt:
        nxt.status = "active"
        session.sm.current_task_id = nxt.id
    return nxt


# ── DECIDING ──────────────────────────────────────────────────────────────────

_DIRECTOR_DECIDE_SYS = """\
你是面试导演，负责根据评分结果决定下一步考察策略。
你可以一次规划多个同级问题（sub_questions），面试官会依次问完。

决策选项（四选一）：
- "deepen"  : 候选人理解不够，需要从多个角度继续考察当前知识点（子节点）
- "pivot"   : 当前答题尚可，但任务还有其他重要角度未覆盖（挂到任务根下的同级）
- "back_up" : 追问已经太细太偏，退回上一层换角度
- "pass"    : 当前任务已充分考察，推进下一任务

决策依据：
- score≤2 且深度<3：优先 deepen，规划 2-3 个具体角度
- score≥4：优先 pass 或 pivot
- 深度≥3：优先 back_up 或 pass
- 本任务已问题数≥4：强制 pass

sub_questions：你要规划的具体问题方向列表（1-3 条）。
每条是一句话的考察焦点，面试官会据此生成正式问题。
pass 时 sub_questions 留空。

输出 JSON（不加代码块）：
{
  "decision": "deepen",
  "reasoning": "候选人完全不知道内存隔离机制...",
  "sub_questions": ["..."]
}"""

_DIRECTOR_DECIDE_USR = """\
当前考察任务：{task_text}
面试官的问题：{question}
候选人的回答：{answer}
评分：{score}/5
评分分析：{reasoning}
当前问题在思维树中的深度：{depth}（0=任务根，1=首问，以此类推）
本任务已问题数：{question_count}
已有待问计划（未来同级节点）：{pending_plan}"""


def _is_under_task(roots: list[ThoughtNode], node: ThoughtNode, task_id: str | None) -> bool:
    if task_id is None:
        return False
    for n in flat(roots):
        if n.id == task_id:
            return node in flat([n])
    return False


async def director_decide(
    session: InterviewSession,
    question_node: ThoughtNode,
    score_result: dict,
) -> dict:
    """DIRECTING 阶段：导演根据评分决定下一步策略。"""
    assert session.sm.state == "DIRECTING"
    task_node = find(session.roots, session.sm.current_task_id)
    task_text = task_node.text if task_node else "（未知）"
    question_count = sum(
        1 for n in flat(session.roots)
        if n.node_type == "question" and _is_under_task(session.roots, n, session.sm.current_task_id)
    )
    parent = find(session.roots, question_node.parent_id)
    pending_siblings = [n.question_intent for n in (parent.children if parent else [])
                        if n.status == "planned" and n.id != question_node.id]
    pending_plan = "、".join(pending_siblings) if pending_siblings else "无"

    text = await _llm(
        [{"role": "user", "content": _DIRECTOR_DECIDE_USR.format(
            task_text=task_text,
            question=question_node.text,
            answer=question_node.answer[:400],
            score=score_result["score"],
            reasoning=score_result["reasoning"][:400],
            depth=question_node.depth,
            question_count=question_count,
            pending_plan=pending_plan,
        )}],
        system=_DIRECTOR_DECIDE_SYS,
    )
    result   = _parse_json(text, default={"decision": "pass", "reasoning": "解析失败", "sub_questions": []})
    decision = result.get("decision", "pass")
    if decision not in ("deepen", "pivot", "back_up", "pass"):
        decision = "pass"
    sub_questions = [q.strip() for q in result.get("sub_questions", []) if q.strip()][:3]
    print(f"[director] decision={decision} score={score_result['score']} depth={question_node.depth} plan={sub_questions}")
    return {
        "decision":      decision,
        "reasoning":     result.get("reasoning", ""),
        "sub_questions": sub_questions,
    }


def next_planned_sibling(session: InterviewSession, node: ThoughtNode) -> ThoughtNode | None:
    parent = find(session.roots, node.parent_id)
    if parent is None:
        return None
    for child in parent.children:
        if child.status == "planned" and child.id != node.id:
            return child
    return None
