"""面试核心编排：start_interview、run_turn、save_session"""
from __future__ import annotations

import time
import json
import uuid
from pathlib import Path

from .models import (
    ThoughtNode, InterviewSession,
    find, flat, add_planned_nodes, tree_to_dict,
)
from .director import director_plan, director_advance, director_decide, next_planned_sibling
from .interviewer import interviewer_ask
from .scorer import scorer_evaluate, rollup_node

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


def _add_question_node(
    session: InterviewSession,
    parent_node: ThoughtNode,
    question_text: str,
    intent: str = "",
) -> ThoughtNode:
    qnode = ThoughtNode(
        id=str(uuid.uuid4()),
        node_type="question",
        text=question_text,
        depth=parent_node.depth + 1,
        status="asking",
        parent_id=parent_node.id,
        question_intent=intent,
    )
    parent_node.children.append(qnode)
    session.sm.current_node_id = qnode.id
    return qnode


def save_session(session: InterviewSession) -> Path:
    """将本次面试的思维树 + 历史对话保存为 JSON，返回文件路径。"""
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = SESSIONS_DIR / f"{ts}_{session.session_id[:8]}.json"
    data = {
        "session_id": session.session_id,
        "saved_at":   ts,
        "jd":         session.jd,
        "direction":  session.direction,
        "sm_final":   session.sm.to_dict(),
        "sm_log":     session.sm.event_log,
        "tree":       tree_to_dict(session.roots),
        "history":    session.history,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[session saved] {path}")
    return path


async def start_interview(session: InterviewSession) -> str:
    """INIT → PLANNING → ASKING → ANSWERING。返回面试官开场问题文本。"""
    sm = session.sm
    assert sm.state == "INIT"

    sm.transition("PLANNING")
    await director_plan(session)

    first_task = session.roots[0]
    first_task.status = "active"
    sm.current_task_id = first_task.id

    sm.transition("ASKING")
    q_text, intent = await interviewer_ask(session, parent_node=first_task, is_first=True)
    qnode = _add_question_node(session, first_task, q_text, intent=intent)
    qnode.status = "answering"

    sm.transition("ANSWERING")
    session.history.append({"role": "assistant", "content": q_text})
    print(f"[interview/start] tasks={len(session.roots)} first_q={q_text[:60]}")
    return q_text


async def run_turn(session: InterviewSession, user_answer: str) -> dict:
    """ANSWERING → SCORING → DIRECTING → ASKING → ANSWERING。返回 {response, sm, tree}。"""
    sm = session.sm
    assert sm.state == "ANSWERING", f"期望 ANSWERING，当前 {sm.state}"

    qnode = find(session.roots, sm.current_node_id)
    if qnode is None:
        raise RuntimeError("current_node_id 指向不存在的节点")

    qnode.answer = user_answer
    qnode.status = "answered"
    session.history.append({"role": "user", "content": user_answer})

    # SCORING
    sm.transition("SCORING")
    score_result    = await scorer_evaluate(session, qnode, user_answer)
    qnode.score     = score_result["score"]
    qnode.reasoning = score_result["reasoning"]
    qnode.feedback  = score_result["feedback"]
    qnode.status    = "scored"
    sm.last_score   = score_result
    print(f"[scorer] q='{qnode.text[:40]}' score={score_result['score']}")

    # DIRECTING
    sm.transition("DIRECTING")
    task_node = find(session.roots, sm.current_task_id)

    # 优先执行已规划的 planned 兄弟节点
    planned_next = next_planned_sibling(session, qnode)
    if planned_next:
        sm.transition("ASKING")
        q_text, intent = await interviewer_ask(
            session, parent_node=qnode, director_focus=planned_next.question_intent,
        )
        planned_next.text            = q_text
        planned_next.question_intent = intent
        planned_next.status          = "answering"
        session.sm.current_node_id   = planned_next.id
        sm.transition("ANSWERING")
        qnode.status = "done"
        session.history.append({"role": "assistant", "content": q_text})
        print(f"[planned] '{planned_next.question_intent[:50]}' → '{q_text[:60]}'")
        return {"response": q_text, "sm": sm.to_dict(), "tree": tree_to_dict(session.roots)}

    # 无 planned 节点，交给导演决策
    director_result = await director_decide(session, qnode, score_result)
    decision        = director_result["decision"]
    sub_questions   = director_result["sub_questions"]
    qnode.verdict       = decision
    qnode.director_note = director_result["reasoning"]

    if decision == "pass":
        if task_node:
            await rollup_node(task_node)
        next_task = await director_advance(session)
        if next_task is None:
            sm.transition("DONE")
            qnode.status = "done"
            closing = "非常感谢你的参与，今天的面试到此结束！请稍等查看评分详情。"
            session.history.append({"role": "assistant", "content": closing})
            save_session(session)
            return {"response": closing, "sm": sm.to_dict(), "tree": tree_to_dict(session.roots)}
        sm.transition("ASKING")
        q_text, intent = await interviewer_ask(session, parent_node=next_task, is_first=True)
        new_q = _add_question_node(session, next_task, q_text, intent=intent)

    else:
        if decision == "deepen":
            plan_parent = qnode
        elif decision == "pivot":
            plan_parent = task_node or qnode
        elif decision == "back_up":
            gp = find(session.roots, qnode.parent_id)
            plan_parent = (find(session.roots, gp.parent_id) if gp and gp.parent_id else gp) or task_node or qnode
            if gp:
                await rollup_node(gp)
        else:
            plan_parent = qnode

        if len(sub_questions) > 1:
            add_planned_nodes(plan_parent, sub_questions[1:])
        first_focus = sub_questions[0] if sub_questions else director_result.get("reasoning", "")
        sm.transition("ASKING")
        q_text, intent = await interviewer_ask(
            session, parent_node=qnode, director_focus=first_focus,
        )
        new_q = _add_question_node(session, plan_parent, q_text, intent=intent)
        new_q.status = "answering"
        sm.transition("ANSWERING")
        qnode.status = "done"
        session.history.append({"role": "assistant", "content": q_text})
        print(f"[director] {decision} plan={len(sub_questions)} next_q='{q_text[:60]}'")
        return {"response": q_text, "sm": sm.to_dict(), "tree": tree_to_dict(session.roots)}

    new_q.status = "answering"
    sm.transition("ANSWERING")
    qnode.status = "done"
    session.history.append({"role": "assistant", "content": q_text})
    print(f"[director] {decision} next_q='{q_text[:60]}'")
    return {"response": q_text, "sm": sm.to_dict(), "tree": tree_to_dict(session.roots)}
