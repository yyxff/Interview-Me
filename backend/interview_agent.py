"""多 Agent 模拟面试 —— 思维树 + 状态机
========================================

状态机：
    INIT → PLANNING → ASKING → ANSWERING → SCORING ─┬→ ASKING      (continue/deep_dive)
                                                     └→ DIRECTING → ASKING  (pass, 下一任务)
                                                                 └→ DONE

思维树（ThoughtNode）：
    根节点 (node_type='task')  ← Director 在 PLANNING 阶段创建
    └── 问题节点 (node_type='question')  ← Interviewer 在 ASKING 阶段创建
        └── 子问题节点 …  (deep_dive/continue 时追加)

Scorer 在每次用户回答后运行，返回 verdict：
    pass       → Director 推进到下一个 task
    continue   → Interviewer 在同层追问（sibling 节点）
    deep_dive  → Interviewer 深挖子问题（child 节点）
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


# ── 状态机 ────────────────────────────────────────────────────────────────────

_TRANSITIONS: dict[str, set[str]] = {
    "INIT":       {"PLANNING"},
    "PLANNING":   {"ASKING"},
    "ASKING":     {"ANSWERING"},
    "ANSWERING":  {"SCORING"},
    "SCORING":    {"ASKING", "DIRECTING"},
    "DIRECTING":  {"ASKING", "DONE"},
    "DONE":       set(),
}


class InterviewSM:
    def __init__(self):
        self.state: str = "INIT"
        self.current_node_id: str | None = None   # 当前问题节点 id
        self.current_task_id: str | None = None   # 当前根任务节点 id
        self.last_score: dict | None = None        # {score, verdict, feedback}
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


# ── 思维树节点 ────────────────────────────────────────────────────────────────

@dataclass
class ThoughtNode:
    id:         str
    node_type:  str          # 'task' | 'question'
    text:       str          # 任务描述 或 问题文本
    answer:     str  = ""    # 候选人的回答（question 节点）
    depth:      int  = 0
    status:     str  = "pending"   # pending/active/asking/answering/scored/done
    score:      int | None = None
    verdict:    str | None = None  # pass/continue/deep_dive
    feedback:   str  = ""
    task_type:  str  = ""          # task 节点：experience/knowledge/…
    children:   list["ThoughtNode"] = field(default_factory=list)
    parent_id:  str | None = None


# ── Session ───────────────────────────────────────────────────────────────────

@dataclass
class InterviewSession:
    session_id:   str
    jd:           str
    direction:    str
    profile_text: str
    sm:           InterviewSM            = field(default_factory=InterviewSM)
    roots:        list[ThoughtNode]      = field(default_factory=list)  # 根任务列表
    history:      list[dict]             = field(default_factory=list)  # 全程对话


# ── 全局状态 ──────────────────────────────────────────────────────────────────

_sessions: dict[str, InterviewSession] = {}
_provider: Any = None


def set_provider(p: Any) -> None:
    global _provider
    _provider = p


def get_session(sid: str) -> InterviewSession | None:
    return _sessions.get(sid)


# ── 树操作 ────────────────────────────────────────────────────────────────────

def _flat(nodes: list[ThoughtNode]) -> list[ThoughtNode]:
    out: list[ThoughtNode] = []
    q = list(nodes)
    while q:
        n = q.pop(0)
        out.append(n)
        q.extend(n.children)
    return out


def _find(roots: list[ThoughtNode], nid: str | None) -> ThoughtNode | None:
    if nid is None:
        return None
    for n in _flat(roots):
        if n.id == nid:
            return n
    return None


def _next_pending_task(roots: list[ThoughtNode]) -> ThoughtNode | None:
    for n in roots:
        if n.status == "pending":
            return n
    return None


def _flat_dict(nodes: list[dict]) -> list[dict]:
    """将序列化后的树（list[dict]）展平为列表，供 main.py 统计使用。"""
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
            "id":        n.id,
            "node_type": n.node_type,
            "text":      n.text,
            "answer":    n.answer[:120] if n.answer else "",
            "depth":     n.depth,
            "status":    n.status,
            "score":     n.score,
            "verdict":   n.verdict,
            "feedback":  n.feedback,
            "task_type": n.task_type,
            "children":  tree_to_dict(n.children),
        }
        for n in nodes
    ]


# ── LLM ───────────────────────────────────────────────────────────────────────

async def _llm(messages: list[dict], system: str = "", timeout: float = 50.0) -> str:
    if _provider is None:
        raise RuntimeError("LLM provider 未注入")
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

_PLAN_SYS = """\
你是技术面试导演。根据候选人 Profile 和 JD，将整场面试拆分为 4-6 个具体的考察任务。
每个任务是一道具体的面试题方向，不是宽泛话题。
task_type 只能是: experience/knowledge/concept/design/debug/scenario

输出 JSON 数组（只输出数组，不加代码块）：
[{"task":"介绍你做过最有挑战的项目，重点讲技术难点","task_type":"experience"},...]"""

_PLAN_USR = "候选人 Profile：\n{profile}\n\n岗位 JD：\n{jd}\n\n考察方向：\n{direction}"


async def director_plan(session: InterviewSession) -> None:
    """PLANNING 阶段：创建根任务节点。"""
    assert session.sm.state == "PLANNING"
    text = await _llm(
        [{"role": "user", "content": _PLAN_USR.format(
            profile=(session.profile_text or "（未提供）")[:2000],
            jd=(session.jd or "（未指定）")[:1000],
            direction=(session.direction or "（未指定）"),
        )}],
        system=_PLAN_SYS,
    )
    raw = _parse_json(text, default=[])
    roots: list[ThoughtNode] = []
    for item in (raw if isinstance(raw, list) else [])[:6]:
        t = item.get("task", "").strip() if isinstance(item, dict) else ""
        if t:
            roots.append(ThoughtNode(id=str(uuid.uuid4()), node_type="task",
                                     text=t, task_type=item.get("task_type", "knowledge")))
    if not roots:
        roots = [ThoughtNode(id=str(uuid.uuid4()), node_type="task", text="介绍你的项目经历")]
    session.roots = roots


async def director_advance(session: InterviewSession) -> ThoughtNode | None:
    """DIRECTING 阶段：标记当前任务完成，返回下一个待处理任务（None=全部完成）。"""
    assert session.sm.state == "DIRECTING"
    cur = _find(session.roots, session.sm.current_task_id)
    if cur:
        cur.status = "done"
    nxt = _next_pending_task(session.roots)
    if nxt:
        nxt.status = "active"
        session.sm.current_task_id = nxt.id
    return nxt


# ── ScorerAgent ───────────────────────────────────────────────────────────────

_SCORER_SYS = """\
你是面试评分员。根据候选人的回答对当前问题进行评分并给出判定。

评分（1-5）：
5=准确完整有深度  4=正确覆盖要点  3=基本正确有遗漏  2=概念模糊  1=不了解

判定规则：
- "pass"    : score≥4，或已充分考察该方向，可推进下一话题
- "continue": score 2-3，回答方向对但不够深入，换角度追问
- "deep_dive": score≤2，存在明显知识盲区，需要深挖基础

只输出 JSON（不加代码块）：{"score":3,"verdict":"continue","feedback":"了解概念但未说明原理"}"""

_SCORER_USR = """\
当前考察任务（上下文）：{task_text}
面试官的问题：{question}
候选人的回答：{answer}
本轮已追问次数：{follow_up_count}（若已超过3次，优先选 pass 推进）"""


async def scorer_evaluate(session: InterviewSession, question_node: ThoughtNode, answer: str) -> dict:
    """SCORING 阶段：评分并给出判定。"""
    assert session.sm.state == "SCORING"
    # 找根任务上下文
    task_node = _find(session.roots, session.sm.current_task_id)
    task_text = task_node.text if task_node else "（未知）"
    # 统计本轮追问次数（当前任务下的问题数）
    follow_up_count = sum(1 for n in _flat(session.roots)
                          if n.node_type == "question" and n.status in ("scored", "done")
                          and _is_under_task(session.roots, n, session.sm.current_task_id))

    text = await _llm(
        [{"role": "user", "content": _SCORER_USR.format(
            task_text=task_text,
            question=question_node.text,
            answer=answer[:600],
            follow_up_count=follow_up_count,
        )}],
        system=_SCORER_SYS,
    )
    result = _parse_json(text, default={"score": 3, "verdict": "continue", "feedback": "解析失败"})
    score   = max(1, min(5, int(result.get("score", 3))))
    verdict = result.get("verdict", "continue")
    if verdict not in ("pass", "continue", "deep_dive"):
        verdict = "continue"
    return {"score": score, "verdict": verdict, "feedback": result.get("feedback", "")}


def _is_under_task(roots: list[ThoughtNode], node: ThoughtNode, task_id: str | None) -> bool:
    """判断 node 是否在 task_id 的子树内。"""
    if task_id is None:
        return False
    for n in _flat(roots):
        if n.id == task_id:
            return node in _flat([n])
    return False


# ── InterviewerAgent ──────────────────────────────────────────────────────────

_INTERVIEWER_SYS = """\
你是一位专业技术面试官，正在考察候选人。

当前考察任务：{task_text}
你要问的问题类型：{question_context}

{profile_section}
规则：
- 输出一道具体的面试问题，语气自然专业
- 问题要针对任务/追问场景，不重复之前问过的问题
- 如果是 deep_dive，聚焦候选人回答中暴露的薄弱点
- 只输出问题本身，不加任何前缀或解释
- 问题长度：1-2句话"""

_INTERVIEWER_USR_INIT = """\
这是对话的开始，请提出关于「{task_text}」的第一个问题。
参考知识（可选）：{rag_context}"""

_INTERVIEWER_USR_FOLLOWUP = """\
上一个问题：{prev_question}
候选人的回答：{prev_answer}
评分员反馈：{feedback}（判定：{verdict}）

请根据判定提出下一个问题：
- continue: 换角度继续追问同话题
- deep_dive: 深挖候选人回答中暴露的薄弱点

参考知识（可选）：{rag_context}"""


async def interviewer_ask(
    session: InterviewSession,
    parent_node: ThoughtNode,
    verdict: str | None = None,
    score_feedback: str = "",
) -> str:
    """ASKING 阶段：生成下一个面试问题，返回问题文本。"""
    assert session.sm.state == "ASKING"
    import rag as _rag

    task_node = _find(session.roots, session.sm.current_task_id)
    task_text = task_node.text if task_node else "（未知）"

    # RAG 检索辅助（对 knowledge/concept/design 类型做检索）
    rag_context = ""
    task_type = task_node.task_type if task_node else "knowledge"
    if task_type in ("knowledge", "concept", "design", "debug"):
        try:
            r = _rag.retrieve_rich(task_text)
            chunks = r.get("knowledge", [])[:2]
            if chunks:
                rag_context = "\n".join(f"[{c['source']}] {c['text'][:200]}" for c in chunks)
        except Exception:
            pass

    profile_section = f"候选人背景：\n{session.profile_text[:800]}\n" if session.profile_text else ""

    if verdict is None:
        # 首个问题
        question_context = f"首问，话题开场（task_type={task_type}）"
        user_msg = _INTERVIEWER_USR_INIT.format(task_text=task_text, rag_context=rag_context or "无")
    else:
        question_context = "deep_dive（深挖薄弱点）" if verdict == "deep_dive" else "continue（换角度追问）"
        user_msg = _INTERVIEWER_USR_FOLLOWUP.format(
            prev_question=parent_node.text,
            prev_answer=parent_node.answer[:300],
            feedback=score_feedback,
            verdict=verdict,
            rag_context=rag_context or "无",
        )

    system = _INTERVIEWER_SYS.format(
        task_text=task_text,
        question_context=question_context,
        profile_section=profile_section,
    )
    q_text = await _llm([{"role": "user", "content": user_msg}], system=system)
    return q_text.strip()


# ── 核心编排 ──────────────────────────────────────────────────────────────────

def _add_question_node(
    session: InterviewSession,
    parent_node: ThoughtNode,
    question_text: str,
) -> ThoughtNode:
    """在思维树上追加一个问题节点，更新 SM。"""
    depth = parent_node.depth + 1
    qnode = ThoughtNode(
        id=str(uuid.uuid4()),
        node_type="question",
        text=question_text,
        depth=depth,
        status="asking",
        parent_id=parent_node.id,
    )
    parent_node.children.append(qnode)
    session.sm.current_node_id = qnode.id
    return qnode


def save_session(session: InterviewSession) -> Path:
    """将本次面试的思维树 + 历史对话保存为 JSON，返回文件路径。"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = SESSIONS_DIR / f"{ts}_{session.session_id[:8]}.json"
    data = {
        "session_id":   session.session_id,
        "saved_at":     ts,
        "jd":           session.jd,
        "direction":    session.direction,
        "sm_final":     session.sm.to_dict(),
        "sm_log":       session.sm.event_log,
        "tree":         tree_to_dict(session.roots),
        "history":      session.history,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[session saved] {path}")
    return path


async def start_interview(session: InterviewSession) -> str:
    """
    INIT → PLANNING → ASKING → ANSWERING。
    返回面试官的开场问题文本。
    """
    sm = session.sm
    assert sm.state == "INIT"

    # PLANNING: Director 拆分任务
    sm.transition("PLANNING")
    await director_plan(session)

    # 激活第一个根任务
    first_task = session.roots[0]
    first_task.status = "active"
    sm.current_task_id = first_task.id

    # ASKING: Interviewer 生成第一个问题
    sm.transition("ASKING")
    q_text = await interviewer_ask(session, parent_node=first_task, verdict=None)
    qnode = _add_question_node(session, first_task, q_text)
    qnode.status = "answering"

    # ANSWERING: 等待用户
    sm.transition("ANSWERING")

    # 记录到全程历史
    session.history.append({"role": "assistant", "content": q_text})
    print(f"[interview/start] tasks={len(session.roots)} first_q={q_text[:60]}")
    return q_text


async def run_turn(session: InterviewSession, user_answer: str) -> dict:
    """
    ANSWERING → SCORING → (ASKING | DIRECTING) → ANSWERING。
    返回 {response, sm, tree}。
    """
    sm = session.sm
    assert sm.state == "ANSWERING", f"期望 ANSWERING，当前 {sm.state}"

    # 找当前问题节点
    qnode = _find(session.roots, sm.current_node_id)
    if qnode is None:
        raise RuntimeError("current_node_id 指向不存在的节点")

    # 记录用户回答
    qnode.answer = user_answer
    qnode.status = "answered"
    session.history.append({"role": "user", "content": user_answer})

    # SCORING
    sm.transition("SCORING")
    score_result = await scorer_evaluate(session, qnode, user_answer)
    qnode.score   = score_result["score"]
    qnode.verdict = score_result["verdict"]
    qnode.feedback = score_result["feedback"]
    qnode.status  = "scored"
    sm.last_score = score_result
    print(f"[scorer] q='{qnode.text[:40]}' score={score_result['score']} verdict={score_result['verdict']}")

    verdict = score_result["verdict"]

    if verdict == "pass":
        # 当前任务通过 → Director 推进
        sm.transition("DIRECTING")
        next_task = await director_advance(session)

        if next_task is None:
            sm.transition("DONE")
            qnode.status = "done"
            closing = "非常感谢你的参与，今天的面试到此结束！请稍等查看评分详情。"
            session.history.append({"role": "assistant", "content": closing})
            save_session(session)
            return {"response": closing, "sm": sm.to_dict(), "tree": tree_to_dict(session.roots)}

        # 下一任务首问
        sm.transition("ASKING")
        q_text = await interviewer_ask(session, parent_node=next_task, verdict=None)
        new_q = _add_question_node(session, next_task, q_text)
        new_q.status = "answering"
        sm.transition("ANSWERING")

    else:
        # continue/deep_dive：都挂在 qnode 下（子节点），逐层深入
        sm.transition("ASKING")
        q_text = await interviewer_ask(session, parent_node=qnode, verdict=verdict,
                                       score_feedback=score_result["feedback"])
        new_q = _add_question_node(session, qnode, q_text)
        new_q.status = "answering"
        sm.transition("ANSWERING")

    qnode.status = "done"
    session.history.append({"role": "assistant", "content": q_text})
    print(f"[interviewer] next_q='{q_text[:60]}'")
    return {"response": q_text, "sm": sm.to_dict(), "tree": tree_to_dict(session.roots)}
