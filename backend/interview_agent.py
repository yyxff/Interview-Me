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
from typing import Any, Callable, Awaitable

SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


# ── ReAct 引擎 ────────────────────────────────────────────────────────────────

_REACT_HEADER = """\
你可以使用以下工具（最多 {max_steps} 次）：
{tool_descs}

每步格式（二选一）：
① 使用工具：
Thought: <推理过程>
Action: <工具名>
Action Input: <查询字符串>

② 直接给出结果（不需要工具时）：
Final Answer: <最终输出>

工具结果会以 Observation: <结果> 形式返回。用完工具次数后必须给 Final Answer。
"""

_REACT_FORCE_FINAL = "你已用完工具调用次数，现在必须给出 Final Answer。"


async def _react_loop(
    init_system: str,
    init_user: str,
    tools: dict[str, Callable[[str], Awaitable[str]]],
    max_steps: int = 3,
    timeout_per_step: float = 40.0,
) -> str:
    """
    通用 ReAct 循环。
    init_system 应包含 _REACT_HEADER（调用方格式化好后传入）。
    返回 Final Answer 的文本。
    """
    msgs: list[dict] = [{"role": "user", "content": init_user}]

    for step in range(max_steps + 1):
        if step == max_steps:
            msgs.append({"role": "user", "content": _REACT_FORCE_FINAL})

        raw = await asyncio.wait_for(_provider.chat(msgs, init_system), timeout_per_step)
        msgs.append({"role": "assistant", "content": raw})

        # 提取 Final Answer
        if "Final Answer:" in raw:
            return raw.split("Final Answer:", 1)[1].strip()

        if step == max_steps:
            return raw.strip()   # force fallback

        # 提取 Action / Action Input
        action_m = re.search(r"Action:\s*(\w+)", raw)
        input_m  = re.search(r"Action Input:\s*[\"']?(.*?)[\"']?\s*$", raw, re.MULTILINE)
        if not action_m:
            return raw.strip()   # no action → treat as final

        tool_name  = action_m.group(1).strip()
        tool_input = input_m.group(1).strip() if input_m else ""

        if tool_name in tools:
            try:
                observation = await asyncio.wait_for(tools[tool_name](tool_input), 20.0)
            except Exception as e:
                observation = f"工具调用失败: {e}"
        else:
            observation = f"未知工具: {tool_name}"

        observation = observation[:600]
        msgs.append({"role": "user", "content": f"Observation: {observation}"})
        print(f"[react] step={step} tool={tool_name} obs={observation[:80]}")

    return raw.strip()


# ── ReAct 工具（由 tools/ 目录统一管理） ───────────────────────────────────────

from tools import build_toolset   # noqa: E402  (placed after dataclass defs below)


# ── 状态机 ────────────────────────────────────────────────────────────────────

_TRANSITIONS: dict[str, set[str]] = {
    "INIT":       {"PLANNING"},
    "PLANNING":   {"ASKING"},
    "ASKING":     {"ANSWERING"},
    "ANSWERING":  {"SCORING"},
    "SCORING":    {"DIRECTING"},          # 总是先交给导演决策
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
    id:             str
    node_type:      str          # 'task' | 'question'
    text:           str          # 任务描述 或 问题文本
    answer:         str  = ""    # 候选人的回答（question 节点）
    depth:          int  = 0
    status:         str  = "pending"   # pending/active/asking/answering/scored/done
    score:          int | None = None
    verdict:        str | None = None  # pass/deepen/pivot/back_up (导演决策)
    feedback:       str  = ""    # 评分员给候选人的简要反馈
    reasoning:      str  = ""    # 评分员 CoT：逐点分析
    director_note:  str  = ""    # 导演决策理由
    question_intent: str = ""    # 面试官出题意图（self-reflection 产物）
    task_type:      str  = ""    # task 节点：experience/knowledge/…
    summary:        str  = ""    # rollup 摘要（pass/back_up 时生成）
    children:       list["ThoughtNode"] = field(default_factory=list)
    parent_id:      str | None = None


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
            "id":             n.id,
            "node_type":      n.node_type,
            "text":           n.text,
            "answer":         n.answer[:120] if n.answer else "",
            "depth":          n.depth,
            "status":         n.status,
            "score":          n.score,
            "verdict":        n.verdict,
            "feedback":       n.feedback,
            "reasoning":      n.reasoning,
            "director_note":  n.director_note,
            "question_intent": n.question_intent,
            "task_type":      n.task_type,
            "summary":        n.summary,
            "children":       tree_to_dict(n.children),
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
    assert session.sm.state == "PLANNING"
    # director_plan 可用全部工具
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
    text = await _react_loop(system, user, tool_fns, max_steps=3, timeout_per_step=50.0)

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
        # 预埋子问题（导演在规划阶段的 fork）
        sub_qs = [q.strip() for q in item.get("sub_questions", []) if isinstance(q, str) and q.strip()]
        _add_planned_nodes(None, task_node, sub_qs)   # parent_node=task_node
        roots.append(task_node)
    if not roots:
        roots = [ThoughtNode(id=str(uuid.uuid4()), node_type="task", text="介绍你的项目经历")]
    session.roots = roots
    print(f"[director_plan] tasks={len(roots)} planned_per_task={[len(r.children) for r in roots]}")


def _skip_planned_descendants(node: ThoughtNode) -> None:
    """将节点下所有 planned 后代标为 skipped（任务推进时清理未执行的计划）。"""
    for n in _flat([node]):
        if n.id != node.id and n.status == "planned":
            n.status = "skipped"
            print(f"[skip] planned node skipped: '{n.question_intent[:50]}'")


async def director_advance(session: InterviewSession) -> ThoughtNode | None:
    """DIRECTING 阶段：标记当前任务完成（清理残留 planned），返回下一个待处理任务。"""
    assert session.sm.state == "DIRECTING"
    cur = _find(session.roots, session.sm.current_task_id)
    if cur:
        _skip_planned_descendants(cur)   # 完成约束：跳过未问的计划节点
        cur.status = "done"
    nxt = _next_pending_task(session.roots)
    if nxt:
        nxt.status = "active"
        session.sm.current_task_id = nxt.id
    return nxt


# ── ScorerAgent ───────────────────────────────────────────────────────────────

_SCORER_SYS = """\
你是面试评分员。你的职责是**客观评分并给出逐点分析**，不做策略决策。

评分标准（1-5分）：
5分：准确完整，有深度，能说明原理并举例
4分：覆盖核心要点，表述清晰，基本无遗漏
3分：方向正确但有明显遗漏，或表述模糊
2分：只了解表面概念，细节错误或不知道原理
1分：答非所问，或完全不了解

输出 JSON（不加代码块）：
{
  "score": 3,
  "reasoning": "候选人提到了进程拥有独立地址空间(+1)，提到了PCB(+1)，但对进程和线程的本质区别（共享地址空间 vs 独立地址空间）完全未提及(-1.5)，对调度时机的描述也不够准确(-0.5)，综合得3分",
  "feedback": "掌握了进程基本定义，但缺少进程与线程的核心区别，以及调度时机的准确描述"
}"""

_SCORER_USR = """\
当前考察任务：{task_text}
面试官的问题：{question}
候选人的回答：{answer}"""


async def scorer_evaluate(session: InterviewSession, question_node: ThoughtNode, answer: str) -> dict:
    """SCORING 阶段：CoT 评分，只评分不做决策。"""
    assert session.sm.state == "SCORING"
    task_node = _find(session.roots, session.sm.current_task_id)
    task_text = task_node.text if task_node else "（未知）"

    text = await _llm(
        [{"role": "user", "content": _SCORER_USR.format(
            task_text=task_text,
            question=question_node.text,
            answer=answer[:800],
        )}],
        system=_SCORER_SYS,
    )
    result   = _parse_json(text, default={"score": 3, "reasoning": "解析失败", "feedback": "解析失败"})
    score    = max(1, min(5, int(result.get("score", 3))))
    return {
        "score":     score,
        "reasoning": result.get("reasoning", ""),
        "feedback":  result.get("feedback", ""),
    }


# ── Rollup Summary ────────────────────────────────────────────────────────────

_ROLLUP_SYS = """\
你是面试复盘助手。用一句话（≤60字）总结以下面试片段的考察结果。
格式：「{考察角度}：{候选人表现要点，含得分/薄弱点}」
只输出这一句话，不加任何前缀或解释。"""

_ROLLUP_USR = """\
考察任务：{task_text}
问答记录：
{qa_lines}"""


def _build_qa_lines(node: ThoughtNode, depth: int = 0) -> str:
    """递归构建节点的问答文本，用于传给 LLM 摘要。"""
    indent = "  " * depth
    lines: list[str] = []
    if node.node_type == "question":
        score_str = f"{node.score}/5" if node.score is not None else "未评分"
        lines.append(f"{indent}Q: {node.text}")
        if node.answer:
            lines.append(f"{indent}A: {node.answer[:200]}")
        lines.append(f"{indent}→ {score_str}  {node.feedback[:80]}")
    for child in node.children:
        lines.append(_build_qa_lines(child, depth + 1))
    return "\n".join(lines)


def _rollup_text(node: ThoughtNode) -> str:
    """
    同步计算节点的 rollup 文本（不写入 node.summary，由调用方决定是否缓存）。
    - 叶子 question 节点：「问 | 答摘要 | 分 | 反馈」
    - 非叶子节点：合并子节点 summary
    """
    if node.node_type == "question" and not node.children:
        score_str = f"{node.score}/5" if node.score is not None else "未评分"
        ans = node.answer[:100] if node.answer else "（无回答）"
        return f"问：{node.text[:60]} | 答：{ans} | {score_str} | {node.feedback[:60]}"

    child_summaries = []
    for c in node.children:
        s = c.summary or _rollup_text(c)
        child_summaries.append(s[:80])
    joined = "；".join(child_summaries)

    if node.node_type == "task":
        return f"[{node.text[:40]}] {joined}"
    return f"问：{node.text[:40]} → 子问：{joined}"


async def rollup_node(node: ThoughtNode) -> str:
    """
    对一个节点做 LLM 摘要，存入 node.summary 并返回。
    如节点只有一个直接叶子，直接用规则摘要跳过 LLM 调用。
    """
    all_nodes = _flat([node])
    q_nodes   = [n for n in all_nodes if n.node_type == "question"]

    # 简单情况：直接规则摘要，不用 LLM
    if len(q_nodes) <= 1:
        node.summary = _rollup_text(node)
        return node.summary

    task_text = node.text if node.node_type == "task" else "（子问）"
    qa_lines  = "\n".join(_build_qa_lines(c) for c in node.children)

    try:
        text = await _llm(
            [{"role": "user", "content": _ROLLUP_USR.format(
                task_text=task_text, qa_lines=qa_lines[:1200]
            )}],
            system=_ROLLUP_SYS,
            timeout=20.0,
        )
        node.summary = text.strip()[:120]
    except Exception:
        node.summary = _rollup_text(node)[:120]

    print(f"[rollup] node='{node.text[:40]}' summary='{node.summary}'")
    return node.summary


def build_prior_context(session: InterviewSession) -> str:
    """
    收集当前 session 中已完成任务的 summary，
    返回注入面试官 prompt 的字符串（固定大小）。
    """
    done_tasks = [n for n in session.roots if n.status == "done" and n.summary]
    if not done_tasks:
        return ""
    lines = [f"• {t.summary}" for t in done_tasks[-3:]]  # 最近 3 个任务
    return "【已考察】\n" + "\n".join(lines)


def _is_under_task(roots: list[ThoughtNode], node: ThoughtNode, task_id: str | None) -> bool:
    if task_id is None:
        return False
    for n in _flat(roots):
        if n.id == task_id:
            return node in _flat([n])
    return False


# ── DirectorAgent（决策） ──────────────────────────────────────────────────────

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
  "reasoning": "候选人完全不知道内存隔离机制，这是进程的核心概念，需要从底层原理、机制实现、实际影响三个角度补全",
  "sub_questions": [
    "进程间为什么要隔离内存，不隔离会有什么问题",
    "操作系统通过什么机制实现进程间内存隔离",
    "虚拟地址空间的作用是什么"
  ]
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


async def director_decide(
    session: InterviewSession,
    question_node: ThoughtNode,
    score_result: dict,
) -> dict:
    """DIRECTING 阶段：导演根据评分决定下一步策略，可规划多个同级问题。"""
    assert session.sm.state == "DIRECTING"
    task_node = _find(session.roots, session.sm.current_task_id)
    task_text = task_node.text if task_node else "（未知）"
    question_count = sum(
        1 for n in _flat(session.roots)
        if n.node_type == "question" and _is_under_task(session.roots, n, session.sm.current_task_id)
    )
    # 当前节点父节点下已有 planned 兄弟节点
    parent = _find(session.roots, question_node.parent_id)
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


def _next_planned_sibling(session: InterviewSession, node: ThoughtNode) -> ThoughtNode | None:
    """返回 node 的父节点下第一个 planned 兄弟节点（按添加顺序）。"""
    parent = _find(session.roots, node.parent_id)
    if parent is None:
        return None
    for child in parent.children:
        if child.status == "planned" and child.id != node.id:
            return child
    return None


def _add_planned_nodes(
    _session_unused: Any,
    parent_node: ThoughtNode,
    focuses: list[str],
) -> list[ThoughtNode]:
    """在 parent_node 下批量添加 planned 子节点（只存意图，问题文本待面试官生成）。"""
    nodes = []
    for focus in focuses:
        n = ThoughtNode(
            id=str(uuid.uuid4()),
            node_type="question",
            text="",                      # 面试官生成后填入
            depth=parent_node.depth + 1,
            status="planned",
            parent_id=parent_node.id,
            question_intent=focus,
        )
        parent_node.children.append(n)
        nodes.append(n)
    return nodes


# ── InterviewerAgent ──────────────────────────────────────────────────────────

_INTERVIEWER_SYS = """\
你是一位专业技术面试官，正在考察候选人。

当前考察任务：{task_text}
出题场景：{question_context}

{profile_section}
规则：
- 问题要有针对性，不重复之前问过的问题
- 语气自然专业，问题长度 1-2 句话
- 如需了解具体知识点细节（如某算法原理、某机制实现），可先搜索知识库

{react_header}

Final Answer 必须是 JSON（不加代码块）：
{{"intent": "考察候选人是否理解进程隔离的底层机制", "question": "进程之间为什么需要隔离内存？操作系统是怎么实现的？"}}"""

_INTERVIEWER_USR_INIT = """\
{prior_context}这是对话的开始，请提出关于「{task_text}」的第一个问题。"""

_INTERVIEWER_USR_FOLLOWUP = """\
上一个问题：{prev_question}
候选人的回答：{prev_answer}
导演指示：{director_focus}"""

_REFLECT_SYS = """\
你是一位严格的面试质量审核员。检查下面这道面试题是否合格：
1. 是否真正考察了出题意图中的知识点？
2. 候选人能否明确理解这道题在问什么？
3. 是否与出题意图匹配，不过宽也不过窄？

如果合格，原样返回这道题。
如果不合格，输出改进后的问题。
只输出最终问题文本，不加任何解释。"""

_REFLECT_USR = """\
出题意图：{intent}
生成的问题：{question}"""


async def interviewer_ask(
    session: InterviewSession,
    parent_node: ThoughtNode,
    director_focus: str = "",
    is_first: bool = False,
) -> tuple[str, str]:
    """
    ASKING 阶段：ReAct 生成面试问题 + self-reflection。
    返回 (question_text, intent)。
    """
    assert session.sm.state == "ASKING"

    task_node = _find(session.roots, session.sm.current_task_id)
    task_text = task_node.text if task_node else "（未知）"
    task_type = task_node.task_type if task_node else "knowledge"

    profile_section = f"候选人背景：\n{session.profile_text[:800]}\n" if session.profile_text else ""

    # 面试官只用 search_knowledge 工具
    tools      = build_toolset(session, ["search_knowledge"])
    tool_fns   = {n: t["fn"] for n, t in tools.items()}
    tool_descs = "\n".join(t["desc"] for t in tools.values())
    react_header = _REACT_HEADER.format(max_steps=2, tool_descs=tool_descs)

    if is_first:
        prior = build_prior_context(session)
        prior_section = prior + "\n\n" if prior else ""
        question_context = f"首问，话题开场（task_type={task_type}）"
        user_msg = _INTERVIEWER_USR_INIT.format(
            task_text=task_text, prior_context=prior_section
        )
    else:
        question_context = f"追问/转向（导演指示：{director_focus or '无'}）"
        user_msg = _INTERVIEWER_USR_FOLLOWUP.format(
            prev_question=parent_node.text,
            prev_answer=parent_node.answer[:300],
            director_focus=director_focus or "根据上下文自行判断",
        )

    system = _INTERVIEWER_SYS.format(
        task_text=task_text,
        question_context=question_context,
        profile_section=profile_section,
        react_header=react_header,
    )
    raw = await _react_loop(system, user_msg, tool_fns, max_steps=2, timeout_per_step=40.0)
    parsed = _parse_json(raw, default={"intent": "", "question": raw.strip()})
    intent   = parsed.get("intent", "")
    draft_q  = parsed.get("question", raw.strip()).strip()

    # Self-reflection：审核问题质量
    if intent and draft_q:
        try:
            final_q = await _llm(
                [{"role": "user", "content": _REFLECT_USR.format(intent=intent, question=draft_q)}],
                system=_REFLECT_SYS,
                timeout=20.0,
            )
            final_q = final_q.strip()
        except Exception:
            final_q = draft_q
    else:
        final_q = draft_q

    print(f"[interviewer] intent='{intent[:50]}' draft='{draft_q[:50]}' final='{final_q[:50]}'")
    return final_q, intent


# ── 核心编排 ──────────────────────────────────────────────────────────────────

def _add_question_node(
    session: InterviewSession,
    parent_node: ThoughtNode,
    question_text: str,
    intent: str = "",
) -> ThoughtNode:
    """在思维树上追加一个问题节点，更新 SM。"""
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

    # ASKING: Interviewer 生成第一个问题（含 self-reflection）
    sm.transition("ASKING")
    q_text, intent = await interviewer_ask(session, parent_node=first_task, is_first=True)
    qnode = _add_question_node(session, first_task, q_text, intent=intent)
    qnode.status = "answering"

    # ANSWERING: 等待用户
    sm.transition("ANSWERING")

    session.history.append({"role": "assistant", "content": q_text})
    print(f"[interview/start] tasks={len(session.roots)} first_q={q_text[:60]}")
    return q_text


async def run_turn(session: InterviewSession, user_answer: str) -> dict:
    """
    ANSWERING → SCORING → DIRECTING → ASKING → ANSWERING。
    返回 {response, sm, tree}。
    """
    sm = session.sm
    assert sm.state == "ANSWERING", f"期望 ANSWERING，当前 {sm.state}"

    qnode = _find(session.roots, sm.current_node_id)
    if qnode is None:
        raise RuntimeError("current_node_id 指向不存在的节点")

    # 记录用户回答
    qnode.answer = user_answer
    qnode.status = "answered"
    session.history.append({"role": "user", "content": user_answer})

    # ── SCORING：评分员 CoT 评分 ──────────────────────────────────────────────
    sm.transition("SCORING")
    score_result = await scorer_evaluate(session, qnode, user_answer)
    qnode.score     = score_result["score"]
    qnode.reasoning = score_result["reasoning"]
    qnode.feedback  = score_result["feedback"]
    qnode.status    = "scored"
    sm.last_score   = score_result
    print(f"[scorer] q='{qnode.text[:40]}' score={score_result['score']}")

    # ── DIRECTING ────────────────────────────────────────────────────────────
    sm.transition("DIRECTING")
    task_node = _find(session.roots, sm.current_task_id)

    # ── 优先执行已规划的 planned 兄弟节点（导演上轮的承诺必须兑现）─────────────
    planned_next = _next_planned_sibling(session, qnode)
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

    # ── 无 planned 节点，才交给导演决策 ──────────────────────────────────────
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
        # 确定新问题挂载的父节点
        if decision == "deepen":
            plan_parent = qnode
        elif decision == "pivot":
            plan_parent = task_node or qnode
        elif decision == "back_up":
            gp = _find(session.roots, qnode.parent_id)
            plan_parent = (_find(session.roots, gp.parent_id) if gp and gp.parent_id else gp) or task_node or qnode
            if gp:
                await rollup_node(gp)
        else:
            plan_parent = qnode

        # 批量注册剩余子问题为 planned，立刻问第一个
        if len(sub_questions) > 1:
            _add_planned_nodes(session, plan_parent, sub_questions[1:])
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
