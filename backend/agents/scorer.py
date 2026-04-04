"""ScorerAgent：CoT 评分 + Rollup 摘要 + 上下文构建"""
from __future__ import annotations

import asyncio
from typing import Any

from .models import ThoughtNode, InterviewSession, flat, find

_provider: Any = None


def set_provider(p: Any) -> None:
    global _provider
    _provider = p


async def _llm(messages: list[dict], system: str = "", timeout: float = 50.0) -> str:
    if _provider is None:
        raise RuntimeError("LLM provider 未注入")
    return await asyncio.wait_for(_provider.chat(messages, system), timeout=timeout)


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
  "reasoning": "候选人提到了...",
  "feedback": "掌握了...，但缺少..."
}"""

_SCORER_USR = """\
当前考察任务：{task_text}
面试官的问题：{question}
候选人的回答：{answer}"""


async def scorer_evaluate(session: InterviewSession, question_node: ThoughtNode, answer: str) -> dict:
    """SCORING 阶段：CoT 评分，只评分不做决策。"""
    from .models import find
    import json, re

    assert session.sm.state == "SCORING"
    task_node = find(session.roots, session.sm.current_task_id)
    task_text = task_node.text if task_node else "（未知）"

    text = await _llm(
        [{"role": "user", "content": _SCORER_USR.format(
            task_text=task_text,
            question=question_node.text,
            answer=answer[:800],
        )}],
        system=_SCORER_SYS,
    )
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)
    try:
        result = json.loads(text)
    except Exception:
        result = {"score": 3, "reasoning": "解析失败", "feedback": "解析失败"}

    score = max(1, min(5, int(result.get("score", 3))))
    return {
        "score":     score,
        "reasoning": result.get("reasoning", ""),
        "feedback":  result.get("feedback", ""),
    }


# ── Rollup ────────────────────────────────────────────────────────────────────

_ROLLUP_SYS = """\
你是面试复盘助手。用一句话（≤60字）总结以下面试片段的考察结果。
格式：「{考察角度}：{候选人表现要点，含得分/薄弱点}」
只输出这一句话，不加任何前缀或解释。"""

_ROLLUP_USR = """\
考察任务：{task_text}
问答记录：
{qa_lines}"""


def _build_qa_lines(node: ThoughtNode, depth: int = 0) -> str:
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
    if node.node_type == "question" and not node.children:
        score_str = f"{node.score}/5" if node.score is not None else "未评分"
        ans = node.answer[:100] if node.answer else "（无回答）"
        return f"问：{node.text[:60]} | 答：{ans} | {score_str} | {node.feedback[:60]}"
    child_summaries = [c.summary or _rollup_text(c) for c in node.children]
    joined = "；".join(s[:80] for s in child_summaries)
    if node.node_type == "task":
        return f"[{node.text[:40]}] {joined}"
    return f"问：{node.text[:40]} → 子问：{joined}"


async def rollup_node(node: ThoughtNode) -> str:
    """对一个节点做 LLM 摘要，存入 node.summary 并返回。"""
    all_nodes = flat([node])
    q_nodes   = [n for n in all_nodes if n.node_type == "question"]

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
    """收集当前 session 中已完成任务的 summary，返回注入面试官 prompt 的字符串。"""
    done_tasks = [n for n in session.roots if n.status == "done" and n.summary]
    if not done_tasks:
        return ""
    lines = [f"• {t.summary}" for t in done_tasks[-3:]]
    return "【已考察】\n" + "\n".join(lines)
