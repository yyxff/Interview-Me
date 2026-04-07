"""ScorerAgent：CoT 评分 + Self-Reflection + Rollup 摘要 + 上下文构建

评分流程（两步）：
  Step 1 CoT       — 强制分步推理：列知识点 → 逐点核查 → 初步评分
  Step 2 Reflect   — 审查初步结果：防止关键词陷阱 / 表述惩罚，必要时修正分数

单步模式（use_reflection=False）退化为原来的一次调用，用于低延迟场景。
"""
from __future__ import annotations

import asyncio
import json
import re
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


def _parse_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON，兼容带代码块的情况。"""
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)
    # 取最后一个 {...} 块（CoT 输出的分析文字在前，JSON 在后）
    matches = re.findall(r'\{[^{}]*\}', text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[-1])
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}


# ── Step 1：CoT 评分 ──────────────────────────────────────────────────────────

_COT_SYS = """\
你是面试评分员，使用逐步分析法打分，确保评分有据可查。

请按以下结构输出（先写分析，再写 JSON）：

【知识点核查】
列出该问题期望考察的 2-4 个核心知识点，\
逐点判断候选人是否提到、是否理解原理（不只是背关键词）。

【遗漏与亮点】
- 遗漏：候选人没提到的重要内容
- 亮点：候选人额外提到的有价值内容（可以没有）

【初步评分】
综合以上，给出 1-5 分及一句理由。

最后输出 JSON（不加代码块）：
{"score": 3, "reasoning": "一句话说明评分依据", "feedback": "对候选人的具体建议"}

评分标准：
5分：准确完整，能说明原理并举例
4分：覆盖核心要点，基本无遗漏
3分：方向正确但有明显遗漏，或表述模糊
2分：只了解表面概念，细节错误或不知原理
1分：答非所问，或完全不了解"""

_COT_USR = """\
当前考察任务：{task_text}
面试官的问题：{question}
候选人的回答：{answer}"""


# ── Step 2：Self-Reflection ───────────────────────────────────────────────────

_REFLECT_SYS = """\
你是面试评分审核员，负责复查评分是否公正准确。

重点检查两类常见偏差：
1. 「关键词陷阱」：候选人说出了术语，但解释不出原理 → 不应给 4-5 分
2. 「表述惩罚」：候选人理解了原理，但表达不够规范或书面化 → 不应因此扣分

审核规则：
- 若评分合理，原样输出 JSON，reasoning 末尾加「（审核通过）」
- 若需修正，输出修正后的 JSON，reasoning 末尾加「（修正：原分X→Y，原因：...）」
- 分数修正幅度通常不超过 1 分；超过 1 分时必须在 reasoning 中详细说明

只输出 JSON，不加代码块、不加任何额外文字。"""

_REFLECT_USR = """\
【CoT 分析过程】
{cot_output}

【初步评分 JSON】
{initial_json}

请审核上述评分是否合理，输出最终 JSON。"""


# ── 主入口 ────────────────────────────────────────────────────────────────────

async def scorer_evaluate(
    session: InterviewSession,
    question_node: ThoughtNode,
    answer: str,
    use_reflection: bool = True,
) -> dict:
    """
    SCORING 阶段：CoT 评分 + （可选）Self-Reflection。

    use_reflection=True （默认）：两步调用，质量更高
    use_reflection=False         ：单步调用，延迟减半，适合低延迟场景
    """
    assert session.sm.state == "SCORING"
    task_node = find(session.roots, session.sm.current_task_id)
    task_text = task_node.text if task_node else "（未知）"

    # ── Step 1：CoT 评分 ──────────────────────────────────────────────────────
    cot_output = await _llm(
        [{"role": "user", "content": _COT_USR.format(
            task_text = task_text,
            question  = question_node.text,
            answer    = answer[:800],
        )}],
        system  = _COT_SYS,
        timeout = 50.0,
    )

    initial = _parse_json(cot_output)
    score   = max(1, min(5, int(initial.get("score", 3))))
    result  = {
        "score":     score,
        "reasoning": initial.get("reasoning", ""),
        "feedback":  initial.get("feedback", ""),
    }

    if not use_reflection:
        return result

    # ── Step 2：Self-Reflection ───────────────────────────────────────────────
    try:
        initial_json_str = json.dumps(
            {"score": score, "reasoning": result["reasoning"], "feedback": result["feedback"]},
            ensure_ascii=False,
        )
        reflect_output = await _llm(
            [{"role": "user", "content": _REFLECT_USR.format(
                cot_output   = cot_output[:1200],
                initial_json = initial_json_str,
            )}],
            system  = _REFLECT_SYS,
            timeout = 30.0,
        )
        reflected = _parse_json(reflect_output)
        if reflected and "score" in reflected:
            final_score = max(1, min(5, int(reflected["score"])))
            result = {
                "score":     final_score,
                "reasoning": reflected.get("reasoning", result["reasoning"]),
                "feedback":  reflected.get("feedback",  result["feedback"]),
            }
    except Exception as e:
        print(f"[scorer] reflection 失败，使用初步评分: {e}")

    return result


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
            system  = _ROLLUP_SYS,
            timeout = 20.0,
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
