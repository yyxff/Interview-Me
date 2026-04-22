"""
score_node：Scorer 评分（Critic-Actor Loop）
"""
from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agents.models import ThoughtNode, find

from ..llm import _build_llm
from ..state import InterviewState, _dict_to_node, _node_to_dict, _parse_json
from ..tools import _make_score_tools


_SCORE_SYSTEM = """\
你是面试评分员，客观评分并给出逐点分析，不做策略决策。

你可以先搜索知识库，获取该题目的标准答案作为评分参考。

评分标准（1-5分）：
5分：准确完整，有深度，能说明原理并举例
4分：覆盖核心要点，表述清晰，基本无遗漏
3分：方向正确但有明显遗漏，或表述模糊
2分：只了解表面概念，细节错误或不知道原理
1分：答非所问，或完全不了解

评分前请在 reasoning 中按以下步骤逐步分析：
1. 候选人实际说了哪些要点（只基于原文，不要推断）
2. 对照标准答案，哪些要点覆盖了，哪些遗漏了
3. 有无明显错误或误解
4. 综合以上，得出分数

输出 JSON（不加代码块）：
{"score": 3, "reasoning": "1)候选人提到了... 2)遗漏了... 3)无明显错误 4)综合得分3", "feedback": "掌握了...，但缺少..."}"""

_SCORE_CRITIC_SYSTEM = """\
你是评分审查员。遇到不确定的技术细节，先搜索知识库核实，再下判断。

检查下面这个评分是否准确公平：
1. reasoning 有没有幻觉（捏造候选人没说过的内容）？
2. reasoning 中引用的技术细节是否准确？如不确定，搜索知识库验证
3. score 与 reasoning 是否自洽？（如"方向正确但有遗漏"打了1分，属于矛盾）
4. feedback 是否具体有建设性，而不是泛泛而谈？

输出 JSON（不加代码块）：
{"approved": true, "critique": "（如不通过，说明具体问题；通过则留空）"}"""

_SCORE_REVISE_SYSTEM = """\
你是面试评分员。审查员对你的初步评分提出了意见，请根据意见修正。

评分标准（1-5分）：
5分：准确完整，有深度，能说明原理并举例
4分：覆盖核心要点，表述清晰，基本无遗漏
3分：方向正确但有明显遗漏，或表述模糊
2分：只了解表面概念，细节错误或不知道原理
1分：答非所问，或完全不了解

针对审查员的意见，在 reasoning 中重新逐步分析：
1. 候选人实际说了哪些要点（只基于原文，不要推断）
2. 对照标准答案，哪些要点覆盖了，哪些遗漏了
3. 有无明显错误或误解
4. 综合以上，得出修正后的分数

输出 JSON（不加代码块）：
{"score": 3, "reasoning": "1)候选人提到了... 2)遗漏了... 3)无明显错误 4)综合得分3", "feedback": "..."}"""

_MAX_CRITIC_ROUNDS = 2


# ── 上下文函数 ────────────────────────────────────────────────────────────────

def _get_score_context(state: InterviewState) -> tuple[list, ThoughtNode | None, ThoughtNode | None]:
    """从 state 还原树，定位当前题和任务节点。"""
    roots = [_dict_to_node(d) for d in state["roots_data"]]
    qnode = find(roots, state["current_question_id"])
    task_node = find(roots, state["current_task_id"])
    return roots, qnode, task_node


def _build_score_prompt(task_node: ThoughtNode | None, qnode: ThoughtNode) -> str:
    task_text = task_node.text if task_node else "（未知）"
    return (
        f"当前考察任务：{task_text}\n"
        f"面试官的问题：{qnode.text}\n"
        f"候选人的回答：{qnode.answer[:600]}"
    )


async def _critic(llm, tools: list, question: str, answer: str, current: dict) -> dict:
    """Critic：审查评分逻辑是否自洽、技术细节是否准确，输出 {approved, critique}。"""
    react_agent = create_react_agent(llm, tools, prompt=SystemMessage(content=_SCORE_CRITIC_SYSTEM))
    result = await react_agent.ainvoke({
        "messages": [HumanMessage(content=(
            f"面试问题：{question}\n"
            f"候选人回答：{answer[:600]}\n"
            f"当前评分：{json.dumps(current, ensure_ascii=False)}"
        ))]
    })
    return _parse_json(result["messages"][-1].content, default={"approved": True, "critique": ""})


async def _revise(llm, tools: list, question: str, answer: str, current: dict, critique: str) -> dict:
    """Actor 修正：ReAct agent 看到 critique 后，可再次搜知识库，然后给出修正评分。"""
    react_agent = create_react_agent(llm, tools, prompt=SystemMessage(content=_SCORE_REVISE_SYSTEM))
    result = await react_agent.ainvoke({
        "messages": [HumanMessage(content=(
            f"面试问题：{question}\n"
            f"候选人回答：{answer[:600]}\n"
            f"你的初步评分：{json.dumps(current, ensure_ascii=False)}\n"
            f"审查员意见：{critique}"
        ))]
    })
    return _parse_json(result["messages"][-1].content, default=current)


# ── Node ──────────────────────────────────────────────────────────────────────

async def score_node(state: InterviewState) -> dict:
    """
    ── Node③：Scorer 评分（Critic-Actor Loop） ─────────────────────────

    流程：
      ① Scorer ReAct：搜索知识库 → 初步评分（RAG + CoT）
      ② Critic：审查评分，输出 {approved, critique}
         - approved → 结束
         - not approved → ③
      ③ Scorer 看 critique 修正评分 → 回到 ②
      （最多 _MAX_CRITIC_ROUNDS 轮）
    """
    t0 = time.time()
    roots, qnode, task_node = _get_score_context(state)
    if qnode is None:
        logger.warning("[score] ⚠ qnode is None, skip")
        return {"last_score": 3}

    logger.info("[score] ▶ start  q='%s'  answer='%s'", qnode.text[:50], qnode.answer[:40])

    llm = _build_llm()
    tools = _make_score_tools(state)

    # ① 初步评分
    logger.info("[score] ① calling scorer ReAct ...  (%.1fs)", time.time()-t0)
    react_agent = create_react_agent(llm, tools, prompt=SystemMessage(content=_SCORE_SYSTEM))
    result = await react_agent.ainvoke({
        "messages": [HumanMessage(content=_build_score_prompt(task_node, qnode))]
    })
    current = _parse_json(
        result["messages"][-1].content,
        default={"score": 3, "reasoning": "解析失败", "feedback": ""},
    )
    logger.info(
        "[score] ① done  (%.1fs)  score=%s\n  reasoning= %s\n  feedback = %s",
        time.time()-t0, current.get('score'),
        current.get('reasoning', '')[:200], current.get('feedback', '')[:200],
    )

    # ② Critic-Actor Loop
    for round_i in range(_MAX_CRITIC_ROUNDS):
        logger.info("[score] ② critic round=%d ...  (%.1fs)", round_i+1, time.time()-t0)
        feedback = await _critic(llm, tools, qnode.text, qnode.answer, current)
        approved = feedback.get("approved", True)
        critique = feedback.get("critique", "")
        logger.info(
            "[score] ② critic done  (%.1fs)  approved=%s\n  critique = %s",
            time.time()-t0, approved, critique,
        )
        if approved:
            break
        logger.info("[score] ③ revise ...  (%.1fs)", time.time()-t0)
        current = await _revise(llm, tools, qnode.text, qnode.answer, current, critique)
        logger.info(
            "[score] ③ revise done  (%.1fs)  score=%s\n  reasoning= %s\n  feedback = %s",
            time.time()-t0, current.get('score'),
            current.get('reasoning', '')[:200], current.get('feedback', '')[:200],
        )

    score = max(1, min(5, int(current.get("score", 3))))
    qnode.score = score
    qnode.reasoning = current.get("reasoning", "")
    qnode.feedback = current.get("feedback", "")
    qnode.status = "scored"

    logger.info("[score] ✔ done  score=%d  total=%.1fs", score, time.time()-t0)

    return {
        "roots_data": [_node_to_dict(r) for r in roots],
        "last_score": score,
    }
