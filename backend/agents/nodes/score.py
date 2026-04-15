"""
score_node：Scorer 评分
"""
from __future__ import annotations

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

输出 JSON（不加代码块）：
{"score": 3, "reasoning": "候选人提到了...", "feedback": "掌握了...，但缺少..."}"""


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


# ── Node ──────────────────────────────────────────────────────────────────────

async def score_node(state: InterviewState) -> dict:
    """
    ── Node③：Scorer 评分 ──────────────────────────────────────────────

    使用 create_react_agent：评分前可主动搜索知识库获取标准答案作为参考。
    """
    roots, qnode, task_node = _get_score_context(state)
    if qnode is None:
        return {"last_score": 3}

    llm = _build_llm()
    tools = _make_score_tools(state)
    react_agent = create_react_agent(llm, tools, prompt=SystemMessage(content=_SCORE_SYSTEM))

    result = await react_agent.ainvoke({
        "messages": [HumanMessage(content=_build_score_prompt(task_node, qnode))]
    })

    score_data = _parse_json(
        result["messages"][-1].content,
        default={"score": 3, "reasoning": "解析失败", "feedback": ""},
    )
    score = max(1, min(5, int(score_data.get("score", 3))))

    qnode.score = score
    qnode.reasoning = score_data.get("reasoning", "")
    qnode.feedback = score_data.get("feedback", "")
    qnode.status = "scored"

    print(f"[score] q='{qnode.text[:40]}' score={score}")

    return {
        "roots_data": [_node_to_dict(r) for r in roots],
        "last_score": score,
    }
