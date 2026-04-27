"""
Headless score_node interface — callable without FastAPI.

Usage:
    from shared.runner import score, ScoreResult

    result = asyncio.run(score("Redis 的持久化机制有哪些？", "有 RDB 和 AOF 两种..."))
    print(result.score, result.reasoning)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import uuid
from dataclasses import dataclass

from agents.models import ThoughtNode
from agents.state import InterviewState, _node_to_dict
from agents.nodes.score import score_node
import agents.nodes.score as _score_mod


@dataclass
class ScoreResult:
    score: int
    reasoning: str
    feedback: str
    modification_count: int = 0  # how many times Critic triggered a revision


def _make_minimal_state(question: str, answer: str) -> InterviewState:
    """Construct the smallest valid InterviewState for score_node."""
    task_id = str(uuid.uuid4())
    q_id = str(uuid.uuid4())

    task_node = ThoughtNode(
        id=task_id, node_type="task", text="eval-task",
        task_type="knowledge",
    )
    q_node = ThoughtNode(
        id=q_id, node_type="question",
        text=question, answer=answer,
        depth=1, status="answered",
        parent_id=task_id,
    )
    task_node.children = [q_node]

    return {
        "session_id": "eval-runner",
        "jd": "",
        "direction": "",
        "profile_text": "",
        "roots_data": [_node_to_dict(task_node)],
        "current_task_id": task_id,
        "current_question_id": q_id,
        "last_score": None,
        "last_verdict": None,
        "last_sub_questions": [],
        "last_director_reasoning": "",
        "messages": [],
    }


async def score(
    question: str,
    answer: str,
    no_critic: bool = False,
) -> ScoreResult:
    """
    Score a single (question, answer) pair using score_node logic.

    Args:
        question: The interview question text.
        answer:   The candidate's answer text.
        no_critic: If True, skip the Critic-Actor loop and use the initial score only.

    Returns:
        ScoreResult with score (1-5), reasoning, and feedback.
    """
    original_rounds = _score_mod._MAX_CRITIC_ROUNDS
    if no_critic:
        _score_mod._MAX_CRITIC_ROUNDS = 0

    try:
        state = _make_minimal_state(question, answer)
        result_update = await score_node(state)
    finally:
        _score_mod._MAX_CRITIC_ROUNDS = original_rounds

    # Extract from updated roots_data
    roots_data = result_update.get("roots_data", state["roots_data"])
    q_node_dict = next(
        (n for root in roots_data for n in _iter_nodes(root)
         if n["id"] == state["current_question_id"]),
        {}
    )

    return ScoreResult(
        score=result_update.get("last_score", 3),
        reasoning=q_node_dict.get("reasoning", ""),
        feedback=q_node_dict.get("feedback", ""),
    )


def _iter_nodes(node_dict: dict):
    yield node_dict
    for child in node_dict.get("children", []):
        yield from _iter_nodes(child)
