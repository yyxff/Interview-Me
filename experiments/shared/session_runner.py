"""
Headless interview session runner — drives the full agent loop without FastAPI.

Usage:
    from shared.session_runner import run_session, SessionResult

    def my_answer_fn(question: str) -> str:
        return "我觉得 Redis 的 RDB 和 AOF..."

    result = asyncio.run(run_session(
        jd="后端工程师，熟悉 Redis",
        direction="Redis 持久化",
        answer_fn=my_answer_fn,
    ))
    print(result.to_dict())
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from langgraph.types import Command

from agents.graph import interview_graph
from agents.state import InterviewState, _dict_to_node


@dataclass
class TurnRecord:
    question: str
    answer: str
    score: int | None
    verdict: str | None
    reasoning: str = ""          # score_node's step-by-step reasoning
    feedback: str = ""           # score_node's improvement feedback
    director_reasoning: str = "" # decide_node's reasoning for this verdict
    roots_snapshot: list = field(default_factory=list)  # full tree after this turn


@dataclass
class SessionResult:
    session_id: str
    jd: str
    direction: str
    turns: list[TurnRecord] = field(default_factory=list)
    roots_data: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    interrupted: bool = False

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "jd": self.jd,
            "direction": self.direction,
            "duration_seconds": round(self.duration_seconds, 1),
            "interrupted": self.interrupted,
            "turns": [
                {
                    "question": t.question,
                    "answer": t.answer,
                    "score": t.score,
                    "verdict": t.verdict,
                    "reasoning": t.reasoning,
                    "feedback": t.feedback,
                    "director_reasoning": t.director_reasoning,
                }
                for t in self.turns
            ],
            "roots_data": self.roots_data,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def _find_scored_node(roots_data: list[dict], question_text: str) -> dict | None:
    """BFS through roots_data to find the question node matching this text."""
    queue = list(roots_data)
    while queue:
        node = queue.pop(0)
        if node.get("node_type") == "question" and node.get("text") == question_text:
            return node
        queue.extend(node.get("children", []))
    return None


async def _run_until_interrupt(input_, config: dict) -> tuple[str | None, dict]:
    """Drive graph until interrupt() or end. Returns (question, last_state_patch)."""
    question: str | None = None
    last_state: dict = {}

    async for chunk in interview_graph.astream(input_, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupts = chunk["__interrupt__"]
            question = interrupts[0].value if interrupts else None
        else:
            for node_output in chunk.values():
                if isinstance(node_output, dict):
                    last_state.update(node_output)

    return question, last_state


async def _get_current_state(config: dict) -> dict:
    snapshot = await interview_graph.aget_state(config)
    return dict(snapshot.values) if snapshot else {}


async def run_session(
    jd: str,
    direction: str,
    answer_fn: Callable[[str], str],
    profile_text: str = "",
    max_turns: int = 30,
    max_tasks: int | None = None,
    max_turns_per_task: int | None = None,
    on_turn: Callable[[int, TurnRecord], None] | None = None,
) -> SessionResult:
    """
    Run a complete mock interview session headlessly.

    Args:
        jd:                  Job description text.
        direction:           Interview focus / topic direction.
        answer_fn:           Callback: given a question string, return the candidate answer.
        profile_text:        Optional candidate profile/resume text.
        max_turns:           Hard limit on total question rounds.
        max_tasks:           Eval-only: truncate plan to first N tasks after planning.
        max_turns_per_task:  Eval-only: force task switch after N questions per task.
        on_turn:             Called after each scored turn.
    """
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    initial_state: InterviewState = {
        "session_id": session_id,
        "jd": jd,
        "direction": direction,
        "profile_text": profile_text,
        "roots_data": [],
        "current_task_id": None,
        "current_question_id": None,
        "last_score": None,
        "last_verdict": None,
        "last_sub_questions": [],
        "last_director_reasoning": "",
        "messages": [],
    }

    result = SessionResult(session_id=session_id, jd=jd, direction=direction)
    t0 = time.time()

    try:
        # First run: plan → ask → interrupt
        question, _ = await _run_until_interrupt(initial_state, config)

        # Eval: truncate tasks to max_tasks after planning
        if max_tasks is not None:
            state = await _get_current_state(config)
            roots = state.get("roots_data", [])
            if len(roots) > max_tasks:
                # Mark excess tasks as skipped so decide_node won't pick them up
                for r in roots[max_tasks:]:
                    r["status"] = "skipped"
                await interview_graph.aupdate_state(config, {"roots_data": roots})

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[interrupted during planning]")
        result.interrupted = True
        result.duration_seconds = time.time() - t0
        return result

    turn_count = 0
    try:
        while question is not None and turn_count < max_turns:
            turn_count += 1
            answer = answer_fn(question)

            # Resume graph with candidate's answer
            resume_input = Command(resume=answer)
            question_next, _ = await _run_until_interrupt(resume_input, config)

            # Read state after this turn to get score + verdict + full tree
            current_state = await _get_current_state(config)
            roots_data = current_state.get("roots_data", [])
            scored_node = _find_scored_node(roots_data, question)
            turn = TurnRecord(
                question=question,
                answer=answer,
                score=current_state.get("last_score"),
                verdict=current_state.get("last_verdict"),
                reasoning=scored_node.get("reasoning", "") if scored_node else "",
                feedback=scored_node.get("feedback", "") if scored_node else "",
                director_reasoning=current_state.get("last_director_reasoning", ""),
                roots_snapshot=roots_data,
            )
            result.turns.append(turn)
            result.roots_data = roots_data  # keep result current for Ctrl+C safety
            if on_turn:
                on_turn(turn_count, turn)

            # Eval: force task switch if this task has hit max_turns_per_task
            if max_turns_per_task is not None and question_next is not None:
                current_task_id = current_state.get("current_task_id")
                task_node = next((r for r in roots_data if r.get("id") == current_task_id), None)
                task_turns = len(task_node.get("children", [])) if task_node else 0
                if task_turns >= max_turns_per_task:
                    await interview_graph.aupdate_state(config, {"last_verdict": "pass"})

            question = question_next

    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"\n[interrupted after {turn_count} turns]")
        result.interrupted = True

    final_state = await _get_current_state(config)
    result.roots_data = final_state.get("roots_data", [])
    result.duration_seconds = time.time() - t0

    return result
