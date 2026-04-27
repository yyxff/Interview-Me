"""
Run a full mock interview session with a synthetic Persona as the candidate.

Usage:
    python3 run_persona_session.py --persona expert --jd "后端工程师，熟悉Redis和MySQL" --direction "Redis持久化机制"
    python3 run_persona_session.py --persona novice --out results/novice_session.json
    python3 run_persona_session.py --persona mixed --verify  # also run behavior expectation checks
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from session_runner import run_session  # noqa: E402
from gemini_client import flash  # noqa: E402

PERSONAS_DIR = Path(__file__).parent / "personas"

ANSWER_SYSTEM = """\
你是一个模拟面试候选人。根据给定的能力画像和风格说明，回答面试官的问题。

重要：
- 严格按照能力等级作答，不要超出能力范围
- 口语化表达，像真实候选人在说话
- 不要解释你的回答策略，直接给出答案
- 回答控制在400字以内，简洁到位"""


def load_persona(name: str) -> dict:
    path = PERSONAS_DIR / f"{name}.yaml"
    if not path.exists():
        available = [p.stem for p in PERSONAS_DIR.glob("*.yaml")]
        raise ValueError(f"Persona '{name}' not found. Available: {available}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_answer_fn(persona: dict) -> callable:
    """Return a function that generates candidate answers based on persona."""
    def answer_fn(question: str) -> str:
        skill_summary = "\n".join(
            f"  {domain}: {level}/5分" for domain, level in persona["skills"].items()
        )
        prompt = (
            f"候选人能力画像：\n{persona['description']}\n\n"
            f"各领域技能等级：\n{skill_summary}\n\n"
            f"回答风格：\n{persona['style']}\n\n"
            f"面试官问题：{question}\n\n"
            f"请按照上述能力画像回答这道面试题："
        )
        return flash(prompt, system=ANSWER_SYSTEM)
    return answer_fn


def verify_expectations(persona: dict, result) -> dict:
    """Check if session behavior matches persona expectations."""
    turns = result.turns
    if not turns:
        return {"passed": False, "reason": "no turns"}

    scores = [t.score for t in turns if t.score is not None]
    verdicts = [t.verdict for t in turns if t.verdict is not None]

    avg_score = mean(scores) if scores else 0
    pass_rate = verdicts.count("pass") / len(verdicts) if verdicts else 0
    deepen_rate = verdicts.count("deepen") / len(verdicts) if verdicts else 0

    checks = {}
    name = persona["name"]

    if name == "expert":
        checks["avg_score>=3.5"] = avg_score >= 3.5
        checks["pass_rate>=60%"] = pass_rate >= 0.6
    elif name == "novice":
        checks["avg_score<=3.0"] = avg_score <= 3.0
        checks["deepen_rate>=40%"] = deepen_rate >= 0.4
    elif name == "mixed":
        # Check score gap between strong/weak domains by keyword matching
        strong = persona.get("strong_domains", [])
        weak = persona.get("weak_domains", [])
        # Can't easily filter by domain without roots_data analysis, so just check overall variance
        if len(scores) >= 4:
            high = max(scores)
            low = min(scores)
            checks["score_gap>=1.5"] = (high - low) >= 1.5
        else:
            checks["score_gap>=1.5"] = None  # not enough data

    passed = all(v for v in checks.values() if v is not None)
    return {
        "passed": passed,
        "checks": checks,
        "stats": {
            "avg_score": round(avg_score, 2),
            "pass_rate": round(pass_rate, 2),
            "deepen_rate": round(deepen_rate, 2),
            "total_turns": len(turns),
        },
    }


def _save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def main_async(args):
    persona = load_persona(args.persona)
    jd = args.jd or "后端工程师，需要熟悉缓存、数据库、操作系统和网络基础"
    direction = args.direction or "Redis、MySQL、操作系统、网络"

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        session_dir = Path(__file__).parent / "results" / f"{persona['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(parents=True, exist_ok=True)
        out_path = session_dir / "session.json"

    print(f"Persona : {persona['name']} — {persona['description']}")
    print(f"Direction: {direction}")
    print(f"Output  : {out_path}")
    print("Planning interview...\n")

    # Q prints immediately when asked, A prints right after generation
    base_answer_fn = make_answer_fn(persona)
    turn_counter = [0]

    def answer_fn(question: str) -> str:
        turn_counter[0] += 1
        print(f"\n{'─'*60}")
        print(f"[第{turn_counter[0]}轮] Q: {question}")
        print("  A: 生成中...", flush=True)
        answer = base_answer_fn(question)
        print(f"\r  A: {answer}")
        return answer

    # Incremental save after every scored turn
    saved_turns: list[dict] = []

    current_roots: list[dict] = []

    def on_turn(n: int, turn):
        nonlocal current_roots
        current_roots = turn.roots_snapshot
        print(f"  → score={turn.score}  verdict={turn.verdict}")
        if turn.reasoning:
            print(f"     reasoning: {turn.reasoning[:120]}")
        if turn.feedback:
            print(f"     feedback:  {turn.feedback[:80]}")
        if turn.director_reasoning:
            print(f"     director:  {turn.director_reasoning[:100]}")
        saved_turns.append({
            "question": turn.question, "answer": turn.answer,
            "score": turn.score, "verdict": turn.verdict,
            "reasoning": turn.reasoning, "feedback": turn.feedback,
            "director_reasoning": turn.director_reasoning,
        })
        _save(out_path, {
            "persona": persona["name"], "jd": jd, "direction": direction,
            "interrupted": True, "turns": saved_turns,
            "roots_data": current_roots,
        })

    result = await run_session(
        jd=jd, direction=direction,
        answer_fn=answer_fn,
        on_turn=on_turn,
    )

    status = "interrupted (partial)" if result.interrupted else "complete"
    print(f"\n{'='*60}")
    print(f"Session {status}: {len(result.turns)} turns, {result.duration_seconds:.0f}s")

    output = result.to_dict()
    output["persona"] = persona["name"]

    if args.verify:
        exp = verify_expectations(persona, result)
        output["expectations"] = exp
        print("\nExpectation checks:")
        for check, passed in exp["checks"].items():
            mark = "✓" if passed else "✗" if passed is not None else "?"
            print(f"  {mark} {check}")
        print(f"Stats: {exp['stats']}")

    _save(out_path, output)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True, choices=["expert", "novice", "mixed"])
    parser.add_argument("--jd", default="", help="Job description")
    parser.add_argument("--direction", default="", help="Interview topic direction")
    parser.add_argument("--out", default="", help="Output path (default: results/<persona>_<ts>.json)")
    parser.add_argument("--verify", action="store_true", help="Run behavior expectation checks")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass  # session_runner already handled the interrupt and saved results


if __name__ == "__main__":
    main()
