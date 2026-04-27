"""
LLM-as-Judge: evaluate a complete interview session on 4 dimensions.

Dimensions (1-5 each):
  coverage       — were the key technical topics actually tested?
  adaptiveness   — did the interviewer adjust strategy based on performance?
  question_quality — were questions clear, non-repetitive, well-paced?
  scoring_fairness — did scores reflect answer quality consistently?

Usage:
    # Judge a single session
    python3 judge_session.py --session results/expert_20260427.json

    # With a checklist of topics that should be covered
    python3 judge_session.py --session results/expert_20260427.json --checklist checklist.yaml

    # Compare two sessions (e.g., different agent versions on same persona)
    python3 judge_session.py --compare results/expert_v1.json results/expert_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from gemini_client import pro  # noqa: E402

JUDGE_SYSTEM = """\
你是一个专业的技术面试质量审查员。你的任务是评估一场技术面试的质量。

请客观评分，给出有依据的判断。每个维度 1-5 分：
5分：非常优秀
4分：良好，有小问题
3分：一般，有明显问题
2分：较差，存在严重问题
1分：很差

输出 JSON（不加代码块），格式如下：
{
  "coverage": 4,
  "adaptiveness": 3,
  "question_quality": 4,
  "scoring_fairness": 3,
  "overall": 4,
  "main_issue": "最主要的一个问题（一句话）",
  "coverage_mode": "checklist",
  "reasoning": {
    "coverage": "理由",
    "adaptiveness": "理由",
    "question_quality": "理由",
    "scoring_fairness": "理由"
  }
}"""

JUDGE_PROMPT_TEMPLATE = """\
以下是一场技术面试的完整记录。

【面试背景】
JD：{jd}
方向：{direction}

{checklist_section}

【面试记录】
{transcript}

请对这场面试按 4 个维度评分：
1. coverage（覆盖度）：{coverage_instruction}
2. adaptiveness（适应性）：面试官是否根据候选人分数/表现调整了策略？低分时是否 deepen，高分时是否推进？
3. question_quality（问题质量）：问题是否清晰无歧义？有无重复提问？难度是否合理递进？
4. scoring_fairness（评分公平性）：分数是否与答案质量一致？有无明显过高或过低？"""


# ── Transcript formatting ─────────────────────────────────────────────────────

def format_transcript(session: dict) -> str:
    turns = session.get("turns", [])
    lines = []
    for i, t in enumerate(turns):
        lines.append(f"[第{i+1}轮]")
        lines.append(f"面试官：{t.get('question', '')}")
        lines.append(f"候选人：{t.get('answer', '')}")
        score = t.get("score")
        verdict = t.get("verdict")
        if score is not None:
            lines.append(f"评分：{score}/5  决策：{verdict}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(session: dict, checklist: list[str] | None) -> str:
    if checklist:
        checklist_section = "【应覆盖的知识点清单】\n" + "\n".join(f"- {item}" for item in checklist)
        coverage_instruction = "以下知识点清单是否被充分考察到？"
        coverage_mode = "checklist"
    else:
        checklist_section = ""
        coverage_instruction = "面试是否覆盖了多个不同技术话题，没有在单一知识点上过度停留？"
        coverage_mode = "diversity"

    transcript = format_transcript(session)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        jd=session.get("jd", "")[:200],
        direction=session.get("direction", ""),
        checklist_section=checklist_section,
        transcript=transcript,
        coverage_instruction=coverage_instruction,
    )
    return prompt, coverage_mode


# ── Judging ───────────────────────────────────────────────────────────────────

def judge(session: dict, checklist: list[str] | None = None) -> dict:
    prompt, coverage_mode = build_prompt(session, checklist)
    raw = pro(prompt, system=JUDGE_SYSTEM, json_mode=True)
    result = json.loads(raw)
    result["coverage_mode"] = coverage_mode
    return result


# ── Compare ───────────────────────────────────────────────────────────────────

def compare(scores_a: dict, scores_b: dict) -> dict:
    dims = ["coverage", "adaptiveness", "question_quality", "scoring_fairness", "overall"]
    delta = {}
    for d in dims:
        a = scores_a.get(d)
        b = scores_b.get(d)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta[d] = round(b - a, 2)
    return delta


# ── Save ──────────────────────────────────────────────────────────────────────

def save_result(result: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def print_scores(scores: dict, title: str = ""):
    if title:
        print(f"\n── {title} ──")
    dims = ["coverage", "adaptiveness", "question_quality", "scoring_fairness", "overall"]
    for d in dims:
        v = scores.get(d, "?")
        bar = "█" * int(v) if isinstance(v, int) else ""
        print(f"  {d:<20} {v}/5  {bar}")
    if scores.get("main_issue"):
        print(f"  main_issue: {scores['main_issue']}")
    if "reasoning" in scores:
        print("\n  Reasoning:")
        for d, r in scores["reasoning"].items():
            print(f"    {d}: {r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Path to session JSON")
    group.add_argument("--compare", nargs=2, metavar=("SESSION_A", "SESSION_B"),
                       help="Compare two sessions (B - A delta)")
    parser.add_argument("--checklist", help="Path to YAML file with list of topics to cover")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    checklist = None
    if args.checklist:
        with open(args.checklist, encoding="utf-8") as f:
            checklist = yaml.safe_load(f)

    if args.compare:
        path_a, path_b = Path(args.compare[0]), Path(args.compare[1])
        session_a = json.loads(path_a.read_text(encoding="utf-8"))
        session_b = json.loads(path_b.read_text(encoding="utf-8"))

        print("Judging session A...")
        scores_a = judge(session_a, checklist)
        print("Judging session B...")
        scores_b = judge(session_b, checklist)

        delta = compare(scores_a, scores_b)
        print_scores(scores_a, title=f"Session A: {path_a.parent.name}")
        print_scores(scores_b, title=f"Session B: {path_b.parent.name}")

        print("\n── Delta (B - A) ──")
        for d, v in delta.items():
            sign = "+" if v > 0 else ""
            symbol = "↑" if v > 0 else "↓" if v < 0 else "="
            print(f"  {d:<20} {sign}{v} {symbol}")

        ts = time.strftime("%Y%m%d_%H%M%S")
        compare_dir = Path(args.out_dir) / f"compare_{ts}"
        output = {
            "timestamp": ts,
            "session_a": str(path_a),
            "session_b": str(path_b),
            "scores_a": scores_a,
            "scores_b": scores_b,
            "delta": delta,
            "judge_model": "gemini-2.5-pro",
        }
        path = save_result(output, compare_dir / "compare.json")

    else:
        session_path = Path(args.session)
        if session_path.is_dir():
            session_path = session_path / "session.json"
        session = json.loads(session_path.read_text(encoding="utf-8"))
        persona = session.get("persona", session_path.stem)

        print(f"Judging session: {session_path.parent.name} ({len(session.get('turns', []))} turns)")
        scores = judge(session, checklist)
        print_scores(scores, title="Judge Results")

        output = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "persona": persona,
            "session_path": str(session_path),
            "scores": scores,
            "judge_model": "gemini-2.5-pro",
        }
        # Save judge.json alongside session.json in the same directory
        path = save_result(output, session_path.parent / "judge.json")

    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
