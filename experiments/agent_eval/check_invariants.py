"""
Behavioral invariant checker for decide_node decisions.

Parses session JSON logs and verifies the 4 rules defined in the decide_node system prompt:
  Rule 1: score <= 2 AND depth < 3  → decision MUST be 'deepen'
  Rule 2: score >= 4                → decision MUST be 'pass' or 'pivot'
  Rule 3: depth >= 3                → decision MUST be 'back_up' or 'pass'
  Rule 4: question_count >= 4       → decision MUST be 'pass'

Usage:
    # Single file
    python3 check_invariants.py --log ../../backend/sessions/20260422_125717_ea21562b_lg.json

    # All sessions
    python3 check_invariants.py --log-dir ../../backend/sessions/
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DecisionRecord:
    session_id: str
    task_text: str
    question_text: str
    score: int | None
    depth: int
    question_count_in_task: int
    decision: str  # verdict field


@dataclass
class RuleResult:
    rule_id: int
    description: str
    total_applicable: int
    violations: int

    @property
    def violation_rate(self) -> float:
        return self.violations / self.total_applicable if self.total_applicable > 0 else 0.0


# ── Parsing ───────────────────────────────────────────────────────────────────

def _iter_nodes(node: dict):
    yield node
    for child in node.get("children", []):
        yield from _iter_nodes(child)


def parse_session(path: Path) -> list[DecisionRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    session_id = data.get("session_id", path.stem)
    tree = data.get("tree", [])
    records = []

    for task_node in tree:
        if task_node.get("node_type") != "task":
            continue
        # Count active questions in this task (not planned/skipped)
        active_questions = [
            c for c in task_node.get("children", [])
            if c.get("node_type") == "question"
            and c.get("status") not in ("planned", "skipped")
        ]
        q_count = len(active_questions)

        for i, q_node in enumerate(active_questions):
            verdict = q_node.get("verdict")
            if not verdict:
                continue  # skip questions without a verdict (last unanswered)
            # question_count at the time of this decision = questions so far (1-indexed)
            records.append(DecisionRecord(
                session_id=session_id,
                task_text=task_node.get("text", "")[:60],
                question_text=q_node.get("text", "")[:60],
                score=q_node.get("score"),
                depth=q_node.get("depth", 0),
                question_count_in_task=i + 1,
                decision=verdict,
            ))

    return records


# ── Rules ─────────────────────────────────────────────────────────────────────

RULES = [
    {
        "id": 1,
        "description": "score<=2 AND depth<3 → MUST be 'deepen'",
        "applies": lambda r: r.score is not None and r.score <= 2 and r.depth < 3,
        "valid": lambda r: r.decision == "deepen",
    },
    {
        "id": 2,
        "description": "score>=4 → MUST be 'pass' or 'pivot'",
        "applies": lambda r: r.score is not None and r.score >= 4,
        "valid": lambda r: r.decision in ("pass", "pivot"),
    },
    {
        "id": 3,
        "description": "depth>=3 → MUST be 'back_up' or 'pass'",
        "applies": lambda r: r.depth >= 3,
        "valid": lambda r: r.decision in ("back_up", "pass"),
    },
    {
        "id": 4,
        "description": "question_count>=4 → MUST be 'pass'",
        "applies": lambda r: r.question_count_in_task >= 4,
        "valid": lambda r: r.decision == "pass",
    },
]


def check_rules(records: list[DecisionRecord]) -> tuple[list[RuleResult], list[dict]]:
    results = []
    violation_details = []

    for rule in RULES:
        applicable = [r for r in records if rule["applies"](r)]
        violations = [r for r in applicable if not rule["valid"](r)]
        results.append(RuleResult(
            rule_id=rule["id"],
            description=rule["description"],
            total_applicable=len(applicable),
            violations=len(violations),
        ))
        for v in violations:
            violation_details.append({
                "rule_id": rule["id"],
                "session_id": v.session_id,
                "task": v.task_text,
                "question": v.question_text,
                "score": v.score,
                "depth": v.depth,
                "q_count": v.question_count_in_task,
                "actual_decision": v.decision,
            })

    return results, violation_details


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(rule_results: list[RuleResult], violation_details: list[dict],
                 session_count: int, decision_count: int):
    print(f"\n{'='*60}")
    print(f"Invariant Check Report")
    print(f"  Sessions analyzed: {session_count}")
    print(f"  Total decisions:   {decision_count}")
    print(f"{'='*60}")

    all_pass = True
    for r in rule_results:
        status = "✓" if r.violations == 0 else "✗"
        rate_str = f"{r.violation_rate:.1%}" if r.total_applicable > 0 else "N/A"
        print(f"\n  Rule {r.rule_id}: {r.description}")
        print(f"    Applicable: {r.total_applicable}  Violations: {r.violations}  Rate: {rate_str}  {status}")
        if r.violations > 0:
            all_pass = False

    if all_pass:
        print(f"\n✓ All invariants passed!")
    else:
        print(f"\n✗ Violations detected:")
        for v in violation_details[:20]:  # cap at 20
            print(f"  [Rule {v['rule_id']}] session={v['session_id'][:8]}  "
                  f"score={v['score']}  depth={v['depth']}  q_count={v['q_count']}  "
                  f"decision={v['actual_decision']}")
            print(f"    task: {v['task']}")
        if len(violation_details) > 20:
            print(f"  ... and {len(violation_details) - 20} more")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--log", help="Path to a single session JSON file")
    group.add_argument("--log-dir", help="Directory of session JSON files")
    args = parser.parse_args()

    if args.log:
        paths = [Path(args.log)]
    else:
        paths = sorted(Path(args.log_dir).glob("*.json"))

    all_records: list[DecisionRecord] = []
    for path in paths:
        try:
            records = parse_session(path)
            all_records.extend(records)
        except Exception as e:
            print(f"[warn] failed to parse {path.name}: {e}")

    if not all_records:
        print("No decision records found. Check that sessions have verdict fields.")
        return

    rule_results, violation_details = check_rules(all_records)
    print_report(rule_results, violation_details, len(paths), len(all_records))


if __name__ == "__main__":
    main()
