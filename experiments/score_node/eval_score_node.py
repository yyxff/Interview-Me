"""
Evaluate score_node accuracy, consistency, and Critic effectiveness.

Usage:
    # Full eval (accuracy + Critic comparison)
    python3 eval_score_node.py --testset testsets/score_node.jsonl

    # Skip Critic loop (measure raw Scorer only)
    python3 eval_score_node.py --testset testsets/score_node.jsonl --no-critic

    # Consistency mode: run each case 3 times, measure variance
    python3 eval_score_node.py --testset testsets/score_node.jsonl --consistency

    # Critic on/off comparison (runs both, prints delta)
    python3 eval_score_node.py --testset testsets/score_node.jsonl --compare-critic
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from runner import score as score_fn  # noqa: E402


# ── Metrics ───────────────────────────────────────────────────────────────────

def weighted_kappa(y_true: list[int], y_pred: list[int], n_classes: int = 5) -> float:
    """Cohen's kappa with linear weights."""
    n = len(y_true)
    if n == 0:
        return 0.0
    labels = list(range(1, n_classes + 1))
    # Observed agreement matrix
    obs = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        obs[t - 1][p - 1] += 1
    # Expected
    row_sums = [sum(obs[i]) for i in range(n_classes)]
    col_sums = [sum(obs[i][j] for i in range(n_classes)) for j in range(n_classes)]
    num = denom = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            w = 1 - abs(labels[i] - labels[j]) / (n_classes - 1)
            expected = row_sums[i] * col_sums[j] / n
            num += w * obs[i][j]
            denom += w * expected
    if denom == 0:
        return 1.0
    # kappa = 1 - (1 - Po) / (1 - Pe) in linear form
    po = num / n
    pe = denom / n
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 1.0


def mae(y_true: list[int], y_pred: list[int]) -> float:
    return mean(abs(t - p) for t, p in zip(y_true, y_pred))


def spearman(y_true: list[int], y_pred: list[int]) -> float:
    n = len(y_true)
    if n < 2:
        return 0.0

    def rank(lst):
        sorted_idx = sorted(range(n), key=lambda i: lst[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and lst[sorted_idx[j]] == lst[sorted_idx[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[sorted_idx[k]] = avg_rank
            i = j + 1
        return r

    rt, rp = rank(y_true), rank(y_pred)
    d2 = sum((a - b) ** 2 for a, b in zip(rt, rp))
    return 1 - 6 * d2 / (n * (n * n - 1))


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    n = len(y_true)
    if n == 0:
        print("  [warn] no successful predictions — check LLM auth and model config")
        return {"n": 0, "kappa": None, "mae": None, "spearman": None,
                "exact_match_rate": None, "soft_match_rate": None}
    return {
        "n": n,
        "kappa": round(weighted_kappa(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "spearman": round(spearman(y_true, y_pred), 4),
        "exact_match_rate": round(sum(t == p for t, p in zip(y_true, y_pred)) / n, 4),
        "soft_match_rate": round(sum(abs(t - p) <= 1 for t, p in zip(y_true, y_pred)) / n, 4),
    }


# ── Loading ───────────────────────────────────────────────────────────────────

def load_testset(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Eval runs ─────────────────────────────────────────────────────────────────

async def run_accuracy(records: list[dict], no_critic: bool = False, no_rag: bool = False) -> tuple[list[int], list[int]]:
    y_true, y_pred = [], []
    parts = []
    if no_critic: parts.append("no-critic")
    if no_rag:    parts.append("no-rag")
    label = "+".join(parts) or "default"
    for i, rec in enumerate(records):
        print(f"  [{i+1}/{len(records)}] {label}  q='{rec['question'][:40]}'")
        try:
            result = await score_fn(rec["question"], rec["answer"], no_critic=no_critic, no_rag=no_rag)
            y_true.append(int(rec["gold_score"]))
            y_pred.append(result.score)
            rag_info = f"  rag_calls={result.rag_calls}" if not no_rag else ""
            print(f"    gold={rec['gold_score']}  pred={result.score}  {'✓' if abs(rec['gold_score']-result.score)<=1 else '✗'}{rag_info}")
        except Exception as e:
            print(f"    [error] {e}")
    return y_true, y_pred


async def run_consistency(records: list[dict], repeats: int = 3) -> dict:
    all_stds, extreme_drifts = [], 0
    for i, rec in enumerate(records):
        print(f"  [{i+1}/{len(records)}] consistency  q='{rec['question'][:40]}'")
        scores = []
        for _ in range(repeats):
            try:
                result = await score_fn(rec["question"], rec["answer"])
                scores.append(result.score)
            except Exception as e:
                print(f"    [error] {e}")
        if len(scores) >= 2:
            sd = stdev(scores)
            all_stds.append(sd)
            if max(scores) - min(scores) >= 2:
                extreme_drifts += 1
            print(f"    scores={scores}  std={sd:.2f}")
    return {
        "n": len(all_stds),
        "mean_std": round(mean(all_stds), 4) if all_stds else None,
        "extreme_drift_rate": round(extreme_drifts / len(all_stds), 4) if all_stds else None,
    }


# ── Results ───────────────────────────────────────────────────────────────────

def save_results(metrics: dict, out_dir: Path, label: str = "score_node") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{label}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return out_path


def print_metrics(metrics: dict, title: str = ""):
    if title:
        print(f"\n── {title} ──")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args):
    records = load_testset(args.testset)
    print(f"Loaded {len(records)} records from {args.testset}")

    results_dir = Path(__file__).parent / "results"
    ts = time.strftime("%Y%m%d_%H%M%S")
    output = {"timestamp": ts, "testset": args.testset, "model": os.environ.get("LLM_MODEL", "default")}

    if args.consistency:
        print("\nRunning consistency evaluation (3 repeats per case)...")
        cons = await run_consistency(records)
        output["consistency"] = cons
        print_metrics(cons, "Consistency")

    elif args.compare_rag:
        print("\nRunning WITH RAG (knowledge base enabled)...")
        yt, yp = await run_accuracy(records, no_critic=args.no_critic)
        m_on = compute_metrics(yt, yp)
        print_metrics(m_on, "RAG ON")

        print("\nRunning WITHOUT RAG (no knowledge base)...")
        yt2, yp2 = await run_accuracy(records, no_critic=args.no_critic, no_rag=True)
        m_off = compute_metrics(yt2, yp2)
        print_metrics(m_off, "RAG OFF")

        delta = {k: round(m_on[k] - m_off.get(k, 0), 4)
                 for k in m_on if isinstance(m_on[k], float)}
        output["rag_on"] = m_on
        output["rag_off"] = m_off
        output["rag_delta"] = delta
        print("\n── RAG contribution (ON - OFF) ──")
        for k, v in delta.items():
            sign = "+" if v > 0 else ""
            print(f"  {k}: {sign}{v}")

    elif args.compare_critic:
        print("\nRunning WITH critic...")
        yt, yp = await run_accuracy(records, no_critic=False)
        m_on = compute_metrics(yt, yp)
        print_metrics(m_on, "Critic ON")

        print("\nRunning WITHOUT critic...")
        yt2, yp2 = await run_accuracy(records, no_critic=True)
        m_off = compute_metrics(yt2, yp2)
        print_metrics(m_off, "Critic OFF")

        delta = {k: round(m_on[k] - m_off.get(k, 0), 4) for k in m_on if isinstance(m_on[k], float)}
        output["critic_on"] = m_on
        output["critic_off"] = m_off
        output["critic_delta"] = delta
        print("\n── Critic contribution (ON - OFF) ──")
        for k, v in delta.items():
            sign = "+" if v > 0 else ""
            print(f"  {k}: {sign}{v}")

    else:
        no_critic = args.no_critic
        no_rag = args.no_rag
        parts = []
        if no_critic: parts.append("no-critic")
        if no_rag:    parts.append("no-rag")
        label_str = "+".join(parts) or "default"
        print(f"\nRunning accuracy eval ({label_str})...")
        yt, yp = await run_accuracy(records, no_critic=no_critic, no_rag=no_rag)
        m = compute_metrics(yt, yp)
        output[label_str] = m
        print_metrics(m, f"Accuracy ({label_str})")

    path = save_results(output, results_dir)
    print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="testsets/score_node.jsonl")
    parser.add_argument("--no-critic", action="store_true", help="Skip Critic-Actor loop")
    parser.add_argument("--no-rag", action="store_true", help="Disable knowledge base access")
    parser.add_argument("--consistency", action="store_true", help="Run 3x per case, measure variance")
    parser.add_argument("--compare-critic", action="store_true", help="Run both critic on/off and compare")
    parser.add_argument("--compare-rag", action="store_true", help="Run both rag on/off and compare")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
