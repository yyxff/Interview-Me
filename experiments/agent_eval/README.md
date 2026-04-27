# Agent End-to-End Evaluation

Evaluates the full interview agent using synthetic Personas and LLM-as-Judge.

## Quick Start

```bash
cd experiments/agent_eval
export GEMINI_API_KEY=your_key
export LLM_PROVIDER=openai-compatible
export LLM_API_KEY=your_llm_key
export LLM_MODEL=your_model

# 1. Run a persona session (generates a full mock interview)
python3 run_persona_session.py --persona expert --verify
python3 run_persona_session.py --persona novice --verify
python3 run_persona_session.py --persona mixed --verify

# 2. Check behavioral invariants on saved session logs
python3 check_invariants.py --log-dir ../../backend/sessions/

# 3. Judge a session with LLM
python3 judge_session.py --session results/expert_20260427_120000

# 4. Compare two versions of the agent on the same persona
python3 judge_session.py --compare results/expert_v1.json results/expert_v2.json
```

## Evaluation Layers

### Layer 1: Behavioral Invariants (zero cost, rule-based)

`check_invariants.py` verifies 4 rules that decide_node should always follow:

| Rule | Condition | Expected decision |
|------|-----------|-------------------|
| 1 | score ≤ 2 AND depth < 3 | deepen |
| 2 | score ≥ 4 | pass or pivot |
| 3 | depth ≥ 3 | back_up or pass |
| 4 | question_count ≥ 4 | pass |

**Baseline (2026-04-27):** 11 sessions, 17 decisions — Rule 2: 1 violation (100% rate on 1 applicable case)

### Layer 2: Persona Behavior Expectations

`run_persona_session.py --verify` checks if sessions match expected behavior:

| Persona | avg_score | pass_rate | deepen_rate |
|---------|-----------|-----------|-------------|
| expert | ≥ 3.5 | ≥ 60% | — |
| novice | ≤ 3.0 | — | ≥ 40% |
| mixed | — | — | score gap ≥ 1.5 |

### Layer 3: LLM-as-Judge (Gemini 2.5 Pro)

`judge_session.py` evaluates 4 dimensions (1-5 each):

| Dimension | What it measures |
|-----------|-----------------|
| coverage | Were key topics tested? |
| adaptiveness | Did strategy adjust to performance? |
| question_quality | Clear, non-repetitive, well-paced? |
| scoring_fairness | Scores match answer quality? |

**Baseline:** _Fill in after first run._

## Personas

| Persona | Skills | Expected behavior |
|---------|--------|-------------------|
| `expert` | All domains 4-5/5 | Short sessions, mostly pass, high scores |
| `novice` | All domains 1-2/5 | Long sessions, many deepen, low scores |
| `mixed` | Redis/MySQL=5, OS/Network=2 | Score gap between domains |

## Files

```
agent_eval/
  check_invariants.py       # Behavioral invariant checker
  run_persona_session.py    # Run full session with synthetic persona
  judge_session.py          # LLM-as-Judge evaluation
  personas/
    expert.yaml             # Persona configs
    novice.yaml
    mixed.yaml
  results/
    *.json                  # Session outputs and judge reports
```
