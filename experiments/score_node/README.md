# Score Node Evaluation

Evaluates `score_node` accuracy, consistency, and Critic-Actor loop effectiveness.

## Quick Start

```bash
cd experiments/score_node
export GEMINI_API_KEY=your_key
export LLM_PROVIDER=openai-compatible   # or anthropic
export LLM_API_KEY=your_llm_key
export LLM_MODEL=your_model

# 1. Build testset (generates ~35 labeled examples via Gemini Flash)
python3 build_testset.py --out testsets/score_node.jsonl

# 2. Review manually — open testsets/score_node.jsonl and check outliers

# 3. Run accuracy evaluation
python3 eval_score_node.py --testset testsets/score_node.jsonl

# 4. Compare Critic on vs off
python3 eval_score_node.py --testset testsets/score_node.jsonl --compare-critic

# 5. Measure score consistency
python3 eval_score_node.py --testset testsets/score_node.jsonl --consistency
```

## Metrics

| Metric | Description | Good threshold |
|--------|-------------|----------------|
| Cohen's κ (weighted) | Inter-rater agreement with gold labels | > 0.6 |
| MAE | Mean absolute error vs gold scores | < 0.8 |
| Spearman ρ | Rank correlation | > 0.7 |
| Soft match rate | Predictions within ±1 of gold | > 0.85 |
| mean_std | Avg std over 3 repeated runs | < 0.5 |
| extreme_drift_rate | Fraction where max - min ≥ 2 | < 0.10 |

## Baselines

_Fill in after first run._

| Date | κ | MAE | Spearman | Notes |
|------|---|-----|----------|-------|
| (pending) | - | - | - | first run |

## Test Case Types

| Type | Description | Target score |
|------|-------------|--------------|
| `perfect` | Complete, deep answer | 5 |
| `poor` | Wrong or no knowledge | 1-2 |
| `partial` | Right direction, gaps | 3 |
| `detail_error` | One wrong detail | 2-3 |
| `off_topic` | Fluent but unrelated | 1 |
| `obscure` | Rare topic, shallow answer | 2 |
| `length_pair` | Short answer, mid quality | 3 |

## Files

```
score_node/
  build_testset.py      # Generate labeled testset via Gemini Flash
  eval_score_node.py    # Run accuracy / consistency / critic eval
  testsets/
    score_node.jsonl    # Labeled test cases (human-reviewed)
  results/
    score_node_*.json   # Eval results by timestamp
```
