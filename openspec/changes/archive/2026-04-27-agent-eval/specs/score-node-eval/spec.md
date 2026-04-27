## ADDED Requirements

### Requirement: Test set construction
系统 SHALL 提供一个测试集构建脚本，生成包含 (question, answer, gold_score, reasoning) 的 JSONL 文件。测试集 MUST 覆盖至少 7 类 case：满分答案、差答案、方向正确但不完整、细节错误、答非所问、知识库无覆盖的冷门题、同质量长短答案对。测试集规模 SHALL 不少于 30 条。

#### Scenario: 生成测试集
- **WHEN** 运行 `python build_testset.py --out testset.jsonl`
- **THEN** 输出 JSONL 文件，每行包含 `question`, `answer`, `gold_score`(1-5), `gold_reasoning`, `case_type` 字段

#### Scenario: 覆盖 case 类型
- **WHEN** 统计 testset.jsonl 中 `case_type` 字段分布
- **THEN** 7 类 case 类型均有至少 3 条记录

### Requirement: 准确性评估
系统 SHALL 计算 score_node 输出与 gold_score 的 Cohen's κ（weighted, linear）、MAE、Spearman ρ 三项指标。

#### Scenario: 运行准确性评估
- **WHEN** 运行 `python eval_score_node.py --testset testset.jsonl`
- **THEN** 打印并保存 `{"kappa": float, "mae": float, "spearman": float}` 到结果 JSON

#### Scenario: κ 基准线
- **WHEN** 首次运行后记录 baseline
- **THEN** 后续版本 κ 下降 > 0.05 SHALL 在报告中标注 regression 警告

### Requirement: 一致性评估
系统 SHALL 对同一 (question, answer) 对重复运行 score_node 3 次，计算分数标准差和极端漂移率（最大分差 ≥ 2 的比例）。

#### Scenario: 一致性测试
- **WHEN** 运行一致性模式 `--consistency`，对测试集每条记录跑 3 次
- **THEN** 输出 `{"mean_std": float, "extreme_drift_rate": float}`

### Requirement: Critic 有效性评估
系统 SHALL 支持 `--no-critic` 模式，跳过 Critic-Actor Loop 直接使用初步评分，并对比 critic-on 和 critic-off 的准确性指标差异。

#### Scenario: 对比 Critic on/off
- **WHEN** 分别运行 `--no-critic` 和默认模式
- **THEN** 报告包含两组 κ、MAE 对比，以及 Critic 触发修改的比例（modification_rate）和修改后变差的比例（over_correction_rate）

### Requirement: 结果持久化
评估结果 SHALL 保存为 `experiments/eval/results/score_node_<timestamp>.json`，包含指标、模型版本、测试集路径、运行时间。

#### Scenario: 结果文件格式
- **WHEN** 评估完成
- **THEN** 结果文件包含 `timestamp`, `metrics`, `model`, `testset_path`, `duration_seconds` 字段
