## ADDED Requirements

### Requirement: Session Transcript 格式化
系统 SHALL 将 session JSON 转换为结构化的面试记录文本，包含：面试背景（候选人技术栈、面试目标）、按轮次排列的问答记录（含 score 和 decision）。

#### Scenario: Transcript 生成
- **WHEN** 输入 session JSON
- **THEN** 输出可读的面试记录文本，每轮包含 Q/A/Score/Decision 四要素

### Requirement: 多维度 LLM 评分
系统 SHALL 调用 Gemini 2.5 Pro，按以下 4 个维度对 transcript 打分（1-5），并输出结构化 JSON：
- `coverage`：核心技术点是否被充分考察（需传入"应考察清单"作为参考）
- `adaptiveness`：面试官是否根据候选人表现调整了策略
- `question_quality`：问题是否清晰、不重复、难度递进合理
- `scoring_fairness`：评分是否与答案质量一致、无明显偏差

每个维度 MUST 附带一句 reasoning。

#### Scenario: 正常评分
- **WHEN** 运行 `python judge_session.py --session session.json --checklist checklist.yaml`
- **THEN** 输出 `{"coverage": int, "adaptiveness": int, "question_quality": int, "scoring_fairness": int, "overall": int, "main_issue": str, "per_dimension_reasoning": {...}}`

#### Scenario: 缺少应考察清单
- **WHEN** 未传入 checklist
- **THEN** coverage 维度降级为"问题覆盖话题多样性"评估，并在输出中标注 `coverage_mode: "diversity"`

### Requirement: 版本对比报告
系统 SHALL 支持对两个 session JSON（不同版本 agent 在同一 Persona 下的结果）生成对比报告，显示各维度分数变化。

#### Scenario: 版本对比
- **WHEN** 运行 `python judge_session.py --compare v1.json v2.json`
- **THEN** 输出各维度 delta（v2 - v1），标注提升/退步

### Requirement: 结果持久化
评估结果 SHALL 保存为 `experiments/eval/results/e2e_judge_<persona>_<timestamp>.json`。

#### Scenario: 结果文件
- **WHEN** 评估完成
- **THEN** 文件包含 `timestamp`, `persona`, `scores`, `judge_model`, `session_path` 字段
