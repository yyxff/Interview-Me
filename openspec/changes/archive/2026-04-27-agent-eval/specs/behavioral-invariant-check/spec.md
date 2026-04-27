## ADDED Requirements

### Requirement: Session 日志解析
系统 SHALL 提供解析器，从 session 日志（LangGraph state 快照中的 `roots_data`）提取决策序列，每条记录包含 (score, depth, question_count_in_task, decision)。

#### Scenario: 解析 session 日志
- **WHEN** 运行 `python check_invariants.py --log session.json`
- **THEN** 内部提取出决策序列列表，每条包含必要字段

### Requirement: 规则合规性检查
系统 SHALL 对每条决策记录验证以下 4 条不变量规则，并报告违反次数和比例：
1. score ≤ 2 且 depth < 3 → decision MUST 是 deepen
2. score ≥ 4 → decision MUST 是 pass 或 pivot
3. depth ≥ 3 → decision MUST 是 back_up 或 pass
4. question_count_in_task ≥ 4 → decision MUST 是 pass

#### Scenario: 合规报告
- **WHEN** 运行不变量检查
- **THEN** 输出每条规则的 `{rule_id, total_applicable, violations, violation_rate}` 列表

#### Scenario: 无违反
- **WHEN** agent 行为完全合规
- **THEN** 所有规则 violation_rate 为 0，输出 "All invariants passed"

#### Scenario: 检测到违反
- **WHEN** 存在规则违反
- **THEN** 列出具体违反的决策记录（含 session_id、step、实际 decision、期望 decision）

### Requirement: 批量分析
系统 SHALL 支持对目录下所有 session 日志批量运行，输出汇总违反率。

#### Scenario: 批量运行
- **WHEN** 运行 `python check_invariants.py --log-dir logs/`
- **THEN** 输出汇总报告，包含总 session 数、总决策数、各规则汇总违反率
