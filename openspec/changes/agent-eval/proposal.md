## Why

目前 interview agent 缺乏系统性评估，无法量化判断每次迭代是提升还是退步。随着 score_node（Critic-Actor Loop）和 decide_node（ReAct 导演）的复杂度增加，需要一套可重复运行的评估框架来支撑后续开发决策。

## What Changes

- 新增 score_node 单节点评估流水线：构建标注测试集，衡量打分准确性、一致性、Critic 有效性
- 新增行为不变量检查器：回放 session 日志，程序化验证 decide_node 的规则合规性
- 新增合成 Persona 测试集：3 类候选人画像（专家/新手/混合），用于端到端 agent 行为验证
- 新增 LLM-as-Judge 端到端评估：用 Gemini 2.5 Pro 对完整 session 按覆盖度/适应性/问题质量/公平性打分
- 新增评估结果存储与对比：每次运行保存 JSON 结果，支持版本间横向对比
- 新增 standalone 接口层：将 score_node 和完整 agent session 从 FastAPI/前端解耦，支持纯 Python 直接调用

## Capabilities

### New Capabilities

- `standalone-interfaces`: score_node 和完整 agent interview 的 headless Python 接口，脱离 FastAPI/前端可独立运行
- `score-node-eval`: score_node 单节点评估——测试集构建、自动打分、准确性/一致性/Critic有效性指标计算
- `behavioral-invariant-check`: 从 session 日志回放 decide_node 决策，程序化检查规则违反
- `persona-simulation`: 合成候选人 Persona，驱动端到端 agent session 生成
- `e2e-llm-judge`: 用 Gemini 对完整 session transcript 按多维度打分，输出结构化 JSON 报告

### Modified Capabilities

## Impact

- 新增 `experiments/score_node/` 目录：score_node 评估脚本、测试集、结果
- 新增 `experiments/agent_eval/` 目录：端到端评估脚本、Persona 配置、session 结果、judge 报告
- `backend/agents/` 新增 `runner.py`（score_node headless 调用）和 `session_runner.py`（完整 interview headless 运行），**不改动现有节点逻辑**
- 依赖 Gemini API（用于 LLM-as-Judge 和标注辅助）
- 依赖现有 session 日志系统（decide_node 行为不变量检查的数据源）
- 依赖现有知识库（score_node 测试集构建）
