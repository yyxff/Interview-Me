## 1. 基础设施

- [x] 1.1 创建目录结构：`experiments/score_node/`（scripts/, results/, testsets/）和 `experiments/agent_eval/`（scripts/, results/, personas/）
- [x] 1.2 确认 session 日志持久化方案（MemorySaver → 文件），确保行为不变量检查有数据源
- [x] 1.3 配置 Gemini API 客户端工具模块（Flash 用于批量标注，2.5 Pro 用于 judge），放在 `experiments/` 共享目录

## 2. Standalone 接口层（解耦 FastAPI）

- [x] 2.1 编写 `backend/agents/runner.py`：暴露 `async score(question, answer, no_critic=False) -> ScoreResult`，构造最小 InterviewState 直接调用 score_node，不依赖 FastAPI
- [x] 2.2 编写 `backend/agents/session_runner.py`：暴露 `async run_session(jd, direction, answer_fn) -> SessionResult`，直接驱动 interview_graph + Command(resume=...) 循环，不依赖 HTTP
- [x] 2.3 验证两个接口在未启动 FastAPI server 的环境下可独立运行（`python -c "import asyncio; from agents.runner import score; ..."` 不报错）
- [x] 2.4 在评估脚本模板中添加 sys.path 配置，使 `from agents.runner import score` 可直接 import

## 3. Score Node 评估

- [x] 3.1 编写 `experiments/score_node/build_testset.py`：调用 Gemini Flash 按 7 类 case 生成 (question, answer, gold_score, reasoning) 四元组，输出 `testsets/score_node.jsonl`
- [ ] 3.2 人工 review testset，修正异常标注（重点检查两端分和低置信度项）
- [x] 3.3 编写 `experiments/score_node/eval_score_node.py`：通过 `runner.score()` 对 testset 运行评估，计算 Cohen's κ、MAE、Spearman ρ
- [x] 3.4 实现 `--no-critic` 模式（调用 `runner.score(no_critic=True)`），对比 critic-on/off 指标差异，记录 modification_rate 和 over_correction_rate
- [x] 3.5 实现 `--consistency` 模式：同一 (q,a) 对跑 3 次，计算 mean_std 和 extreme_drift_rate
- [x] 3.6 结果保存为 `experiments/score_node/results/score_node_<timestamp>.json`，记录 baseline 数值

## 4. 行为不变量检查

- [x] 4.1 编写 session 日志解析器：从 `roots_data` 提取 (score, depth, question_count, decision) 决策序列
- [x] 4.2 编写 `experiments/agent_eval/check_invariants.py`：验证 4 条规则，输出违反率报告
- [x] 4.3 支持 `--log-dir` 批量模式，汇总多个 session 的违反率
- [x] 4.4 对现有已保存的 session 日志跑一遍，记录当前 baseline 违反率

## 5. 合成 Persona 测试

- [x] 5.1 编写 3 个 Persona 配置文件：`experiments/agent_eval/personas/expert.yaml`, `novice.yaml`, `mixed.yaml`
- [x] 5.2 编写 Persona 答案生成器：根据 Persona 配置和问题，调用 Gemini Flash 生成符合能力等级的答案
- [x] 5.3 编写 `experiments/agent_eval/run_persona_session.py`：用 Persona 答案生成器作为 `answer_fn`，调用 `session_runner.run_session()` 生成完整 session JSON
- [x] 5.4 编写行为预期验证逻辑：检查 expert/novice/mixed session 是否符合预设行为预期（平均分、决策分布）

## 6. 端到端 LLM-as-Judge

- [x] 6.1 编写 session transcript 格式化器：将 session JSON 转为结构化面试记录文本
- [x] 6.2 编写 Judge prompt：包含应考察清单注入、4 维度评分要求、JSON 输出格式
- [x] 6.3 编写 `experiments/agent_eval/judge_session.py`：调用 Gemini 2.5 Pro，输出 4 维度分数 + reasoning，保存为 `results/e2e_judge_<persona>_<timestamp>.json`
- [x] 6.4 实现 `--compare` 模式：对比两个 session 的 judge 结果，输出各维度 delta
- [ ] 6.5 对 3 个 Persona session 各跑一次 judge，记录初始 baseline 报告

## 7. 文档

- [x] 7.1 在 `experiments/score_node/README.md` 记录：脚本用途、运行方法、指标含义、baseline 数值
- [x] 7.2 在 `experiments/agent_eval/README.md` 记录：Persona 设计、各脚本用途、运行顺序、baseline 报告
