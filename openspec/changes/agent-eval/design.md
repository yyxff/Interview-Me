## Context

Interview agent 由 plan → ask → score → decide 四个节点组成。score_node 使用 Critic-Actor Loop + RAG 打分，decide_node 使用 ReAct 导演动态调整面试策略。当前缺乏任何量化评估，无法判断迭代方向。

评估框架不修改 agent 节点逻辑，独立放在 `experiments/score_node/`（单节点评估）和 `experiments/agent_eval/`（端到端评估）下，与现有 `experiments/rag/` 并列。

## Goals / Non-Goals

**Goals:**
- score_node 和完整 agent interview 可脱离 FastAPI/前端独立运行（headless）
- score_node 可量化评估（准确性、一致性、Critic 有效性）
- decide_node 行为合规性可程序化检查
- 端到端 session 质量可通过 LLM-as-Judge 纵向对比
- 所有评估结果保存为 JSON，支持版本间对比

**Non-Goals:**
- 不构建 CI 自动触发（人工按需跑）
- 不评估 plan_node 和 ask_node（优先级低）
- 不做 UI 可视化 dashboard
- 不修改 score_node / decide_node 等节点的内部逻辑

## Decisions

### D1：score_node 测试集构建方式

**选择：Gemini Flash 批量生成初稿 + 人工 review**

- Gemini 2.0 Flash 生成 (question, answer, gold_score, reasoning) 四元组
- 人工只审核异常项（score 分布两端 + Gemini 自信度低的）
- 目标 30 条，覆盖 7 类 case（满分/差答/方向对但不完整/细节错/答非所问/冷门题/长短答案）

替代方案：纯人工标注 → 成本高，人工一致性也不完美，不选。

### D2：Critic 有效性评估设计

**选择：对比 Loop on/off 两个模式的 κ**

- 在测试集上分别跑"有 Critic"和"无 Critic（只用初步评分）"
- 对比两者 Cohen's κ，判断 Critic 是否真的提升了准确性
- 同时记录修改率（触发比例）和"过修正率"（改后更差的比例）

### D3：行为不变量检查数据源

**选择：从现有 session 日志 JSON 中解析 ThoughtNode 树**

- 每个 session 已有完整 `roots_data` 快照（LangGraph checkpointer 的状态）
- 写解析器提取 (score, depth, question_count, decision) 四元组序列
- 程序化验证 decide_node 系统 prompt 中写明的 4 条规则

### D4：LLM-as-Judge 模型选择

**选择：Gemini 2.5 Pro 主评**

- Agent 自身用 Gemini，cross-family 偏差不是主要问题（评估过程质量，非偏好比较）
- Gemini 2.5 Pro 推理能力足够解析长 transcript
- Flash 用于 score_node 批量标注（便宜快），2.5 Pro 用于端到端评估（质量优先）

### D6：Standalone 接口架构

**选择：在 `backend/agents/` 新增两个薄包装模块，评估脚本通过 sys.path 直接 import**

```
backend/agents/
  runner.py          ← score_node headless 调用
                        score(question, answer, no_critic=False) -> ScoreResult
  session_runner.py  ← 完整 interview headless 驱动
                        run_session(jd, answer_fn) -> SessionResult
                        answer_fn: (question: str) -> str  # 由 Persona 生成器实现
```

- `runner.py`：直接构造最小 `InterviewState`，调用 `score_node(state)`，返回 score/reasoning/feedback，不依赖 FastAPI
- `session_runner.py`：直接调用 `interview_graph.ainvoke()` + `Command(resume=...)`，在 Python 中处理 interrupt，不依赖 HTTP 请求。`answer_fn` 是回调，可接入 Persona 生成器或人工输入
- 评估脚本通过 `sys.path.insert(0, "../../backend")` 直接 import，无需安装包

替代方案：把评估脚本放在 backend/ 内部 → 污染 production 代码，不选。

### D5：端到端评估触发方式

**选择：合成 Persona 驱动，人工按需跑**

- 3 个固定 Persona（Expert / Novice / Mixed），答案模板固定，保证可重复性
- 每次重大版本迭代前手动执行，输出对比报告
- Persona 答案用 Gemini Flash 按模板生成，不人工撰写

## Risks / Trade-offs

- **[Score 测试集规模小]** → 30 条统计功效有限；κ 置信区间宽。缓解：先建立基准，后续增量扩充。
- **[Persona 答案固定]** → 无法测试 agent 对真实人类语言多样性的鲁棒性。缓解：后续引入真实匿名 session 作为补充。
- **[Gemini-as-Judge 自评]** → Agent 和 Judge 同家族，可能有轻微偏袒。缓解：评估过程质量（覆盖度/适应性）比评估输出偏好，偏差影响小；必要时加 GPT-4o 交叉验证。
- **[日志格式依赖]** → 行为不变量检查依赖 LangGraph state 的 `roots_data` 字段格式。若 ThoughtNode 结构变化，解析器需同步更新。
- **[InterviewState 最小构造]** → `runner.py` 需要手动构造合法的 `InterviewState`，若 state 结构变化需同步维护。缓解：`runner.py` 集中在一处构造，变化时改一个文件。

## Open Questions

- session 日志当前持久化到哪里？MemorySaver 在进程重启后丢失，端到端评估需要确认日志存储方案。
- score_node 测试集中"知识库没有的冷门问题"这一类，如何界定"冷门"？建议从知识库覆盖率低的章节里选。
