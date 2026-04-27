## ADDED Requirements

### Requirement: Score node headless 调用接口
系统 SHALL 提供 `experiments/shared/runner.py`，暴露 `score()` 函数，接受 question、answer、可选 no_critic 参数，通过 sys.path 直接 import backend 的 `score_node` 逻辑并返回结构化结果，不依赖 FastAPI 或任何 HTTP 层。该文件不属于后端生产代码。

#### Scenario: 直接调用 score
- **WHEN** 从 eval 脚本执行 `from runner import score; result = await score(question, answer)`
- **THEN** 返回包含 `score`(int)、`reasoning`(str)、`feedback`(str) 的对象，与 score_node 内部逻辑完全一致

#### Scenario: 跳过 Critic
- **WHEN** 调用 `await score(question, answer, no_critic=True)`
- **THEN** 跳过 Critic-Actor Loop，仅返回初步评分结果

#### Scenario: 无 FastAPI 依赖
- **WHEN** 在未启动 FastAPI server 的环境中调用 `score()`
- **THEN** 正常执行，不抛出任何与路由或 HTTP 相关的异常

### Requirement: 完整 interview headless 运行接口
系统 SHALL 提供 `experiments/shared/session_runner.py`，暴露 `run_session()` 函数，接受 jd（职位描述）、direction（方向）和 `answer_fn` 回调，通过 sys.path 直接 import backend 的 `interview_graph`，驱动完整 plan → ask → score → decide 循环，不依赖 HTTP 请求或前端。该文件不属于后端生产代码。

#### Scenario: Persona 驱动完整 session
- **WHEN** 调用 `await run_session(jd="...", direction="...", answer_fn=persona_answer_fn)`
- **THEN** 完成完整面试循环直到 `__end__`，返回包含所有轮次 question/answer/score/decision 的 `SessionResult`

#### Scenario: answer_fn 回调接口
- **WHEN** `answer_fn(question: str) -> str` 被调用
- **THEN** 返回候选人答案字符串，session_runner 将其作为 LangGraph interrupt 的 resume 值

#### Scenario: Session 结果可序列化
- **WHEN** `run_session()` 返回 `SessionResult`
- **THEN** 可直接调用 `.to_dict()` 或 `json.dumps()` 序列化为 JSON，无循环引用或不可序列化字段

#### Scenario: 无 FastAPI 依赖
- **WHEN** 在未启动 FastAPI server 的环境中调用 `run_session()`
- **THEN** 正常执行，不依赖任何 HTTP 层

### Requirement: 评估脚本通过 shared 模块 import
评估脚本（`experiments/score_node/` 和 `experiments/agent_eval/` 下）SHALL 通过 `sys.path` 添加 `experiments/shared/` 后 import `runner` 和 `session_runner`，不需要安装额外 Python 包，不直接修改后端代码。

#### Scenario: 评估脚本路径配置
- **WHEN** 在 `experiments/score_node/` 或 `experiments/agent_eval/` 下运行任意评估脚本
- **THEN** 脚本头部的 `sys.path` 配置使 `from runner import score` 成功 import
