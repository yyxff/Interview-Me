## ADDED Requirements

### Requirement: Persona 定义
系统 SHALL 提供至少 3 个固定 Persona 配置文件（YAML），定义候选人的技术能力分布：
- `expert`：所有知识点均高度掌握，答案准确完整，预期平均分 4-5
- `novice`：所有知识点均浅层了解，答案方向对但细节缺失，预期平均分 2-3
- `mixed`：部分知识点精通（预期 4-5），部分不了解（预期 1-2），可配置强弱领域

#### Scenario: Persona 配置加载
- **WHEN** 加载 `personas/expert.yaml`
- **THEN** 得到包含技术域能力等级映射的配置对象

### Requirement: Persona 答案生成
系统 SHALL 根据 Persona 配置和面试官问题，用 Gemini Flash 生成符合该能力等级的候选人答案，答案风格 SHALL 符合真实口语化表达。

#### Scenario: Expert 答案生成
- **WHEN** Persona 为 expert，问题为技术题
- **THEN** 生成的答案涵盖核心要点，有原理说明，无明显遗漏

#### Scenario: Novice 答案生成
- **WHEN** Persona 为 novice，问题为技术题
- **THEN** 生成的答案方向大致正确，但缺少细节和原理

### Requirement: 端到端 Session 生成
系统 SHALL 支持以 Persona 驱动完整 agent session：在每次 ask_node interrupt 时，自动调用 Persona 答案生成器填充候选人回答，直到 session 结束。

#### Scenario: 完整 session 生成
- **WHEN** 运行 `python run_persona_session.py --persona expert --output session.json`
- **THEN** 生成完整 session，包含所有轮次的问题、答案、分数、决策，保存为 JSON

### Requirement: 行为预期验证
系统 SHALL 对 Persona session 做基本行为预期检查：
- expert session：平均分 SHALL ≥ 3.5，pass 决策比例 SHALL ≥ 60%
- novice session：平均分 SHALL ≤ 3.0，deepen 决策比例 SHALL ≥ 40%
- mixed session：强弱领域平均分差 SHALL ≥ 1.5

#### Scenario: Expert session 验证
- **WHEN** 对 expert persona session 运行预期检查
- **THEN** 输出是否通过各项预期，不通过时列出实际值

#### Scenario: Novice session 验证
- **WHEN** 对 novice persona session 运行预期检查
- **THEN** 输出是否通过各项预期，不通过时列出实际值
