# 架构决策记录（ADR）

> 记录项目中每一个值得思考的技术选型与权衡，供后续回顾、扩展时参考。

---

## 23. Graph RAG：不使用社区摘要（Community Summaries）

**决策：** 图谱检索只使用"实体/关系向量搜索 + BFS 1跳展开"，不引入 Microsoft GraphRAG 的社区摘要机制。

**社区摘要是什么：**
通过 Leiden 聚类算法将知识图谱实体划分为若干"社区"，对每个社区用 LLM 生成摘要，用于回答"整个知识库的主题是什么"之类的全局查询。

**为什么不用：**

1. **查询类型不匹配**：面试问答场景几乎全是局部查询（"解释 MVCC"、"进程 vs 线程"），社区摘要的价值在于全局查询（"这个知识库的核心主题是什么"），这类问题在面试准备场景中极少出现。

2. **知识库规模不够**：社区结构在 100+ 文档、10000+ 节点时才有意义。目前只有 2 个文件、约 2100 个节点，聚类结果无非是"OS 相关"和"网络相关"，直接从文件名就能得到，无需 LLM 额外计算。

3. **成本与收益不匹配**：每个社区需要一次 LLM 调用生成摘要，知识库更新后需重建。对当前规模，这是纯粹的额外成本。

**什么时候值得引入：**
知识库扩展到 50+ 文档、用户开始问"我应该重点复习哪些方向"之类的宏观问题时，再评估。

---

## 1. 前端框架：React + TypeScript + Vite

**选择：** React 18 + TypeScript + Vite

**为什么选 React？**
- 状态机（idle → listening → processing → speaking）天然适合 React 的状态驱动模型，每个状态对应不同 UI 表现。
- hooks 体系让语音识别、TTS 等副作用逻辑可以封装为独立 hook，易于单独测试和替换。
- 生态丰富，后续引入 VAD 库、Waveform 可视化、WebRTC 等都有成熟方案。

**为什么不选 Vue/Svelte？**
- Vue/Svelte 同样可行，但 React 在语音/音频处理领域的社区资源（@ricky0123/vad-web、wavesurfer-react 等）更完整。
- 团队/个人如果后续要接入 Vercel AI SDK 或 Next.js，React 一致性更好。

**为什么选 Vite 而非 CRA/Next.js？**
- Vite 冷启动 < 1s，HMR 极快，骨架开发体验好。
- 纯 SPA 即可满足需求，不需要 SSR（面试室不需要 SEO）。
- CRA 已停止维护。

---

## 2. 语音识别（STT）：Web Speech API（浏览器原生）

**选择：** `window.SpeechRecognition` / `webkitSpeechRecognition`

**为什么选浏览器原生？**
- **零延迟**：直接在浏览器本地运行（Chrome 调用 Google 云端识别），无需自建 WebSocket 流。
- **零成本**：不需要 Whisper 服务器，不产生 API 费用，骨架阶段首选。
- **连续识别**：`continuous: true` + `interimResults: true` 天然支持实时流式输出中间结果，打字机效果免费获得。

**缺点与权衡：**
| 维度 | Web Speech API | Whisper（自建/云端）|
|------|---------------|-------------------|
| 延迟 | 极低（< 200ms）| 高（500ms - 2s）|
| 准确率 | 普通话较好，方言差 | 多语言/方言均强 |
| 成本 | 免费 | 有费用或算力成本 |
| 控制权 | 低（浏览器黑盒） | 高（可调参数）|
| 离线 | 不支持 | 支持（本地模型）|

**骨架阶段用 Web Speech API，生产环境考虑替换为：**
- Whisper via WebSocket（实时流 → 更准确，可离线）
- AssemblyAI / Deepgram（云端流式 STT，延迟低，效果好）

**Chrome-Only 问题：**
- Web Speech API 在 Firefox/Safari 支持有限（Safari 部分支持）。
- 骨架阶段可接受；生产环境需改为 Whisper 或第三方 SDK 解决跨浏览器问题。

---

## 3. 语音合成（TTS）：Web Speech API SpeechSynthesis

**选择：** `window.speechSynthesis`

**为什么？**
- 与 STT 同源，零依赖，快速出声，适合骨架验证对话流程。
- 中文语音在 macOS/Chrome 上质量尚可（`zh-CN` 使用系统 Tingting/Meijia 语音）。

**已知问题与处理：**
- **Chrome 15s Bug**：Chrome 的 `SpeechSynthesis` 在说话约 15 秒后会静默停止（已知 bug，多年未修）。
  - **处理方式**：在 `useTTS` 中每 10 秒执行一次 `pause()` + `resume()` 刷新，规避此 bug。
- **语音质量有限**：机械感较强。

**生产环境替换方案：**
- **ElevenLabs**：音质极佳，有情感，按字符计费。
- **Azure TTS / Google TTS**：企业级稳定性，中文效果好。
- **Edge TTS（免费）**：微软 Edge 浏览器的神经网络 TTS，质量远超 Web Speech API，可通过非官方 API 调用。

---

## 4. VAD（语音活动检测）：静音计时器 + onspeechstart

**选择：** 利用 Web Speech API 的 `onspeechstart` 事件 + 1.5 秒静音计时器判定轮次边界

**为什么不用专用 VAD 库（如 @ricky0123/vad-web）？**
- `@ricky0123/vad-web` 基于 Silero VAD ONNX 模型，精度极高，适合生产。
- 但它需要 WASM + ONNX Runtime（包体 ~3MB），增加骨架复杂度。
- 骨架阶段用 `onspeechstart` + 静音计时器已能跑通完整对话流，验证逻辑正确性更重要。

**静音计时器的业务逻辑：**
```
用户说话 → 每次收到 Final Result 就重置 1.5s 计时器
→ 1.5s 内无新 Final Result → 判定本轮说完 → 发送后端
```
1.5 秒是面试场景的合理停顿时长：
- 太短（< 1s）：用户思考时被误判为结束，打断体验差。
- 太长（> 3s）：响应感觉迟钝。

**生产环境升级路径：**
- 接入 `@ricky0123/vad-web` 做真正的端点检测（End-Point Detection）。
- 或在后端用 Whisper 的流式模式做 server-side VAD。

---

## 5. 打断检测（Interruption Detection）

**选择：** `onspeechstart` 回调中检测当前状态，若为 `speaking` 则立即 `cancel()` TTS

**为什么这样做？**
- 模拟真实人类对话：听者开口时说话者应停止。
- 打断后重新处理用户输入，避免 AI 回答被覆盖。

**实现细节：**
```
用户开口（onspeechstart）
  └→ 若 state === 'speaking'
       └→ speechSynthesis.cancel()
       └→ setState('listening')
       └→ 继续识别用户说的话
```

**权衡：**
- 当前方案有约 200ms 延迟（`onspeechstart` 触发比实际说话滞后）。
- 如果接入 `@ricky0123/vad-web`，可将延迟降低到 ~80ms。
- 业务上，200ms 延迟在面试场景完全可接受（比按钮好得多）。

---

## 6. 后端框架：FastAPI（Python）

**选择：** FastAPI + Uvicorn

**为什么选 Python 而非 Node.js？**
- AI/ML 生态在 Python 最完整（Anthropic SDK、LangChain、向量数据库等）。
- 后续要做：简历解析（PyPDF2）、评分模型（sklearn/transformers）等，Python 是首选。

**为什么选 FastAPI 而非 Flask/Django？**
| | FastAPI | Flask | Django |
|--|---------|-------|--------|
| 异步支持 | 原生 async/await | 需 quart | 部分 |
| 类型安全 | Pydantic 自动校验 | 手动 | 手动 |
| 开发速度 | 快 | 快 | 慢（重） |
| 适合 AI Agent | ✅ | ⚠️ | ❌ |

- FastAPI 的 `async def` 适合后续接入流式响应（SSE / WebSocket）。
- Pydantic v2 自动校验请求体，减少 bug。

---

## 7. AI 模型：Claude Opus 4.6（可选，有 Placeholder 兜底）

**选择：** `claude-opus-4-6`，通过 ANTHROPIC_API_KEY 环境变量启用

**为什么选 Claude 而非 GPT-4？**
- 指令遵循能力强，中文对话自然。
- 面试场景需要角色扮演（面试官），Claude 在指令跟随上表现稳定。
- Anthropic 的安全设计使其不容易"出戏"。

**为什么做 Placeholder 兜底？**
- 骨架阶段希望不依赖 API Key 也能跑起来，验证语音对话流程本身。
- 降低上手门槛：clone → install → run，立即可以测试 UI。

**后续可替换为：**
- GPT-4o（更快的流式响应）
- 本地 LLM via Ollama（零成本，适合离线场景）
- 混合方案：快速回复用小模型，深度追问用大模型

---

## 8. 状态管理：React 本地状态 + Ref 模式

**选择：** `useState` + `useRef`，不引入 Zustand/Redux

**为什么不用全局状态管理库？**
- 骨架阶段状态简单（5 个顶级状态），引入 Zustand 是过度工程。
- 面试状态是线性状态机，不需要跨组件共享复杂状态。

**Ref 模式的必要性：**
```
问题：语音识别回调（onFinalResult）在异步闭包中捕获了 state
     → 闭包内的 state 永远是创建时的值（stale closure）

解决：用 useRef 存储需要在回调中读取的最新值
     stateRef.current = state  // 同步更新
     在回调中用 stateRef.current 而不是 state
```
这是 React + 浏览器原生 API 组合时的经典问题，不是设计缺陷，而是必要的模式。

---

## 9. 语言设置：硬编码 `zh-CN`

**当前：** STT 和 TTS 都固定为中文

**为什么先硬编码？**
- 骨架阶段明确场景（中文技术面试），减少变量。
- Web Speech API 的语言设置在运行时不能热切换（需要重启 recognition 实例）。

**后续扩展方案：**
- 在设置面板中选择语言，重启 recognition 时传入新 lang 参数。
- 在 `langRef` 中存储当前语言，`initRecognition` 读取最新值（`useSpeechRecognition` 已预留此设计）。

---

## 10. 流式响应传输：SSE vs WebSocket

**选择：** SSE（Server-Sent Events）用于 LLM 流式回复

**方案对比：**
| | SSE（当前）| WebSocket |
|--|-----------|-----------|
| 连接方式 | 每轮对话一个 HTTP 长连接 | 全程一个持久连接 |
| 方向 | 单向（服务器推） | 双向 |
| 实现复杂度 | 低（FastAPI `StreamingResponse`）| 中（需管理连接生命周期、重连逻辑）|
| 协议 | 标准 HTTP，CDN/代理天然支持 | 需要 Upgrade 握手，部分代理要额外配置 |
| 适合场景 | 请求-响应型流式（当前架构）| 实时双向通信 |

**为什么现阶段选 SSE？**
- 交互模式是"用户说 → AI 回"，天然的请求-响应结构，SSE 完全够用。
- 实现极简：FastAPI `StreamingResponse` + 前端 `fetch().body.getReader()`，无额外依赖。
- 端到端真流式：LLM token 到达后端后立即 `yield`，不等全量响应。

**何时应该升级到 WebSocket？**
- 后端接管 STT（客户端把音频流实时推给服务器识别）
- 后端接管 TTS（服务器生成音频流实时推给客户端播放）
- 需要服务器主动发起消息（如：面试计时提醒、实时评分推送）
- 需要在同一连接上并发处理多个流（语音 + 文本）

---

## 11. 回声消除（Echo Cancellation）：TTS 期间停麦

**问题：** 浏览器的 `SpeechRecognition` API 是黑盒，无法传入经过处理的 AudioStream，也无法感知当前音量来源（扬声器还是人声）。TTS 播放时麦克风会收到扬声器的声音，触发误识别。

**方案对比：**

| 方案 | 效果 | 工作量 |
|------|------|--------|
| **A. 智能抑制**（短文本阈值过滤） | 减少自打断，但阈值难以精确，长回声仍可触发 | 小 |
| **B. TTS 期间停麦**（当前）| 彻底解决，代价是 AI 说话期间无法语音打断 | 小 |
| **C. getUserMedia + WebAudio VAD** | 用回声消除的 MediaStream 做自定义 VAD | 中 |
| **D. 全链路 WebSocket**（服务端 VAD + AEC）| 接近 GPT Voice 效果，服务端控制全部音频逻辑 | 大 |

**当前选择 B 的原因：**
- 方案 A 实测阈值不稳定，中文 TTS 回声经常超过字符阈值。
- 面试场景下，AI 说话期间用户不打断是合理的交互预期。
- 方案 C/D 需要替换整个音频管道，超出当前骨架阶段的范围。

**升级路径：**
- 接入 `@ricky0123/vad-web`（Silero VAD）+ `getUserMedia` 自定义流，实现真正的回声消除，同时保留打断能力。
- 或整体迁移到 WebSocket + 服务端音频处理（见 Decision 10 升级条件）。

---

## 12. RAG 向量索引策略：Q&A 多问题扩展

**选择：** 对每个原文 chunk，用 LLM 生成 N 个不同角度的问题，每个问题单独向量化，但 `document` 字段存储原文。

**为什么不直接向量化原文？**
- 用户提问是问句语义，原文是陈述语义，两者在向量空间中距离较远，直接匹配召回率低。
- 用问题 embed 后，用户问"Redis 怎么做持久化"能精准命中"Redis 持久化有哪些方案"这个问题对应的 chunk，而不需要和大段原文做相似度比较。

**实现关键点：**
- ChromaDB `add()` 的 `embeddings` 参数接受预计算向量，`documents` 存原文。
- 检索时返回的是原文（不是问题），检索质量由问题向量决定，展示内容由原文决定。
- 每个 chunk 生成 4 个问题 → 4 个向量，向量数 = chunk 数 × 4。
- `retrieve()` 按 `chunk_id` 去重，避免同一 chunk 被计入多次。

**对照实验：** v1 版本用原文直接向量化（已保存为 `chroma_db_v1_raw_text`），v2 用 Q&A 向量化。通过 `check_retrieval.py` 对同一批查询比较 Recall@K。

---

## 13. Q&A 对持久化：缓存到 `.qa.json`

**选择：** 每本书索引时，把 LLM 生成的所有 Q&A 对保存到 `knowledge/{name}.qa.json`，重新索引时优先从文件加载，不再调用 LLM。

**为什么要缓存？**
- LLM 生成 Q&A 的成本：一本 272 个 chunk 的书 × 4 问题 = 1088 次 API 调用，费用和时间不可忽视。
- 数据库损坏、迁移、A/B 对比实验时需要重建索引，若每次都重新生成 Q&A，成本翻倍。
- `.qa.json` 也是实验数据资产，可用于离线分析 LLM 生成质量。

**文件格式：**
```json
[
  {
    "chunk_id": "book_0",
    "chapter":  "第一章",
    "text":     "原文...",
    "questions": ["问题1", "问题2", "问题3", "问题4"]
  }
]
```

---

## 14. 知识库文件结构：单大文件 + 伴生文件

**选择（待实施）：** 保留一个 `{name}.md`（供人阅读），配套生成伴生文件：
- `{name}.chunks.json`：预切分结构（含 chapter 层级），重建索引时跳过重新切块
- `{name}.qa.json`：LLM 生成的 Q&A 对缓存（见 Decision 13）

**为什么不做"每章一个文件"的目录树？**
- Web 查看器目前按单文件设计，目录树需要重写 UI 导航逻辑。
- EPUB 的章节数量不固定（几十到几百），目录树对用户心智负担更重。
- 伴生文件方案对现有 viewer 零改动，仅索引层感知到层级信息。

**为什么不只存 JSON 不存 .md？**
- `.md` 是人类可读的文本，Web Viewer 直接展示，调试 EPUB 转换质量时不需要反序列化。
- JSON 是机器可读的结构，索引逻辑使用。两者职责分离，互不干扰。

**层级信息的来源：**
- EPUB spine 顺序 + `<h1>/<h2>` 标签 → 章/节
- 当前 `_chunk_markdown()` 已从 `## ` 标题提取 `chapter` 字段
- 后续可从 ebooklib 的 `toc`（目录树）提取更精确的层级路径

---

## 15. 实验对比与指标追踪

**原则：** 每次改变核心算法（切块策略、向量化方式、检索参数）时，先保存当前数据库快照，再实施改动，事后用同一批查询对比指标。

**快照命名规范：**
```
backend/chroma_db_v{版本}_{策略描述}/
例：chroma_db_v1_raw_text/
    chroma_db_v2_qa_embed/
    chroma_db_v3_hierarchical/
```

**评估工具：** `backend/check_retrieval.py` 支持指定查询，输出命中 chunk 的 source/chapter/question，可用于：
- 手动评估 Precision（返回的 chunk 是否相关）
- 比较不同版本对同一问题的召回差异

**后续可量化的指标：**
| 指标 | 含义 | 如何收集 |
|------|------|---------|
| Recall@3 | top-3 里有多少包含正确答案 | 人工标注 benchmark 查询集 |
| MRR | 第一个正确结果的排名倒数 | 同上 |
| 索引耗时 | 总 LLM 调用时间 | 已在 progress 中记录 |
| 向量数/chunk | 扩展比例 | `knowledge_count() / chunk_count` |

---

---

## 16. 向量数据库：继续使用 ChromaDB，暂不迁移 Qdrant

**选择：** 保持 ChromaDB（内嵌模式）

**考量过 Qdrant 的时机：** Graph RAG 架构讨论时，评估是否应该切换。

**结论：保持 ChromaDB，原因如下：**
- 当前规模：几本技术书籍，预估 ≤ 5000 chunks。ChromaDB 内嵌模式轻松应对。
- Qdrant 优势（Named Vectors 多空间、高级 payload 过滤、生产规模）在当前场景无法触发。
- Qdrant 需要独立进程（Docker），增加本地开发和部署运维成本。
- Graph RAG 的图存储不依赖向量库（JSON + NetworkX），两者互不干扰。

**迁移 Qdrant 的触发条件：**
- 向量数 > 100 万（当前规模不会达到）
- 需要 Sparse + Dense 混合检索（BM25 + 语义）
- 需要多租户 / 生产部署

---

## 17. Graph RAG 架构：自实现 vs LightRAG

**选择：** 自实现轻量图层，不引入 LightRAG

**为什么考虑 LightRAG？**
LightRAG 自动从文本抽取 entity + relation（LLM 驱动），支持 local/global/hybrid 三种查询模式，global 模式适合"出几道相关题"这类宏观问法。

**为什么最终选择自实现？**

| 维度 | LightRAG | 自实现 |
|------|----------|--------|
| 抽取成本 | 每 chunk 一次 LLM 调用，一本书 500+ 次 | 同等成本，但流程可控 |
| 中文技术文档效果 | prompt 质量参差，需大量调试 | 可针对性优化 |
| 代码质量/可维护性 | 开源项目，接入侵入性强 | 完全掌控 |
| 与现有架构融合 | 需要并行维护两套系统 | 直接集成 |
| 适配书籍结构 | 通用设计，未针对章节层级 | 可利用 H1/H2/H3 结构 |

**自实现的核心思路（仿 LightRAG）：**
- Graph chunk → LLM → `(entity, relation, entity)` 三元组
- 节点：知识实体（"进程调度"、"PCB"、"上下文切换"）
- 边：关系类型（"包含"、"依赖"、"对比"、"引申"）+ 来源 chunk_id
- 存储：JSON（节点表 + 边表）+ NetworkX 内存图
- 查询：BFS/DFS 从命中节点展开 1-2 跳，补充关联上下文

**两种场景的价值：**
- **学习场景**：命中"进程调度" → 沿边找到"上下文切换 → 内核栈 → PCB"，呈现知识链路
- **面试场景**：回答完 A → 图中 A 的邻居 = 高频连问候选，面试官追问更自然

---

## 18. 双 RAG 流水线架构

**选择：** Vector RAG + Graph RAG 并行，共享同一份 chunk 语料

**两条流水线：**

```
同一份原始文本
       │
   Chunking（切分）
       │
  ┌────┴─────────┐
  │              │
  ↓              ↓
Vector RAG     Graph RAG
chunk →        chunk →
LLM 生成       LLM 抽取
QA 对     →    (entity, relation, entity) →
embed     →    图存储（JSON + NetworkX）
ChromaDB

查询时：两路召回 → 结果级 chunk_id 去重 → merge → 送 LLM context
```

**"去重"发生在哪里？**
- 两条流水线消费同一文本，产出不同数据结构（向量 vs 图边），**数据层无冲突**。
- 去重仅发生在**查询结果合并**时：Vector 召回 + Graph 召回可能包含相同 chunk，按 `chunk_id` 过滤掉重复项即可，一行代码解决。

**两种 chunk 大小需求不同（见 Decision 19）：**
- Vector RAG 需要小 chunk（精确语义匹配）
- Graph RAG 需要大 chunk（实体关系上下文更完整）

---

## 19. Chunking 策略：双层切分 + Graph 层 Overlap

**当前方案的问题（`_chunk_markdown`，字符数 ≤ 800）：**
1. 按 `##` 硬性分割，边界处上下文丢失
2. 用字符数而非 token 数做上限（中文 1 字 ≈ 1.5 token，导致实际 token 量波动大）
3. 对 Graph RAG 不友好：500 字符的 chunk 里实体关系太稀疏，三元组抽取质量差

**待实施方案：双层 chunk**

| 层级 | 用途 | 大小目标 | Overlap |
|------|------|---------|---------|
| Vector chunk | QA 生成 + embed，精确检索 | ~300 token | 无（可选 50 token） |
| Graph chunk | 实体/关系抽取，构建图 | ~800–1200 token（完整 ## 节） | 相邻 chunk 末尾 ~100 token 首尾重叠 |

**为什么 Vector 层不需要 Overlap？**
- Overlap 会产生重复向量，增加向量库体积，且对精确问答检索提升有限。
- 若某句话被切断，生成 QA 时 LLM 会跳过该段落，不会生成错误问题。

**为什么 Graph 层需要 Overlap？**
- "A 依赖 B"这一关系可能横跨两个 chunk 的边界，没有 overlap 就会被切断，抽取不到这条边。
- Overlap ~100 token = 一段完整的陈述句，足以覆盖跨段关系。

**Overlap 实现方式：**
- 切分时记录每个 chunk 的末尾 N token
- 下一个 chunk 开头拼接上一个 chunk 的末尾 N token
- Overlap 部分仅用于 LLM 输入，不作为该 chunk 的 `text` 字段存储（避免数据冗余）

**大小度量改为 token 数（tiktoken）：**
- 用 `tiktoken` 或 `transformers` tokenizer 准确计算 token 数
- 目标：Vector chunk ≤ 400 token，Graph chunk ≤ 1200 token（留 LLM 处理余量）

**切分依据保持结构优先：**
- 首先按 `##`（H2）边界分节
- 节内按段落切分，不打断段落
- 超长节（> 目标 token）才做段落级滑窗切分

**下一步：** 确定双层 chunk 设计后，重写 `_chunk_markdown()`，两个 pipeline 分别调用不同大小参数的同一函数。

---

---

## 20. 多 Agent 面试系统：ReAct + 工具目录化

**已实施（2026-03-22）**

- `director_plan()` 和 `interviewer_ask()` 均接入 `_react_loop()`
  - Director 规划阶段：可调用 `search_knowledge` / `search_profile` / `search_past_sessions`（最多 3 步）
  - Interviewer 提问阶段：可调用 `search_knowledge`（最多 2 步），动态决定是否查知识库
- 工具目录化：`backend/tools/` 每个工具独立模块，`build_toolset(session, names)` 控制每个 agent 的工具权限
- `director_plan()` 的 JSON 支持 `sub_questions[]`，开局就建好计划树（预埋子问题）

**Rollup Summary（上下文压缩）：**
- `ThoughtNode.summary` 字段
- `pass` 时对完成任务做 LLM 摘要，`back_up` 时对退出子树做摘要
- 面试官开始新任务首问时注入最近 3 个任务摘要（固定大小，不随对话增长）

**Planned 节点 bug 修复（关键）：**
- `run_turn()` 先检查 planned 兄弟节点，有则直接执行，不调 director — 导演的承诺必须兑现
- `director_advance()` 调用 `_skip_planned_descendants()` 清理残留 planned → skipped，防止僵尸节点

---

## 21. 知识库问答：Fork 对话树 + 标签页窗口系统

**非线性对话树（初版，2026-03-22）：**
- 每个 Q+A 是 `ConvNode` 树节点，可在任意节点派生新分支
- `getPath()` 从叶到根回溯，`submit()` 只用当前路径上下文（无跨分支污染）
- `handleSummarize()` 只总结当前路径

**标签页窗口系统（2026-03-29）：**
- 新增 `ConvTab` 类型 `{ id, label, currentId }`，`tabs[]` + `activeTabId` 管理多窗口
- "⑂ 分叉" 按钮出现在每条 AI 回答上，点击 → 以该节点为起点创建新标签页
- 右侧面板的分支树节点上也有 ⑂ 按钮，hover 时显示
- 不同标签页共享同一棵 `nodes` 树（ConvNode），共享分支点前的对话，各自管理 `currentId` 决定自己的上下文路径

**右侧面板重构（2026-03-29）：**
- 移除浮动 `ConvTree` 组件，移除独立 `NotesSidebar`
- 新 `RightPanel` 组件整合两个标签："对话树"（可视化分支树 + ⑂ 操作）和"知识笔记"
- 分支树节点：点击 → 当前标签页跳转该节点；⑂ → fork 出新标签页
- InfoPanel（面试页实时树）属性标签同步为文字标签（与复盘页一致）

**TODO（待实施）：**
- 每个标签页/分支可独立一键总结为知识笔记

---

## 后续路线图（骨架 → 生产）

| 阶段 | 升级项 | 影响 |
|------|--------|------|
| MVP | 接入 Claude API，完善 System Prompt | 真实面试体验 |
| v1 | 替换 TTS → ElevenLabs/Edge TTS | 更自然的语音 |
| v1 | 替换 VAD → @ricky0123/vad-web | 更准确的断句 |
| v2 | STT 替换 → Whisper WebSocket 流 | 更准确的识别，跨浏览器 |
| v2 | 流式响应（SSE）→ 边生成边朗读 | 大幅降低首字节延迟 |
| v3 | 面试评分模块（结构化反馈） | 核心业务价值 |
| v3 | 简历上传 → 个性化提问 | 差异化竞争力 |

---

## RAG 检索策略实验（2026-04-02）

### 背景

知识库检索存在两类问题：
- **事实型（O）**：问题考察某概念本身，答案在描述该概念的 chunk 里
- **关系型（GE）**：问题问"A 依赖/触发/通过什么"，答案是未出现在问题中的实体 B

这两类问题对检索方法的需求完全相反，测试集为 O/GE 各 30 题 1:1 混合。

### 关键发现：Reranker 在关系型问题失效

Cross-encoder（bge-reranker-base）训练于 MS-MARCO，学的是 query↔chunk 文字表面相关性。

```
事实型："TCP三次握手是什么" → chunk 讲三次握手 → 文字重合 → 高分 ✓
关系型：query 提 A → GT chunk 讲 B → A≠B → reranker 打低分 ✗
```

RRF 也天然偏向 bi+graph 双命中的 chunk，GT 在 graph-only 时分数低被挤出候选池。

### 各方案对比

| 方法 | O Score@5 | GE Score@5 | 综合MRR | 备注 |
|------|-----------|-----------|---------|------|
| bi-encoder | 0.833 | 0.333 | 0.429 | 基线 |
| rrf+rerank | 0.833 | 0.333 | 0.468 | 原系统 |
| graph-only | 0.300 | 1.000 | 0.489 | 关系型完美，事实型差 |
| 加权RRF(k_graph=20) | 0.500 | 0.867 | — | 解GE但破O |
| HyDE+加权RRF | 0.467 | 0.833 | — | 改善有限 |
| LLM路由(few-shot) | — | — | 0.468 | GE路由准确率0%，失败 |
| 实体距离路由(0.26) | 0.467 | 0.967 | 0.605 | 需额外路由判断 |
| **rrf_path_rr** ✅ | **0.833** | **0.933** | **0.767** | **上线方案** |

### 最终方案：证据链 Reranking（Contextualized Reranking）

**原理**：把 graph 检索到的关系边拼成自然语言，前置到对应 chunk 文本再送 reranker：

```
[图谱路径] TCP Keepalive --依赖--> SO_KEEPALIVE。应用程序需通过 setsockopt 设置才能启用。

[原始chunk] SO_KEEPALIVE 是一个 socket 选项...
```

Reranker 同时看到 A、关系动词、B，不再因"query 提 A、chunk 讲 B"而误判。

**实现**：`retrieve_rich()` 新增 `path_map` 参数，`main.py` 在调用前从 `graph_result.relations` 构建路径文本并传入。无需路由，单一管道同时覆盖两类问题。

**效果**（vs 基线 rrf+rerank）：
- 综合 Score@5：0.583 → **0.883**（+51%）
- 综合 MRR：0.468 → **0.767**（+64%）
- O 策略 Score@5 不变（0.833），GE Score@5 0.333 → 0.933

### 实验脚本

```
backend/
├── eval_01_baseline_O.py       # O策略多指标基线（n=94）
├── eval_02_baseline_GE.py      # GE测试集生成 + 多路检索评测
├── eval_03_combined.py         # O+GE 1:1 混合评测（含各路由方案）
├── debug_01_inspect_ge.py      # 单题逐路召回调试
└── eval_archive/               # 历史实验脚本
```

日志目录：`backend/eval_logs/`，关键文件：
- `metrics_o_20260402_181139.json` — O基线（n=94）
- `combined_20260402_232951.json` — 最终对比（含rrf_path_rr）
