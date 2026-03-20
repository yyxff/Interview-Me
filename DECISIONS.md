# 架构决策记录（ADR）

> 记录项目中每一个值得思考的技术选型与权衡，供后续回顾、扩展时参考。

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
