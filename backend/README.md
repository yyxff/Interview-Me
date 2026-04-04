# Backend 架构说明

## 目录结构

```
backend/
├── main.py                  # 入口：FastAPI app、provider 注入、startup 任务
├── llm/                     # LLM 提供商抽象层
├── routes/                  # HTTP 路由（每个文件一组业务接口）
├── agents/                  # 多 Agent 模拟面试引擎
├── rag/                     # 向量检索增强生成（ChromaDB）
├── graph_rag/               # 图谱增强检索（NetworkX + ChromaDB）
├── tools/                   # Agent 工具注册中心
├── sessions/                # 面试会话 JSON 存档（运行时生成）
├── knowledge/               # 知识库文档（.epub/.md/.pdf）
└── interview_agent.py       # 向后兼容 shim → agents/
```

---

## 启动流程（main.py）

1. `build_provider()` 读取环境变量构造 LLM 客户端
2. 将 provider 注入各路由模块（`set_provider()`）
3. 注册 7 个 APIRouter，挂载到 FastAPI app
4. `startup` 事件异步触发：向量索引 → QA 缓存回填 → 图谱索引

**环境变量：**

| 变量 | 说明 |
|------|------|
| `LLM_PROVIDER` | `anthropic` / `openai` / `openai-compatible` |
| `LLM_API_KEY` | API Key |
| `LLM_MODEL` | 模型名，默认 `claude-opus-4-6` |
| `LLM_BASE_URL` | openai-compatible 时必填（DeepSeek、Ollama 等） |

---

## llm/

LLM 提供商的抽象与工厂。

| 文件 | 作用 |
|------|------|
| `provider.py` | `LLMProvider` ABC（`chat` / `stream_chat`）；`AnthropicProvider`、`OpenAIProvider`；`build_provider()` 工厂函数 |

各 Agent 和路由模块均通过 `set_provider(p)` 注入，不直接导入具体实现。

---

## routes/

每个文件对应一组 HTTP 接口，通过 `APIRouter` 挂载。

| 文件 | 前缀 | 核心接口 |
|------|------|---------|
| `chat.py` | — | `POST /chat`、`POST /chat/stream` — 普通问答（不含 RAG） |
| `qa.py` | — | `POST /qa/stream` — 知识库精准问答（Graph RAG + RRF + rerank，SSE 流式） |
| `knowledge.py` | — | `/health`、`/upload/resume`、`/upload/knowledge`、`/knowledge/list`、`/rag/status`、`/rag/index-progress`、`/profile/*` |
| `graph.py` | — | `/graph/index`、`/graph/stats`、`/graph/full`、`/entity-chunk`、`/chunk/{id}` |
| `notes.py` | — | `/qa/summarize`、`/notes/*` — 笔记的保存、索引、增删查 |
| `qa_sessions.py` | — | `/qa-sessions/*` — QA 会话的持久化与检索 |
| `interview.py` | — | `/interview/start`、`/interview/chat`、`/interview/session/*`、`/interview/results` |

---

## agents/

模拟面试的多 Agent 引擎，状态机驱动。

### 状态机

```
INIT → PLANNING → ASKING → ANSWERING → SCORING → DIRECTING → ASKING → ...
                                                             ↘ DONE
```

### 文件说明

| 文件 | 作用 |
|------|------|
| `models.py` | 数据结构：`InterviewSM`（状态机）、`ThoughtNode`（问题/任务节点）、`InterviewSession`；树操作工具函数（`flat`、`find`、`tree_to_dict` 等） |
| `react.py` | 通用 ReAct 循环引擎（`react_loop`）：Thought → Action → Observation → Final Answer |
| `director.py` | DirectorAgent：`director_plan`（PLANNING 阶段，规划任务列表）、`director_decide`（每轮决策：deepen/pivot/back_up/pass）、`director_advance`（推进到下一任务） |
| `interviewer.py` | InterviewerAgent：`interviewer_ask`（ReAct 生成问题 + self-reflection 二次校验） |
| `scorer.py` | ScorerAgent：`scorer_evaluate`（CoT 打分 1-5）、`rollup_node`（任务段总结）、`build_prior_context`（已完成任务上下文） |
| `orchestrator.py` | 编排层：`start_interview`、`run_turn`、`save_session`；调用上面三个 Agent，串联完整一轮流程 |

### 思维树（ThoughtNode）

```
roots[0]: task "操作系统"      (node_type=task)
  ├─ question "进程与线程的区别"  (node_type=question, score=4)
  │    └─ question "追问：用户态线程怎么调度"  (status=answering)
  └─ question [planned]        (status=planned, question_intent="...")
roots[1]: task "网络协议"
  └─ ...
```

每道题的分数、反馈、导演决策都存在节点上，最终序列化为 JSON 存档。

---

## rag/

向量数据库（ChromaDB）的所有操作。

| 文件 | 作用 |
|------|------|
| `client.py` | 常量（路径、top-k 参数）；懒加载单例（`_get_client`、`_get_ef`、`_get_reranker`、3 个 collection getter）；`rerank()`、`is_available()` |
| `chunking.py` | `_chunk_markdown`（按 `##` 分块，≤800 字）、`_chunk_pdf`（合并短段落，~600 字）、`_load_or_build_chunks`（读缓存 `.chunks.json`） |
| `documents.py` | `epub_to_markdown`（EPUB → Markdown）、`ingest_epub`（写入 knowledge/ 并清旧索引）、`index_resume`（PDF 简历 → resume collection） |
| `indexing.py` | `AdaptiveSemaphore`（TCP 慢启动 + AIMD 并发控制）；`index_knowledge_with_qa`（全量索引：chunk + LLM 生成 QA 对）；`backfill_qa_cache`、`index_knowledge` |
| `retrieval.py` | `retrieve_rich`（主路径：bi-encoder + graph RRF 融合 + cross-encoder rerank）；`retrieve`（简单检索）；`_safe_query`、`_dedupe_chunks` |
| `notes.py` | 笔记的存储（`.md` + `.meta.json`）、索引到 notes collection、CRUD 操作 |
| `profile.py` | 候选人简介的读写（`save_profile`、`get_profile_text`、`profile_status`） |

### 精准问答检索管线（`/qa/stream`）

```
用户问题
  → rewrite_query（LLM 重写，提取关键词）
  → retrieve_rich：
       ├─ 向量检索（knowledge collection）
       ├─ Graph RAG 实体/关系检索（graph_rag.retrieve_graph）
       └─ RRF 融合 → cross-encoder rerank → top-k chunks
  → 拼接 chunks + graph summary → LLM 流式生成答案
```

---

## graph_rag/

知识图谱的构建与检索，与向量 RAG 并行，通过 RRF 融合。

| 文件 | 作用 |
|------|------|
| `store.py` | 常量（图目录、BFS 跳数）；懒加载单例（entities/relations ChromaDB collection、NetworkX DiGraph）；`_load_all_graphs_into_nx`（启动时加载全部 `.graph.json`） |
| `extractor.py` | LLM 从单个 chunk 提取实体和关系（`_extract_entities_relations`）；JSON-lines 解析 |
| `builder.py` | `index_knowledge_graph`：扫描 knowledge/ → 提取 → 合并图 → 存 `.graph.json` → 索引到 ChromaDB；实体/关系去重与合并 |
| `retrieval.py` | `retrieve_graph`：实体+关系向量检索 + BFS 邻域扩展 + cosine 剪枝；`get_subgraph_for_viz`（前端可视化子图）；`graph_stats` |

---

## tools/

Agent 调用的工具，通过 `build_toolset(session, ["tool_name"])` 按需组装。

| 文件 | 工具名 | 作用 |
|------|--------|------|
| `search_knowledge.py` | `search_knowledge` | 向量检索知识库，供 InterviewerAgent 查证知识点 |
| `search_profile.py` | `search_profile` | 检索候选人简历/背景，供 DirectorAgent 定制问题 |
| `search_past_sessions.py` | `search_past_sessions` | 检索历史面试记录，避免重复考察 |

注册新工具：新建 `tools/<name>.py`，实现 `make(session) -> {"desc": str, "fn": async fn}`，并在 `tools/__init__.py` 的 `REGISTRY` 中加一行。

---

## 向后兼容

`interview_agent.py` 是一个 shim，将旧的 `import interview_agent` 透明地转发到 `agents/` 包，老代码无需修改。
