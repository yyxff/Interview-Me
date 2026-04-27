## Context

当前架构中笔记（Note）有独立的 notes collection，`_fetch_notes()` 在检索时单独召回并拼入 system prompt，与向量路和图谱路完全隔离。这导致：
- 笔记不参与 RRF 融合和 cross-encoder rerank，无法与文档 chunk 竞争排名
- 图谱检索无法发现笔记中的实体/关系，BFS 扩展不会跨越笔记
- 笔记质量好坏对检索结果无影响，强制出现在参考资料中

目标：将笔记视为一等公民的 chunk，让两条检索路都能自然找到它。

## Goals / Non-Goals

**Goals:**
- `index_note()` 额外将笔记 chunk 写入 knowledge collection（bi-encoder 向量检索）
- `index_note()` 额外对笔记做实体/关系抽取，写入 graph RAG（entity/relation Qdrant + NetworkX）
- `delete_note()` 同步清理 knowledge collection 和 graph RAG 中的对应记录
- 笔记在 knowledge collection 的 chunk_id 格式为 `note:<note_id>`，source 字段为 `"笔记"`

**Non-Goals:**
- 不改变 notes collection 的写入逻辑（现有笔记召回路径保持兼容）
- 不修改检索流水线（retrieve_rich、retrieve_graph）本身
- 不对笔记做进一步分块（整篇笔记作为单个 chunk）
- 不引入异步图索引进度跟踪（沿用 graph_rag builder 现有模式）

## Decisions

### D1: chunk_id 格式为 `note:<note_id>`

knowledge collection 中现有 chunk 的 ID 来自文档解析，格式多样。笔记使用 `note:` 前缀可以：
- 在 delete_note() 中精确定位并删除，无需 `where` 过滤
- 在检索日志和前端 source 展示中一眼识别来源

备选方案：用 `where={"source": "笔记"}` 过滤删除 — 有误删其他 source 字段相同文档的风险，排除。

### D2: 图谱使用 `note:<note_id>` 作为 source 键

`_build_graph_for_source()` 以 `source` 为命名空间生成 entity_id / relation_id，存储文件名为 `{source}.graph.json`。笔记使用 `note:<note_id>` 作为 source，可自然隔离，删除时按 source 清理 Qdrant 即可（`_clear_vectors_for_source(col, f"note:{note_id}")`）。

备选方案：所有笔记共享一个 `notes` source — 删除单条笔记时需要精细过滤，引入复杂度，排除。

### D3: 图索引同步调用（非异步）

`index_note()` 已是同步函数，在后台线程中运行。graph_rag 的抽取（`_extract_entities_relations`）是异步的，需要 `asyncio.run()` 或 `asyncio.get_event_loop().run_until_complete()` 包装。

选择 `asyncio.run()` — 后台线程没有运行中的 event loop，`asyncio.run()` 是最简洁的方式。

### D4: 复用 builder.py 现有函数

`_build_graph_for_source()`、`_index_graph_to_qdrant()`、`_save_graph()` 都可以直接复用，只需在 notes.py 中调用。需要暴露这些内部函数（或在 graph_rag `__init__.py` 中导出）。

## Risks / Trade-offs

- [Risk] 笔记内容短，实体抽取可能返回空结果 → 正常情况，空图不写入，不影响功能
- [Risk] 旧版笔记（已存在）不会自动补索引 → 接受，用户可删除重建或提供迁移脚本（本 change 不包含）
- [Risk] `asyncio.run()` 在已有 event loop 的线程中调用会报错 → 后台线程无 loop，安全；如未来改为 async 调用需调整

## Migration Plan

1. 部署后，新建笔记自动走双路索引
2. 旧笔记不受影响，继续通过 notes collection 召回
3. 旧笔记如需进入新路径，用户删除后重建即可（无自动迁移）
