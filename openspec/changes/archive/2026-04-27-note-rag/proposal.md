## Why

笔记（Notes）目前只写入独立的 notes collection，无法被向量检索（knowledge collection）和图谱检索（graph RAG）发现。用户期望笔记和文档 chunk 一样，能被两条检索路径找到，而不是通过硬编码的 notes collection 单独召回。

## What Changes

- `index_note()` 在写入 notes collection 后，额外将笔记作为单个 chunk 索引到 knowledge collection（向量 RAG）
- `index_note()` 额外对笔记文本做实体/关系抽取，写入 graph RAG（entity/relation store + graph vector index）
- `delete_note()` 同步删除 knowledge collection 和 graph RAG 中的对应记录
- 笔记在 knowledge collection 中的 `source` 字段标记为 `"笔记"`，`chunk_id` 格式为 `note:<note_id>`，方便识别与去重

## Capabilities

### New Capabilities

- `note-knowledge-index`: 笔记作为单 chunk 被索引进 knowledge collection（bi-encoder 向量），与文档 chunk 一起参与 RRF + rerank 召回
- `note-graph-index`: 笔记文本做 LLM 实体/关系抽取，写入 graph RAG pipeline，参与图谱检索与 BFS 扩展

### Modified Capabilities

- `note-lifecycle`: delete_note() 现在额外清理 knowledge collection 和 graph RAG 中的笔记记录

## Impact

- `backend/rag/notes.py`: `index_note()`、`delete_note()` 修改
- `backend/graph_rag/builder.py`: 需要暴露或复用单 chunk 图谱索引入口
- `backend/graph_rag/extractor.py`: 复用 `_extract_entities_relations()` 对笔记文本做抽取
- 不影响前端、API 路由或其他检索逻辑
