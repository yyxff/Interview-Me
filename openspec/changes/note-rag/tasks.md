## 1. 暴露 graph_rag builder 内部函数

- [ ] 1.1 在 `backend/graph_rag/__init__.py` 中导出 `_build_graph_for_source`、`_index_graph_to_qdrant`、`_save_graph`、`_clear_vectors_for_source`

## 2. 扩展 index_note() 写入 knowledge collection

- [ ] 2.1 在 `backend/rag/notes.py` 中 import `_get_knowledge_col`
- [ ] 2.2 在 `index_note()` 完成 notes collection 写入后，将整篇笔记以 `note:<note_id>` 为 chunk_id 写入 knowledge collection（先删旧再写入，幂等）

## 3. 扩展 index_note() 写入 graph RAG

- [ ] 3.1 在 `backend/rag/notes.py` 中 import graph_rag builder 函数和 extractor
- [ ] 3.2 在 `index_note()` 中对笔记文本调用 `_extract_entities_relations()`（单 chunk，chunk_id = `note:<note_id>`），用 `asyncio.run()` 包装
- [ ] 3.3 调用 `_build_graph_for_source(source=f"note:{note_id}", extracted=[...])`、`_save_graph()`、`_index_graph_to_qdrant()` 写入图谱
- [ ] 3.4 实体为空时静默跳过，异常捕获后打印日志但不影响 knowledge collection 写入

## 4. 扩展 delete_note() 清理所有索引

- [ ] 4.1 在 `delete_note()` 中按 id `note:<note_id>` 删除 knowledge collection 中的记录
- [ ] 4.2 在 `delete_note()` 中调用 `_clear_vectors_for_source(ent_col, f"note:{note_id}")` 和 `_clear_vectors_for_source(rel_col, f"note:{note_id}")` 清理 graph RAG Qdrant
- [ ] 4.3 删除 `<graph_dir>/note:<note_id>.graph.json` 文件（若存在）
- [ ] 4.4 每步清理独立 try/except，互不阻断
