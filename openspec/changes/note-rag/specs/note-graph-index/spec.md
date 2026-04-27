## ADDED Requirements

### Requirement: 笔记实体关系抽取写入 graph RAG
`index_note()` SHALL 对笔记文本执行实体/关系抽取，将结果写入 graph RAG（Qdrant entity/relation collection + NetworkX 图 + graph.json 文件）。

- 使用 source = `note:<note_id>` 作为图谱命名空间
- 复用 `_build_graph_for_source()`、`_index_graph_to_qdrant()`、`_save_graph()` 现有实现
- 抽取结果为空（实体为 0）时不写入，静默跳过

#### Scenario: 笔记内容包含可抽取实体
- **WHEN** 笔记文本包含技术概念或实体，调用 `index_note()`
- **THEN** graph RAG 的 entity/relation Qdrant collection 中出现 source 为 `note:<note_id>` 的记录，且 `note:<note_id>.graph.json` 文件被创建

#### Scenario: 笔记内容不包含可识别实体
- **WHEN** LLM 抽取返回空实体列表
- **THEN** 不写入 Qdrant，不创建 graph.json，不抛出异常

#### Scenario: 图谱索引失败（LLM 超时等）
- **WHEN** 实体抽取过程中发生异常
- **THEN** 异常被捕获并打印错误日志，knowledge collection 的笔记索引不受影响
