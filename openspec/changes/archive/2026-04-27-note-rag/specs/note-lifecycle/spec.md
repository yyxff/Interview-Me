## MODIFIED Requirements

### Requirement: 删除笔记同步清理所有索引
`delete_note()` SHALL 在删除磁盘文件和 notes collection 的同时，清理 knowledge collection 和 graph RAG 中的对应记录。

- 按 chunk_id `note:<note_id>` 删除 knowledge collection 中的记录
- 按 source `note:<note_id>` 删除 graph RAG entity/relation Qdrant collection 中的记录
- 删除 `<note_id>.graph.json` 文件（若存在）
- 任何单步清理失败 SHALL 被捕获，不阻断其余清理步骤
- 若笔记不存在，返回 False（行为与现在一致）

#### Scenario: 删除已双路索引的笔记
- **WHEN** 调用 `delete_note(note_id)`，该笔记已在 knowledge collection 和 graph RAG 中建立索引
- **THEN** knowledge collection 中 `note:<note_id>` 条目被删除，graph RAG 中 source 为 `note:<note_id>` 的 entity/relation 被删除，graph.json 文件被删除，返回 True

#### Scenario: 删除仅有旧格式索引（仅 notes collection）的笔记
- **WHEN** 笔记仅存在于 notes collection，knowledge collection 和 graph RAG 中无对应记录
- **THEN** 清理静默跳过（无报错），返回 True

#### Scenario: 删除不存在的笔记
- **WHEN** note_id 对应的 .md 文件不存在
- **THEN** 返回 False，不执行任何清理
