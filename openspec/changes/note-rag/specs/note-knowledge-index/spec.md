## ADDED Requirements

### Requirement: 笔记索引到 knowledge collection
`index_note()` 在完成 notes collection 索引后，SHALL 将整篇笔记作为单个 chunk 写入 knowledge collection。

- chunk_id 格式为 `note:<note_id>`
- `source` 字段为 `"笔记"`，`chapter` 字段为笔记标题
- 若同 chunk_id 已存在，先删除再写入（幂等）

#### Scenario: 新笔记首次索引
- **WHEN** 调用 `index_note(note_id, title, text)`
- **THEN** knowledge collection 中出现 id 为 `note:<note_id>` 的条目，text 为完整笔记内容，source 为 `"笔记"`

#### Scenario: 重新索引已存在笔记
- **WHEN** 对同一 note_id 再次调用 `index_note()`
- **THEN** knowledge collection 中只保留一条该笔记的记录（无重复）

#### Scenario: RAG 不可用时
- **WHEN** `is_available()` 返回 False
- **THEN** `index_note()` 静默跳过 knowledge collection 写入，不抛出异常
