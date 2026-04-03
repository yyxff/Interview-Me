# eval_archive — 已归档的实验脚本

这些脚本在实验阶段发挥了作用，但最终未被采纳为生产策略或已被更新版本取代。
保留在此供回溯参考，日志见 `../eval_logs/` 各对应子目录。

---

| 脚本 | 对应 eval_logs 子目录 | 实验内容 | 归档原因 |
|------|----------------------|----------|----------|
| `eval_retrieval.py` | `01_early_baseline/` | 最早期基线：从 chunks 抽 30% 原文做查询，测 bi/rrf/graph 三路 Hit@1/3 和 MRR | 测试集构造粗糙（用原文做 query），已被专项测试集替代 |
| `eval_graph_testsets.py` | `03_graph_eval/` | Graph RAG 专项：测 GA（单边场景化题）和 GB（二跳路径题）两类题型，对比 bi/rrf/graph | 测试集设计和 GT 标注方式后来被 eval_02_baseline_GE.py 重新设计取代 |
| `eval_mixed_og.py` | `04_mixed_og/` | O+GA+GB 混合集早期版本：1:1 比例采样，统一评分规则 | 被更精简的 eval_03_combined.py 替代（GE 重新定义为边级别） |
| `eval_routing.py` | `05_routing_v1/` | LLM 路由 v1：零样本 + 少样本 LLM 分类，将查询路由到 rrf 或 graph | 路由准确率天花板低（GE 和 O 句式相同），结论：LLM 文本分类不可靠 |
| `eval_routing_v2.py` | `06_routing_v2/` | 实体距离路由（Plan A）+ 置信度自适应路由（Plan C）的早期实现 | Plan A 最终在 eval_03_combined.py 重新实现并做了阈值扫描；Plan C 未采纳 |
| `eval_graph_rerank.py` | `07_graph_rerank/` | graph BFS 召回 → cross-encoder rerank，对比 graph-only 和 oracle 上界 | reranker 在跨实体查询（GE）上反效果，问题根源在此被确认；被 rrf_path_rr 解决 |
| `eval_hyde.py` | `08_hyde/` | HyDE：LLM 生成假想答案，用假想答案向量做 graph 实体检索 | 对 GE 无明显提升（GE 答案实体 LLM 未必能猜中），不采纳 |
