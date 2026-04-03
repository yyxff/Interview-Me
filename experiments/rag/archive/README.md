# archive — 已归档的实验脚本

这些脚本在实验阶段发挥了作用，但最终未被采纳为生产策略或已被更新版本取代。
保留在此供回溯参考，日志见 `../logs/` 各对应子目录。

---

| 脚本 | 对应 logs 子目录 | 实验内容 | 归档原因 |
|------|-----------------|----------|----------|
| `eval_01_baseline_O.py` | `09_baseline_O/` | O 类题精准基线，测 bi/rrf/graph | 被 `eval.py --dataset O` 替代 |
| `eval_02_baseline_GE.py` | `10_baseline_GE/` | GE 类题基线（含 LLM 问题生成），测多种检索变体 | 生成部分拆入 `gen_ge_testset.py`，评测被 `eval.py --dataset GE` 替代 |
| `eval_03_combined.py` | `11_combined/` | O+GE 混合综合对比，含 Plan A 阈值扫描 | 被 `eval.py --dataset combined` 替代 |
| `eval_retrieval.py` | `01_early_baseline/` | 最早期基线：用 chunk 原文做 query | 测试集构造粗糙，已被专项测试集替代 |
| `eval_graph_testsets.py` | `03_graph_eval/` | Graph RAG 专项：测 GA/GB 两类题型，对比 bi/rrf/graph | 测试集设计后来被 GE 策略重新定义取代 |
| `eval_mixed_og.py` | `04_mixed_og/` | O+GA+GB 混合集早期版本 | 被 eval_03_combined.py 替代（GE 重新定义） |
| `eval_routing.py` | `05_routing_v1/` | LLM 路由 v1：零样本 + 少样本分类 | 路由准确率天花板低，不可靠 |
| `eval_routing_v2.py` | `06_routing_v2/` | Plan A 实体距离路由 + Plan C 置信度路由早期实现 | Plan A 整合进 `methods.py` 和 `eval.py`；Plan C 未采纳 |
| `eval_graph_rerank.py` | `07_graph_rerank/` | graph BFS → cross-encoder rerank，对比 graph-only 和 oracle 上界 | reranker 在跨实体查询上反效果，问题根源在此确认；被 rrf_path_rr 解决 |
| `eval_hyde.py` | `08_hyde/` | HyDE：LLM 生成假想答案做 graph 实体检索 | 对 GE 无明显提升，不采纳 |
| `debug_01_inspect_ge.py` | — | 逐题查看各路召回的 chunk 内容和 GT 排名 | 功能合并进 `eval.py --debug` |
