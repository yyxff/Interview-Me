# experiments/rag — RAG 检索实验

RAG 检索策略的完整实验目录，与后端工程代码分离。

## 目录结构

```
rag/
  scripts/      # 当前可用的脚本
  archive/      # 已归档的历史脚本（有 README）
  logs/         # 实验日志（JSON），按类别分子目录（有 README）
  testsets/     # 测试集（有 README）
  report.md     # 完整实验报告，含所有指标数据
```

---

## 前置条件

```bash
conda activate interview-me

# 脚本会自动读取 backend/.env，确认已配置
# 向量库、图数据库等依赖需已初始化（backend 正常可用即可）
```

---

## scripts/ 说明

| 脚本 | 用途 |
|------|------|
| `methods.py` | 所有检索方法实现，供 `eval.py` import 使用，不直接运行 |
| `eval.py` | 评测主脚本，通过 `--dataset` 和 `--methods` 参数灵活选择；`--debug` 可逐题查看召回详情 |
| `gen_ge_testset.py` | 生成 GE 类测试集（调 LLM，结果缓存在 logs/routing_cache/） |

---

## 复现实验

所有脚本从 `scripts/` 目录运行。

### 评测：eval.py

通过 `--dataset` 选测试集，`--methods` 选检索方法：

```bash
cd scripts

# O 类题（事实型），测默认 4 种方法
python eval.py --dataset O

# GE 类题（关系型），测所有方法
python eval.py --dataset GE --methods all

# O+GE 混合，测核心方法（最常用）
python eval.py --dataset combined --methods bi rrf graph rrf_path_rr

# 加入 Plan A 路由对比（会额外加载知识图谱，慢一些）
python eval.py --dataset combined --methods bi rrf rrf_path_rr plan_a --threshold 0.26

# 自定义题数、种子、测试集路径
python eval.py --dataset GE --methods bi graph rrf_path_rr --n 20 --seed 7 \
  --ge-testset ../testsets/testset_GE_20260402_185222.json
```

**可选方法（--methods）：**

| 方法名 | 说明 |
|--------|------|
| `bi` | 纯 bi-encoder 向量检索 |
| `rrf` | bi+graph RRF → cross-encoder rerank（旧基线） |
| `graph` | 纯 graph BFS，不做 rerank |
| `rrf_nr` | bi+graph RRF，跳过 rerank |
| `rrf_w` | 加权 RRF（graph 权重更高），不做 rerank |
| `graph_rr` | graph BFS → cross-encoder rerank |
| `hyde` | HyDE：LLM 生成假设答案后做 bi 检索 |
| `hyde_rrf_w` | HyDE bi + graph → 加权 RRF |
| `graph_path_rr` | graph BFS → 路径文本前置 → rerank |
| `rrf_path_rr` | bi+graph RRF → 路径文本前置 → rerank ★生产方案 |
| `routed` | LLM 路由：factual→rrf，relational→graph |
| `plan_a` | 实体距离路由：近邻实体有边→graph，否则→rrf |
| `all` | 以上全部 |

日志写入 `logs/eval_<dataset>_<timestamp>.json`

---

### 生成新 GE 测试集：gen_ge_testset.py

```bash
cd scripts

# 生成 30 题（优先使用缓存）
python gen_ge_testset.py --n 30

# 强制重新生成（清除 LLM 缓存）
python gen_ge_testset.py --n 30 --regen
```

测试集保存到 `testsets/testset_GE_<timestamp>.json`

---

### Debug：逐题查看召回详情

加 `--debug` 即可，不写日志，默认只看 2 题：

```bash
cd scripts

# 随机 2 题，看 bi / graph / rrf_path_rr 各路的 chunk 内容和 GT 排名
python eval.py --dataset GE --methods bi graph rrf_path_rr --debug

# 指定题号
python eval.py --dataset GE --methods bi graph rrf_path_rr --debug --ids 3 11

# 看 1 题，更多方法
python eval.py --dataset GE --methods all --debug --n 1
```

---

## 查看历史日志

```bash
# 最终采纳方案的完整实验结果
cat logs/11_combined/combined_20260402_232951.json | python3 -m json.tool

# O 类基线
cat logs/09_baseline_O/metrics_o_20260402_181139.json | python3 -m json.tool
```

完整指标说明和所有实验数据见 `report.md`。
