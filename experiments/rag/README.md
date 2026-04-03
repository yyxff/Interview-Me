# experiments/rag — RAG 检索实验

RAG 检索策略的完整实验目录，与后端工程代码分离。

## 目录结构

```
rag/
  scripts/      # 当前可用的评测脚本
  archive/      # 已归档的历史脚本（有各自 README）
  logs/         # 实验日志（JSON），按类别分子目录（有 README）
  testsets/     # 测试集（有 README）
  report_RAG.md # 完整实验报告，含所有指标数据
```

---

## 前置条件

所有脚本依赖后端服务的 Python 环境和 `.env` 配置：

```bash
# 激活 conda 环境
conda activate interview-me

# 确认后端 .env 已配置（脚本会自动读取）
cat ../../backend/.env

# 确认后端服务已启动（脚本直接调用后端模块，不需要 HTTP 服务）
# 但需要向量库、图数据库等依赖已初始化
```

---

## scripts/ 说明

| 脚本 | 用途 |
|------|------|
| `eval_01_baseline_O.py` | O 类题（事实型）基线评测，测 bi / rrf+rerank / graph 三路，n=94 |
| `eval_02_baseline_GE.py` | GE 类题（关系型）基线评测，含多种变体（rrf_no_rerank / weighted_rrf / graph+rerank 等），n=30；首次运行会调 LLM 生成题目并缓存 |
| `eval_03_combined.py` | O+GE 1:1 混合集综合对比，覆盖所有策略（基线 / rrf_path_rr / Plan A 路由 / LLM 路由），是主实验脚本 |
| `debug_01_inspect_ge.py` | 调试工具，查看单道 GE 题在各路检索下的 top-k 返回详情，用于排查检索失败原因 |

---

## 复现实验

所有脚本从 `scripts/` 目录运行。

### 1. O 类题基线（事实型，n=94）

测试 bi-encoder / rrf+rerank / graph 三路在 O 类题上的表现。

```bash
cd scripts
python eval_01_baseline_O.py
```

日志写入 `logs/09_baseline_O/metrics_o_<timestamp>.json`

---

### 2. GE 类题基线（关系型，n=30）

测试各方法在图谱边关系题上的表现，包含 bi / rrf / rrf_no_rerank / rrf_weighted / graph / graph+rerank。

```bash
cd scripts

# 使用已有测试集（推荐，可复现）
python eval_02_baseline_GE.py --n 30 --top-k 5

# 强制重新用 LLM 生成新题目（会覆盖缓存）
python eval_02_baseline_GE.py --regen
```

日志写入 `logs/10_baseline_GE/graph_edge_<timestamp>.json`

---

### 3. O+GE 综合对比（各 30 题，主实验）

在 O:GE=1:1 混合集上对比所有策略，包含：
- 基线：bi / rrf / graph
- 进阶：graph_path_rr / **rrf_path_rr**（最终采纳方案）
- 路由：Plan A 实体距离路由（多阈值扫描）
- 实验性：LLM 路由（few-shot）

```bash
cd scripts

# 使用默认测试集和参数（n=30，seed=42）
python eval_03_combined.py

# 自定义参数
python eval_03_combined.py --n 20 --seed 123

# 指定测试集路径
python eval_03_combined.py \
  --o-testset ../testsets/testset_O_20260331_224445.json \
  --ge-testset ../testsets/testset_GE_20260402_185222.json
```

日志写入 `logs/11_combined/combined_<timestamp>.json`

---

### 4. 调试：查看单题各路召回详情

```bash
cd scripts

# 随机抽 2 题查看
python debug_01_inspect_ge.py

# 指定题号（从 1 开始）
python debug_01_inspect_ge.py --ids 3 11

# 指定种子和展示 top-k
python debug_01_inspect_ge.py --seed 7 --top-k 5
```

---

## 查看历史日志

```bash
# 查看最新综合实验结果
cat logs/11_combined/combined_20260402_232951.json | python3 -m json.tool

# 查看 O 类基线
cat logs/09_baseline_O/metrics_o_20260402_181139.json | python3 -m json.tool
```

完整指标说明和所有实验数据见 `report_RAG.md`。
