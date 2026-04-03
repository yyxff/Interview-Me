"""
图谱 RAG 测试集生成脚本

策略:
  GA — 单边场景化问法
      取图谱边 (subject --predicate--> object)，用 LLM 生成不直接出现实体名的场景题。
      GT: 该边的 source_chunk_id（单 chunk 命中）

  GB — 二跳路径跳过中间节点
      取路径 A --r1--> B --r2--> C（A/C 在不同 chunk），问题提 A 和 C 但不提 B。
      GT: e1 + e2 两条边的 source_chunk_id（两个 chunk 都要命中）

  GC — 双模态互补题（RRF 测试集）
      C1 来自 merged_O.json（bi 容易命中），C2 通过图谱 1 跳可达（bi 难命中）。
      问题显式提及 C1 的核心概念（触发 bi 检索），隐藏连接实体（逼 graph 上场）。
      GT: [C1, C2]，Full=1.0 / Partial=0.5 / Miss=0，理论上 RRF 应该赢。

数据存放:
  eval_logs/testsets/
    testset_GA_<timestamp>.json   — 本次生成的 GA 题
    testset_GB_<timestamp>.json   — 本次生成的 GB 题
    testset_GC_<timestamp>.json   — 本次生成的 GC 题
    graph_registry.json           — 去重索引，key = edge_id 或 path_id

运行:
  conda run -n interview-me python3 gen_graph_testsets.py [--ga 50] [--gb 20] [--gc 50] [--dry-run]
  再次运行会跳过 registry 中已生成的 edge/path，安全追加。
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

GRAPH_DIR     = BACKEND_DIR / "graph"
TESTSETS_DIR  = BACKEND_DIR / "eval_logs" / "testsets"
REGISTRY_FILE = TESTSETS_DIR / "graph_registry.json"
TESTSETS_DIR.mkdir(parents=True, exist_ok=True)

# predicate 白名单——对区分度有意义的关系类型
INTERESTING_PREDICATES = {
    "对比", "依赖", "触发", "解决", "优化", "替代", "实现", "导致",
    "基于", "使用", "采用", "避免", "防止", "保证", "确保", "控制", "减少",
}

# ── LLM 调用（复用 .env 配置，openai-compatible）─────────────────────────────

def _make_llm_client():
    """从环境变量构造 OpenAI client（支持 openai-compatible）。"""
    # 尝试读 .env
    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    provider  = os.environ.get("LLM_PROVIDER", "openai-compatible")
    api_key   = os.environ.get("LLM_API_KEY", "")
    base_url  = os.environ.get("LLM_BASE_URL", "")
    model     = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

    from openai import AsyncOpenAI
    if provider == "anthropic":
        # anthropic 也可通过 openai-compatible 接口调
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.anthropic.com/v1")
    elif base_url:
        client = AsyncOpenAI(api_key=api_key or "placeholder", base_url=base_url)
    else:
        client = AsyncOpenAI(api_key=api_key)
    return client, model


async def llm_generate(client, model: str, prompt: str) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4000,
    )
    raw = resp.choices[0].message.content.strip()
    # 尝试从 JSON 中解析 question 字段
    try:
        # 去掉可能的 markdown 代码块
        text = raw.strip("` \n")
        if text.startswith("json"):
            text = text[4:].strip()
        obj = json.loads(text)
        return obj.get("question", raw).strip()
    except Exception:
        return raw


# ── 图数据加载 ────────────────────────────────────────────────────────────────

def load_edges() -> list[dict]:
    """加载所有图谱的边，过滤到有意义的 predicate，补充 relation_id。"""
    edges = []
    for f in sorted(GRAPH_DIR.glob("*.graph.json")):
        g = json.loads(f.read_text(encoding="utf-8"))
        source = g.get("source", f.stem)
        nodes  = g.get("nodes", {})
        for edge in g.get("edges", []):
            pred = edge.get("predicate", "")
            if pred not in INTERESTING_PREDICATES:
                continue
            # 补充节点描述
            subj_desc = edge.get("subject_desc") or nodes.get(edge.get("subject", ""), {}).get("description", "")
            obj_desc  = edge.get("object_desc")  or nodes.get(edge.get("object",  ""), {}).get("description", "")
            edges.append({
                "relation_id":     edge.get("relation_id", ""),
                "subject":         edge.get("subject", ""),
                "subject_desc":    subj_desc[:60],
                "predicate":       pred,
                "object":          edge.get("object", ""),
                "object_desc":     obj_desc[:60],
                "description":     edge.get("description", "")[:100],
                "source_chunk_id": edge.get("source_chunk_id", ""),
                "source":          source,
            })
    return edges


def find_gc_pairs(edges: list[dict], chunk_to_questions: dict | None = None) -> list[dict]:
    """
    找 GC 对：
    - C1: 图谱边 e1 的 source_chunk（问题里会显式提到 e1.subject，bi 能命中）
    - C2: 通过连接实体 O 1 跳可达的不同 chunk（问题隐藏 O，bi 难以直接命中）
    结构: S --r1--> O（source=C1），O --r2--> X（source=C2）
    """
    if chunk_to_questions is None:
        chunk_to_questions = {}

    edges_by_subject: dict[str, list] = defaultdict(list)
    for e in edges:
        edges_by_subject[e["subject"]].append(e)

    pairs = []
    seen: set[str] = set()

    for e1 in edges:
        C1 = e1["source_chunk_id"]
        if not C1:
            continue

        O = e1["object"]  # 连接实体（隐藏）

        for e2 in edges_by_subject.get(O, []):
            C2 = e2["source_chunk_id"]
            if not C2 or C2 == C1:
                continue

            pair_id = f"{e1['relation_id']}::{e2['relation_id']}"
            if pair_id in seen:
                continue
            seen.add(pair_id)

            pairs.append({
                "pair_id":              pair_id,
                "C1":                   C1,
                "C2":                   C2,
                "connecting_entity":    O,
                "e1":                   e1,
                "e2":                   e2,
                "c1_sample_questions":  chunk_to_questions.get(C1, [])[:2],
            })

    return pairs


def find_two_hop_paths(edges: list[dict]) -> list[dict]:
    """
    找跨 chunk 的二跳路径 A->B->C。
    返回每条路径的 dict，含两条边的完整信息。
    """
    edge_by_subject = defaultdict(list)
    for e in edges:
        edge_by_subject[e["subject"]].append(e)

    paths = []
    seen  = set()
    for e1 in edges:
        B = e1["object"]
        for e2 in edge_by_subject.get(B, []):
            C = e2["object"]
            # 去掉环、同 chunk（同 chunk 用向量就够了）
            if C == e1["subject"]:
                continue
            if e1["source_chunk_id"] == e2["source_chunk_id"]:
                continue
            if not e1["source_chunk_id"] or not e2["source_chunk_id"]:
                continue
            path_id = f"{e1['relation_id']}::{e2['relation_id']}"
            if path_id in seen:
                continue
            seen.add(path_id)
            paths.append({
                "path_id":    path_id,
                "A":          e1["subject"],
                "A_desc":     e1["subject_desc"],
                "r1":         e1["predicate"],
                "B":          e1["object"],
                "B_desc":     e1["object_desc"],
                "r2":         e2["predicate"],
                "C":          e2["object"],
                "C_desc":     e2["object_desc"],
                "e1_chunk_id": e1["source_chunk_id"],
                "e2_chunk_id": e2["source_chunk_id"],
                "e1_desc":    e1["description"],
                "e2_desc":    e2["description"],
                "source":     e1["source"],
            })
    return paths


# ── Prompt 模板 ───────────────────────────────────────────────────────────────

def build_prompt_ga(edge: dict) -> str:
    return f"""\
你是一个资深技术面试官，正在设计一道考察候选人深度理解的面试题。

已知以下知识图谱关系：
  主体：{edge['subject']}（{edge['subject_desc']}）
  关系：{edge['predicate']}
  客体：{edge['object']}（{edge['object_desc']}）
  关系说明：{edge['description']}

请生成一道技术面试题，要求：
1. 题目中不直接出现"{edge['subject']}"或"{edge['object']}"这两个词
2. 以"场景/现象/问题"切入，例如"当...时，为什么..."或"...是如何做到..."
3. 完整回答必须同时涉及主体和客体才算完整
4. 面向 2-3 年经验的后端/系统方向候选人
5. 必须是以"？"结尾的完整疑问句，不超过 60 字
6. 只输出题目本身，不要任何解释和前缀

输出格式：仅输出一个 JSON 对象，不要任何额外文字：
{{"question": "完整的面试题，以？结尾"}}

示例（仅示意格式，不要照抄内容）：
{{"question": "多核CPU中，某个核心写入数据后，其他核心如何得知缓存已失效？"}}"""


def build_prompt_gc(pair: dict) -> str:
    e1 = pair["e1"]
    e2 = pair["e2"]
    O  = pair["connecting_entity"]
    c1_q = pair["c1_sample_questions"][0] if pair["c1_sample_questions"] else ""
    ref_line = f'\n  参考问法示例（仅供理解 A 的知识范围，不要照抄）：{c1_q}' if c1_q else ""
    return f"""\
你是一个资深技术面试官，需要设计一道同时考察两个关联知识点的面试题。

【知识点 A】（语义搜索容易命中，无需你提供关键词，候选人背景知识即可触发）
  核心概念：{e1['subject']}（{e1['subject_desc']}）{ref_line}

【A 与 B 的关联路径】（B 是通过关系才能找到的知识，语义搜索难以直接命中）
  {e1['subject']} --[{e1['predicate']}]--> {O}（{e1['object_desc']}）
    说明：{e1['description']}
  {O} --[{e2['predicate']}]--> {e2['object']}（{e2['object_desc']}）
    说明：{e2['description']}

【知识点 B】
  核心概念：{e2['object']}（{e2['object_desc']}）

生成一道面试题，要求：
1. 题目中可以提到"{e1['subject']}"（A 的核心概念），帮助语义检索命中知识点 A
2. 不能直接提到"{O}"（隐含中间连接概念，让图谱检索发挥作用找到 B）
3. 完整回答必须同时用到知识点 A 和 B 才算完整
4. 问法是"为什么"、"如何"或"什么条件下"，考察理解，不要问"什么是"
5. 面向 2-3 年经验的后端/系统方向候选人
6. 完整疑问句，以"？"结尾，不超过 60 字

输出格式：仅输出一个 JSON 对象：
{{"question": "完整的面试题，以？结尾"}}

示例（仅示意格式，不要照抄内容）：
{{"question": "TCP 三次握手后，为什么操作系统还需要维护 TIME_WAIT 状态才能保证后续连接的数据不被污染？"}}"""


def build_prompt_gb(path: dict) -> str:
    return f"""\
你是一个资深技术面试官，正在设计一道考察候选人多跳推理能力的面试题。

已知以下二跳知识路径：
  {path['A']}（{path['A_desc']}）
    --[{path['r1']}]-->
  {path['B']}（{path['B_desc']}）
    --[{path['r2']}]-->
  {path['C']}（{path['C_desc']}）

补充说明：
  第一跳：{path['e1_desc']}
  第二跳：{path['e2_desc']}

请生成一道面试题，要求：
1. 题目中可以提到"{path['A']}"和"{path['C']}"，但绝对不能出现"{path['B']}"（B 是隐含的中间概念）
2. 完整回答需要经过 B 才能解释 A 和 C 的关联
3. 问法是"为什么"或"如何"，考察理解深度，不要问"什么是"
4. 面向 2-3 年经验的后端/系统方向候选人
5. 必须是以"？"结尾的完整疑问句，不超过 60 字
6. 只输出题目本身，不要任何解释和前缀

输出格式：仅输出一个 JSON 对象，不要任何额外文字：
{{"question": "完整的面试题，以？结尾"}}

示例（仅示意格式，不要照抄内容）：
{{"question": "边缘触发模式下，为什么文件描述符必须设为非阻塞才能避免进程永久阻塞？"}}"""


# ── Registry ─────────────────────────────────────────────────────────────────

def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {}


def save_registry(reg: dict) -> None:
    REGISTRY_FILE.write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── 生成主流程 ────────────────────────────────────────────────────────────────

async def generate_ga(edges: list[dict], n: int, registry: dict,
                      client, model: str, dry_run: bool) -> list[dict]:
    """生成 GA 策略题目，跳过已在 registry 中的 edge。"""
    # 过滤已生成
    todo = [e for e in edges if f"GA::{e['relation_id']}" not in registry and e["relation_id"]]
    random.shuffle(todo)
    todo = todo[:n]

    questions = []
    for i, edge in enumerate(todo):
        key = f"GA::{edge['relation_id']}"
        print(f"  [GA {i+1}/{len(todo)}] {edge['subject']} --{edge['predicate']}--> {edge['object']}", flush=True)
        if dry_run:
            q_text = f"[DRY-RUN] 关于{edge['subject']}和{edge['object']}的题目"
        else:
            try:
                prompt  = build_prompt_ga(edge)
                q_text  = await llm_generate(client, model, prompt)
            except Exception as exc:
                print(f"    ⚠ LLM 失败: {exc}")
                continue

        print(f"    → {q_text}")
        item = {
            "id":                   key,
            "strategy":             "GA",
            "question":             q_text,
            "ground_truth_chunk_ids": [edge["source_chunk_id"]],
            "source_edge": {
                "subject":         edge["subject"],
                "subject_desc":    edge["subject_desc"],
                "predicate":       edge["predicate"],
                "object":          edge["object"],
                "object_desc":     edge["object_desc"],
                "description":     edge["description"],
                "source_chunk_id": edge["source_chunk_id"],
                "relation_id":     edge["relation_id"],
            },
            "source":               edge["source"],
        }
        questions.append(item)
        registry[key] = {
            "strategy": "GA",
            "relation_id": edge["relation_id"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "llm_model": model,
        }
    return questions


async def generate_gb(paths: list[dict], n: int, registry: dict,
                      client, model: str, dry_run: bool) -> list[dict]:
    """生成 GB 策略题目，跳过已在 registry 中的路径。"""
    todo = [p for p in paths if f"GB::{p['path_id']}" not in registry]
    random.shuffle(todo)
    todo = todo[:n]

    questions = []
    for i, path in enumerate(todo):
        key = f"GB::{path['path_id']}"
        print(f"  [GB {i+1}/{len(todo)}] {path['A']} --{path['r1']}--> {path['B']} --{path['r2']}--> {path['C']}", flush=True)
        if dry_run:
            q_text = f"[DRY-RUN] {path['A']}和{path['C']}通过什么机制关联？"
        else:
            try:
                prompt = build_prompt_gb(path)
                q_text = await llm_generate(client, model, prompt)
            except Exception as exc:
                print(f"    ⚠ LLM 失败: {exc}")
                continue

        print(f"    → {q_text}")
        item = {
            "id":       key,
            "strategy": "GB",
            "question": q_text,
            "ground_truth_chunk_ids": list(dict.fromkeys(
                [path["e1_chunk_id"], path["e2_chunk_id"]]
            )),
            "path": {
                "A":          path["A"],
                "A_desc":     path["A_desc"],
                "r1":         path["r1"],
                "B":          path["B"],
                "B_desc":     path["B_desc"],
                "r2":         path["r2"],
                "C":          path["C"],
                "C_desc":     path["C_desc"],
                "e1_chunk_id": path["e1_chunk_id"],
                "e2_chunk_id": path["e2_chunk_id"],
            },
            "source": path["source"],
        }
        questions.append(item)
        registry[key] = {
            "strategy": "GB",
            "path_id":  path["path_id"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "llm_model": model,
        }
    return questions


async def generate_gc(pairs: list[dict], n: int, registry: dict,
                      client, model: str, dry_run: bool) -> list[dict]:
    """生成 GC 策略题目（双模态互补），跳过已在 registry 中的 pair。"""
    todo = [p for p in pairs if f"GC::{p['pair_id']}" not in registry]
    random.shuffle(todo)
    todo = todo[:n]

    questions = []
    for i, pair in enumerate(todo):
        key = f"GC::{pair['pair_id']}"
        e1, e2, O = pair["e1"], pair["e2"], pair["connecting_entity"]
        print(
            f"  [GC {i+1}/{len(todo)}] {e1['subject']} --{e1['predicate']}--> "
            f"{O} --{e2['predicate']}--> {e2['object']}",
            flush=True,
        )
        if dry_run:
            q_text = f"[DRY-RUN] {e1['subject']}在{e2['predicate']}场景中如何影响{e2['object']}？"
        else:
            try:
                prompt  = build_prompt_gc(pair)
                q_text  = await llm_generate(client, model, prompt)
            except Exception as exc:
                print(f"    ⚠ LLM 失败: {exc}")
                continue

        print(f"    → {q_text}")
        item = {
            "id":       key,
            "strategy": "GC",
            "question": q_text,
            "ground_truth_chunk_ids": list(dict.fromkeys([pair["C1"], pair["C2"]])),
            "pair": {
                "C1":               pair["C1"],
                "C2":               pair["C2"],
                "connecting_entity": O,
                "e1_subject":        e1["subject"],
                "e1_predicate":      e1["predicate"],
                "e1_object":         O,
                "e1_description":    e1["description"],
                "e2_subject":        O,
                "e2_predicate":      e2["predicate"],
                "e2_object":         e2["object"],
                "e2_description":    e2["description"],
            },
        }
        questions.append(item)
        registry[key] = {
            "strategy":     "GC",
            "pair_id":      pair["pair_id"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "llm_model":    model,
        }
    return questions


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ga",      type=int, default=50,    help="GA 题数上限")
    parser.add_argument("--gb",      type=int, default=44,    help="GB 题数上限（受路径数限制）")
    parser.add_argument("--gc",      type=int, default=0,     help="GC 题数上限（双模态互补）")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",     help="不调 LLM，只打印路径")
    args = parser.parse_args()

    random.seed(args.seed)

    print("[gen] 加载图谱数据 ...")
    edges = load_edges()
    paths = find_two_hop_paths(edges)
    print(f"  有意义边: {len(edges)}  |  2-hop 路径: {len(paths)}")

    # GC: 可选地从 merged_O 加载参考问法（丰富 prompt 上下文，不作为过滤条件）
    gc_pairs: list[dict] = []
    if args.gc > 0:
        chunk_to_questions: dict[str, list] = defaultdict(list)
        merged_o_file = TESTSETS_DIR / "merged_O.json"
        if merged_o_file.exists():
            merged_o = json.loads(merged_o_file.read_text(encoding="utf-8"))
            for q in merged_o.get("questions", []):
                cid = q.get("chunk_id", "")
                if cid:
                    chunk_to_questions[cid].append(q.get("question", ""))
        gc_pairs = find_gc_pairs(edges, chunk_to_questions)
        print(f"  GC 候选对: {len(gc_pairs)}")

    registry = load_registry()
    print(f"  已生成 (registry): {len(registry)} 条\n")

    client, model = _make_llm_client()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── GA ───────────────────────────────────────────────────────────────────
    print(f"=== 生成 GA（单边场景化，目标 {args.ga} 题）===")
    ga_questions = await generate_ga(edges, args.ga, registry, client, model, args.dry_run)

    if ga_questions:
        ga_file = TESTSETS_DIR / f"testset_GA_{ts}.json"
        ga_file.write_text(json.dumps({
            "meta": {
                "strategy":       "GA",
                "strategy_name":  "图谱单边场景化",
                "n_questions":    len(ga_questions),
                "seed":           args.seed,
                "timestamp":      datetime.now().isoformat(timespec="seconds"),
                "llm_model":      model,
                "note": (
                    "取图谱边(subject--predicate-->object)，LLM 生成不直接出现实体名的场景题。"
                    "GT=单条边的 source_chunk_id，命中算法：检索结果包含该 chunk_id 即命中。"
                ),
            },
            "questions": ga_questions,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  GA 已写入 {ga_file}  ({len(ga_questions)} 题)")
    else:
        ga_file = None
        print("\n  GA: 无新题生成（均已在 registry 中）")

    # ── GB ───────────────────────────────────────────────────────────────────
    print(f"\n=== 生成 GB（二跳路径，目标 {args.gb} 题，实际可用 {len(paths)} 条路径）===")
    gb_questions = await generate_gb(paths, args.gb, registry, client, model, args.dry_run)

    if gb_questions:
        gb_file = TESTSETS_DIR / f"testset_GB_{ts}.json"
        gb_file.write_text(json.dumps({
            "meta": {
                "strategy":       "GB",
                "strategy_name":  "图谱二跳跳过中间节点",
                "n_questions":    len(gb_questions),
                "seed":           args.seed,
                "timestamp":      datetime.now().isoformat(timespec="seconds"),
                "llm_model":      model,
                "note": (
                    "取路径 A--r1-->B--r2-->C（跨 chunk），题目提 A/C 不提 B。"
                    "GT=两条边各自的 source_chunk_id，全命中才算满分（partial credit 算 0.5）。"
                ),
            },
            "questions": gb_questions,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  GB 已写入 {gb_file}  ({len(gb_questions)} 题)")
    else:
        gb_file = None
        print("\n  GB: 无新题生成（均已在 registry 中）")

    # ── GC ───────────────────────────────────────────────────────────────────
    gc_questions: list[dict] = []
    gc_file = None
    if args.gc > 0 and gc_pairs:
        print(f"\n=== 生成 GC（双模态互补，目标 {args.gc} 题，候选 {len(gc_pairs)} 对）===")
        gc_questions = await generate_gc(gc_pairs, args.gc, registry, client, model, args.dry_run)

        if gc_questions:
            gc_file = TESTSETS_DIR / f"testset_GC_{ts}.json"
            gc_file.write_text(json.dumps({
                "meta": {
                    "strategy":      "GC",
                    "strategy_name": "双模态互补题",
                    "n_questions":   len(gc_questions),
                    "seed":          args.seed,
                    "timestamp":     datetime.now().isoformat(timespec="seconds"),
                    "llm_model":     model,
                    "note": (
                        "C1 来自 merged_O（bi 容易命中），C2 通过图谱 1 跳可达（bi 难命中）。"
                        "问题显式提及 C1 核心概念，隐藏连接实体，逼图谱检索找到 C2。"
                        "GT=[C1,C2]，Full=1.0/Partial=0.5/Miss=0；理论上 RRF 应当胜出。"
                    ),
                },
                "questions": gc_questions,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n  GC 已写入 {gc_file}  ({len(gc_questions)} 题)")
        else:
            print("\n  GC: 无新题生成（均已在 registry 中）")

    # ── 更新 registry ────────────────────────────────────────────────────────
    save_registry(registry)

    # ── 汇总 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"GA: {len(ga_questions)} 题  |  GB: {len(gb_questions)} 题  |  GC: {len(gc_questions)} 题")
    print(f"Registry 已更新: {REGISTRY_FILE}")
    if ga_file:
        print(f"GA 文件: {ga_file.name}")
    if gb_file:
        print(f"GB 文件: {gb_file.name}")
    if gc_file:
        print(f"GC 文件: {gc_file.name}")


if __name__ == "__main__":
    asyncio.run(main())
