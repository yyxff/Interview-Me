"""
gen_ge_testset.py — GE 类题（图谱边关系题）测试集生成

从知识图谱中选 A --[predicate]--> B 的边，用 LLM 生成自然语言问题：
"A [predicate] 什么？"（问题只出现 A，不出现 B）
Ground truth = 边的来源 chunk（source_chunk_id）

生成的测试集保存到 ../testsets/testset_GE_<timestamp>.json。
LLM 生成结果缓存到 ../logs/routing_cache/ge_questions.json，避免重复调用。

用法：
  python gen_ge_testset.py --n 30

  # 强制重新生成（清除缓存）
  python gen_ge_testset.py --n 30 --regen
"""

import argparse, asyncio, json, os, random, sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR  = Path(__file__).parent
RAG_DIR      = SCRIPTS_DIR.parent
BACKEND_DIR  = RAG_DIR.parent.parent / "backend"
TESTSETS_DIR = RAG_DIR / "testsets"
CACHE_DIR    = RAG_DIR / "logs" / "routing_cache"
sys.path.insert(0, str(BACKEND_DIR))

CACHE_DIR.mkdir(parents=True, exist_ok=True)
GE_CACHE = CACHE_DIR / "ge_questions.json"

# 加载 .env
env_file = BACKEND_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import rag
import graph_rag as _gr


# ── 候选边收集 ────────────────────────────────────────────────────────────────

BAD_CHUNK_KWS = ["一、前言", "适合什么群体", "要怎么阅读", "质量如何"]

def build_chunk_index() -> dict[str, str]:
    idx = {}
    for f in sorted(rag.KNOWLEDGE_DIR.glob("*.chunks.json")):
        src = f.stem.replace(".chunks", "")
        for i, c in enumerate(json.loads(f.read_text("utf-8"))):
            idx[f"{src}_{i}"] = c.get("text", "")
    return idx

def collect_candidate_edges(chunk_index: dict) -> list[dict]:
    """从所有 .graph.json 中收集高质量跨-chunk 边。"""
    edges = []
    for gf in sorted((_gr.GRAPH_DIR).glob("*.graph.json")):
        g = json.loads(gf.read_text("utf-8"))
        nodes = g.get("nodes", {})
        for e in g.get("edges", []):
            subj, obj, pred = e["subject"], e["object"], e["predicate"]
            rel_desc = e.get("description", "")
            if not rel_desc or len(rel_desc) < 15 or subj == obj:
                continue
            src_cid  = e.get("source_chunk_id", "")
            if not src_cid:
                continue
            src_text = chunk_index.get(src_cid, "")
            if len(src_text) < 150:
                continue
            if any(kw in src_text[:200] for kw in BAD_CHUNK_KWS):
                continue
            edges.append({
                "subject":   subj,
                "predicate": pred,
                "object":    obj,
                "obj_chunk": src_cid,
                "subj_desc": nodes.get(subj, {}).get("description", "")[:80],
                "obj_desc":  nodes.get(obj,  {}).get("description", "")[:80],
                "rel_desc":  rel_desc,
            })
    return edges


# ── LLM 问题生成 ──────────────────────────────────────────────────────────────

_GEN_PROMPT = """\
你是一个面试题出题助手。根据以下知识图谱中的一条关系，出一道面试问题。

关系信息：
  主体（A）：{subject}（{subj_desc}）
  谓词：{predicate}
  客体（B）：{object}（{obj_desc}）
  关系描述：{rel_desc}

要求：
1. 问题必须提到 A（主体），并体现"[predicate]"这个关系或动作
2. 问题**绝对不能**出现 B（客体）的名称或明显暗示 B 的词语
3. 问题要自然，像面试官会问的技术问题
4. 20-50 字，以"？"结尾
5. 只输出问题本身，不要任何前缀或解释

示例（仅格式参考，不要照抄）：
  A=HTTP/2, predicate=解决, B=队头阻塞 → "HTTP/2 引入了哪些机制来解决 HTTP/1.1 中严重影响性能的传输问题？"
  A=QUIC,   predicate=依赖, B=UDP      → "QUIC 协议在传输层选择了哪种底层协议作为基础？"

现在请出题："""


async def _gen_one(edge: dict, client, model: str) -> str:
    prompt = _GEN_PROMPT.format(**{k: edge[k] for k in
                                   ("subject","subj_desc","predicate","object","obj_desc","rel_desc")})
    try:
        from openai import AsyncOpenAI
        resp = await client.chat.completions.create(
            model=model, max_tokens=800, temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content
        if content is None:
            content = str(resp.choices[0].message)
        text = content.strip()
        return "" if edge["object"] in text else text
    except Exception as e:
        print(f"\n  [gen error] {e}")
        return ""


async def generate_questions(edges: list[dict], concurrency: int = 6) -> list[dict]:
    cache: dict = {}
    if GE_CACHE.exists():
        try:
            cache = json.loads(GE_CACHE.read_text("utf-8"))
        except Exception:
            pass

    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key  = os.environ.get("LLM_API_KEY", "x") or "x",
        base_url = os.environ.get("LLM_BASE_URL") or None,
    )
    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
    sem   = asyncio.Semaphore(concurrency)

    need   = [e for e in edges if e["obj_chunk"] not in cache]
    cached = [e for e in edges if e["obj_chunk"] in cache]
    print(f"[gen] 缓存命中 {len(cached)}，需生成 {len(need)} 条")

    async def gen(e):
        async with sem:
            return e, await _gen_one(e, client, model)

    for e, q in await asyncio.gather(*[gen(e) for e in need]):
        if q:
            cache[e["obj_chunk"]] = {"question": q, "edge": e}

    GE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")

    results = []
    for e in edges:
        item = cache.get(e["obj_chunk"])
        if item and item.get("question"):
            results.append({
                "id":                    f"GE::{e['obj_chunk']}",
                "strategy":              "GE",
                "question":              item["question"],
                "ground_truth_chunk_ids":[e["obj_chunk"]],
                "subject":               e["subject"],
                "predicate":             e["predicate"],
                "object":                e["object"],
                "source":                e["obj_chunk"].split("_")[0],
            })
    return results


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=30, help="生成题目数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regen", action="store_true", help="清除缓存，强制重新生成")
    args = parser.parse_args()

    if args.regen and GE_CACHE.exists():
        GE_CACHE.unlink()
        print("[gen] 已清除问题缓存")

    print("[gen] 构建 chunk 索引...")
    chunk_index = build_chunk_index()
    print(f"  {len(chunk_index)} chunks")

    print("[gen] 收集候选边...")
    edges = collect_candidate_edges(chunk_index)
    random.seed(args.seed)
    random.shuffle(edges)

    # 控制谓词多样性：每个谓词最多 5 条
    selected, pred_cnt = [], {}
    for e in edges:
        p = e["predicate"]
        if pred_cnt.get(p, 0) < 5:
            selected.append(e)
            pred_cnt[p] = pred_cnt.get(p, 0) + 1
        if len(selected) >= args.n * 2:
            break

    print(f"  候选边: {len(edges)}  选取: {len(selected)}")
    print(f"  谓词分布: {pred_cnt}")

    print("\n[gen] 生成问题（异步 LLM）...")
    questions = asyncio.run(generate_questions(selected))
    questions = questions[:args.n]
    print(f"  生成 {len(questions)} 题")

    if not questions:
        print("错误：没有生成任何问题，请检查 LLM 配置")
        return

    # 保存测试集
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = TESTSETS_DIR / f"testset_GE_{ts}.json"
    TESTSETS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "description": "GE 策略：图谱边关系题（问 A predicate 什么，GT = source_chunk_id）",
        "n": len(questions),
        "questions": questions,
    }, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n[gen] 测试集已保存: {path.name}")

    # 样题预览
    print("\n── 样题预览 ────────────────────────────────────────────")
    for q in questions[:5]:
        print(f"  [{q['subject']} --{q['predicate']}--> {q['object']}]")
        print(f"  Q: {q['question']}")
        print(f"  GT: {q['ground_truth_chunk_ids'][0]}")
        print()


if __name__ == "__main__":
    main()
