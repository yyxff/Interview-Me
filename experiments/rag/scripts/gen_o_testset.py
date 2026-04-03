"""
gen_testset.py — 用 LLM 生成 RAG 评估测试集（支持增量生成）

从 .chunks.json 采样 chunk，让 LLM 生成问题，输出与 eval_qa_retrieval.py 兼容的测试集。
问题由 LLM 根据 chunk 内容生成，但不是简单复述，而是经过语义抽象——比直接用 .qa.json 更接近真实查询。

三种 prompt 策略（--strategy A/B/C）：
  A  面试官盲问：不引用原文词汇，模拟真实面试提问风格
  B  关键词回避：先提取核心术语，再要求问题不出现这些词
  C  场景化提问：把知识点转化为实际工作遇到的问题

增量生成：
  每个 (chunk_id, strategy) 组合只生成一次，结果记录在 registry.json。
  扩大规模时（如从 50 增到 200）只生成新的 chunk，已有数据不重复消耗 API。
  --force 强制重新生成所有（包括已有的）。

运行方式：
  # 首次生成 50 个 chunk 的测试集（策略 A）
  LLM_API_KEY=sk-... python3 gen_testset.py --strategy A --n 50

  # 之后扩大到 200 个，只会生成新的 150 个
  LLM_API_KEY=sk-... python3 gen_testset.py --strategy A --n 200

  # 查看已生成情况
  python3 gen_testset.py --list

  # 使用 OpenAI-compatible（如 DeepSeek）
  LLM_PROVIDER=openai-compatible LLM_BASE_URL=https://api.deepseek.com/v1 \\
  LLM_API_KEY=sk-... LLM_MODEL=deepseek-chat \\
  python3 gen_testset.py --strategy A --n 50

  # 使用 Ollama（本地，无需 key）
  LLM_PROVIDER=openai-compatible LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5 \\
  python3 gen_testset.py --strategy A --n 20

数据文件（均在 eval_logs/testsets/）：
  registry.json                     — 全局注册表，记录每个 (chunk_id, strategy) 的生成记录
  testset_{strategy}_{ts}.json      — 单次运行新增的问题（可叠加）
  merged_{strategy}.json            — 某策略所有历史问题的合并视图（--merge 时生成）
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 路径解析 ─────────────────────────────────────────────────────────────────
_SCRIPT_BACKEND = Path(__file__).parent


def _resolve_backend_dir(cli_arg: str | None) -> Path:
    if cli_arg:
        return Path(cli_arg).resolve()
    cwd = Path.cwd()
    if (cwd / "chroma_db").exists():
        return cwd
    return _SCRIPT_BACKEND


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--backend-dir", default=None)
_known, _ = _pre.parse_known_args()
BACKEND_DIR = _resolve_backend_dir(_known.backend_dir)
sys.path.insert(0, str(BACKEND_DIR))

KNOWLEDGE_DIR = BACKEND_DIR / "knowledge"
TESTSET_DIR = BACKEND_DIR / "eval_logs" / "testsets"
TESTSET_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_PATH = TESTSET_DIR / "registry.json"


# ── Registry（增量去重的核心）────────────────────────────────────────────────

def load_registry() -> dict:
    """
    Registry 结构：
    {
      "chunk_id::strategy": {
        "chunk_id": str,
        "strategy": str,
        "n_questions": int,
        "testset_file": str,       # 问题存储在哪个文件
        "timestamp": str,
        "llm_model": str
      },
      ...
    }
    """
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_registry(registry: dict) -> None:
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def registry_key(chunk_id: str, strategy: str) -> str:
    return f"{chunk_id}::{strategy}"


def get_processed_ids(registry: dict, strategy: str) -> set[str]:
    """返回该策略下已处理过的 chunk_id 集合。"""
    return {
        v["chunk_id"]
        for k, v in registry.items()
        if v.get("strategy") == strategy
    }


# ── Prompt 策略 ───────────────────────────────────────────────────────────────

STRATEGIES = {
    "O": {
        "name": "原始QA生成",
        "description": "与 rag.py 索引时完全相同的 prompt，作为对照基线",
        "system": "你是一位技术面试题生成助手，根据知识点生成有代表性、不同问法的面试问题。",
        "prompt": "根据以下技术内容，生成{n}个不同角度的面试问题。\n只输出问题本身，每行一个，不加序号和任何前缀符号。\n\n内容：\n{chunk}",
    },
    "A": {
        "name": "面试官盲问",
        "description": "模拟面试官不看原文直接提问，禁止引用原文标题/术语，考语义理解",
        "prompt": """你是一位资深技术面试官，正在考察候选人对以下知识点的掌握程度。

请根据下面的技术内容，生成 {n} 个面试问题。

要求：
1. 用面试官的自然口吻提问，**不能直接引用原文中的标题、章节名、或特定专有名词**
2. 可以对概念重新描述，比如不说"MESI协议"，改问"多核CPU如何保证各核心缓存的数据一致性"
3. 每个问题的答案必须能从原文中找到
4. 问题难度适中，能区分理解深浅
5. 直接输出问题，每行一个，不要编号、不要其他格式

技术内容：
{chunk}""",
    },

    "B": {
        "name": "关键词回避",
        "description": "先提取原文核心术语，再要求问题不出现这些词，强制语义泛化",
        "prompt": """你需要为以下技术内容生成测试问题，用于评估信息检索系统。

第一步：从原文提取 5-8 个核心专有名词（如协议名、算法名、数据结构名等）。
第二步：生成 {n} 个问题，**问题中必须回避这些专有名词**，改用更通用的描述。

例如：原文有"LRU算法" → 不问"LRU算法是什么"，改问"当内存不足时，操作系统如何决定淘汰哪个内存页？"

要求：
- 问题的答案能从原文找到
- 用抽象/通用的角度提问，而非直接引用术语
- 输出格式：先输出提取的关键词（一行，逗号分隔），然后空一行，再输出问题列表（每行一个）

技术内容：
{chunk}""",
    },

    "C": {
        "name": "场景化提问",
        "description": "把知识点转化为实际工作/故障排查场景，最贴近真实用户查询",
        "prompt": """将以下技术内容转化为实际工作场景中的提问，生成 {n} 个问题。

要求：
1. 以实际遇到的问题/现象来提问，例如：
   - "系统出现了 XX 现象，可能是什么原因？怎么排查？"
   - "在做 XX 需求时，为什么要用 XX 方法而不是 XX？"
   - "如果不用 XX 机制，会发生什么问题？"
2. 不要暴露原文的章节标题
3. 答案能从原文中找到
4. 每行一个问题，不要编号

技术内容：
{chunk}""",
    },
}


# ── LLM 客户端 ────────────────────────────────────────────────────────────────

async def call_llm(prompt: str, system: str | None = None) -> str:
    provider_name = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")
    base_url = os.environ.get("LLM_BASE_URL") or None

    if provider_name == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        kwargs = dict(model=model or "claude-3-5-haiku-20241022", max_tokens=1024,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        resp = await client.messages.create(**kwargs)
        return resp.content[0].text

    elif provider_name in ("openai", "openai-compatible"):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key or "ollama", base_url=base_url)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await client.chat.completions.create(
            model=model or "gpt-4o-mini", max_tokens=1024, messages=messages,
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"未知 LLM_PROVIDER: {provider_name}")


# ── 解析 LLM 输出 ─────────────────────────────────────────────────────────────

def parse_questions(raw: str, strategy: str) -> list[str]:
    lines = raw.strip().splitlines()
    if strategy == "B":
        blank_idx = next((i for i, l in enumerate(lines) if not l.strip()), None)
        if blank_idx is not None:
            lines = lines[blank_idx + 1:]

    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for prefix in ["- ", "* ", "Q:", "问："]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        line = re.sub(r"^\d+[\.、\)]\s*", "", line)
        if len(line) > 5:
            questions.append(line)
    return questions


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_chunks(min_len: int = 100) -> list[dict]:
    chunks = []
    for f in sorted(KNOWLEDGE_DIR.glob("*.chunks.json")):
        stem = f.name.replace(".chunks.json", "")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[load] 跳过 {f.name}: {e}")
            continue
        for idx, chunk in enumerate(data):
            text = chunk.get("text", "").strip()
            if len(text) >= min_len:
                chunks.append({
                    "chunk_id":   f"{stem}_{idx}",
                    "chunk_text": text,
                    "source":     stem,
                })
    return chunks


# ── --list 命令 ───────────────────────────────────────────────────────────────

def cmd_list():
    registry = load_registry()
    if not registry:
        print("Registry 为空，尚未生成任何测试集。")
        return

    from collections import defaultdict
    by_strategy = defaultdict(list)
    for v in registry.values():
        by_strategy[v["strategy"]].append(v)

    for strategy in sorted(by_strategy):
        entries = by_strategy[strategy]
        n_chunks = len(entries)
        n_questions = sum(e["n_questions"] for e in entries)
        models = set(e.get("llm_model", "?") for e in entries)
        latest = max(e["timestamp"] for e in entries)
        print(f"策略 {strategy} ({STRATEGIES[strategy]['name']})")
        print(f"  chunk 数: {n_chunks}  问题数: {n_questions}  最新: {latest}")
        print(f"  模型: {', '.join(models)}")
        print()


# ── --merge 命令 ──────────────────────────────────────────────────────────────

def cmd_merge(strategy: str):
    """合并某策略的所有历史问题到一个文件，供 eval 脚本使用。"""
    registry = load_registry()
    all_questions = []
    seen_files = set()

    for v in registry.values():
        if v.get("strategy") != strategy:
            continue
        fpath = TESTSET_DIR / v["testset_file"]
        if fpath in seen_files or not fpath.exists():
            continue
        seen_files.add(fpath)
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            all_questions.extend(data.get("questions", []))
        except Exception as e:
            print(f"[merge] 跳过 {fpath.name}: {e}")

    out = TESTSET_DIR / f"merged_{strategy}.json"
    result = {
        "meta": {
            "strategy": strategy,
            "strategy_name": STRATEGIES[strategy]["name"],
            "n_questions": len(all_questions),
            "source_files": [f.name for f in seen_files],
            "merged_at": datetime.now().isoformat(timespec="seconds"),
        },
        "questions": all_questions,
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] 策略 {strategy}：{len(all_questions)} 题 → {out}")
    return out


# ── 主生成流程 ────────────────────────────────────────────────────────────────

async def generate(args):
    strategy_cfg = STRATEGIES[args.strategy]
    registry = load_registry()
    processed_ids = get_processed_ids(registry, args.strategy)

    print(f"[gen] 策略: {args.strategy} — {strategy_cfg['name']}")
    print(f"[gen] {strategy_cfg['description']}")

    all_chunks = load_chunks(min_len=args.min_len)
    print(f"[gen] 知识库共 {len(all_chunks)} 个有效 chunk（≥{args.min_len} 字）")

    if args.force:
        print(f"[gen] --force 模式：忽略 registry，重新生成所有")
        candidates = all_chunks
    else:
        skipped = [c for c in all_chunks if c["chunk_id"] in processed_ids]
        candidates = [c for c in all_chunks if c["chunk_id"] not in processed_ids]
        if skipped:
            print(f"[gen] 已跳过 {len(skipped)} 个已处理 chunk（使用 --force 可重新生成）")

    if len(candidates) == 0:
        print(f"[gen] 没有新 chunk 需要处理。当前策略 {args.strategy} 已有 {len(processed_ids)} 个 chunk。")
        if args.merge:
            cmd_merge(args.strategy)
        return

    random.seed(args.seed)
    need = max(0, args.n - len(processed_ids))  # 还需要多少个新 chunk
    if not args.force and need <= 0:
        print(f"[gen] 目标 {args.n} 个 chunk 已满足（已有 {len(processed_ids)} 个），无需生成。")
        print(f"[gen] 若要扩大规模，请增大 --n 参数。")
        if args.merge:
            cmd_merge(args.strategy)
        return

    sample_size = need if not args.force else args.n
    sample = random.sample(candidates, min(sample_size, len(candidates)))
    print(f"[gen] 目标: {args.n} 个 chunk，已有: {len(processed_ids)}，本次新增: {len(sample)} 个")
    print(f"[gen] 每 chunk {args.q} 题，预计新增 ~{len(sample) * args.q} 题\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"testset_{args.strategy}_{ts}.json"
    out_path = TESTSET_DIR / out_filename

    new_questions = []
    failed = 0
    t0 = time.time()

    for i, chunk in enumerate(sample):
        elapsed = time.time() - t0
        eta = (elapsed / i * (len(sample) - i)) if i > 0 else 0
        eta_str = f"ETA {eta/60:.1f}min" if i > 0 else "..."
        print(
            f"[{i+1:>3}/{len(sample)}] {eta_str}  {chunk['source'][:20]}"
            f"  chunk={chunk['chunk_id'].rsplit('_', 1)[-1]}",
            end="  ",
        )

        prompt = strategy_cfg["prompt"].format(
            n=args.q,
            chunk=chunk["chunk_text"][:1500],
        )
        system = strategy_cfg.get("system")

        try:
            raw = await call_llm(prompt, system=system)
            questions = parse_questions(raw, args.strategy)
            print(f"→ {len(questions)} 题")

            for q in questions:
                new_questions.append({
                    "question":   q,
                    "chunk_id":   chunk["chunk_id"],
                    "chunk_text": chunk["chunk_text"],
                    "source":     chunk["source"],
                    "strategy":   args.strategy,
                })

            # 立即写入 registry（断点续跑安全）
            rkey = registry_key(chunk["chunk_id"], args.strategy)
            registry[rkey] = {
                "chunk_id":    chunk["chunk_id"],
                "strategy":    args.strategy,
                "n_questions": len(questions),
                "testset_file": out_filename,
                "timestamp":   datetime.now().isoformat(timespec="seconds"),
                "llm_provider": os.environ.get("LLM_PROVIDER", "anthropic"),
                "llm_model":   os.environ.get("LLM_MODEL", "default"),
            }
            save_registry(registry)

        except Exception as e:
            print(f"→ 失败: {e}")
            failed += 1
            if args.fail_fast:
                raise

        if args.delay > 0:
            await asyncio.sleep(args.delay)

    elapsed = round(time.time() - t0, 1)
    print(f"\n[gen] 完成：{len(new_questions)} 题，来自 {len(sample) - failed}/{len(sample)} 个 chunk，耗时 {elapsed}s")

    # 写本次输出文件
    output = {
        "meta": {
            "strategy":      args.strategy,
            "strategy_name": strategy_cfg["name"],
            "n_chunks":      len(sample) - failed,
            "n_questions":   len(new_questions),
            "seed":          args.seed,
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "llm_provider":  os.environ.get("LLM_PROVIDER", "anthropic"),
            "llm_model":     os.environ.get("LLM_MODEL", "default"),
        },
        "questions": new_questions,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen] 本次数据 → {out_path}")

    # 可选：自动合并
    if args.merge:
        cmd_merge(args.strategy)

    # 打印 registry 汇总
    all_processed = get_processed_ids(registry, args.strategy)
    print(f"[gen] Registry 累计：策略 {args.strategy} 共 {len(all_processed)} 个 chunk 已处理")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="用 LLM 生成 RAG 评估测试集（支持增量）")
    parser.add_argument("--strategy",    choices=["O", "A", "B", "C"], default="O",
                        help="Prompt 策略（A=面试官盲问, B=关键词回避, C=场景化，默认 A）")
    parser.add_argument("--n",           type=int, default=50,
                        help="目标 chunk 总数（含已有），默认 50；扩大时只生成新增部分")
    parser.add_argument("--q",           type=int, default=3,
                        help="每个 chunk 生成题数（默认 3）")
    parser.add_argument("--seed",        type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("--min-len",     type=int, default=100,
                        help="chunk 最短字数（默认 100）")
    parser.add_argument("--delay",       type=float, default=0.5,
                        help="每次 API 调用后等待秒数（默认 0.5）")
    parser.add_argument("--force",       action="store_true",
                        help="忽略 registry，强制重新生成所有 chunk")
    parser.add_argument("--merge",       action="store_true",
                        help="生成完成后自动合并所有历史数据到 merged_{strategy}.json")
    parser.add_argument("--list",        action="store_true",
                        help="查看 registry 中已生成情况，不生成新数据")
    parser.add_argument("--fail-fast",   action="store_true",
                        help="遇到 LLM 错误立即退出")
    parser.add_argument("--backend-dir", default=None,
                        help="后端目录（已在启动时解析）")
    args = parser.parse_args()

    if args.list:
        cmd_list()
        return

    asyncio.run(generate(args))


if __name__ == "__main__":
    main()
