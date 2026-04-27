"""
Build a labeled testset for score_node evaluation.

Calls Gemini Flash to generate (question, answer, gold_score, gold_reasoning) tuples
across 7 case types, then saves to testsets/score_node.jsonl.

Usage:
    python3 build_testset.py --out testsets/score_node.jsonl [--per-case 5]

Review the output manually and correct outliers before running eval_score_node.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from gemini_client import flash  # noqa: E402

SYSTEM_PROMPT = """\
你是技术面试评分专家。你的任务是生成面试评分测试案例。

评分标准（1-5分）：
5分：准确完整，有深度，能说明原理并举例
4分：覆盖核心要点，表述清晰，基本无遗漏
3分：方向正确但有明显遗漏，或表述模糊
2分：只了解表面概念，细节错误或不知道原理
1分：答非所问，或完全不了解

对每个案例输出 JSON（不加代码块）：
{
  "question": "面试问题",
  "answer": "候选人的回答（模拟真实口语，不要太书面）",
  "gold_score": 3,
  "gold_reasoning": "1)候选人提到了... 2)遗漏了... 3)综合给分X"
}"""

CASE_PROMPTS: dict[str, str] = {
    "perfect":
        "生成一个技术面试问答案例，候选人回答准确完整有深度，应得5分。"
        "话题：Redis/MySQL/操作系统/网络/分布式系统 中随机选一个。",
    "poor":
        "生成一个技术面试问答案例，候选人回答完全不了解或答非所问，应得1-2分。"
        "话题：Redis/MySQL/操作系统/网络/分布式系统 中随机选一个。",
    "partial":
        "生成一个技术面试问答案例，候选人方向正确但有明显遗漏，应得3分。"
        "这是最容易漂移的区间，请确保遗漏点明确。",
    "detail_error":
        "生成一个技术面试问答案例，候选人整体理解正确但某个关键细节有错误，应得2-3分。"
        "请在 gold_reasoning 中明确指出错误点。",
    "off_topic":
        "生成一个技术面试问答案例，候选人答案听起来流畅但完全答非所问，应得1分。"
        "候选人的答案要听起来自信，但和问题毫不相关。",
    "obscure":
        "生成一个技术面试问答案例，问题考察相对冷门的知识点（知识库可能没有标准答案），"
        "候选人回答只了解皮毛，应得2分。话题建议：JVM GC细节/Raft算法细节/Linux内核参数。",
    "length_pair":
        "生成一个技术面试问答案例，候选人回答质量为3分（方向对但不完整）。"
        "答案要比较简短（2-3句话），因为我们要测试长短答案对评分的影响。",
}


def generate_case(case_type: str, prompt: str) -> dict | None:
    full_prompt = f"{prompt}\n\n输出单个 JSON 对象。"
    try:
        raw = flash(full_prompt, system=SYSTEM_PROMPT, json_mode=True)
        data = json.loads(raw)
        data["case_type"] = case_type
        # Validate required fields
        for key in ("question", "answer", "gold_score", "gold_reasoning"):
            if key not in data:
                print(f"  [warn] missing field '{key}' in {case_type} case, skipping")
                return None
        if not (1 <= int(data["gold_score"]) <= 5):
            print(f"  [warn] invalid gold_score={data['gold_score']}, skipping")
            return None
        return data
    except Exception as e:
        print(f"  [error] {case_type}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="testsets/score_node.jsonl")
    parser.add_argument("--per-case", type=int, default=5,
                        help="Number of examples per case type (default: 5, min 30 total)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for case_type, prompt in CASE_PROMPTS.items():
        print(f"Generating {args.per_case}x '{case_type}'...")
        for i in range(args.per_case):
            record = generate_case(case_type, prompt)
            if record:
                records.append(record)
                print(f"  [{i+1}/{args.per_case}] score={record['gold_score']} ✓")

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(records)} records saved to {out_path}")
    print("Case distribution:")
    from collections import Counter
    for ct, n in sorted(Counter(r["case_type"] for r in records).items()):
        print(f"  {ct}: {n}")
    print("\nNext: review the file manually, then run eval_score_node.py")


if __name__ == "__main__":
    main()
