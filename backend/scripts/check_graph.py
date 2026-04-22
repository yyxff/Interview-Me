#!/usr/bin/env python3
"""
图 RAG 查询脚本：输入查询词，显示命中的节点和边信息。

用法：
  python check_graph.py "进程调度"
  python check_graph.py            # 进入交互模式
  python check_graph.py --top 8 "FCFS和RR的区别"
"""

import sys
import argparse

# ── 颜色输出 ────────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def h(text, color=BOLD):
    return f"{color}{text}{RESET}"


def run_query(query: str, entity_top_k: int, relation_top_k: int):
    import graph_rag
    import rag as rag_mod

    # 临时覆盖 top-k
    orig_ent = graph_rag.ENTITY_TOP_K
    orig_rel = graph_rag.RELATION_TOP_K
    graph_rag.ENTITY_TOP_K   = entity_top_k
    graph_rag.RELATION_TOP_K = relation_top_k

    try:
        result = graph_rag.retrieve_graph(query)
    finally:
        graph_rag.ENTITY_TOP_K   = orig_ent
        graph_rag.RELATION_TOP_K = orig_rel

    entities   = result.get("entities", [])
    relations  = result.get("relations", [])
    chunk_ids  = result.get("source_chunk_ids", [])
    summary    = result.get("graph_summary", "")

    # ── 实体 ──────────────────────────────────────────────────────────────
    print(f"\n{h('═══ 命中实体', CYAN)} ({len(entities)} 个)")
    if not entities:
        print(f"  {DIM}（无）{RESET}")
    for i, e in enumerate(entities, 1):
        name   = e.get("name", "")
        etype  = e.get("entity_type", "")
        desc   = e.get("description", "")
        cids   = e.get("source_chunk_ids", [])
        print(f"  {h(f'[{i}]', BOLD)} {h(name, BOLD)}  {DIM}[{etype}]{RESET}")
        # description 里可能已含 "name（type）：desc" 格式，直接截断显示
        short_desc = desc.split("：", 1)[-1] if "：" in desc else desc
        if short_desc:
            print(f"      {DIM}{short_desc[:120]}{RESET}")
        if cids:
            print(f"      来源 chunk: {', '.join(cids[:3])}"
                  + (f" … 共{len(cids)}个" if len(cids) > 3 else ""))

    # ── 关系 ──────────────────────────────────────────────────────────────
    print(f"\n{h('═══ 命中关系', YELLOW)} ({len(relations)} 条)")
    if not relations:
        print(f"  {DIM}（无）{RESET}")
    for i, r in enumerate(relations, 1):
        subj  = r.get("subject", "")
        pred  = r.get("predicate", "")
        obj   = r.get("object", "")
        desc  = r.get("description", "")
        cid   = r.get("source_chunk_id", "")
        print(f"  {h(f'[{i}]', BOLD)} {h(subj, BOLD)} "
              f"--{h(pred, YELLOW)}--> {h(obj, BOLD)}")
        # 截取关系 description 中 ：后的内容
        short_desc = desc.split("：", 1)[-1] if "：" in desc else desc
        if short_desc:
            print(f"      {DIM}{short_desc[:120]}{RESET}")
        if cid:
            print(f"      来源 chunk: {cid}")

    # ── 图谱摘要 ──────────────────────────────────────────────────────────
    if summary:
        print(f"\n{h('═══ 图谱摘要', GREEN)}")
        for line in summary.splitlines():
            print(f"  {line}")

    # ── 关联 chunk ids ─────────────────────────────────────────────────────
    print(f"\n{h('═══ 关联 chunk IDs', DIM)} ({len(chunk_ids)} 个)")
    if chunk_ids:
        for cid in chunk_ids[:10]:
            print(f"  {DIM}{cid}{RESET}")
        if len(chunk_ids) > 10:
            print(f"  {DIM}… 还有 {len(chunk_ids) - 10} 个{RESET}")
    else:
        print(f"  {DIM}（无）{RESET}")


def main():
    parser = argparse.ArgumentParser(description="图 RAG 查询工具")
    parser.add_argument("query", nargs="?", default=None, help="查询词（省略则进入交互模式）")
    parser.add_argument("--top-entity", type=int, default=5, metavar="N",
                        help="实体 top-k（默认 5）")
    parser.add_argument("--top-rel", type=int, default=3, metavar="N",
                        help="关系 top-k（默认 3）")
    args = parser.parse_args()

    # 导入时已在 backend/ 下，确保路径正确
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    if args.query:
        run_query(args.query, args.top_entity, args.top_rel)
    else:
        print(f"{h('图 RAG 交互查询', CYAN)}  （输入 q 退出）")
        while True:
            try:
                q = input("\n> 查询: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in ("q", "quit", "exit"):
                break
            run_query(q, args.top_entity, args.top_rel)


if __name__ == "__main__":
    main()
