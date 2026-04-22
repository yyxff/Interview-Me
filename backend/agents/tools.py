"""
工具定义（供 create_react_agent 使用）

设计：每个 tool 单独定义一次，_make_xxx_tools 通过组合来声明各 node 的权限。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.tools import tool

from .state import InterviewState, SESSIONS_DIR, _flat_dict


# ── 单个 Tool 工厂 ────────────────────────────────────────────────────────────

def _search_knowledge_tool():
    @tool
    async def search_knowledge(query: str) -> str:
        """搜索知识库，获取与查询相关的技术内容片段。适合查找技术原理、概念定义。"""
        print(f"[tool:search_knowledge] query={query!r}")
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            import rag
            result = rag.retrieve_rich(query)
            if not result:
                out = "知识库无相关内容"
                print(f"[tool:search_knowledge] → {out}")
                return out
            out = "\n---\n".join(
                f"[{r.get('source', '?')}]\n{r.get('text', '')}" for r in result
            )
            print(f"[tool:search_knowledge] → {len(result)} chunks, preview: {out[:200]!r}")
            return out
        except Exception as e:
            out = f"搜索失败: {e}"
            print(f"[tool:search_knowledge] → ERROR: {e}")
            return out
    return search_knowledge


def _search_profile_tool(profile: str):
    @tool
    async def search_profile(query: str) -> str:
        """在候选人简历中搜索与查询相关的信息。适合查找候选人的项目经历、技能。"""
        print(f"[tool:search_profile] query={query!r}")
        if not profile:
            out = "（未提供简历）"
            print(f"[tool:search_profile] → {out}")
            return out
        lines = [ln for ln in profile.splitlines() if query.lower() in ln.lower()]
        out = "\n".join(lines[:15]) if lines else "简历中未找到相关内容"
        print(f"[tool:search_profile] → {len(lines)} lines matched, preview: {out[:200]!r}")
        return out
    return search_profile


def _search_past_sessions_tool():
    @tool
    async def search_past_sessions(query: str) -> str:
        """搜索历史面试记录中候选人的薄弱点（低分问题）。"""
        print(f"[tool:search_past_sessions] query={query!r}")
        try:
            weak = []
            for p in sorted(SESSIONS_DIR.glob("*.json"))[-5:]:
                data = json.loads(p.read_text(encoding="utf-8"))
                for node in _flat_dict(data.get("tree", [])):
                    if (node.get("score") or 5) <= 2 and query.lower() in (node.get("text") or "").lower():
                        weak.append(f"Q: {node['text'][:80]} (score={node['score']})")
            out = "\n".join(weak[:5]) if weak else "未找到相关历史薄弱点"
            print(f"[tool:search_past_sessions] → {len(weak)} weak items found, preview: {out[:200]!r}")
            return out
        except Exception as e:
            out = f"查询失败: {e}"
            print(f"[tool:search_past_sessions] → ERROR: {e}")
            return out
    return search_past_sessions


def _explore_concept_tool():
    @tool
    async def explore_concept(entity: str) -> str:
        """
        联想工具：给定一个技术概念关键词，在知识图谱中做 BFS，
        返回与该概念直接相关的邻居概念、关系描述。
        适合发现关联知识点，决定下一步考察哪个方向。
        示例：explore_concept("GC") → 返回 Stop-the-World、引用计数、G1 等关联概念。
        """
        print(f"[tool:explore_concept] entity={entity!r}")
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from graph_rag.retrieval import explore_concept_bfs
            result = explore_concept_bfs(entity)
            out = result["summary"]
            print(f"[tool:explore_concept] → preview: {out[:200]!r}")
            return out
        except Exception as e:
            out = f"图谱查询失败: {e}"
            print(f"[tool:explore_concept] → ERROR: {e}")
            return out
    return explore_concept


# ── Node 工具组合 ─────────────────────────────────────────────────────────────

def _make_plan_tools(state: InterviewState) -> list:
    """plan：知识库 + 简历 + 历史薄弱点，制定完整考察计划。"""
    return [
        _search_knowledge_tool(),
        _search_profile_tool(state["profile_text"]),
        _search_past_sessions_tool(),
    ]


def _make_ask_tools(state: InterviewState) -> list:
    """ask：知识库 + 简历 + 历史薄弱点 + 图谱联想，出有针对性的题。"""
    return [
        _search_knowledge_tool(),
        _search_profile_tool(state["profile_text"]),
        _search_past_sessions_tool(),
        _explore_concept_tool(),
    ]


def _make_score_tools(state: InterviewState) -> list:
    """score：仅知识库，获取标准答案作为评分参考。"""
    return [
        _search_knowledge_tool(),
    ]
