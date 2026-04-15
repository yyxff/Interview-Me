"""
工具定义（供 create_react_agent 使用）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.tools import tool

from .state import InterviewState, SESSIONS_DIR, _flat_dict


# ══════════════════════════════════════════════════════════════════════════════
# ④ 工具定义（供 create_react_agent 使用）
# ══════════════════════════════════════════════════════════════════════════════

def _make_plan_tools(state: InterviewState) -> list:
    """
    为 plan 阶段创建工具列表。

    关键模式：工具用闭包捕获 state 里的数据，而不是依赖全局 session 对象。
    这样工具是无状态的纯函数（输入 query → 输出 str），对测试和复用友好。

    @tool 装饰器做了三件事：
      1. 把函数包装成 LangChain BaseTool 对象
      2. 用函数签名和 docstring 生成工具描述（LLM 读这个来决定何时调用）
      3. 支持 function calling API（模型直接输出 JSON 调用，不用解析文本）
    """
    profile = state["profile_text"]

    @tool
    async def search_knowledge(query: str) -> str:
        """搜索知识库，获取与查询相关的技术内容片段。适合查找技术原理、概念定义。"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            import rag  # 复用原有 RAG 模块
            result = rag.retrieve_rich(query)
            if not result:
                return "知识库无相关内容"
            return "\n---\n".join(
                f"[{r.get('source', '?')}]\n{r.get('text', '')}" for r in result
            )
        except Exception as e:
            return f"搜索失败: {e}"

    @tool
    async def search_profile(query: str) -> str:
        """在候选人简历中搜索与查询相关的信息。适合查找候选人的项目经历、技能。"""
        if not profile:
            return "（未提供简历）"
        lines = [ln for ln in profile.splitlines() if query.lower() in ln.lower()]
        return "\n".join(lines[:15]) if lines else "简历中未找到相关内容"

    @tool
    async def search_past_sessions(query: str) -> str:
        """搜索历史面试记录中候选人的薄弱点（低分问题）。"""
        try:
            weak = []
            for p in sorted(SESSIONS_DIR.glob("*.json"))[-5:]:
                data = json.loads(p.read_text(encoding="utf-8"))
                for node in _flat_dict(data.get("tree", [])):
                    if (node.get("score") or 5) <= 2 and query.lower() in (node.get("text") or "").lower():
                        weak.append(f"Q: {node['text'][:80]} (score={node['score']})")
            return "\n".join(weak[:5]) if weak else "未找到相关历史薄弱点"
        except Exception as e:
            return f"查询失败: {e}"

    return [search_knowledge, search_profile, search_past_sessions]


def _make_ask_tools(state: InterviewState) -> list:
    """为 ask 阶段创建工具（只需 search_knowledge）。"""

    @tool
    async def search_knowledge(query: str) -> str:
        """搜索知识库，获取与查询相关的技术内容。出题前可用来了解知识点细节。"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            import rag
            result = rag.retrieve_rich(query)
            if not result:
                return "知识库无相关内容"
            return "\n---\n".join(
                f"[{r.get('source', '?')}]\n{r.get('text', '')}" for r in result
            )
        except Exception as e:
            return f"搜索失败: {e}"

    return [search_knowledge]
