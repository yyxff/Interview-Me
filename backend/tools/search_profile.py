"""Tool: search_profile — 在候选人简历中搜索相关经验"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interview_agent import InterviewSession


DESC = (
    "search_profile: 在候选人简历中搜索相关技术经验\n"
    '  用法：search_profile("MySQL 项目")'
)


def make(session: "InterviewSession") -> dict:
    async def fn(query: str) -> str:
        text = session.profile_text or ""
        if not text:
            return "未上传候选人简历"
        keywords = query.split()
        lines = [l for l in text.split("\n") if any(kw in l for kw in keywords)]
        snippet = "\n".join(lines[:20]) if lines else text[:600]
        return snippet[:600]

    return {"desc": DESC, "fn": fn}
