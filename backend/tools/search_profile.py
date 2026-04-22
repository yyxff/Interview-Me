"""Tool: search_profile — 查看候选人简历"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interview_agent import InterviewSession

logger = logging.getLogger(__name__)

DESC = (
    "search_profile: 在候选人简历中搜索相关技术经验\n"
    '  用法：search_profile("MySQL 项目")'
)


def search_profile(profile: str, query: str) -> str:
    """返回候选人完整简历。"""
    logger.debug("search_profile query=%r (%d chars)", query, len(profile))
    return profile


# ── 老式接口（供旧版 interview_agent 使用） ───────────────────────────────────
def make(session: "InterviewSession") -> dict:
    async def fn(query: str) -> str:
        return search_profile(session.profile_text or "", query)
    return {"desc": DESC, "fn": fn}
