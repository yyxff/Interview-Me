"""Tool: search_past_sessions — 在历史面试记录中找薄弱点"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interview_agent import InterviewSession

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"

DESC = (
    "search_past_sessions: 搜索历史面试记录，找出候选人在某个话题上的薄弱点\n"
    '  用法：search_past_sessions("并发控制 锁")'
)


def _flat_dict(nodes: list) -> list:
    result = []
    for n in nodes:
        result.append(n)
        result.extend(_flat_dict(n.get("children", [])))
    return result


def search_past_sessions(query: str, exclude_session_id: str = "") -> str:
    """搜索历史面试记录中的低分问题（score ≤ 2）。"""
    logger.debug("search_past_sessions query=%r", query)
    keywords = query.split()
    results: list[str] = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True)[:5]:
        if exclude_session_id and exclude_session_id[:8] in f.name:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            weak = []
            for node in _flat_dict(data.get("tree", [])):
                score = node.get("score")
                if score and score <= 2 and node.get("text"):
                    blob = node["text"] + node.get("feedback", "")
                    if any(kw in blob for kw in keywords):
                        weak.append(
                            f"  Q: {node['text'][:80]} | {score}/5 | {node.get('feedback', '')[:60]}"
                        )
            if weak:
                results.append(f"[{data.get('saved_at', f.stem)}]:\n" + "\n".join(weak[:3]))
        except Exception:
            pass
    out = "\n\n".join(results) if results else "未找到相关历史薄弱点记录"
    logger.debug("search_past_sessions → %d session(s) matched", len(results))
    return out


# ── 老式接口（供旧版 interview_agent 使用） ───────────────────────────────────
def make(session: "InterviewSession") -> dict:
    async def fn(query: str) -> str:
        return search_past_sessions(query, exclude_session_id=session.session_id)
    return {"desc": DESC, "fn": fn}
