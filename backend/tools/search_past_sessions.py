"""Tool: search_past_sessions — 在历史面试记录中找薄弱点"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interview_agent import InterviewSession


DESC = (
    "search_past_sessions: 搜索历史面试记录，找出候选人在某个话题上的薄弱点\n"
    '  用法：search_past_sessions("并发控制 锁")'
)


def make(session: "InterviewSession") -> dict:
    async def fn(query: str) -> str:
        from interview_agent import SESSIONS_DIR, _flat_dict

        files = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
        results: list[str] = []
        keywords = query.split()
        for f in files[:5]:
            if session.session_id[:8] in f.name:
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
        return "\n\n".join(results) if results else "未找到相关历史薄弱点记录"

    return {"desc": DESC, "fn": fn}
