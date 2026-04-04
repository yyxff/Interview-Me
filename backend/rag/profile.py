"""Profile（用户简历/自我介绍，全局持久）"""
from __future__ import annotations

import re
from pathlib import Path

PROFILE_DIR = Path(__file__).parent.parent / "profile"
PROFILE_DIR.mkdir(exist_ok=True)

_PROFILE_FILE = "profile.md"


def save_profile(md_text: str) -> None:
    if not md_text.strip():
        raise ValueError("内容不能为空")
    (PROFILE_DIR / _PROFILE_FILE).write_text(md_text, encoding="utf-8")
    print(f"[profile] 已保存 {len(md_text)} 字")


def get_profile_text() -> str | None:
    path = PROFILE_DIR / _PROFILE_FILE
    return path.read_text(encoding="utf-8") if path.exists() else None


def profile_status() -> dict:
    path = PROFILE_DIR / _PROFILE_FILE
    if not path.exists():
        return {"uploaded": False, "size": 0}
    text     = path.read_text(encoding="utf-8")
    sections = re.findall(r'^##\s+(.+)', text, re.MULTILINE)
    return {"uploaded": True, "size": len(text), "sections": sections}
