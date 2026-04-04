"""通用 ReAct 循环引擎"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Awaitable

_REACT_HEADER = """\
你可以使用以下工具（最多 {max_steps} 次）：
{tool_descs}

每步格式（二选一）：
① 使用工具：
Thought: <推理过程>
Action: <工具名>
Action Input: <查询字符串>

② 直接给出结果（不需要工具时）：
Final Answer: <最终输出>

工具结果会以 Observation: <结果> 形式返回。用完工具次数后必须给 Final Answer。
"""

_REACT_FORCE_FINAL = "你已用完工具调用次数，现在必须给出 Final Answer。"

# 注入 provider（由 agents/__init__.py 统一注入）
_provider: Any = None


def set_provider(p: Any) -> None:
    global _provider
    _provider = p


async def react_loop(
    init_system: str,
    init_user: str,
    tools: dict[str, Callable[[str], Awaitable[str]]],
    max_steps: int = 3,
    timeout_per_step: float = 40.0,
) -> str:
    """
    通用 ReAct 循环。
    init_system 应包含 _REACT_HEADER（调用方格式化好后传入）。
    返回 Final Answer 的文本。
    """
    msgs: list[dict] = [{"role": "user", "content": init_user}]

    for step in range(max_steps + 1):
        if step == max_steps:
            msgs.append({"role": "user", "content": _REACT_FORCE_FINAL})

        raw = await asyncio.wait_for(_provider.chat(msgs, init_system), timeout_per_step)
        msgs.append({"role": "assistant", "content": raw})

        if "Final Answer:" in raw:
            return raw.split("Final Answer:", 1)[1].strip()

        if step == max_steps:
            return raw.strip()

        action_m = re.search(r"Action:\s*(\w+)", raw)
        input_m  = re.search(r"Action Input:\s*[\"']?(.*?)[\"']?\s*$", raw, re.MULTILINE)
        if not action_m:
            return raw.strip()

        tool_name  = action_m.group(1).strip()
        tool_input = input_m.group(1).strip() if input_m else ""

        if tool_name in tools:
            try:
                observation = await asyncio.wait_for(tools[tool_name](tool_input), 20.0)
            except Exception as e:
                observation = f"工具调用失败: {e}"
        else:
            observation = f"未知工具: {tool_name}"

        observation = observation[:600]
        msgs.append({"role": "user", "content": f"Observation: {observation}"})
        print(f"[react] step={step} tool={tool_name} obs={observation[:80]}")

    return raw.strip()
