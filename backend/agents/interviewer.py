"""InterviewerAgent：生成面试问题 + self-reflection"""
from __future__ import annotations

import asyncio
from typing import Any

from .models import ThoughtNode, InterviewSession, find
from .react import _REACT_HEADER, react_loop
from .scorer import build_prior_context

_provider: Any = None


def set_provider(p: Any) -> None:
    global _provider
    _provider = p


async def _llm(messages: list[dict], system: str = "", timeout: float = 50.0) -> str:
    if _provider is None:
        raise RuntimeError("LLM provider 未注入")
    return await asyncio.wait_for(_provider.chat(messages, system), timeout=timeout)


def _parse_json(text: str, default: Any) -> Any:
    import json, re
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'[\[\{].*[\]\}]', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return default


_INTERVIEWER_SYS = """\
你是一位专业技术面试官，正在考察候选人。

当前考察任务：{task_text}
出题场景：{question_context}

{profile_section}
规则：
- 问题要有针对性，不重复之前问过的问题
- 语气自然专业，问题长度 1-2 句话
- 如需了解具体知识点细节（如某算法原理、某机制实现），可先搜索知识库

{react_header}

Final Answer 必须是 JSON（不加代码块）：
{{"intent": "考察候选人是否理解进程隔离的底层机制", "question": "进程之间为什么需要隔离内存？操作系统是怎么实现的？"}}"""

_INTERVIEWER_USR_INIT = """\
{prior_context}这是对话的开始，请提出关于「{task_text}」的第一个问题。"""

_INTERVIEWER_USR_FOLLOWUP = """\
上一个问题：{prev_question}
候选人的回答：{prev_answer}
导演指示：{director_focus}"""

_REFLECT_SYS = """\
你是一位严格的面试质量审核员。检查下面这道面试题是否合格：
1. 是否真正考察了出题意图中的知识点？
2. 候选人能否明确理解这道题在问什么？
3. 是否与出题意图匹配，不过宽也不过窄？

如果合格，原样返回这道题。
如果不合格，输出改进后的问题。
只输出最终问题文本，不加任何解释。"""

_REFLECT_USR = """\
出题意图：{intent}
生成的问题：{question}"""


async def interviewer_ask(
    session: InterviewSession,
    parent_node: ThoughtNode,
    director_focus: str = "",
    is_first: bool = False,
) -> tuple[str, str]:
    """ASKING 阶段：ReAct 生成面试问题 + self-reflection。返回 (question_text, intent)。"""
    from tools import build_toolset

    assert session.sm.state == "ASKING"
    task_node = find(session.roots, session.sm.current_task_id)
    task_text = task_node.text if task_node else "（未知）"
    task_type = task_node.task_type if task_node else "knowledge"
    profile_section = f"候选人背景：\n{session.profile_text[:800]}\n" if session.profile_text else ""

    tools      = build_toolset(session, ["search_knowledge"])
    tool_fns   = {n: t["fn"] for n, t in tools.items()}
    tool_descs = "\n".join(t["desc"] for t in tools.values())
    react_header = _REACT_HEADER.format(max_steps=2, tool_descs=tool_descs)

    if is_first:
        prior = build_prior_context(session)
        prior_section    = prior + "\n\n" if prior else ""
        question_context = f"首问，话题开场（task_type={task_type}）"
        user_msg = _INTERVIEWER_USR_INIT.format(task_text=task_text, prior_context=prior_section)
    else:
        question_context = f"追问/转向（导演指示：{director_focus or '无'}）"
        user_msg = _INTERVIEWER_USR_FOLLOWUP.format(
            prev_question=parent_node.text,
            prev_answer=parent_node.answer[:300],
            director_focus=director_focus or "根据上下文自行判断",
        )

    system = _INTERVIEWER_SYS.format(
        task_text=task_text,
        question_context=question_context,
        profile_section=profile_section,
        react_header=react_header,
    )
    raw = await react_loop(system, user_msg, tool_fns, max_steps=2, timeout_per_step=40.0)
    parsed  = _parse_json(raw, default={"intent": "", "question": raw.strip()})
    intent  = parsed.get("intent", "")
    draft_q = parsed.get("question", raw.strip()).strip()

    # Self-reflection
    if intent and draft_q:
        try:
            final_q = await _llm(
                [{"role": "user", "content": _REFLECT_USR.format(intent=intent, question=draft_q)}],
                system=_REFLECT_SYS,
                timeout=20.0,
            )
            final_q = final_q.strip()
        except Exception:
            final_q = draft_q
    else:
        final_q = draft_q

    print(f"[interviewer] intent='{intent[:50]}' draft='{draft_q[:50]}' final='{final_q[:50]}'")
    return final_q, intent
