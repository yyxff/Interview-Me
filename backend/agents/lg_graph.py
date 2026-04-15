"""
LangGraph 版面试 Agent
======================

将原来的 手写状态机 + 手写ReAct循环 替换为 LangGraph 标准模式。
新文件，原来的代码完全保留不动。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangGraph 五个核心概念（贯穿本文件）：

  ① State (TypedDict)
     所有节点共享的"公告板"，节点读取状态、返回更新字典。
     Annotated[list, add_messages] 告诉框架用"追加"而非"覆盖"更新消息。

  ② Node（普通 async 函数）
     每个 Agent 就是 async def node(state: State) -> dict。
     只需返回"本轮要变更的字段"，其余字段框架自动保留。

  ③ interrupt()
     在节点内调用 interrupt(value) → 图暂停、状态持久化、value 返回给调用者。
     下次用 Command(resume=answer) 调用图时，从 interrupt() 行继续执行。

  ④ create_react_agent
     LangGraph 内置 ReAct 子图，用模型原生 function calling 替代手写文本解析。
     自动完成：LLM 调用 → 工具调用 → 观测 → LLM → … 直到模型停止调用工具。

  ⑤ Conditional Edge（条件边）
     add_conditional_edges(node, fn, map)：fn(state) 返回 key，map 映射到下一个节点。
     实现动态路由，替代原来 orchestrator.py 里的大段 if-else。

  ⑥ MemorySaver（Checkpointer）
     每步执行后自动持久化整个 state。
     用 thread_id 标识会话，跨 HTTP 请求恢复状态，无需手动管理 _sessions 字典。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

图结构示意：

  START → [plan] → [ask] ──interrupt()──▶ 等待用户回答
                     ▲                          │
                     │              Command(resume=answer)
                     │                          ▼
                     │                       [score]
                     │                          │
                     │                       [decide]
                     │                          │
                     └──── "ask" ───────────────┘
                                                │
                                             "end"
                                                ▼
                                              END
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt  # noqa: F401  (Command 供路由层使用)
from typing_extensions import TypedDict

# 复用原有的树操作工具（纯 Python 逻辑，与状态机无关）
from agents.models import (
    ThoughtNode,
    add_planned_nodes,
    find,
    flat,
    next_pending_task,
    tree_to_dict,
)

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# ① State 定义
# ══════════════════════════════════════════════════════════════════════════════

class InterviewState(TypedDict):
    """
    LangGraph State = 节点间共享的"公告板"（TypedDict）。

    每个节点返回一个 dict，只包含"本轮要更新的字段"：
      - 普通字段：直接覆盖（最新值生效）
      - messages：Annotated + add_messages → 框架自动"追加"而非覆盖

    为什么用 TypedDict 而不是 dataclass？
      → LangGraph 需要能序列化/反序列化 state（持久化到 checkpointer），
        TypedDict 里只放 JSON 兼容的类型，ThoughtNode 对象存成 dict。
    """
    # ── 会话基本信息 ──────────────────────────────────────────────────
    session_id: str
    jd: str
    direction: str
    profile_text: str

    # ── 思维树（序列化为 dict，方便 JSON 持久化）─────────────────────
    roots_data: list[dict]           # ThoughtNode 的完整 dict 表示
    current_task_id: str | None      # 当前正在考察的任务节点 id
    current_question_id: str | None  # 当前最新问题节点 id

    # ── 上一轮结果（节点间传递数据）──────────────────────────────────
    last_score: int | None
    last_verdict: str | None         # deepen | pivot | back_up | pass
    last_sub_questions: list[str]    # 导演规划的子问题方向
    last_director_reasoning: str     # 导演的推理过程（供下一轮 ask 参考）

    # ── 对话历史 ─────────────────────────────────────────────────────
    # Annotated + add_messages：每次节点返回 messages 时框架自动追加
    # 而不是覆盖，这是 LangGraph 最常用的 Reducer 模式
    messages: Annotated[list[BaseMessage], add_messages]


# ══════════════════════════════════════════════════════════════════════════════
# 内部工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _node_to_dict(n: ThoughtNode) -> dict:
    """完整序列化 ThoughtNode（不截断 answer，供 state 存储）。"""
    return {
        "id": n.id, "node_type": n.node_type, "text": n.text,
        "answer": n.answer, "depth": n.depth, "status": n.status,
        "score": n.score, "verdict": n.verdict, "feedback": n.feedback,
        "reasoning": n.reasoning, "director_note": n.director_note,
        "question_intent": n.question_intent, "task_type": n.task_type,
        "summary": n.summary, "parent_id": n.parent_id,
        "children": [_node_to_dict(c) for c in n.children],
    }


def _dict_to_node(d: dict) -> ThoughtNode:
    """从 dict 还原 ThoughtNode（递归）。"""
    children = [_dict_to_node(c) for c in d.get("children", [])]
    return ThoughtNode(
        id=d["id"], node_type=d["node_type"], text=d["text"],
        answer=d.get("answer", ""), depth=d.get("depth", 0),
        status=d.get("status", "pending"), score=d.get("score"),
        verdict=d.get("verdict"), feedback=d.get("feedback", ""),
        reasoning=d.get("reasoning", ""), director_note=d.get("director_note", ""),
        question_intent=d.get("question_intent", ""), task_type=d.get("task_type", ""),
        summary=d.get("summary", ""), children=children,
        parent_id=d.get("parent_id"),
    )


def _flat_dict(nodes: list[dict]) -> list[dict]:
    """将 dict 格式的树展平。"""
    out: list[dict] = []
    q = list(nodes)
    while q:
        n = q.pop(0)
        out.append(n)
        q.extend(n.get("children", []))
    return out


def _parse_json(text: str, default: Any) -> Any:
    """宽松解析 LLM 输出的 JSON（去掉代码块标记）。"""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"[\[\{].*[\]\}]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return default


# ══════════════════════════════════════════════════════════════════════════════
# LLM 工厂（读取与原系统完全相同的环境变量）
# ══════════════════════════════════════════════════════════════════════════════

def _build_llm():
    """
    根据环境变量返回 LangChain LLM 对象。

    LangChain LLM 与原来自定义 LLMProvider 的区别：
      - 支持 .ainvoke([messages])   ← 直接调用
      - 支持 .bind_tools(tools)     ← 绑定工具（create_react_agent 用这个）
      - 支持 .astream([messages])   ← 流式输出
    """
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")
    base_url = os.environ.get("LLM_BASE_URL")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-opus-4-6",
            api_key=api_key,
            max_tokens=4096,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=api_key or "ollama",
            base_url=base_url,
            max_tokens=4096,
        )


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
            import sys
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
            import sys
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


# ══════════════════════════════════════════════════════════════════════════════
# ② Node 函数
# ══════════════════════════════════════════════════════════════════════════════

# ── plan_node ─────────────────────────────────────────────────────────────────

_PLAN_SYSTEM = """\
你是技术面试导演。根据候选人 Profile 和 JD，将整场面试拆分为 4-6 个具体的考察任务。
每个任务是一道具体的面试题方向，不是宽泛话题。
task_type 只能是: experience/knowledge/concept/design/debug/scenario

对每个任务，可在 sub_questions 中预规划 1-3 个具体子问题方向（考察角度）。

你可以先搜索知识库/简历/历史记录，了解候选人背景，再制定计划。

Final Answer 必须是 JSON 数组（只输出数组，不加代码块）：
[{"task":"介绍你做过最有挑战的项目","task_type":"experience","sub_questions":["项目背景","你的职责","技术挑战"]},...]"""


async def plan_node(state: InterviewState) -> dict:
    """
    ── Node①：Director 规划任务 ────────────────────────────────────────

    演示要点：create_react_agent 作为"子图"调用
    ┌─────────────────────────────────────────────────────────────────┐
    │  create_react_agent(llm, tools, prompt=system_msg)             │
    │    ↓                                                            │
    │  内部自动循环：                                                  │
    │    LLM 输出 → 有工具调用？→ 执行工具 → 追加观测 → LLM 输出 → …  │
    │    直到 LLM 不再调用工具 → 返回最终消息                          │
    │                                                                 │
    │  对比原来的 react_loop：用文本解析 Action/Observation            │
    │  现在：用模型原生 function calling，更健壮、更准确               │
    └─────────────────────────────────────────────────────────────────┘
    """
    llm = _build_llm()
    tools = _make_plan_tools(state)

    # ④ create_react_agent：传入 LLM + 工具列表 + 系统提示
    react_agent = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=_PLAN_SYSTEM),
    )

    user_content = (
        f"候选人 Profile：\n{state['profile_text'][:2000]}\n\n"
        f"岗位 JD：\n{state['jd'][:1000]}\n\n"
        f"考察方向：\n{state['direction']}"
    )

    # 调用子图。注意：这里的 ainvoke 是子图内部的，不是父图的
    # 子图有自己的消息列表，执行完毕后返回包含所有消息的 state
    result = await react_agent.ainvoke({
        "messages": [HumanMessage(content=user_content)]
    })

    # 最后一条 AI 消息 = 最终输出（任务规划 JSON）
    final_text = result["messages"][-1].content
    raw = _parse_json(final_text, default=[])

    # 解析 JSON → 创建 ThoughtNode 树
    roots: list[ThoughtNode] = []
    for item in (raw if isinstance(raw, list) else [])[:6]:
        t = item.get("task", "").strip() if isinstance(item, dict) else ""
        if not t:
            continue
        task_node = ThoughtNode(
            id=str(uuid.uuid4()), node_type="task",
            text=t, task_type=item.get("task_type", "knowledge"),
        )
        sub_qs = [q.strip() for q in item.get("sub_questions", []) if isinstance(q, str) and q.strip()]
        add_planned_nodes(task_node, sub_qs)
        roots.append(task_node)

    if not roots:
        roots = [ThoughtNode(id=str(uuid.uuid4()), node_type="task", text="介绍你的项目经历")]

    roots[0].status = "active"
    print(f"[plan] tasks={len(roots)} planned={[len(r.children) for r in roots]}")

    # ② 节点只返回"本次要更新的字段"，框架负责合并进 state
    return {
        "roots_data": [_node_to_dict(r) for r in roots],
        "current_task_id": roots[0].id,
        "current_question_id": None,
        "last_verdict": None,
        "last_sub_questions": [],
        "last_director_reasoning": "",
    }


# ── ask_node ──────────────────────────────────────────────────────────────────

_INTERVIEWER_SYSTEM = """\
你是一位专业技术面试官。

当前考察任务：{task_text}
出题场景：{question_context}
{profile_section}
规则：
- 问题要有针对性，不重复已问过的问题
- 语气自然专业，问题长度 1-2 句话
- 如需了解具体知识点细节，可先搜索知识库

输出 JSON（不加代码块）：
{"intent": "考察意图描述", "question": "具体的面试问题"}"""

_REFLECT_SYSTEM = """\
你是面试质量审核员。检查下面这道面试题是否合格：
1. 是否真正考察了出题意图中的知识点？
2. 候选人能否明确理解这道题在问什么？
只输出最终问题文本，不加任何解释。"""


async def ask_node(state: InterviewState) -> dict:
    """
    ── Node②：Interviewer 出题 + ③ interrupt() 等待回答 ───────────────

    演示要点：interrupt() 实现 Human-in-the-Loop
    ┌─────────────────────────────────────────────────────────────────┐
    │  正常代码：                                                      │
    │    answer = some_function()   # 同步等待                        │
    │                                                                 │
    │  LangGraph interrupt：                                          │
    │    answer = interrupt(question)  # 暂停图、持久化状态            │
    │    # ↑ 这行之后的代码在下次 ainvoke(Command(resume=...)) 时执行  │
    │                                                                 │
    │  对应两个 HTTP 请求：                                            │
    │    POST /v2/interview/start → 图运行到 interrupt → 返回问题      │
    │    POST /v2/interview/chat  → Command(resume=answer) → 继续     │
    └─────────────────────────────────────────────────────────────────┘
    """
    roots = [_dict_to_node(d) for d in state["roots_data"]]
    task_node = find(roots, state["current_task_id"])
    if task_node is None:
        raise RuntimeError("current_task_id 指向不存在的节点")

    task_text = task_node.text
    task_type = task_node.task_type
    profile_section = (
        f"候选人背景：\n{state['profile_text'][:800]}\n"
        if state["profile_text"] else ""
    )

    # 判断是否是该任务的第一题
    asked_questions = [
        n for n in flat([task_node])
        if n.node_type == "question" and n.status not in ("planned", "skipped")
    ]
    is_first = len(asked_questions) == 0
    director_focus = state.get("last_director_reasoning", "") or ""

    if is_first:
        question_context = f"首问，话题开场（task_type={task_type}）"
        user_content = f"这是对话的开始，请提出关于「{task_text}」的第一个问题。"
    else:
        question_context = f"追问/转向（导演指示：{director_focus or '无'}）"
        cur_qnode = find(roots, state["current_question_id"])
        user_content = (
            f"上一个问题：{cur_qnode.text if cur_qnode else ''}\n"
            f"候选人的回答：{(cur_qnode.answer[:300] if cur_qnode else '')}\n"
            f"导演指示：{director_focus or '根据上下文自行判断'}"
        )

    system = _INTERVIEWER_SYSTEM.format(
        task_text=task_text,
        question_context=question_context,
        profile_section=profile_section,
    )

    # 用 create_react_agent 生成问题（可选调用 search_knowledge 工具）
    llm = _build_llm()
    tools = _make_ask_tools(state)
    react_agent = create_react_agent(llm, tools, prompt=SystemMessage(content=system))
    result = await react_agent.ainvoke({"messages": [HumanMessage(content=user_content)]})

    final_text = result["messages"][-1].content
    parsed = _parse_json(final_text, default={"intent": "", "question": final_text.strip()})
    intent = parsed.get("intent", "")
    draft_q = parsed.get("question", final_text.strip()).strip()

    # Self-reflection：LLM 验证问题是否符合考察意图（复用原逻辑）
    final_q = draft_q
    if intent and draft_q:
        try:
            reflect = await llm.ainvoke([
                SystemMessage(content=_REFLECT_SYSTEM),
                HumanMessage(content=f"出题意图：{intent}\n生成的问题：{draft_q}"),
            ])
            final_q = reflect.content.strip() or draft_q
        except Exception:
            pass

    # 在思维树中创建新的问题节点
    new_qnode = ThoughtNode(
        id=str(uuid.uuid4()), node_type="question",
        text=final_q, depth=task_node.depth + 1,
        status="asking", parent_id=task_node.id,
        question_intent=intent,
    )
    task_node.children.append(new_qnode)

    print(f"[ask] intent='{intent[:50]}' q='{final_q[:60]}'")

    # ③ interrupt() ────────────────────────────────────────────────────────────
    # 执行到这里：图暂停，final_q 作为"中断值"返回给 HTTP 调用者。
    # 框架把当前完整 state 存入 checkpointer（按 thread_id 索引）。
    #
    # 下次调用 graph.ainvoke(Command(resume=user_answer), config) 时：
    #   → 框架从 checkpointer 恢复 state
    #   → 从 interrupt() 这行继续执行
    #   → user_answer 就是 Command(resume=...) 里传入的值
    user_answer: str = interrupt(final_q)
    # ──────────────────────────────────────────────────────────────────────────

    # 恢复执行后：把用户答案记录到节点
    new_qnode.answer = user_answer
    new_qnode.status = "answered"

    return {
        "roots_data": [_node_to_dict(r) for r in roots],
        "current_question_id": new_qnode.id,
        # add_messages reducer：这里返回的 messages 会被"追加"到已有列表
        "messages": [AIMessage(content=final_q), HumanMessage(content=user_answer)],
    }


# ── score_node ────────────────────────────────────────────────────────────────

_SCORE_SYSTEM = """\
你是面试评分员，客观评分并给出逐点分析，不做策略决策。

评分标准（1-5分）：
5分：准确完整，有深度，能说明原理并举例
4分：覆盖核心要点，表述清晰，基本无遗漏
3分：方向正确但有明显遗漏，或表述模糊
2分：只了解表面概念，细节错误或不知道原理
1分：答非所问，或完全不了解

输出 JSON（不加代码块）：
{"score": 3, "reasoning": "候选人提到了...", "feedback": "掌握了...，但缺少..."}"""


async def score_node(state: InterviewState) -> dict:
    """
    ── Node③：Scorer 评分 ──────────────────────────────────────────────

    演示要点：最简单的 LangGraph 节点 —— 直接 llm.ainvoke()，不需要工具
    ┌─────────────────────────────────────────────────────────────────┐
    │  llm.ainvoke([SystemMessage(...), HumanMessage(...)])           │
    │  ↑ 这是 LangChain LLM 的标准调用方式                            │
    │  返回 AIMessage，.content 取文本内容                             │
    └─────────────────────────────────────────────────────────────────┘
    """
    roots = [_dict_to_node(d) for d in state["roots_data"]]
    qnode = find(roots, state["current_question_id"])
    task_node = find(roots, state["current_task_id"])

    if qnode is None:
        return {"last_score": 3}

    task_text = task_node.text if task_node else "（未知）"

    llm = _build_llm()
    result = await llm.ainvoke([
        SystemMessage(content=_SCORE_SYSTEM),
        HumanMessage(content=(
            f"当前考察任务：{task_text}\n"
            f"面试官的问题：{qnode.text}\n"
            f"候选人的回答：{qnode.answer[:600]}"
        )),
    ])

    score_data = _parse_json(
        result.content,
        default={"score": 3, "reasoning": "解析失败", "feedback": ""},
    )
    score = max(1, min(5, int(score_data.get("score", 3))))

    qnode.score = score
    qnode.reasoning = score_data.get("reasoning", "")
    qnode.feedback = score_data.get("feedback", "")
    qnode.status = "scored"

    print(f"[score] q='{qnode.text[:40]}' score={score}")

    return {
        "roots_data": [_node_to_dict(r) for r in roots],
        "last_score": score,
    }


# ── decide_node ───────────────────────────────────────────────────────────────

_DECIDE_SYSTEM = """\
你是面试导演。根据评分结果决定下一步策略。

决策选项（四选一）：
- "deepen"  : 候选人理解不够，从多个角度继续考察当前知识点（下钻）
- "pivot"   : 当前答题尚可，但任务还有其他重要角度未覆盖（同级）
- "back_up" : 追问已经太细太偏，退回上一层换角度
- "pass"    : 当前任务已充分考察，推进下一任务

决策依据：
- score≤2 且深度<3：优先 deepen，规划 2-3 个具体角度
- score≥4：优先 pass 或 pivot
- 深度≥3：优先 back_up 或 pass
- 本任务已问题数≥4：强制 pass

输出 JSON（不加代码块）：
{"decision": "deepen", "reasoning": "理由", "sub_questions": ["子问题方向1","子问题方向2"]}"""


async def decide_node(state: InterviewState) -> dict:
    """
    ── Node④：Director 决策 ────────────────────────────────────────────

    演示要点：节点不做路由，只更新 state；路由由条件边的函数决定
    ┌─────────────────────────────────────────────────────────────────┐
    │  节点职责：把 verdict 写入 state["last_verdict"]                │
    │  路由职责：route_after_decide(state) 读 last_verdict 返回 key   │
    │                                                                 │
    │  好处：节点纯粹负责业务逻辑，路由逻辑集中在一处，清晰可维护      │
    └─────────────────────────────────────────────────────────────────┘
    """
    roots = [_dict_to_node(d) for d in state["roots_data"]]
    qnode = find(roots, state["current_question_id"])
    task_node = find(roots, state["current_task_id"])

    if qnode is None or task_node is None:
        return {"last_verdict": "pass", "last_sub_questions": [], "last_director_reasoning": "节点不存在"}

    task_text = task_node.text
    question_count = sum(
        1 for n in flat([task_node])
        if n.node_type == "question" and n.status not in ("planned", "skipped")
    )
    parent = find(roots, qnode.parent_id)
    pending_plan = "、".join(
        n.question_intent for n in (parent.children if parent else [])
        if n.status == "planned" and n.id != qnode.id
    ) or "无"

    llm = _build_llm()
    result = await llm.ainvoke([
        SystemMessage(content=_DECIDE_SYSTEM),
        HumanMessage(content=(
            f"当前考察任务：{task_text}\n"
            f"面试官的问题：{qnode.text}\n"
            f"候选人的回答：{qnode.answer[:400]}\n"
            f"评分：{state['last_score']}/5\n"
            f"评分分析：{qnode.reasoning[:400]}\n"
            f"当前问题深度：{qnode.depth}\n"
            f"本任务已问题数：{question_count}\n"
            f"已有待问计划：{pending_plan}"
        )),
    ])

    d = _parse_json(result.content, default={"decision": "pass", "reasoning": "解析失败", "sub_questions": []})
    decision = d.get("decision", "pass")
    if decision not in ("deepen", "pivot", "back_up", "pass"):
        decision = "pass"
    sub_questions = [q.strip() for q in d.get("sub_questions", []) if q.strip()][:3]

    qnode.verdict = decision
    qnode.director_note = d.get("reasoning", "")
    qnode.status = "done"

    # 根据决策更新树结构
    updated_task_id = state["current_task_id"]

    if decision == "pass":
        # 标记当前任务完成，推进到下一个
        task_node.status = "done"
        next_task = next_pending_task(roots)
        if next_task:
            next_task.status = "active"
            updated_task_id = next_task.id
        else:
            updated_task_id = None  # 面试结束标志

    elif decision in ("deepen", "pivot", "back_up"):
        # 在树中添加 planned 子节点（面试官下一题的方向）
        if decision == "deepen":
            plan_parent = qnode
        elif decision == "pivot":
            plan_parent = task_node
        else:  # back_up
            gp = find(roots, qnode.parent_id)
            plan_parent = (
                find(roots, gp.parent_id) if gp and gp.parent_id else gp
            ) or task_node
        if sub_questions:
            add_planned_nodes(plan_parent, sub_questions)

    print(f"[decide] verdict={decision} score={state['last_score']} depth={qnode.depth}")

    return {
        "roots_data": [_node_to_dict(r) for r in roots],
        "current_task_id": updated_task_id,
        "last_verdict": decision,
        "last_sub_questions": sub_questions,
        "last_director_reasoning": d.get("reasoning", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 条件边路由函数
# ══════════════════════════════════════════════════════════════════════════════

def route_after_decide(state: InterviewState) -> Literal["ask", "__end__"]:
    """
    ── 条件边路由函数 ───────────────────────────────────────────────────

    从 decide 节点出发后，走哪条路？

    规则：
      - verdict == "pass" 且没有更多任务（current_task_id is None）→ 结束面试
      - 其余所有情况 → 继续出题

    注意：这个函数只做路由决策，不改 state，副作用为零。
    返回值是字符串 key，框架用 add_conditional_edges 里的 map 来查目标节点。
    """
    verdict = state.get("last_verdict", "pass")
    current_task_id = state.get("current_task_id")

    if verdict == "pass" and current_task_id is None:
        return "__end__"   # 所有任务完成 → 结束
    else:
        return "ask"       # 继续追问 or 进入下一任务


# ══════════════════════════════════════════════════════════════════════════════
# 图构建 + 编译
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    ── 把节点和边组装成可运行的图 ──────────────────────────────────────

    图结构：
        START → plan → ask ←──────────────────┐
                        │                     │
                   (interrupt)                │
                        │                     │
                      score                   │
                        │                   "ask"
                      decide                  │
                        │                     │
              ┌─────────┴──────────┐          │
           "ask"              "__end__"        │
              │                   │           │
              └───────────────────┘          END
                                  └──→ END

    ⑥ Checkpointer（MemorySaver）：
       每次节点执行后自动把 state 快照存入内存。
       用 config["configurable"]["thread_id"] 区分不同会话。
       生产环境替换成 SqliteSaver / AsyncPostgresSaver 即可，代码不变。
    """
    workflow = StateGraph(InterviewState)

    # 注册节点（name → 函数）
    workflow.add_node("plan", plan_node)
    workflow.add_node("ask", ask_node)
    workflow.add_node("score", score_node)
    workflow.add_node("decide", decide_node)

    # 固定边（始终走这条路）
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "ask")     # 规划完 → 出第一道题
    workflow.add_edge("ask", "score")    # 问题发出、用户回答后 → 评分
    workflow.add_edge("score", "decide") # 评分后 → 导演决策

    # ⑤ 条件边：decide 之后，由 route_after_decide 函数决定走哪条路
    workflow.add_conditional_edges(
        "decide",            # 从这个节点出发
        route_after_decide,  # 路由函数，返回字符串 key
        {                    # key → 目标节点 的映射表
            "ask": "ask",
            "__end__": END,
        },
    )

    # ⑥ 挂载 Checkpointer，每步自动存档
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ── 全局单例（应用启动时初始化一次，所有请求共用这个图实例）────────────────────
interview_graph = build_graph()
