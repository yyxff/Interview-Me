"""
图构建 + 编译 + 全局单例
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import ask_node, decide_node, plan_node, route_after_decide, score_node
from .state import InterviewState


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
