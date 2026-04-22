"""知识库 QA 应用层

职责：query 改写 → 混合检索（vector + graph）→ 构建 system prompt → 流式回答
对外暴露两个接口：prepare_context() 和 stream_response()
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, TYPE_CHECKING

import rag

if TYPE_CHECKING:
    from llm.provider import LLMProvider

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_QA_SYSTEM_PROMPT = """\
你是一位技术知识助手，专注于软件工程领域的问题解答。

## 参考资料使用规范
1. 先判断每条参考资料与用户问题是否相关，**不相关的资料直接忽略**，不要强行引用
2. 若所有资料均与问题无关，直接说明"知识库中暂无相关内容"，然后用通用知识回答并明确标注
3. 若资料部分相关，只提取相关部分；引用时用 [1][2] 等序号标注
4. **严禁**将无关资料强行套用到答案中"""

_REWRITE_PROMPT = """\
根据对话历史，将用户的最新问题改写为一个独立、完整的检索查询。

规则：
- 补全指代词（"它"→具体名词、"这个"→具体概念）
- 补全省略的主语或对比对象
- 保留问题的核心意图，不要增加额外约束
- 如果问题已经完整独立，原样返回
- 只输出改写后的问题，不要解释、不要引号

对话历史（最近两轮）：
{history}

用户问题：{question}"""

_MAX_HISTORY = 12


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class QAContext:
    """prepare_context() 的返回值，stream_response() 的输入。"""
    original_query:   str
    retrieval_query:  str
    sources:          list[dict] = field(default_factory=list)
    graph_viz:        dict       = field(default_factory=dict)
    system:           str        = ""
    messages:         list[dict] = field(default_factory=list)


# ── 内部函数 ──────────────────────────────────────────────────────────────────

async def _rewrite_query(question: str, history: list[dict],
                         provider: LLMProvider) -> str:
    recent = history[-4:]
    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else 'AI'}: {m['content'][:200]}"
        for m in recent
    )
    prompt = _REWRITE_PROMPT.format(history=history_text, question=question)
    try:
        result = await asyncio.wait_for(
            provider.chat([{"role": "user", "content": prompt}], system=""),
            timeout=8.0,
        )
        rewritten = result.strip().strip('"').strip("'")
        return rewritten if rewritten else question
    except Exception:
        return question


def _retrieve(query: str) -> tuple[list[dict], list[dict], dict, str]:
    """混合检索：vector + graph RAG。
    返回 (knowledge_chunks, note_chunks, graph_viz, graph_summary)。
    """
    graph_result    = rag.retrieve_graph(query)
    graph_chunk_ids = graph_result.get("source_chunk_ids", [])
    graph_summary   = graph_result.get("graph_summary", "")

    graph_extra: list[dict] = []
    graph_viz: dict = {}
    try:
        import graph_rag as _gr
        if graph_chunk_ids:
            graph_extra = _gr.get_chunks_by_ids(graph_chunk_ids)
        if graph_result.get("entities"):
            graph_viz = _gr.get_subgraph_for_viz(query)
    except Exception:
        pass

    path_map: dict[str, str] = {}
    for r in graph_result.get("relations", []):
        cid = r.get("source_chunk_id", "")
        if cid and cid not in path_map:
            path_map[cid] = (
                f"[图谱路径] {r['subject']} --{r['predicate']}--> {r['object']}。"
                f"{r.get('description', '')}"
            )

    result    = rag.retrieve_rich(query, extra_chunks=graph_extra or None,
                                  path_map=path_map or None)
    knowledge = result["knowledge"]
    notes     = result.get("notes", [])

    retrieval_log = result.get("retrieval_log", [])
    be_stage     = next((e for e in retrieval_log if e.get("_be_stage")),  None)
    rrf_stage    = next((e for e in retrieval_log if e.get("_rrf_stage")), None)
    final_chunks = [e for e in retrieval_log
                    if not e.get("_summary") and not e.get("_be_stage") and not e.get("_rrf_stage")]

    if be_stage:
        lines = [f"[rag] bi-encoder 召回 {len(be_stage['candidates'])} 个候选"]
        for cid, dist in be_stage["candidates"]:
            lines.append(f"  {cid}  dist={dist:.4f}")
        logger.debug("\n".join(lines))

    entities       = graph_result.get("entities", [])
    relations      = graph_result.get("relations", [])
    bfs_chunk_ids  = graph_result.get("bfs_chunk_ids", [])
    if entities or relations:
        lines = [f"[graph rag] 直接命中 {len(entities)} 实体 / {len(relations)} 关系"]
        for e in entities:
            cids = ", ".join(e.get("source_chunk_ids", []))
            lines.append(f"  entity    {e['name']}  chunks=[{cids}]")
        for r in relations:
            cid = r.get("source_chunk_id", "")
            lines.append(f"  relation  {r['subject']} --{r['predicate']}--> {r['object']}  chunk={cid}")
        if bfs_chunk_ids:
            lines.append(f"  bfs 扩展 {len(bfs_chunk_ids)} 个 chunk")
        logger.debug("\n".join(lines))

    if rrf_stage:
        lines = [f"[rrf] 合并后候选池 {len(rrf_stage['candidates'])} 个（进入 rerank）"]
        for cid, score in rrf_stage["candidates"]:
            lines.append(f"  {cid}  rrf={score:.5f}")
        logger.debug("\n".join(lines))

    if final_chunks:
        lines = [f"[rerank] 最终输出 {len(final_chunks)} 个"]
        for entry in final_chunks:
            via   = entry.get("via_graph", False)
            gonly = entry.get("graph_only", False)
            tag   = (" [graph]" if gonly else " [vec] [graph]") if via else " [vec]"
            lines.append(
                f"  {entry.get('chunk_id', '')}  bi={entry.get('bi_dist', -1):.4f}"
                f"  rerank={entry.get('rerank_score', -1):.4f}{tag}"
            )
        logger.debug("\n".join(lines))

    return knowledge, notes, graph_viz, graph_summary


def _build_system(knowledge: list[dict], notes: list[dict], graph_summary: str) -> str:
    system = _QA_SYSTEM_PROMPT
    if knowledge:
        refs = "\n\n---\n\n".join(
            f"[{i+1}] 来源：{c['source']} > {c.get('path') or c['chapter']}\n{c['text']}"
            for i, c in enumerate(knowledge)
        )
        system += f"\n\n## 参考资料\n{refs}"
    if notes:
        note_refs = "\n\n---\n\n".join(
            f"笔记《{c['chapter']}》\n{c['text']}" for c in notes
        )
        system += f"\n\n## 我的知识笔记\n{note_refs}"
    if graph_summary:
        system += f"\n\n## 图谱关联上下文\n{graph_summary}"
    return system


# ── 公开接口 ──────────────────────────────────────────────────────────────────

async def prepare_context(question: str, history: list[dict],
                          provider: LLMProvider | None) -> QAContext:
    """改写 query → 检索 → 构建 system prompt，返回 QAContext。"""
    logger.info("[qa] question: %s", question)

    retrieval_query = question
    if provider is not None and history:
        retrieval_query = await _rewrite_query(question, history, provider)

    if retrieval_query != question:
        logger.info("[qa] rewrite: %s", retrieval_query)

    knowledge, notes, graph_viz, graph_summary = _retrieve(retrieval_query)
    logger.info(
        "[qa] done  knowledge=%d notes=%d graph_viz_nodes=%d",
        len(knowledge), len(notes), len(graph_viz.get("nodes", [])),
    )

    all_sources = knowledge + notes
    sources = [
        {
            "source":    c["source"],
            "path":      c.get("path", ""),
            "chapter":   c["chapter"],
            "chunk_id":  c["chunk_id"],
            "text":      c["text"],
            "via_graph": c.get("via_graph", False),
        }
        for c in all_sources
    ]

    messages = history[-_MAX_HISTORY:]
    messages = messages + [{"role": "user", "content": question}]

    return QAContext(
        original_query=question,
        retrieval_query=retrieval_query,
        sources=sources,
        graph_viz=graph_viz,
        system=_build_system(knowledge, notes, graph_summary),
        messages=messages,
    )


async def stream_response(ctx: QAContext,
                          provider: LLMProvider | None) -> AsyncIterator[str]:
    """流式生成 LLM 回答，yield 文本片段。"""
    if provider is None:
        yield "请配置 LLM_PROVIDER 和 LLM_API_KEY"
        return
    async for chunk in provider.stream_chat(ctx.messages, ctx.system):
        yield chunk
