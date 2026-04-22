"""路由：/qa/stream（知识库 QA 精准模式）"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import rag
from llm.provider import LLMProvider

router = APIRouter()
logger = logging.getLogger(__name__)

_provider: LLMProvider | None = None


def set_provider(p: LLMProvider | None) -> None:
    global _provider
    _provider = p


QA_SYSTEM_PROMPT = """你是一位技术知识助手，专注于软件工程领域的问题解答。

## 参考资料使用规范
1. 先判断每条参考资料与用户问题是否相关，**不相关的资料直接忽略**，不要强行引用
2. 若所有资料均与问题无关，直接说明"知识库中暂无相关内容"，然后用通用知识回答并明确标注
3. 若资料部分相关，只提取相关部分；引用时用 [1][2] 等序号标注
4. **严禁**将无关资料强行套用到答案中"""

REWRITE_PROMPT = """根据对话历史，将用户的最新问题改写为一个独立、完整的检索查询。

规则：
- 补全指代词（"它"→具体名词、"这个"→具体概念）
- 补全省略的主语或对比对象
- 保留问题的核心意图，不要增加额外约束
- 如果问题已经完整独立，原样返回
- 只输出改写后的问题，不要解释、不要引号

对话历史（最近两轮）：
{history}

用户问题：{question}"""

MAX_HISTORY_MESSAGES = 12


def trim_history(history: list) -> list:
    return history[-MAX_HISTORY_MESSAGES:]


async def rewrite_query(question: str, history: list[dict]) -> str:
    """用最近两轮对话上下文改写检索 query。失败时降级返回原问题。"""
    if _provider is None:
        return question
    recent = history[-4:]
    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else 'AI'}: {m['content'][:200]}"
        for m in recent
    )
    prompt = REWRITE_PROMPT.format(history=history_text, question=question)
    try:
        result = await asyncio.wait_for(
            _provider.chat([{"role": "user", "content": prompt}], system=""),
            timeout=8.0,
        )
        rewritten = result.strip().strip('"').strip("'")
        return rewritten if rewritten else question
    except Exception:
        return question


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QARequest(BaseModel):
    message: str
    history: list[HistoryItem] = []


@router.post("/qa/stream")
async def qa_stream(req: QARequest):
    """基于知识库的问答：先发 sources 事件，再流式发 LLM 回复。"""
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    retrieval_query = await rewrite_query(req.message, history_dicts)

    if retrieval_query != req.message:
        logger.info("[rewrite] '%s' → '%s'", req.message, retrieval_query)

    # Graph RAG 检索
    graph_result    = rag.retrieve_graph(retrieval_query)
    graph_chunk_ids = graph_result.get("source_chunk_ids", [])
    graph_summary   = graph_result.get("graph_summary", "")

    graph_log = graph_result.get("graph_log", [])
    if graph_log:
        logger.debug("[graph] %d hits:", len(graph_log))
        for entry in graph_log:
            if entry.get("type") == "entity":
                logger.debug("  entity  %s  dist=%.4f", entry.get('name',''), entry.get('dist',-1))
            else:
                logger.debug("  relation  %s  dist=%.4f", entry.get('triple',''), entry.get('dist',-1))

    graph_extra: list[dict] = []
    graph_viz: dict = {}
    try:
        import graph_rag as _gr
        if graph_chunk_ids:
            graph_extra = _gr.get_chunks_by_ids(graph_chunk_ids)
        if graph_result.get("entities"):
            graph_viz = _gr.get_subgraph_for_viz(retrieval_query)
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
    if graph_summary:
        logger.debug("[graph] summary → %s", graph_summary)

    result    = rag.retrieve_rich(retrieval_query, extra_chunks=graph_extra or None,
                                  path_map=path_map or None)
    knowledge = result["knowledge"]
    notes     = result.get("notes", [])

    retrieval_log = result.get("retrieval_log", [])
    summary = next((e for e in retrieval_log if e.get("_summary")), None)
    if summary:
        logger.debug(
            "[retrieval] bi=%s + graph=%s(new=%s, overlap=%s) → rerank_input=%s → final=%s",
            summary['be_candidates'], summary['graph_extra'],
            summary['graph_new'], summary['graph_overlap'],
            summary['rerank_input'], summary['final_output'],
        )
    for entry in retrieval_log:
        if entry.get("_summary"):
            continue
        tag = " [graph-only]" if entry.get("graph_only") else (" [graph+vec]" if entry.get("via_graph") else "")
        q   = f"  Q:{entry['question']}" if entry.get("question") else ""
        logger.debug(
            "  %s  bi=%.4f  rrf=%.5f  rerank=%.4f%s%s",
            entry.get('chunk_id',''), entry.get('bi_dist',-1),
            entry.get('rrf_score',0), entry.get('rerank_score',-1), tag, q,
        )

    all_sources = knowledge + notes
    sources_payload = [
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

    messages = [{"role": m.role, "content": m.content} for m in trim_history(req.history)]
    messages.append({"role": "user", "content": req.message})

    system = QA_SYSTEM_PROMPT
    if knowledge:
        refs = "\n\n---\n\n".join(
            f"[{i+1}] 来源：{c['source']} > {c.get('path') or c['chapter']}\n{c['text']}"
            for i, c in enumerate(knowledge)
        )
        system += f"\n\n## 参考资料\n{refs}"
    if notes:
        note_refs = "\n\n---\n\n".join(
            f"笔记《{c['chapter']}》\n{c['text']}"
            for c in notes
        )
        system += f"\n\n## 我的知识笔记\n{note_refs}"
    if graph_summary:
        system += f"\n\n## 图谱关联上下文\n{graph_summary}"

    async def _generate():
        first_event: dict = {'sources': sources_payload}
        if retrieval_query != req.message:
            first_event['rewritten_query'] = retrieval_query
        yield f"data: {json.dumps(first_event, ensure_ascii=False)}\n\n"
        if graph_viz and graph_viz.get("nodes"):
            yield f"data: {json.dumps({'graph': graph_viz}, ensure_ascii=False)}\n\n"

        if _provider is None:
            yield f"data: {json.dumps({'text': '请配置 LLM_PROVIDER 和 LLM_API_KEY'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            async for chunk in _provider.stream_chat(messages, system):
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
