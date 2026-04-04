"""路由：/graph/*  /chunk/*  /entity-chunk"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

import rag
from llm.provider import LLMProvider

router = APIRouter()

_provider: LLMProvider | None = None


def set_provider(p: LLMProvider | None) -> None:
    global _provider
    _provider = p


@router.post("/graph/index")
async def graph_index(force: bool = False):
    """启动图索引（后台任务）。force=True 时删除旧图重建。"""
    if _provider is None:
        raise HTTPException(status_code=503, detail="LLM 未配置，图抽取需要 LLM")
    if not rag.is_available():
        raise HTTPException(status_code=503, detail="RAG 模块未安装")
    try:
        import graph_rag as _gr
    except ImportError:
        raise HTTPException(status_code=503, detail="graph_rag 模块不可用（缺少 networkx？）")

    if force:
        for f in _gr.GRAPH_DIR.glob("*.graph.json"):
            f.unlink(missing_ok=True)
        try:
            client = rag._get_client()
            for col_name in (_gr._GRAPH_ENTITIES_COL, _gr._GRAPH_RELATIONS_COL):
                try:
                    client.delete_collection(col_name)
                except Exception:
                    pass
        except Exception:
            pass

    async def _run():
        try:
            await _gr.index_knowledge_graph(_provider)
        except Exception as e:
            print(f"[graph/index] 索引失败: {e}")

    asyncio.create_task(_run())
    return {"ok": True, "force": force}


@router.get("/graph/index-progress")
async def graph_index_progress():
    try:
        import graph_rag as _gr
        return _gr.get_graph_index_progress()
    except ImportError:
        return {"status": "unavailable"}


@router.get("/graph/stats")
async def graph_stats():
    try:
        import graph_rag as _gr
        return _gr.graph_stats()
    except ImportError:
        return {"nodes": 0, "edges": 0, "entity_vectors": 0, "relation_vectors": 0}


@router.get("/graph/full")
def graph_full():
    """返回全量知识图谱，供前端可视化。"""
    try:
        import graph_rag as _gr
        G = _gr._get_nx_graph()
        nodes = [
            {
                "id":          name,
                "entity_type": attrs.get("entity_type", "概念"),
                "description": attrs.get("description", ""),
                "source":      attrs.get("source", ""),
            }
            for name, attrs in G.nodes(data=True)
        ]
        edges = [
            {
                "source":    u,
                "target":    v,
                "predicate": data.get("predicate", ""),
            }
            for u, v, data in G.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}


@router.get("/entity-chunk")
def entity_chunk(name: str):
    """按实体名在知识库做向量检索，返回最相关的 chunk。"""
    try:
        col = rag._get_knowledge_col()
        if col.count() == 0:
            raise HTTPException(status_code=404, detail="knowledge collection empty")
        results = rag._safe_query(col, name, n=1)
        if not results:
            raise HTTPException(status_code=404, detail="no chunk found")
        doc, meta = results[0]
        return {
            "chunk_id": meta.get("chunk_id", ""),
            "text":     meta.get("text", doc),
            "source":   meta.get("source", ""),
            "path":     meta.get("path", ""),
            "chapter":  meta.get("chapter", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chunk/{chunk_id:path}")
def chunk_get(chunk_id: str):
    """按 chunk_id 返回原文，供前端图谱节点点击时展示。"""
    try:
        import graph_rag as _gr
        chunks = _gr.get_chunks_by_ids([chunk_id])
        if not chunks:
            raise HTTPException(status_code=404, detail="chunk not found")
        c = chunks[0]
        return {
            "chunk_id": chunk_id,
            "text":     c["text"],
            "source":   c.get("source", ""),
            "path":     c.get("path", ""),
            "chapter":  c.get("chapter", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
