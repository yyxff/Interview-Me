"""Graph RAG — 图构建、持久化、Qdrant 索引、主索引入口"""
from __future__ import annotations

import asyncio
import json
import time as _time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm.provider import LLMProvider

import rag
from .store import (
    GRAPH_DIR, GRAPH_CONCURRENCY_INITIAL,
    _get_entities_col, _get_relations_col,
    _get_nx_graph, _graph_index_progress,
)
from .extractor import _build_context_window, _extract_entities_relations


# ── 图数据结构 ────────────────────────────────────────────────────────────────

def _build_entity_embed_text(name: str, attrs: dict) -> str:
    return f"{name}（{attrs.get('entity_type', '概念')}）：{attrs.get('description', '')}"


def _build_relation_embed_text(edge: dict) -> str:
    subj      = edge["subject"]
    sdesc     = edge.get("subject_desc", "")
    pred      = edge["predicate"]
    obj       = edge["object"]
    odesc     = edge.get("object_desc", "")
    desc      = edge.get("description", "")
    subj_part = f"{subj}（{sdesc}）" if sdesc else subj
    obj_part  = f"{obj}（{odesc}）"  if odesc else obj
    return f"{subj_part}--{pred}-->{obj_part}：{desc}"


def _build_graph_for_source(
    source: str,
    extracted: list[tuple[list[dict], list[dict], str]],
) -> dict:
    """从抽取结果构建图数据结构。同名实体合并，同边取最后描述。"""
    nodes: dict[str, dict] = {}
    edge_map: dict[tuple[str, str], dict] = {}
    ent_idx = rel_idx = 0

    for entities, relations, chunk_id in extracted:
        for ent in entities:
            name = ent["name"]
            if name in nodes:
                existing = nodes[name]
                if chunk_id not in existing["source_chunk_ids"]:
                    existing["source_chunk_ids"].append(chunk_id)
                new_desc = ent["description"]
                if new_desc and new_desc not in existing["description"]:
                    combined = existing["description"] + "；" + new_desc
                    existing["description"] = combined[:200]
            else:
                nodes[name] = {
                    "entity_type":      ent.get("entity_type", "概念"),
                    "description":      ent["description"],
                    "source_chunk_ids": [chunk_id],
                    "entity_id":        f"ent_{source}_{ent_idx}",
                }
                ent_idx += 1

        for rel in relations:
            subj = rel["subject"]
            obj  = rel["object"]
            key  = (subj, obj)
            if key in edge_map:
                edge_map[key].update({
                    "subject_desc":    rel.get("subject_desc", edge_map[key].get("subject_desc", "")),
                    "predicate":       rel["predicate"],
                    "object_desc":     rel.get("object_desc", edge_map[key].get("object_desc", "")),
                    "description":     rel.get("description", ""),
                    "source_chunk_id": chunk_id,
                })
            else:
                edge_map[key] = {
                    "subject":         subj,
                    "subject_desc":    rel.get("subject_desc", ""),
                    "predicate":       rel["predicate"],
                    "object":          obj,
                    "object_desc":     rel.get("object_desc", ""),
                    "description":     rel.get("description", ""),
                    "source_chunk_id": chunk_id,
                    "relation_id":     f"rel_{source}_{rel_idx}",
                }
                rel_idx += 1

    return {"source": source, "nodes": nodes, "edges": list(edge_map.values())}


def _save_graph(source: str, graph: dict) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_DIR / f"{source}.graph.json"
    path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[graph_rag] 图已保存: {path.name} ({len(graph['nodes'])} 节点, {len(graph['edges'])} 边)")


def _load_graph(source: str) -> dict | None:
    path = GRAPH_DIR / f"{source}.graph.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clear_vectors_for_source(col, source: str) -> None:
    try:
        existing = col.get(where={"source": source})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except Exception:
        pass


def _index_graph_to_qdrant(graph: dict) -> None:
    """将图的实体和关系分别向量化存入 Qdrant（Dense + Sparse 双向量）。"""
    source  = graph["source"]
    ent_col = _get_entities_col()
    rel_col = _get_relations_col()

    _clear_vectors_for_source(ent_col, source)
    _clear_vectors_for_source(rel_col, source)

    ent_ids, ent_docs, ent_metas = [], [], []
    for name, attrs in graph.get("nodes", {}).items():
        embed_text = _build_entity_embed_text(name, attrs)
        if not embed_text.strip():
            continue
        ent_ids.append(attrs["entity_id"])
        ent_docs.append(embed_text)
        ent_metas.append({
            "name":                 name,
            "entity_type":          attrs.get("entity_type", "概念"),
            "source_chunk_ids_csv": ",".join(attrs.get("source_chunk_ids", [])),
            "entity_id":            attrs["entity_id"],
            "source":               source,
        })
    if ent_ids:
        ent_col.add(ids=ent_ids, documents=ent_docs, metadatas=ent_metas)

    rel_ids, rel_docs, rel_metas = [], [], []
    for edge in graph.get("edges", []):
        embed_text = _build_relation_embed_text(edge)
        if not embed_text.strip():
            continue
        rel_ids.append(edge["relation_id"])
        rel_docs.append(embed_text)
        rel_metas.append({
            "subject":         edge["subject"],
            "subject_desc":    edge.get("subject_desc", ""),
            "predicate":       edge["predicate"],
            "object":          edge["object"],
            "object_desc":     edge.get("object_desc", ""),
            "source_chunk_id": edge["source_chunk_id"],
            "relation_id":     edge["relation_id"],
            "source":          source,
        })
    if rel_ids:
        rel_col.add(ids=rel_ids, documents=rel_docs, metadatas=rel_metas)

    print(f"[graph_rag] Qdrant 索引: {source} → {len(ent_ids)} 实体向量, {len(rel_ids)} 关系向量")


# ── 主索引入口 ────────────────────────────────────────────────────────────────

async def index_knowledge_graph(provider: "LLMProvider") -> dict:
    """扫描 KNOWLEDGE_DIR/*.md，对未建图的 source 执行完整的抽取→建图→索引流程。"""
    global _nx_graph

    if not rag.is_available():
        return {"error": "RAG 不可用"}
    if not rag.KNOWLEDGE_DIR.exists():
        return {"error": "knowledge/ 目录不存在"}

    pending: list[tuple[str, list[dict]]] = []
    for md_file in sorted(rag.KNOWLEDGE_DIR.glob("*.md")):
        source = md_file.stem
        if (GRAPH_DIR / f"{source}.graph.json").exists():
            continue
        chunks = rag._load_or_build_chunks(md_file, source)
        if chunks:
            pending.append((source, chunks))

    if not pending:
        _graph_index_progress["status"] = "idle"
        return {"sources_processed": 0, "total_entities": 0, "total_relations": 0}

    total_chunks = sum(len(c) for _, c in pending)
    cached_done  = 0
    for source, _ in pending:
        p = GRAPH_DIR / f"{source}.graph.partial.json"
        if p.exists():
            try:
                cached_done += len(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass

    _graph_index_progress.update({
        "status": "running", "chunks_done": cached_done, "chunks_total": total_chunks,
        "entities": 0, "relations": 0, "elapsed_s": 0.0, "eta_s": None,
        "concurrency": GRAPH_CONCURRENCY_INITIAL, "error": None,
    })
    t_start = _time.monotonic()
    total_entities = total_relations = 0

    try:
        sem = rag.AdaptiveSemaphore(initial=GRAPH_CONCURRENCY_INITIAL, ssthresh=8)
        for source, chunks in pending:
            _graph_index_progress["source"] = source
            GRAPH_DIR.mkdir(parents=True, exist_ok=True)
            partial_path  = GRAPH_DIR / f"{source}.graph.partial.json"
            partial_cache: dict[str, dict] = {}
            if partial_path.exists():
                try:
                    partial_cache = json.loads(partial_path.read_text(encoding="utf-8"))
                    print(f"[graph_rag] 读取断点缓存: {partial_path.name} ({len(partial_cache)}/{len(chunks)} chunks)")
                except Exception:
                    partial_cache = {}

            chunks_done_ref = [_graph_index_progress["chunks_done"]]

            async def _extract_one(i: int) -> tuple[list[dict], list[dict], str]:
                chunk_id = f"{source}_{i}"
                ents: list[dict] = []
                rels: list[dict] = []
                if chunk_id in partial_cache:
                    cached = partial_cache[chunk_id]
                    ents = cached.get("entities", [])
                    rels = cached.get("relations", [])
                else:
                    try:
                        async with sem:
                            ents, rels = await _extract_entities_relations(
                                _build_context_window(chunks, i), provider
                            )
                    except Exception as e:
                        wait = min(10, 3 + sem.limit)
                        print(f"[graph_rag] chunk {i} 限速/超时，等待 {wait}s: {e}")
                        await asyncio.sleep(wait)
                    partial_cache[chunk_id] = {"entities": ents, "relations": rels}
                    partial_path.write_text(json.dumps(partial_cache, ensure_ascii=False), encoding="utf-8")

                chunks_done_ref[0] += 1
                done    = chunks_done_ref[0]
                elapsed = _time.monotonic() - t_start
                eta     = elapsed / done * (total_chunks - done) if done else None
                _graph_index_progress.update({
                    "chunks_done": done, "elapsed_s": round(elapsed, 1),
                    "eta_s": round(eta, 0) if eta is not None else None,
                    "concurrency": sem.limit,
                })
                return ents, rels, chunk_id

            results   = await asyncio.gather(*[_extract_one(i) for i in range(len(chunks))])
            graph     = _build_graph_for_source(source, list(results))
            _save_graph(source, graph)
            _index_graph_to_qdrant(graph)
            if partial_path.exists():
                partial_path.unlink()

            n_ents  = len(graph.get("nodes", {}))
            n_rels  = len(graph.get("edges", []))
            total_entities  += n_ents
            total_relations += n_rels
            _graph_index_progress["entities"]  += n_ents
            _graph_index_progress["relations"] += n_rels

        from . import store as _store
        _store._nx_graph = None
        _store._get_nx_graph()
        _graph_index_progress["status"] = "done"

    except Exception as e:
        print(f"[graph_rag] 索引出错: {e}")
        _graph_index_progress.update({"status": "error", "error": str(e)})
        raise

    return {
        "sources_processed": len(pending),
        "total_entities":    total_entities,
        "total_relations":   total_relations,
    }
