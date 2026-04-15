"""Graph RAG — 图查询、子图可视化、统计"""
from __future__ import annotations

import rag
from .store import (
    GRAPH_DIR, ENTITY_TOP_K, RELATION_TOP_K, BFS_HOPS,
    _ENTITY_DEDUP_NORM,
    _get_entities_col, _get_relations_col, _get_nx_graph,
)

# BFS 相似度过滤参数
BFS_NEIGHBOR_TOP_N       = 10
BFS_SIMILARITY_THRESHOLD = 0.45


def _empty_graph_result() -> dict:
    return {"entities": [], "relations": [], "source_chunk_ids": [], "graph_summary": ""}


def _bfs_neighbors(G, start_nodes: list[str], hops: int) -> list[str]:
    """BFS 展开，返回邻居节点的 source_chunk_ids（向后兼容，无相似度过滤）。"""
    visited   = set(start_nodes)
    frontier  = set(n for n in start_nodes if G.has_node(n))
    chunk_ids: list[str] = []
    seen_ids:  set[str]  = set()
    for _ in range(hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for nb in list(G.successors(node)) + list(G.predecessors(node)):
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
                    for cid in G.nodes[nb].get("source_chunk_ids", []):
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            chunk_ids.append(cid)
        frontier = next_frontier
        if not frontier:
            break
    return chunk_ids


def _bfs_neighbors_scored(
    G, start_nodes: list[str], hops: int, query_embedding: list[float],
) -> list[str]:
    """BFS + 余弦相似度剪枝，保留 top-N 最相关邻居节点的 chunk_ids。"""
    import numpy as np

    visited   = set(start_nodes)
    frontier  = set(n for n in start_nodes if G.has_node(n))
    neighbors: list[str] = []

    for _ in range(hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for nb in list(G.successors(node)) + list(G.predecessors(node)):
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
                    neighbors.append(nb)
        frontier = next_frontier
        if not frontier:
            break

    if not neighbors:
        return []

    ef = rag._get_ef()
    try:
        neighbor_embeddings = ef(neighbors)
    except Exception:
        chunk_ids: list[str] = []
        seen: set[str] = set()
        for nb in neighbors[:BFS_NEIGHBOR_TOP_N]:
            for cid in G.nodes[nb].get("source_chunk_ids", []):
                if cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)
        return chunk_ids

    q_vec  = np.array(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q_vec) or 1.0
    scored: list[tuple[float, str]] = []
    for nb, nb_emb in zip(neighbors, neighbor_embeddings):
        nb_vec  = np.array(nb_emb, dtype=np.float32)
        nb_norm = np.linalg.norm(nb_vec)
        if nb_norm == 0:
            continue
        sim = float(np.dot(q_vec, nb_vec) / (q_norm * nb_norm))
        if sim >= BFS_SIMILARITY_THRESHOLD:
            scored.append((sim, nb))

    scored.sort(reverse=True)
    top_neighbors = [nb for _, nb in scored[:BFS_NEIGHBOR_TOP_N]]
    chunk_ids = []
    seen = set()
    for nb in top_neighbors:
        for cid in G.nodes[nb].get("source_chunk_ids", []):
            if cid not in seen:
                seen.add(cid)
                chunk_ids.append(cid)
    return chunk_ids


def _build_graph_summary(entities: list[dict], relations: list[dict], G) -> str:
    if not entities and not relations:
        return ""
    lines: list[str] = []
    if entities:
        parts = [f"{e['name']}（{e['entity_type']}）" for e in entities[:4]]
        lines.append("发现实体：" + "、".join(parts))
    if relations:
        parts = [f"{r['subject']} --{r['predicate']}--> {r['object']}" for r in relations[:3]]
        lines.append("关键关系：" + "；".join(parts))
    entity_names = {e["name"] for e in entities}
    neighbors: set[str] = set()
    for e in entities[:3]:
        node = e["name"]
        if G.has_node(node):
            for nb in list(G.successors(node)) + list(G.predecessors(node)):
                if nb not in entity_names:
                    neighbors.add(nb)
    if neighbors:
        lines.append("关联概念：" + "、".join(list(neighbors)[:5]))
    return "\n".join(lines)


def retrieve_graph(query: str) -> dict:
    """图增强检索，返回 {entities, relations, source_chunk_ids, graph_summary}。"""
    if not rag.is_available():
        return _empty_graph_result()

    try:
        ent_col = _get_entities_col()
        rel_col = _get_relations_col()
        if ent_col.count() == 0:
            return _empty_graph_result()

        raw_ents = rag._safe_query(ent_col, query, ENTITY_TOP_K, return_distances=True)
        entities: list[dict] = []
        graph_log: list[dict] = []
        for doc, meta, dist in raw_ents:
            chunk_ids = [c for c in meta.get("source_chunk_ids_csv", "").split(",") if c]
            entities.append({
                "name":             meta.get("name", ""),
                "entity_type":      meta.get("entity_type", ""),
                "description":      doc,
                "source_chunk_ids": chunk_ids,
            })
            graph_log.append({"type": "entity", "name": meta.get("name", ""), "dist": round(dist, 4)})

        def _norm_name(n: str) -> str:
            return n.translate(_ENTITY_DEDUP_NORM).lower()

        deduped: list[dict] = []
        seen_norms: list[str] = []
        for e in entities:
            n = _norm_name(e["name"])
            if not any(n in sn or sn in n for sn in seen_norms):
                deduped.append(e)
                seen_norms.append(n)
        entities = deduped

        relations: list[dict] = []
        if rel_col.count() > 0:
            raw_rels = rag._safe_query(rel_col, query, RELATION_TOP_K, return_distances=True)
            for doc, meta, dist in raw_rels:
                relations.append({
                    "subject":         meta.get("subject", ""),
                    "predicate":       meta.get("predicate", ""),
                    "object":          meta.get("object", ""),
                    "description":     doc,
                    "source_chunk_id": meta.get("source_chunk_id", ""),
                })
                graph_log.append({
                    "type":   "relation",
                    "triple": f"{meta.get('subject','')} --{meta.get('predicate','')}--> {meta.get('object','')}",
                    "dist":   round(dist, 4),
                })

        G = _get_nx_graph()
        all_chunk_ids: list[str] = []
        seen: set[str] = set()

        for r in relations:
            cid = r["source_chunk_id"]
            if cid and cid not in seen:
                seen.add(cid)
                all_chunk_ids.append(cid)

            # relation 涉及的两个实体节点的 chunk
            for node_name in (r["subject"], r["object"]):
                if G.has_node(node_name):
                    for cid in G.nodes[node_name].get("source_chunk_ids", []):
                        if cid not in seen:
                            seen.add(cid)
                            all_chunk_ids.append(cid)

        for e in entities:
            for cid in e["source_chunk_ids"]:
                if cid not in seen:
                    seen.add(cid)
                    all_chunk_ids.append(cid)

        entity_names = [e["name"] for e in entities]
        try:
            query_emb     = rag._get_ef()([query])[0]
            bfs_chunk_ids = _bfs_neighbors_scored(G, entity_names, BFS_HOPS, query_emb)
        except Exception:
            bfs_chunk_ids = _bfs_neighbors(G, entity_names, BFS_HOPS)
        for cid in bfs_chunk_ids:
            if cid not in seen:
                seen.add(cid)
                all_chunk_ids.append(cid)

        graph_summary = _build_graph_summary(entities, relations, G)
        return {
            "entities":         entities,
            "relations":        relations,
            "source_chunk_ids": all_chunk_ids,
            "graph_summary":    graph_summary,
            "graph_log":        graph_log,
        }

    except Exception as e:
        print(f"[graph_rag] retrieve_graph 失败: {e}")
        return _empty_graph_result()


def get_chunks_by_ids(chunk_ids: list[str]) -> list[dict]:
    """按 chunk_id 从 .chunks.json 批量取原文。"""
    if not chunk_ids:
        return []
    by_source: dict[str, list[tuple[int, str]]] = {}
    for cid in chunk_ids:
        parts = cid.rsplit("_", 1)
        if len(parts) == 2:
            source, idx_str = parts
            try:
                by_source.setdefault(source, []).append((int(idx_str), cid))
            except ValueError:
                continue

    result: list[dict] = []
    for source, idx_cid_pairs in by_source.items():
        md_file = rag.KNOWLEDGE_DIR / f"{source}.md"
        if not md_file.exists():
            continue
        try:
            chunks = rag._load_or_build_chunks(md_file, source)
        except Exception:
            continue
        for idx, cid in idx_cid_pairs:
            if 0 <= idx < len(chunks):
                chunk = dict(chunks[idx])
                chunk["chunk_id"] = cid
                result.append(chunk)
    return result


def get_subgraph_for_viz(query: str) -> dict:
    """返回可视化用的局部子图（used=直接命中，adjacent=BFS邻居）。"""
    if not rag.is_available():
        return {"nodes": [], "edges": []}
    try:
        ent_col = _get_entities_col()
        rel_col = _get_relations_col()
        if ent_col.count() == 0:
            return {"nodes": [], "edges": []}

        raw_ents = rag._safe_query(ent_col, query, ENTITY_TOP_K)
        used_names: set[str] = set()
        used_chunk_ids: dict[str, list[str]] = {}
        for _, meta in raw_ents:
            name = meta.get("name", "")
            if name:
                used_names.add(name)
                cids = [c for c in meta.get("source_chunk_ids_csv", "").split(",") if c]
                used_chunk_ids[name] = list(dict.fromkeys(used_chunk_ids.get(name, []) + cids))

        used_rel_keys: set[tuple[str, str]] = set()
        if rel_col.count() > 0:
            raw_rels = rag._safe_query(rel_col, query, RELATION_TOP_K)
            used_rel_keys = {
                (meta.get("subject", ""), meta.get("object", ""))
                for _, meta in raw_rels
            }

        G = _get_nx_graph()
        adjacent_names: set[str] = set()
        for name in used_names:
            if name in G:
                adjacent_names.update(G.successors(name))
                adjacent_names.update(G.predecessors(name))
        adjacent_names -= used_names
        all_names = used_names | adjacent_names

        nodes = []
        for name in all_names:
            attrs     = dict(G.nodes[name]) if name in G else {}
            chunk_ids = used_chunk_ids.get(name) or attrs.get("source_chunk_ids", [])
            nodes.append({
                "id":               name,
                "entity_type":      attrs.get("entity_type", "概念"),
                "description":      attrs.get("description", ""),
                "source_chunk_ids": chunk_ids,
                "used":             name in used_names,
                "adjacent":         name in adjacent_names,
            })

        edges = [
            {
                "source":    u, "target": v,
                "predicate": data.get("predicate", ""),
                "used":      (u, v) in used_rel_keys,
            }
            for u, v, data in G.edges(data=True)
            if u in all_names and v in all_names
        ]
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        print(f"[graph_rag] get_subgraph_for_viz 失败: {e}")
        return {"nodes": [], "edges": []}


def graph_stats() -> dict:
    try:
        G       = _get_nx_graph()
        ent_col = _get_entities_col() if rag.is_available() else None
        rel_col = _get_relations_col() if rag.is_available() else None
        graph_files = [f.name for f in GRAPH_DIR.glob("*.graph.json")] if GRAPH_DIR.exists() else []
        return {
            "nodes":            G.number_of_nodes(),
            "edges":            G.number_of_edges(),
            "entity_vectors":   ent_col.count() if ent_col else 0,
            "relation_vectors": rel_col.count() if rel_col else 0,
            "graph_files":      graph_files,
        }
    except Exception as e:
        return {"error": str(e)}
