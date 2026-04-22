"""Graph RAG — Qdrant 单例（通过 rag._get_client() 适配器）、NetworkX 图、进度追踪"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import rag  # 复用 _get_client, _get_ef, is_available, KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

# ── 路径常量 ──────────────────────────────────────────────────────────────────

GRAPH_DIR = Path(__file__).parent.parent / "graph"

_GRAPH_ENTITIES_COL  = "graph_entities"
_GRAPH_RELATIONS_COL = "graph_relations"

# 检索参数
ENTITY_TOP_K   = 10
RELATION_TOP_K = 6
BFS_HOPS       = 1

# 实体去重：去除空格/全角空格/间隔号
_ENTITY_DEDUP_NORM = str.maketrans("", "", " \t\u3000·・")

# 图索引并发初始值
GRAPH_CONCURRENCY_INITIAL = 1

# ── 单例 ─────────────────────────────────────────────────────────────────────

_entities_col:  object = None
_relations_col: object = None
_nx_graph:      object = None


def _get_entities_col():
    global _entities_col
    if _entities_col is None:
        _entities_col = rag._get_client().get_or_create_collection(
            _GRAPH_ENTITIES_COL, embedding_function=rag._get_ef()
        )
    return _entities_col


def _get_relations_col():
    global _relations_col
    if _relations_col is None:
        _relations_col = rag._get_client().get_or_create_collection(
            _GRAPH_RELATIONS_COL, embedding_function=rag._get_ef()
        )
    return _relations_col


def _get_nx_graph():
    global _nx_graph
    if _nx_graph is None:
        _nx_graph = _load_all_graphs_into_nx()
    return _nx_graph


def _load_all_graphs_into_nx():
    import networkx as nx
    G = nx.DiGraph()
    if not GRAPH_DIR.exists():
        return G
    for gf in sorted(GRAPH_DIR.glob("*.graph.json")):
        try:
            g = json.loads(gf.read_text(encoding="utf-8"))
            source = g.get("source", gf.stem)
            for name, attrs in g.get("nodes", {}).items():
                if G.has_node(name):
                    G.nodes[name]["source_chunk_ids"].extend(attrs.get("source_chunk_ids", []))
                else:
                    G.add_node(name, source=source, **attrs)
            for edge in g.get("edges", []):
                subj    = edge["subject"]
                obj     = edge["object"]
                src_cid = edge.get("source_chunk_id", "")
                for n, desc_key in ((subj, "subject_desc"), (obj, "object_desc")):
                    if not G.has_node(n):
                        G.add_node(n, entity_type="概念",
                                   description=edge.get(desc_key, ""),
                                   source_chunk_ids=[src_cid] if src_cid else [],
                                   entity_id="")
                    elif src_cid and src_cid not in G.nodes[n].get("source_chunk_ids", []):
                        G.nodes[n]["source_chunk_ids"].append(src_cid)
                G.add_edge(subj, obj, **{k: v for k, v in edge.items()
                                          if k not in ("subject", "object")})
        except Exception as e:
            logger.warning("[graph_rag] 加载图失败 %s: %s", gf.name, e)
    logger.info("[graph_rag] 图加载完成: %d 节点, %d 边", G.number_of_nodes(), G.number_of_edges())
    return G


# ── 进度追踪 ─────────────────────────────────────────────────────────────────

_graph_index_progress: dict = {
    "status":       "idle",
    "source":       "",
    "chunks_done":  0,
    "chunks_total": 0,
    "entities":     0,
    "relations":    0,
    "elapsed_s":    0.0,
    "eta_s":        None,
    "concurrency":  GRAPH_CONCURRENCY_INITIAL,
    "error":        None,
}


def get_graph_index_progress() -> dict:
    return dict(_graph_index_progress)
