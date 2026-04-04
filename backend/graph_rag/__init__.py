"""graph_rag 包 — 知识图谱构建与检索"""
from .store import (
    GRAPH_DIR,
    _GRAPH_ENTITIES_COL, _GRAPH_RELATIONS_COL,
    ENTITY_TOP_K, RELATION_TOP_K, BFS_HOPS,
    _get_entities_col, _get_relations_col,
    _get_nx_graph,
    get_graph_index_progress,
)
from .builder import (
    _build_entity_embed_text, _build_relation_embed_text,
    _build_graph_for_source, _save_graph, _load_graph,
    _index_graph_to_chroma,
    index_knowledge_graph,
)
from .retrieval import (
    retrieve_graph, get_chunks_by_ids,
    get_subgraph_for_viz, graph_stats,
)

__all__ = [
    # store
    "GRAPH_DIR", "_GRAPH_ENTITIES_COL", "_GRAPH_RELATIONS_COL",
    "_get_entities_col", "_get_relations_col", "_get_nx_graph",
    "get_graph_index_progress",
    # builder
    "index_knowledge_graph",
    "_save_graph", "_load_graph", "_index_graph_to_chroma",
    # retrieval
    "retrieve_graph", "get_chunks_by_ids",
    "get_subgraph_for_viz", "graph_stats",
]
