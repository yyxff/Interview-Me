"""Tool: explore_concept — 知识图谱 BFS 联想"""
from __future__ import annotations

import logging

from graph_rag.retrieval import explore_concept_bfs

logger = logging.getLogger(__name__)

DESC = (
    "explore_concept: 在知识图谱中联想与某概念直接相关的邻居概念和关系。"
    "适合决定下一步考察哪个方向。\n"
    '  用法：explore_concept("GC") → 返回 Stop-the-World、引用计数、G1 等关联概念'
)


def explore_concept(entity: str) -> str:
    """图谱 BFS 联想，返回与给定概念直接相关的邻居节点和关系描述。"""
    logger.debug("explore_concept entity=%r", entity)
    result = explore_concept_bfs(entity)
    out = result["summary"]
    logger.debug(
        "explore_concept matched=%r, %d neighbors",
        result.get("matched"), len(result.get("neighbors", [])),
    )
    return out
