"""
Graph RAG 模块

知识图谱构建：
  - 对每个 chunk 建上下文窗口（同 H2 内前后各1 chunk）
  - LLM 单次调用抽取实体 + 关系（含 subject_desc/object_desc 语义增强）
  - 实体合并：同名实体跨 chunk 合并 source_chunk_ids + 描述
  - 持久化：backend/graph/{source}.graph.json

向量化（增强三元组表示）：
  - 实体："{name}（{entity_type}）：{description}"
  - 关系："{subject}（{subject_desc}）--{predicate}-->{object}（{object_desc}）：{description}"

ChromaDB 集合：
  - graph_entities  — 实体描述向量
  - graph_relations — 关系增强三元组向量

查询：
  - 实体向量搜索 → BFS 1跳展开 → 收集 source_chunk_ids
  - 关系向量搜索 → 收集 source_chunk_ids
  - 合并去重，返回 graph_summary 注入 LLM system prompt
"""
from __future__ import annotations

import asyncio
import json
import re
import time as _time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import LLMProvider

import rag  # 复用 _get_client, _get_ef, is_available, KNOWLEDGE_DIR

# ── 路径常量 ──────────────────────────────────────────────────────────────────

GRAPH_DIR = Path(__file__).parent / "graph"

_GRAPH_ENTITIES_COL  = "graph_entities"
_GRAPH_RELATIONS_COL = "graph_relations"

# 检索参数
ENTITY_TOP_K   = 5
RELATION_TOP_K = 3
BFS_HOPS       = 1

# 图索引自适应并发起始值（使用 rag.AdaptiveSemaphore，TCP 慢启动风格）
GRAPH_CONCURRENCY_INITIAL = 1

# ── 单例 ─────────────────────────────────────────────────────────────────────

_entities_col:  object = None
_relations_col: object = None
_nx_graph:      object = None   # networkx.DiGraph

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

# ── ChromaDB 集合懒加载 ───────────────────────────────────────────────────────

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


# ── NetworkX 图懒加载 ─────────────────────────────────────────────────────────

def _get_nx_graph():
    """首次调用时加载所有 .graph.json 到内存图，后续复用。"""
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
                subj = edge["subject"]
                obj  = edge["object"]
                src_cid = edge.get("source_chunk_id", "")
                # 确保边的端点节点存在，用边的 source_chunk_id 和 desc 补全缺失信息
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
            print(f"[graph_rag] 加载图失败 {gf.name}: {e}")
    print(f"[graph_rag] 图加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G


# ── 抽取 ─────────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
你是一个知识图谱抽取助手，专门处理计算机系统/网络领域的技术文档。

从下面的技术文本中抽取：
1. 【实体】：技术概念、算法、数据结构、机制、系统组件
2. 【关系】：实体之间的明确关系

输出规则：
- 每行一条 JSON，不加代码块或注释
- 实体行格式：{{"type":"entity","name":"进程","entity_type":"概念","description":"操作系统资源分配的基本单位，拥有独立地址空间和PCB"}}
- 关系行格式：{{"type":"relation","subject":"进程切换","subject_desc":"进程间CPU控制权转移过程","predicate":"依赖","object":"PCB","object_desc":"存储进程状态信息的数据结构","description":"进程切换时需要将当前进程状态保存至PCB，并从目标进程PCB中恢复状态"}}
- entity_type 只能是：概念/算法/数据结构/机制/组件
- predicate 必须是动词短语，≤6字（包含/依赖/对比/触发/属于/实现/优化/替代）
- name/subject/object ≤15字，使用标准术语
- description ≤80字；subject_desc/object_desc ≤20字，一句话点明核心含义
- 只输出文本中明确存在的实体和关系，不要推断
- 每个实体最多输出一次；关系最多5条

文本：
{context}"""


def _build_context_window(chunks: list[dict], i: int) -> str:
    """
    以 chunk i 为中心，向前后各扩展 1 个同 H2 的 chunk 构成上下文窗口。
    不跨 H2 边界，防止跨节引入噪声关系。
    """
    center_h2 = chunks[i].get("h2", "")
    parts: list[str] = []
    if i > 0 and chunks[i - 1].get("h2", "") == center_h2:
        parts.append(chunks[i - 1]["text"])
    parts.append(chunks[i]["text"])
    if i < len(chunks) - 1 and chunks[i + 1].get("h2", "") == center_h2:
        parts.append(chunks[i + 1]["text"])
    return "\n\n".join(parts)


def _parse_extraction_response(response: str) -> tuple[list[dict], list[dict]]:
    """
    解析 JSON-lines 响应，每行独立尝试解析，失败行静默跳过。
    返回 (entities, relations)。
    """
    entities: list[dict] = []
    relations: list[dict] = []
    # 清除 LLM 可能包裹的 markdown 代码块
    response = re.sub(r"```[a-z]*\n?", "", response)
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if obj.get("type") == "entity":
                name = (obj.get("name") or "").strip()
                desc = (obj.get("description") or "").strip()
                if name and desc:
                    entities.append({
                        "name":        name,
                        "entity_type": obj.get("entity_type", "概念"),
                        "description": desc[:200],
                    })
            elif obj.get("type") == "relation":
                subj = (obj.get("subject") or "").strip()
                obj_ = (obj.get("object") or "").strip()
                pred = (obj.get("predicate") or "").strip()
                if subj and obj_ and pred:
                    relations.append({
                        "subject":      subj,
                        "subject_desc": (obj.get("subject_desc") or "").strip()[:50],
                        "predicate":    pred[:10],
                        "object":       obj_,
                        "object_desc":  (obj.get("object_desc") or "").strip()[:50],
                        "description":  (obj.get("description") or "").strip()[:200],
                    })
        except (json.JSONDecodeError, KeyError):
            continue
    return entities, relations


async def _extract_entities_relations(
    context: str,
    provider: "LLMProvider",
) -> tuple[list[dict], list[dict]]:
    """单次 LLM 调用，从上下文窗口中抽取实体和关系。失败时返回空列表。"""
    prompt = _EXTRACTION_PROMPT.format(context=context[:2000])
    try:
        response = await asyncio.wait_for(
            provider.chat(
                messages=[{"role": "user", "content": prompt}],
                system="你是知识图谱抽取助手，只输出JSON行，不加任何注释或解释。",
            ),
            timeout=rag.LLM_TIMEOUT,
        )
        return _parse_extraction_response(response)
    except asyncio.TimeoutError:
        raise RuntimeError(f"LLM timeout after {rag.LLM_TIMEOUT}s")
    except Exception as e:
        if rag._is_ratelimit(e):
            raise   # 向上传播，触发自适应回退
        print(f"[graph_rag] 抽取失败(非限速): {e}")
        return [], []


# ── 图构建 ───────────────────────────────────────────────────────────────────

def _build_entity_embed_text(name: str, attrs: dict) -> str:
    """实体的增强嵌入文本：name + entity_type + description 拼接。"""
    return f"{name}（{attrs.get('entity_type', '概念')}）：{attrs.get('description', '')}"


def _build_relation_embed_text(edge: dict) -> str:
    """关系的增强嵌入文本：语义增强三元组，两端实体各带简介。"""
    subj  = edge["subject"]
    sdesc = edge.get("subject_desc", "")
    pred  = edge["predicate"]
    obj   = edge["object"]
    odesc = edge.get("object_desc", "")
    desc  = edge.get("description", "")
    subj_part = f"{subj}（{sdesc}）" if sdesc else subj
    obj_part  = f"{obj}（{odesc}）"  if odesc else obj
    return f"{subj_part}--{pred}-->{obj_part}：{desc}"


def _build_graph_for_source(
    source: str,
    extracted: list[tuple[list[dict], list[dict], str]],
) -> dict:
    """
    从抽取结果构建图数据结构。
    extracted: [(entities, relations, chunk_id), ...]
    同名实体合并 source_chunk_ids + description（用"；"拼接，上限200字）。
    同 (subject, object) 边：后出现的 description 覆盖。
    """
    nodes: dict[str, dict] = {}
    edge_map: dict[tuple[str, str], dict] = {}  # (subject, object) → edge

    ent_idx = 0
    rel_idx = 0

    for entities, relations, chunk_id in extracted:
        for ent in entities:
            name = ent["name"]
            if name in nodes:
                existing = nodes[name]
                if chunk_id not in existing["source_chunk_ids"]:
                    existing["source_chunk_ids"].append(chunk_id)
                # 合并描述（避免重复，加分号）
                new_desc = ent["description"]
                if new_desc and new_desc not in existing["description"]:
                    combined = existing["description"] + "；" + new_desc
                    existing["description"] = combined[:200]
            else:
                nodes[name] = {
                    "entity_type":     ent.get("entity_type", "概念"),
                    "description":     ent["description"],
                    "source_chunk_ids": [chunk_id],
                    "entity_id":       f"ent_{source}_{ent_idx}",
                }
                ent_idx += 1

        for rel in relations:
            subj = rel["subject"]
            obj  = rel["object"]
            key  = (subj, obj)
            if key in edge_map:
                # 后出现的描述覆盖（更靠后的 chunk 通常更详细）
                edge_map[key].update({
                    "subject_desc": rel.get("subject_desc", edge_map[key].get("subject_desc", "")),
                    "predicate":    rel["predicate"],
                    "object_desc":  rel.get("object_desc", edge_map[key].get("object_desc", "")),
                    "description":  rel.get("description", ""),
                    "source_chunk_id": chunk_id,
                })
            else:
                edge_map[key] = {
                    "subject":        subj,
                    "subject_desc":   rel.get("subject_desc", ""),
                    "predicate":      rel["predicate"],
                    "object":         obj,
                    "object_desc":    rel.get("object_desc", ""),
                    "description":    rel.get("description", ""),
                    "source_chunk_id": chunk_id,
                    "relation_id":    f"rel_{source}_{rel_idx}",
                }
                rel_idx += 1

    return {
        "source": source,
        "nodes":  nodes,
        "edges":  list(edge_map.values()),
    }


def _save_graph(source: str, graph: dict) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_DIR / f"{source}.graph.json"
    path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[graph_rag] 图已保存: {path.name} "
          f"({len(graph['nodes'])} 节点, {len(graph['edges'])} 边)")


def _load_graph(source: str) -> dict | None:
    path = GRAPH_DIR / f"{source}.graph.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── ChromaDB 索引 ─────────────────────────────────────────────────────────────

def _clear_chroma_for_source(col, source: str) -> None:
    """删除该 source 已有的所有向量（重建时调用）。"""
    try:
        existing = col.get(where={"source": source})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except Exception:
        pass


def _index_graph_to_chroma(graph: dict) -> None:
    """将图的实体和关系分别向量化存入 ChromaDB。"""
    source = graph["source"]
    ent_col = _get_entities_col()
    rel_col = _get_relations_col()

    _clear_chroma_for_source(ent_col, source)
    _clear_chroma_for_source(rel_col, source)

    # ── 实体 ──────────────────────────────────────────────────────────────────
    ent_ids:   list[str]  = []
    ent_docs:  list[str]  = []
    ent_metas: list[dict] = []

    for name, attrs in graph.get("nodes", {}).items():
        embed_text = _build_entity_embed_text(name, attrs)
        if not embed_text.strip():
            continue
        chunk_ids_csv = ",".join(attrs.get("source_chunk_ids", []))
        ent_ids.append(attrs["entity_id"])
        ent_docs.append(embed_text)
        ent_metas.append({
            "name":                 name,
            "entity_type":          attrs.get("entity_type", "概念"),
            "source_chunk_ids_csv": chunk_ids_csv,
            "entity_id":            attrs["entity_id"],
            "source":               source,
        })

    if ent_ids:
        ent_col.add(ids=ent_ids, documents=ent_docs, metadatas=ent_metas)

    # ── 关系 ──────────────────────────────────────────────────────────────────
    rel_ids:   list[str]  = []
    rel_docs:  list[str]  = []
    rel_metas: list[dict] = []

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

    print(f"[graph_rag] Chroma 索引: {source} → "
          f"{len(ent_ids)} 实体向量, {len(rel_ids)} 关系向量")


# ── 主索引入口 ────────────────────────────────────────────────────────────────

async def index_knowledge_graph(provider: "LLMProvider") -> dict:
    """
    扫描 KNOWLEDGE_DIR/*.md，对未建图的 source 执行完整的抽取→建图→索引流程。
    已有 .graph.json 的 source 自动跳过（删除文件可强制重建）。
    """
    global _graph_index_progress, _nx_graph

    if not rag.is_available():
        return {"error": "RAG 不可用"}
    if not rag.KNOWLEDGE_DIR.exists():
        return {"error": "knowledge/ 目录不存在"}

    # 找出需要处理的 source
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

    # 统计已有断点缓存的 chunk 数量（用于进度条初始值）
    cached_done = 0
    for source, _ in pending:
        p = GRAPH_DIR / f"{source}.graph.partial.json"
        if p.exists():
            try:
                cached_done += len(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass

    _graph_index_progress.update({
        "status":       "running",
        "chunks_done":  cached_done,
        "chunks_total": total_chunks,
        "entities":     0,
        "relations":    0,
        "elapsed_s":    0.0,
        "eta_s":        None,
        "concurrency":  GRAPH_CONCURRENCY_INITIAL,
        "error":        None,
    })
    t_start = _time.monotonic()
    total_entities = total_relations = 0

    try:
        sem = rag.AdaptiveSemaphore(initial=GRAPH_CONCURRENCY_INITIAL, ssthresh=8)
        for source, chunks in pending:
            _graph_index_progress["source"] = source

            # ── 断点续传：加载已完成 chunk 的抽取缓存 ──────────────────────
            GRAPH_DIR.mkdir(parents=True, exist_ok=True)
            partial_path  = GRAPH_DIR / f"{source}.graph.partial.json"
            partial_cache: dict[str, dict] = {}   # chunk_id → {entities, relations}
            if partial_path.exists():
                try:
                    partial_cache = json.loads(partial_path.read_text(encoding="utf-8"))
                    print(f"[graph_rag] 读取断点缓存: {partial_path.name}"
                          f"  ({len(partial_cache)}/{len(chunks)} chunks 已完成)")
                except Exception:
                    partial_cache = {}

            # chunks_done_ref 从已缓存数量开始（断点续传时进度连续）
            chunks_done_ref = [_graph_index_progress["chunks_done"]]

            async def _extract_one(i: int) -> tuple[list[dict], list[dict], str]:
                chunk_id = f"{source}_{i}"
                context  = _build_context_window(chunks, i)
                ents: list[dict] = []
                rels: list[dict] = []

                if chunk_id in partial_cache:
                    # 已有缓存，跳过 LLM
                    cached = partial_cache[chunk_id]
                    ents   = cached.get("entities", [])
                    rels   = cached.get("relations", [])
                else:
                    try:
                        async with sem:
                            ents, rels = await _extract_entities_relations(context, provider)
                    except Exception as e:
                        wait = min(10, 3 + sem.limit)
                        print(f"[graph_rag] chunk {i} 限速/超时，等待 {wait}s: {e}")
                        await asyncio.sleep(wait)
                    # 立即写入断点缓存（asyncio 单线程，无需锁）
                    partial_cache[chunk_id] = {"entities": ents, "relations": rels}
                    partial_path.write_text(
                        json.dumps(partial_cache, ensure_ascii=False),
                        encoding="utf-8",
                    )

                chunks_done_ref[0] += 1
                done      = chunks_done_ref[0]
                elapsed   = _time.monotonic() - t_start
                remaining = total_chunks - done
                eta       = elapsed / done * remaining if done else None
                _graph_index_progress.update({
                    "chunks_done":  done,
                    "elapsed_s":    round(elapsed, 1),
                    "eta_s":        round(eta, 0) if eta is not None else None,
                    "concurrency":  sem.limit,
                })
                if done % 20 == 0:
                    print(f"[graph_rag] {source}: {done}/{total_chunks} "
                          f"chunks done (并发={sem.limit}, "
                          f"ssthresh={sem.stats()['ssthresh']})")
                return ents, rels, chunk_id

            results   = await asyncio.gather(*[_extract_one(i) for i in range(len(chunks))])
            extracted = list(results)

            graph = _build_graph_for_source(source, extracted)
            _save_graph(source, graph)
            _index_graph_to_chroma(graph)

            # 完成后删除断点缓存
            if partial_path.exists():
                partial_path.unlink()

            n_ents = len(graph.get("nodes", {}))
            n_rels = len(graph.get("edges", []))
            total_entities += n_ents
            total_relations += n_rels
            _graph_index_progress["entities"]  += n_ents
            _graph_index_progress["relations"] += n_rels

        # 重新加载内存图
        _nx_graph = None
        _get_nx_graph()
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


def get_graph_index_progress() -> dict:
    return dict(_graph_index_progress)


# ── 查询 ─────────────────────────────────────────────────────────────────────

def _empty_graph_result() -> dict:
    return {"entities": [], "relations": [], "source_chunk_ids": [], "graph_summary": ""}


def _bfs_neighbors(G, start_nodes: list[str], hops: int) -> list[str]:
    """
    从 start_nodes 出发做 BFS（双向：successors + predecessors）。
    返回邻居节点的 source_chunk_ids 去重列表（不含 start_nodes 自身）。
    """
    visited   = set(start_nodes)
    frontier  = set(n for n in start_nodes if G.has_node(n))
    chunk_ids: list[str] = []
    seen_ids:  set[str]  = set()

    for _ in range(hops):
        next_frontier: set[str] = set()
        for node in frontier:
            neighbors = list(G.successors(node)) + list(G.predecessors(node))
            for nb in neighbors:
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


def _build_graph_summary(
    entities: list[dict],
    relations: list[dict],
    G,
) -> str:
    """
    生成 ≤3 行的图谱上下文摘要，注入 LLM system prompt。
    """
    if not entities and not relations:
        return ""

    lines: list[str] = []

    if entities:
        parts = [f"{e['name']}（{e['entity_type']}）" for e in entities[:4]]
        lines.append("发现实体：" + "、".join(parts))

    if relations:
        parts = [f"{r['subject']} --{r['predicate']}--> {r['object']}"
                 for r in relations[:3]]
        lines.append("关键关系：" + "；".join(parts))

    # BFS 邻居（不在入口实体集合中）
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
    """
    图增强检索。
    返回 {entities, relations, source_chunk_ids, graph_summary}。
    ChromaDB 未索引时优雅降级，返回空结果。
    """
    if not rag.is_available():
        return _empty_graph_result()

    try:
        ent_col = _get_entities_col()
        rel_col = _get_relations_col()

        if ent_col.count() == 0:
            return _empty_graph_result()

        # Step 1: 实体向量检索
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
            graph_log.append({
                "type": "entity",
                "name": meta.get("name", ""),
                "dist": round(dist, 4),
            })

        # Step 2: 关系向量检索
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
                    "type":      "relation",
                    "triple":    f"{meta.get('subject','')} --{meta.get('predicate','')}--> {meta.get('object','')}",
                    "dist":      round(dist, 4),
                })

        # Step 3: 合并去重——只取命中实体 + 命中关系的 chunk（不展开 BFS 邻居）
        G = _get_nx_graph()
        all_chunk_ids: list[str] = []
        seen: set[str] = set()

        for e in entities:
            for cid in e["source_chunk_ids"]:
                if cid not in seen:
                    seen.add(cid)
                    all_chunk_ids.append(cid)

        for r in relations:
            cid = r["source_chunk_id"]
            if cid and cid not in seen:
                seen.add(cid)
                all_chunk_ids.append(cid)

        # Step 4: 图谱摘要（BFS 邻居只用于摘要文字，不加入 chunk 上下文）
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
    """
    按 chunk_id 从 .chunks.json 中批量取原文。
    chunk_id 格式："{source}_{index}"（最后一个 _ 后为整数索引）。
    返回 chunk dict 列表，每个 dict 追加 chunk_id 字段。
    """
    if not chunk_ids:
        return []

    # 按 source 分组，避免重复加载同一文件
    by_source: dict[str, list[tuple[int, str]]] = {}
    for cid in chunk_ids:
        # rsplit 只拆最后一个 _，source 名称可能含 _
        parts = cid.rsplit("_", 1)
        if len(parts) == 2:
            source, idx_str = parts
            try:
                idx = int(idx_str)
                by_source.setdefault(source, []).append((idx, cid))
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
    """
    返回可视化用的局部子图。
    - used=True  ：被向量检索直接命中的实体/关系
    - adjacent=True：BFS 1跳邻居（未直接命中）
    """
    if not rag.is_available():
        return {"nodes": [], "edges": []}
    try:
        ent_col = _get_entities_col()
        rel_col = _get_relations_col()
        if ent_col.count() == 0:
            return {"nodes": [], "edges": []}

        raw_ents = rag._safe_query(ent_col, query, ENTITY_TOP_K)
        # 从 ChromaDB 结果直接拿 chunk_ids（比 NetworkX 属性更准确）
        used_names: set[str] = set()
        used_chunk_ids: dict[str, list[str]] = {}  # name → chunk_ids
        for _, meta in raw_ents:
            name = meta.get("name", "")
            if name:
                used_names.add(name)
                cids = [c for c in meta.get("source_chunk_ids_csv", "").split(",") if c]
                used_chunk_ids.setdefault(name, [])
                used_chunk_ids[name] = list(dict.fromkeys(used_chunk_ids[name] + cids))

        used_rel_keys: set[tuple[str, str]] = set()
        if rel_col.count() > 0:
            raw_rels = rag._safe_query(rel_col, query, RELATION_TOP_K)
            used_rel_keys = {
                (meta.get("subject", ""), meta.get("object", ""))
                for _, meta in raw_rels
            }

        G = _get_nx_graph()

        # BFS 1-hop neighbors of used nodes
        adjacent_names: set[str] = set()
        for name in used_names:
            if name in G:
                adjacent_names.update(G.successors(name))
                adjacent_names.update(G.predecessors(name))
        adjacent_names -= used_names

        all_names = used_names | adjacent_names

        nodes = []
        for name in all_names:
            attrs = dict(G.nodes[name]) if name in G else {}
            if name in used_chunk_ids:
                chunk_ids = used_chunk_ids[name]
            else:
                chunk_ids = attrs.get("source_chunk_ids", [])
            nodes.append({
                "id":               name,
                "entity_type":      attrs.get("entity_type", "概念"),
                "description":      attrs.get("description", ""),
                "source_chunk_ids": chunk_ids,
                "used":             name in used_names,
                "adjacent":         name in adjacent_names,
            })

        edges = []
        for u, v, data in G.edges(data=True):
            if u in all_names and v in all_names:
                edges.append({
                    "source":    u,
                    "target":    v,
                    "predicate": data.get("predicate", ""),
                    "used":      (u, v) in used_rel_keys,
                })

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        print(f"[graph_rag] get_subgraph_for_viz 失败: {e}")
        return {"nodes": [], "edges": []}


def graph_stats() -> dict:
    """返回图谱统计信息，供 /graph/stats 端点使用。"""
    try:
        G = _get_nx_graph()
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
