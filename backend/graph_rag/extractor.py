"""Graph RAG — LLM 实体/关系抽取"""
from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm.provider import LLMProvider

import rag  # 复用 LLM_TIMEOUT, _is_ratelimit

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
    """以 chunk i 为中心，向前后各扩展 1 个同 H2 的 chunk 构成上下文窗口。"""
    center_h2 = chunks[i].get("h2", "")
    parts: list[str] = []
    if i > 0 and chunks[i - 1].get("h2", "") == center_h2:
        parts.append(chunks[i - 1]["text"])
    parts.append(chunks[i]["text"])
    if i < len(chunks) - 1 and chunks[i + 1].get("h2", "") == center_h2:
        parts.append(chunks[i + 1]["text"])
    return "\n\n".join(parts)


def _parse_extraction_response(response: str) -> tuple[list[dict], list[dict]]:
    """解析 JSON-lines 响应，失败行静默跳过。返回 (entities, relations)。"""
    entities: list[dict] = []
    relations: list[dict] = []
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
    """单次 LLM 调用，抽取实体和关系。失败时返回空列表。"""
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
            raise
        print(f"[graph_rag] 抽取失败(非限速): {e}")
        return [], []
