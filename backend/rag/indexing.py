"""知识库索引：QA 生成、向量写入、缓存管理"""
from __future__ import annotations

import asyncio
import time as _time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm.provider import LLMProvider

from .client import (
    is_available, KNOWLEDGE_DIR, QA_PER_CHUNK,
    _get_knowledge_col,
)
from .chunking import _load_or_build_chunks

# ── 索引进度 ───────────────────────────────────────────────────────────────────

_index_progress: dict = {
    "status":        "idle",
    "file":          "",
    "chunks_done":   0,
    "chunks_total":  0,
    "vectors_added": 0,
    "elapsed_s":     0.0,
    "eta_s":         None,
    "error":         None,
}


def get_index_progress() -> dict:
    return dict(_index_progress)


# ── 自适应并发信号量 ────────────────────────────────────────────────────────────

class AdaptiveSemaphore:
    """TCP 风格自适应并发控制（慢启动 + AIMD）。"""
    MIN = 1
    MAX = 64

    def __init__(self, initial: int = 1, ssthresh: int = 8):
        self._limit   = initial
        self._ssthresh = ssthresh
        self._active  = 0
        self._streak  = 0
        self._cond    = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    async def __aenter__(self):
        async with self._cond:
            while self._active >= self._limit:
                await self._cond.wait()
            self._active += 1
        return self

    async def __aexit__(self, exc_type, *_):
        success = exc_type is None
        async with self._cond:
            was_full = self._active >= self._limit
            self._active -= 1
            self._adjust(success, was_full)
            self._cond.notify_all()
        return False

    def _adjust(self, success: bool, was_full: bool) -> None:
        if not success:
            new_thresh = max(self.MIN, self._limit // 2)
            print(f"[adaptive] 失败 → ssthresh={new_thresh}, 并发 {self._limit}→{new_thresh}")
            self._ssthresh = new_thresh
            self._limit    = new_thresh
            self._streak   = 0
            return
        if not was_full:
            return
        self._streak += 1
        if self._streak < self._limit:
            return
        self._streak = 0
        if self._limit < self._ssthresh:
            new   = min(self.MAX, self._limit * 2)
            phase = "慢启动×2"
        else:
            new   = min(self.MAX, self._limit + 1)
            phase = "线性+1"
        if new != self._limit:
            print(f"[adaptive] {phase} → 并发 {self._limit}→{new} (ssthresh={self._ssthresh})")
        self._limit = new

    def stats(self) -> dict:
        return {"limit": self._limit, "ssthresh": self._ssthresh, "active": self._active}


# ── LLM 问题生成 ───────────────────────────────────────────────────────────────

LLM_TIMEOUT = 60.0
_RATELIMIT_KEYWORDS = ("rate", "quota", "limit", "overload", "capacity", "timeout", "429")


def _is_ratelimit(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in str(e) or any(kw in msg for kw in _RATELIMIT_KEYWORDS)


async def _generate_questions(chunk_text: str, provider: "LLMProvider", n: int = QA_PER_CHUNK) -> list[str]:
    """调用 LLM 对 chunk 生成 n 个不同角度的面试问题。限速/超时错误向上抛出。"""
    prompt = (
        f"根据以下技术内容，生成{n}个不同角度的面试问题。\n"
        "只输出问题本身，每行一个，不加序号和任何前缀符号。\n\n"
        f"内容：\n{chunk_text[:800]}"
    )
    try:
        response = await asyncio.wait_for(
            provider.chat(
                messages=[{"role": "user", "content": prompt}],
                system="你是一位技术面试题生成助手，根据知识点生成有代表性、不同问法的面试问题。",
            ),
            timeout=LLM_TIMEOUT,
        )
        return [line.strip() for line in response.strip().split('\n') if line.strip()][:n]
    except asyncio.TimeoutError:
        raise RuntimeError(f"LLM timeout after {LLM_TIMEOUT}s")
    except Exception as e:
        if _is_ratelimit(e):
            raise
        print(f"[rag] 问题生成失败(非限速): {e}")
        return []


# ── 知识库索引 ─────────────────────────────────────────────────────────────────

async def index_knowledge_with_qa(provider: "LLMProvider | None" = None) -> int:
    """扫描 knowledge/，对每个 chunk 生成多问题向量（若有 provider）。返回新增 chunk 数。"""
    global _index_progress
    import json as _json

    if not is_available() or not KNOWLEDGE_DIR.exists():
        return 0

    col = _get_knowledge_col()
    pending: list[tuple[Path, list[dict]]] = []
    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        existing = col.get(where={"source": md_file.stem}, limit=1)
        if existing["ids"]:
            continue
        chunks = _load_or_build_chunks(md_file, md_file.stem)
        if chunks:
            pending.append((md_file, chunks))

    total_chunks = sum(len(c) for _, c in pending)
    if total_chunks == 0:
        _index_progress.update({"status": "idle"})
        return 0

    mode = f"Q&A×{QA_PER_CHUNK}" if provider else "直接"
    _index_progress.update({
        "status": "running", "chunks_done": 0, "chunks_total": total_chunks,
        "vectors_added": 0, "elapsed_s": 0.0, "eta_s": None, "error": None,
    })
    t_start   = _time.monotonic()
    total_new = 0

    try:
        sem = AdaptiveSemaphore(initial=1, ssthresh=8)

        for md_file, chunks in pending:
            source   = md_file.stem
            qa_path  = md_file.with_suffix(".qa.json")
            _index_progress["file"] = md_file.name

            qa_cache: dict[str, list[str]] = {}
            if qa_path.exists():
                try:
                    qa_cache = _json.loads(qa_path.read_text(encoding="utf-8"))
                    cached = sum(1 for k in qa_cache if k.startswith(source + "_"))
                    print(f"[rag] 读取 QA 缓存: {qa_path.name}  ({cached}/{len(chunks)} chunks)")
                except Exception:
                    qa_cache = {}

            done_ref = [_index_progress["chunks_done"]]

            if provider:
                print(f"[rag] QA 生成({mode}): {md_file.name}  ({len(chunks)} chunks)")

                async def _gen_one(i: int) -> None:
                    chunk_id = f"{source}_{i}"
                    if chunk_id in qa_cache:
                        done_ref[0] += 1
                        return
                    qs: list[str] = []
                    try:
                        async with sem:
                            qs = await _generate_questions(chunks[i]["text"], provider)
                    except Exception as e:
                        wait = min(10, 3 + sem.limit)
                        print(f"[rag] chunk {i} 限速/超时，等待 {wait}s: {e}")
                        await asyncio.sleep(wait)
                    qa_cache[chunk_id] = qs
                    qa_path.write_text(_json.dumps(qa_cache, ensure_ascii=False, indent=2), encoding="utf-8")
                    done_ref[0] += 1
                    done    = done_ref[0]
                    elapsed = _time.monotonic() - t_start
                    eta     = elapsed / done * (total_chunks - done) if done else None
                    _index_progress.update({
                        "chunks_done": done, "elapsed_s": round(elapsed, 1),
                        "eta_s": round(eta, 0) if eta is not None else None,
                    })

                await asyncio.gather(*[_gen_one(i) for i in range(len(chunks))])

            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict] = []
            for i, chunk in enumerate(chunks):
                chunk_id   = f"{source}_{i}"
                chunk_text = chunk["text"]
                chapter    = chunk.get("chapter", "")
                path       = chunk.get("path", chapter)
                questions  = qa_cache.get(chunk_id, []) if provider else []

                if questions:
                    for j, q in enumerate(questions):
                        ids.append(f"{chunk_id}_q{j}")
                        documents.append(q)
                        metadatas.append({
                            "source": source, "path": path,
                            "h1": chunk.get("h1", ""), "h2": chunk.get("h2", ""), "h3": chunk.get("h3", ""),
                            "chapter": chapter, "chunk_id": chunk_id, "question": q, "text": chunk_text,
                        })
                else:
                    ids.append(chunk_id)
                    documents.append(chunk_text)
                    metadatas.append({
                        "source": source, "path": path,
                        "h1": chunk.get("h1", ""), "h2": chunk.get("h2", ""), "h3": chunk.get("h3", ""),
                        "chapter": chapter, "chunk_id": chunk_id, "text": chunk_text,
                    })
                if not provider:
                    done_ref[0] += 1
                    done    = done_ref[0]
                    elapsed = _time.monotonic() - t_start
                    eta     = elapsed / done * (total_chunks - done) if done else None
                    _index_progress.update({
                        "chunks_done": done, "elapsed_s": round(elapsed, 1),
                        "eta_s": round(eta, 0) if eta is not None else None,
                    })
                    print(f"[rag]   chunk {done}/{total_chunks}  {source} > {chapter[:30]}")

            if ids:
                col.add(ids=ids, documents=documents, metadatas=metadatas)
                _index_progress["vectors_added"] += len(ids)
                total_new += len(chunks)
                print(f"[rag] 完成: {md_file.name} → {len(chunks)} chunks, {len(ids)} vectors")

        _index_progress["status"] = "done"
    except Exception as e:
        print(f"[rag] 索引出错: {e}")
        _index_progress.update({"status": "error", "error": str(e)})

    return total_new


def backfill_qa_cache() -> dict[str, int]:
    """从 ChromaDB 回填 .qa.json 缓存文件。"""
    import json as _json
    if not is_available() or not KNOWLEDGE_DIR.exists():
        return {}

    col     = _get_knowledge_col()
    results: dict[str, int] = {}

    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        source  = md_file.stem
        qa_path = md_file.with_suffix(".qa.json")
        if qa_path.exists():
            continue
        try:
            data = col.get(where={"source": source}, include=["metadatas"])
        except Exception:
            continue
        if not data["metadatas"]:
            continue

        qa_cache: dict[str, list[str]] = {}
        for meta in data["metadatas"]:
            cid = meta.get("chunk_id", "")
            q   = meta.get("question", "")
            if cid and q:
                qa_cache.setdefault(cid, [])
                if q not in qa_cache[cid]:
                    qa_cache[cid].append(q)

        if qa_cache:
            qa_path.write_text(_json.dumps(qa_cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[rag] 回填 QA 缓存: {qa_path.name}  ({len(qa_cache)} chunks)")
            results[source] = len(qa_cache)

    return results


def index_knowledge() -> int:
    """同步退化版本（无 LLM），供外部同步调用。"""
    import asyncio as _asyncio
    return _asyncio.run(index_knowledge_with_qa(provider=None))
