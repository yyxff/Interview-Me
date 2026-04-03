"""
RAG 模块 - 向量检索增强生成

知识库:  backend/knowledge/*.md  (全局，启动时自动索引)
简历库:  用户上传 PDF，按 session_id 隔离
向量库:  ChromaDB (backend/chroma_db/, 本地持久化)
Embedding: BAAI/bge-small-zh-v1.5 (本地，首次使用自动下载 ~130MB)

索引策略:
  - 若传入 LLM provider，对每个 chunk 生成 N 个问题，每个问题单独向量化
    但 document 仍存原文 → 检索时返回原文给 LLM，同时去重
  - 若无 provider，直接用原文向量化（退化模式）
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import LLMProvider

# ── 路径常量 ───────────────────────────────────────────────────────────────────

CHROMA_PATH   = Path(__file__).parent / "chroma_db"
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
NOTES_DIR     = Path(__file__).parent / "notes"

# ChromaDB 集合名
_KNOWLEDGE_COL = "knowledge"
_RESUME_COL    = "resumes"
_NOTES_COL     = "notes"
# 检索参数
KNOWLEDGE_TOP_K = 3   # 去重后最多返回几条原文
RESUME_TOP_K    = 2

# 每个 chunk 生成的问题数量
QA_PER_CHUNK = 4

# ── 懒加载单例 ─────────────────────────────────────────────────────────────────

_chroma_client  = None
_embed_fn       = None
_reranker       = None
_knowledge_col  = None
_resume_col     = None
_notes_col      = None
_rag_available  = None

# ── 索引进度 ───────────────────────────────────────────────────────────────────

import time as _time

_index_progress: dict = {
    "status":        "idle",   # idle | running | done | error
    "file":          "",
    "chunks_done":   0,
    "chunks_total":  0,
    "vectors_added": 0,
    "elapsed_s":     0.0,
    "eta_s":         None,     # None = 未知
    "error":         None,
}


def get_index_progress() -> dict:
    return dict(_index_progress)


def is_available() -> bool:
    global _rag_available
    if _rag_available is not None:
        return _rag_available
    try:
        import chromadb                    # noqa: F401
        import sentence_transformers       # noqa: F401
        import pdfplumber                  # noqa: F401
        _rag_available = True
    except ImportError:
        _rag_available = False
    return _rag_available


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _chroma_client


def _get_ef():
    global _embed_fn
    if _embed_fn is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        print("[rag] 加载 Embedding 模型 BAAI/bge-small-zh-v1.5 ...")
        _embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5"
        )
        print("[rag] Embedding 模型加载完成")
    return _embed_fn


def _get_reranker():
    """获取 Cross-encoder Reranker（首次调用会下载模型 ~280MB）。"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("[rag] 加载 Reranker 模型 BAAI/bge-reranker-base ...")
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
        print("[rag] Reranker 加载完成")
    return _reranker


def rerank(query: str, chunks: list[tuple[str, dict]]) -> list[tuple[str, dict, float]]:
    """
    用 cross-encoder 对候选 chunks 重排序。
    输入：[(doc, meta), ...]
    返回：[(doc, meta, score), ...] 按 score 降序
    """
    if not chunks:
        return []
    try:
        reranker = _get_reranker()
        pairs  = [(query, doc) for doc, _ in chunks]
        scores = reranker.predict(pairs).tolist()
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [(doc, meta, score) for (doc, meta), score in ranked]
    except Exception as e:
        print(f"[rag] Reranker 失败，退化为原顺序: {e}")
        return [(doc, meta, 0.0) for doc, meta in chunks]


def _get_knowledge_col():
    global _knowledge_col
    if _knowledge_col is None:
        _knowledge_col = _get_client().get_or_create_collection(
            _KNOWLEDGE_COL, embedding_function=_get_ef()
        )
    return _knowledge_col


def _get_resume_col():
    global _resume_col
    if _resume_col is None:
        _resume_col = _get_client().get_or_create_collection(
            _RESUME_COL, embedding_function=_get_ef()
        )
    return _resume_col


def _get_notes_col():
    global _notes_col
    if _notes_col is None:
        _notes_col = _get_client().get_or_create_collection(
            _NOTES_COL, embedding_function=_get_ef()
        )
    return _notes_col



# ── 文本切块 ───────────────────────────────────────────────────────────────────

def _chunk_markdown(text: str, source: str) -> list[dict]:
    """
    按 ## 标题切割 Markdown，每块 ≤800 字符。
    线性扫描追踪 H1/H2/H3 层级，每个 chunk 携带完整路径。
    返回 [{"text", "source", "path", "h1", "h2", "h3", "chapter"}, ...]
    """
    chunks: list[dict] = []
    h1 = h2 = h3 = ""

    # 按行扫描，遇到 ## 边界则切节
    lines = text.split('\n')
    sections: list[tuple[str, str, str, str]] = []  # (section_text, h1, h2, h3_at_start)
    current_lines: list[str] = []
    current_h1 = current_h2 = current_h3 = ""

    for line in lines:
        stripped = line.strip()
        # H1（单 #，不包含 ##）— 先 flush 旧节再更新 h1
        if re.match(r'^# [^#]', stripped):
            if current_lines:
                sections.append(('\n'.join(current_lines), current_h1, current_h2, current_h3))
                current_lines = []
            current_h1 = stripped.lstrip('#').strip()
            current_h2 = current_h3 = ""
            # H1 行本身不加入 chunk 内容（只是章节标题）
        # H2 分节边界
        elif re.match(r'^## [^#]', stripped):
            if current_lines:
                sections.append(('\n'.join(current_lines), current_h1, current_h2, current_h3))
            current_h2 = stripped.lstrip('#').strip()
            current_h3 = ""
            current_lines = [line]
        else:
            # 记录 H3（分段内部，不作为切割边界）
            if re.match(r'^### [^#]', stripped):
                current_h3 = stripped.lstrip('#').strip()
            current_lines.append(line)

    if current_lines:
        sections.append(('\n'.join(current_lines), current_h1, current_h2, current_h3))

    def _make_path(h1v: str, h2v: str, h3v: str) -> str:
        return " > ".join(p for p in [h1v, h2v, h3v] if p)

    for section_text, s_h1, s_h2, _ in sections:
        section_text = section_text.strip()
        if not section_text:
            continue

        # 在分段过程中实时追踪 h3
        running_h3 = ""

        def _make_chunk(text_body: str, cur_h3: str) -> dict:
            path = _make_path(s_h1, s_h2, cur_h3)
            return {
                "text":    text_body,
                "source":  source,
                "path":    path,
                "h1":      s_h1,
                "h2":      s_h2,
                "h3":      cur_h3,
                "chapter": s_h2,  # 向后兼容
            }

        if len(section_text) <= 800:
            # 先扫一遍取最末 h3
            for ln in section_text.split('\n'):
                if re.match(r'^### [^#]', ln.strip()):
                    running_h3 = ln.strip().lstrip('#').strip()
            chunks.append(_make_chunk(section_text, running_h3))
        else:
            sec_lines = section_text.split('\n', 1)
            header = sec_lines[0] if len(sec_lines) > 1 else ''
            body   = sec_lines[1] if len(sec_lines) > 1 else sec_lines[0]
            paragraphs = re.split(r'\n{2,}', body)

            current = header
            current_h3_local = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                # 更新 h3 状态
                for ln in para.split('\n'):
                    if re.match(r'^### [^#]', ln.strip()):
                        current_h3_local = ln.strip().lstrip('#').strip()
                if len(current) + len(para) + 2 > 800 and current != header:
                    chunks.append(_make_chunk(current.strip(), current_h3_local))
                    current = header + '\n\n' + para
                else:
                    current += '\n\n' + para
            if current.strip() and current.strip() != header.strip():
                chunks.append(_make_chunk(current.strip(), current_h3_local))

    return chunks


def _load_or_build_chunks(md_file: Path, source: str) -> list[dict]:
    """
    若 .chunks.json 存在则直接加载，否则切分并持久化。
    重新切分时删除旧 .chunks.json 即可。
    """
    import json as _json
    chunks_path = md_file.with_suffix('.chunks.json')
    if chunks_path.exists():
        try:
            return _json.loads(chunks_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    chunks = _chunk_markdown(md_file.read_text(encoding='utf-8'), source)
    chunks_path.write_text(
        _json.dumps(chunks, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return chunks


def _chunk_pdf(path: str) -> list[str]:
    """解析简历 PDF，合并段落到 ~600 字符。"""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
    full_text = "\n\n".join(pages_text)

    raw_paras = [p.strip() for p in re.split(r'\n{2,}', full_text) if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in raw_paras:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= 600:
            current += "\n\n" + para
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)

    return chunks


# ── EPUB → Markdown ───────────────────────────────────────────────────────────

def epub_to_markdown(file_bytes: bytes) -> str:
    """将 EPUB 字节流转换为 Markdown 字符串。"""
    import ebooklib
    from ebooklib import epub
    import html2text
    from bs4 import BeautifulSoup

    converter = html2text.HTML2Text()
    converter.ignore_links  = True
    converter.ignore_images = True
    converter.body_width    = 0
    converter.unicode_snob  = True

    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        book = epub.read_epub(tmp_path, options={"ignore_ncx": True})
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    sections: list[str] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_bytes = item.get_content()
        if not html_bytes:
            continue
        soup = BeautifulSoup(html_bytes, "html.parser")
        if len(soup.get_text().strip()) < 100:
            continue
        md = converter.handle(html_bytes.decode("utf-8", errors="replace")).strip()
        if md:
            sections.append(md)

    return "\n\n---\n\n".join(sections)


def ingest_epub(file_bytes: bytes, name: str) -> Path:
    """将 EPUB 转为 Markdown 保存到 knowledge/<name>.md，清除旧的向量索引。"""
    md_path = KNOWLEDGE_DIR / f"{name}.md"
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    if md_path.exists() and is_available():
        col = _get_knowledge_col()
        existing = col.get(where={"source": name})
        if existing["ids"]:
            col.delete(ids=existing["ids"])

    markdown = epub_to_markdown(file_bytes)
    md_path.write_text(markdown, encoding="utf-8")
    return md_path


# ── 自适应并发信号量（TCP 慢启动 + AIMD）────────────────────────────────────────

class AdaptiveSemaphore:
    """
    TCP 风格自适应并发控制：
    - 慢启动：每完成一整轮（limit 次满载成功）就翻倍，直到 ssthresh
    - 拥塞避免：超过 ssthresh 后每轮 +1
    - 失败（限速/超时）：ssthresh = limit//2，limit 降至 ssthresh，切回拥塞避免
    """
    MIN = 1
    MAX = 64

    def __init__(self, initial: int = 1, ssthresh: int = 8):
        self._limit   = initial
        self._ssthresh = ssthresh
        self._active  = 0
        self._streak  = 0          # 本轮满载成功次数
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
        return False   # 不吞异常

    def _adjust(self, success: bool, was_full: bool) -> None:
        if not success:
            new_thresh = max(self.MIN, self._limit // 2)
            print(f"[adaptive] 失败 → ssthresh={new_thresh}, "
                  f"并发 {self._limit}→{new_thresh}")
            self._ssthresh = new_thresh
            self._limit    = new_thresh
            self._streak   = 0
            return

        if not was_full:
            return   # 没跑满，不计入本轮

        self._streak += 1
        if self._streak < self._limit:
            return   # 本轮还没满

        # 完成一整轮，决定是翻倍还是 +1
        self._streak = 0
        if self._limit < self._ssthresh:
            new = min(self.MAX, self._limit * 2)
            phase = "慢启动×2"
        else:
            new = min(self.MAX, self._limit + 1)
            phase = "线性+1"
        if new != self._limit:
            print(f"[adaptive] {phase} → 并发 {self._limit}→{new} "
                  f"(ssthresh={self._ssthresh})")
        self._limit = new

    def stats(self) -> dict:
        return {
            "limit":    self._limit,
            "ssthresh": self._ssthresh,
            "active":   self._active,
        }


# ── LLM 问题生成 ───────────────────────────────────────────────────────────────

LLM_TIMEOUT = 60.0   # 单次 LLM 调用超时秒数

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
            raise   # 向上传播，让自适应信号量记为失败
        print(f"[rag] 问题生成失败(非限速): {e}")
        return []


# ── 索引操作 ───────────────────────────────────────────────────────────────────

async def index_knowledge_with_qa(provider: "LLMProvider | None" = None) -> int:
    """
    扫描 knowledge/，对每个 chunk 生成多问题向量（若有 provider）。
    - 向量用问题 embed，document 存原文，metadata 含 chunk_id/chapter/question
    - 无 provider 时直接用原文向量化（退化模式）
    返回本次新增的 chunk 数。
    """
    global _index_progress

    if not is_available():
        return 0
    if not KNOWLEDGE_DIR.exists():
        return 0

    # 统计需要处理的总 chunk 数
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
        "status":        "running",
        "chunks_done":   0,
        "chunks_total":  total_chunks,
        "vectors_added": 0,
        "elapsed_s":     0.0,
        "eta_s":         None,
        "error":         None,
    })
    t_start = _time.monotonic()
    total_new = 0

    try:
        import json as _json
        sem = AdaptiveSemaphore(initial=1, ssthresh=8)

        for md_file, chunks in pending:
            source  = md_file.stem
            qa_path = md_file.with_suffix(".qa.json")
            _index_progress["file"] = md_file.name

            # ── 加载 QA 缓存（断点续传）───────────────────────────────────
            qa_cache: dict[str, list[str]] = {}
            if qa_path.exists():
                try:
                    qa_cache = _json.loads(qa_path.read_text(encoding="utf-8"))
                    cached = sum(1 for k in qa_cache if k.startswith(source + "_"))
                    print(f"[rag] 读取 QA 缓存: {qa_path.name}  ({cached}/{len(chunks)} chunks)")
                except Exception:
                    qa_cache = {}

            done_ref = [_index_progress["chunks_done"]]

            # ── 阶段1：并发生成 QA（缓存命中则跳过 LLM）────────────────────
            if provider:
                print(f"[rag] QA 生成({mode}): {md_file.name}  ({len(chunks)} chunks)")

                async def _gen_one(i: int) -> None:
                    chunk_id = f"{source}_{i}"
                    if chunk_id in qa_cache:
                        # 已缓存：直接更新进度
                        done_ref[0] += 1
                        return
                    chunk_text = chunks[i]["text"]
                    qs: list[str] = []
                    try:
                        async with sem:
                            qs = await _generate_questions(chunk_text, provider)
                    except Exception as e:
                        wait = min(10, 3 + sem.limit)
                        print(f"[rag] chunk {i} 限速/超时，等待 {wait}s: {e}")
                        await asyncio.sleep(wait)
                    qa_cache[chunk_id] = qs
                    qa_path.write_text(
                        _json.dumps(qa_cache, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    done_ref[0] += 1
                    done    = done_ref[0]
                    elapsed = _time.monotonic() - t_start
                    eta     = elapsed / done * (total_chunks - done) if done else None
                    _index_progress.update({
                        "chunks_done": done,
                        "elapsed_s":   round(elapsed, 1),
                        "eta_s":       round(eta, 0) if eta is not None else None,
                    })

                await asyncio.gather(*[_gen_one(i) for i in range(len(chunks))])

            # ── 阶段2：从 qa_cache 构建向量列表，批量写入 ChromaDB ─────────
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
                            "source":   source,
                            "path":     path,
                            "h1":       chunk.get("h1", ""),
                            "h2":       chunk.get("h2", ""),
                            "h3":       chunk.get("h3", ""),
                            "chapter":  chapter,
                            "chunk_id": chunk_id,
                            "question": q,
                            "text":     chunk_text,
                        })
                else:
                    ids.append(chunk_id)
                    documents.append(chunk_text)
                    metadatas.append({
                        "source":   source,
                        "path":     path,
                        "h1":       chunk.get("h1", ""),
                        "h2":       chunk.get("h2", ""),
                        "h3":       chunk.get("h3", ""),
                        "chapter":  chapter,
                        "chunk_id": chunk_id,
                        "text":     chunk_text,
                    })

                if not provider:
                    done_ref[0] += 1
                    done    = done_ref[0]
                    elapsed = _time.monotonic() - t_start
                    eta     = elapsed / done * (total_chunks - done) if done else None
                    _index_progress.update({
                        "chunks_done": done,
                        "elapsed_s":   round(elapsed, 1),
                        "eta_s":       round(eta, 0) if eta is not None else None,
                    })
                    print(f"[rag]   chunk {done}/{total_chunks}  {source} > {chapter[:30]}")

            if ids:
                col.add(ids=ids, documents=documents, metadatas=metadatas)
                _index_progress["vectors_added"] += len(ids)
                total_new += len(chunks)
                print(f"[rag] 完成: {md_file.name} → {len(chunks)} chunks, "
                      f"{len(ids)} vectors (并发峰值={sem.limit})")

        _index_progress["status"] = "done"
    except Exception as e:
        print(f"[rag] 索引出错: {e}")
        _index_progress.update({"status": "error", "error": str(e)})

    return total_new


def backfill_qa_cache() -> dict[str, int]:
    """
    从 ChromaDB 回填 .qa.json 缓存文件（用于已索引但缺少缓存文件的 source）。
    不调用 LLM，直接读取 ChromaDB metadata 中的 question 字段。
    返回 {source: chunk_count} 字典。
    """
    import json as _json
    if not is_available() or not KNOWLEDGE_DIR.exists():
        return {}

    col     = _get_knowledge_col()
    results: dict[str, int] = {}

    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        source   = md_file.stem
        qa_path  = md_file.with_suffix(".qa.json")
        if qa_path.exists():
            continue   # 已有缓存，跳过

        # 取出该 source 全部向量的 metadata
        try:
            data = col.get(where={"source": source}, include=["metadatas"])
        except Exception:
            continue
        if not data["metadatas"]:
            continue

        # 按 chunk_id 聚合 question
        qa_cache: dict[str, list[str]] = {}
        for meta in data["metadatas"]:
            cid = meta.get("chunk_id", "")
            q   = meta.get("question", "")
            if cid and q:
                qa_cache.setdefault(cid, [])
                if q not in qa_cache[cid]:
                    qa_cache[cid].append(q)

        if qa_cache:
            qa_path.write_text(
                _json.dumps(qa_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[rag] 回填 QA 缓存: {qa_path.name}  ({len(qa_cache)} chunks)")
            results[source] = len(qa_cache)

    return results


def index_knowledge() -> int:
    """同步退化版本（无 LLM），供外部同步调用。"""
    import asyncio
    return asyncio.run(index_knowledge_with_qa(provider=None))


# ── 检索操作 ───────────────────────────────────────────────────────────────────

def _rrf_merge(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Reciprocal Rank Fusion：合并多路排序列表。
    ranked_lists: 每个子列表是 chunk_id 按排名顺序排列（最好在前）。
    返回按 RRF 分数降序排列的 chunk_id 列表（去重）。
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            if cid:
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


# cosine distance 阈值：ChromaDB 默认用 L2，bge 系列用余弦距离
# 余弦距离 ∈ [0, 2]，0=完全相同，越大越不相关
# 0.9 约对应余弦相似度 ~0.55，经验值，可按实际效果调整
SIMILARITY_DISTANCE_THRESHOLD = 0.70  # L2距离，bge归一化向量下≈cosine_sim>0.755


def _safe_query(
    col, query: str, n: int, where: dict | None = None, return_distances: bool = False
) -> list[tuple[str, dict]] | list[tuple[str, dict, float]]:
    """查询，返回 [(document, metadata), ...] 列表，距离超过阈值的结果会被过滤。
    return_distances=True 时返回 [(document, metadata, distance), ...]。
    """
    try:
        kwargs: dict = {
            "query_texts": [query],
            "n_results":   n,
            "include":     ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = col.query(**kwargs)
        docs      = results.get("documents",  [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]
        if return_distances:
            return [
                (d, m, dist) for d, m, dist in zip(docs, metas, distances)
                if d and dist < SIMILARITY_DISTANCE_THRESHOLD
            ]
        return [
            (d, m) for d, m, dist in zip(docs, metas, distances)
            if d and dist < SIMILARITY_DISTANCE_THRESHOLD
        ]
    except Exception:
        return []


def _dedupe_chunks(raw: list[tuple[str, dict]], limit: int) -> list[tuple[str, dict]]:
    """按 chunk_id 去重，最多返回 limit 条。"""
    seen: set[str] = set()
    result: list[tuple[str, dict]] = []
    for doc, meta in raw:
        chunk_id = meta.get("chunk_id", doc[:30])
        if chunk_id not in seen:
            seen.add(chunk_id)
            result.append((doc, meta))
            if len(result) >= limit:
                break
    return result


def retrieve_rich(
    query: str,
    session_id: str | None = None,
    extra_chunks: list[dict] | None = None,
    top_k: int | None = None,
    path_map: dict[str, str] | None = None,
) -> dict:
    """
    精准模式检索流程：
      bi-encoder 召回 → 阈值过滤 → 去重
      + extra_chunks（图谱召回）
      → RRF 融合排序 → cross-encoder 重排（含证据链前置）→ 取 top-K

    extra_chunks: 图谱召回的 chunk 列表，每项含 {text, source, path, chapter, chunk_id}，
                  按图谱命中排名顺序传入（最相关在前）。
    path_map: {chunk_id: 证据链文本}，有路径的 chunk 在 rerank 前将路径文本前置，
              帮助 cross-encoder 理解关系型查询（A→B 中 query 提 A、chunk 讲 B 的情形）。

    返回:
        {
            "knowledge": [{"text":..., "source":..., "path":..., "chapter":..., "chunk_id":...,
                           "question":..., "via_graph":...}],
            "resume":    [...],
        }
    """
    if not is_available():
        return {"knowledge": [], "resume": []}

    k = top_k if top_k is not None else KNOWLEDGE_TOP_K
    knowledge: list[dict] = []
    retrieval_log: list[dict] = []   # 供调用方打日志
    try:
        col = _get_knowledge_col()
        if col.count() > 0:
            # Step 1: bi-encoder 召回
            n_candidates = min(k * QA_PER_CHUNK * 2, col.count())
            raw_with_dist = _safe_query(col, query, n_candidates, return_distances=True)
            raw = [(d, m) for d, m, _ in raw_with_dist]
            candidates = _dedupe_chunks(raw, limit=k * 3)
            dist_map = {m.get("chunk_id", ""): dist for d, m, dist in raw_with_dist}

            # Step 2: 构建 chunk 数据字典（chunk_id → (text, meta)）
            cand_map: dict[str, tuple[str, dict]] = {
                m.get("chunk_id", ""): (m.get("text", doc), m)
                for doc, m in candidates
                if m.get("chunk_id")
            }
            be_rank = list(cand_map.keys())  # bi-encoder 排名顺序

            # Step 3: 合并图谱 extra_chunks（按传入顺序为排名）
            extra_map: dict[str, dict] = {}
            graph_rank: list[str] = []
            if extra_chunks:
                for c in extra_chunks:
                    cid = c.get("chunk_id", "")
                    if cid:
                        extra_map[cid] = c
                        graph_rank.append(cid)
                        if cid not in cand_map:
                            # 补充进候选字典，meta 对齐 ChromaDB meta 格式
                            cand_map[cid] = (c["text"], {
                                "chunk_id": cid,
                                "text":     c["text"],
                                "source":   c.get("source", ""),
                                "path":     c.get("path", ""),
                                "chapter":  c.get("chapter", ""),
                                "question": "",
                            })

            # Step 4: RRF 融合，同时保留分数供日志
            be_only  = len(be_rank)
            gph_only = sum(1 for cid in graph_rank if cid not in set(be_rank))
            overlap  = sum(1 for cid in graph_rank if cid in set(be_rank))
            if graph_rank:
                rrf_scores = {}
                for ranked_list in [be_rank, graph_rank]:
                    for rank, cid in enumerate(ranked_list, start=1):
                        if cid:
                            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank)
                merged_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)
            else:
                rrf_scores = {}
                merged_ids = be_rank

            # Step 5: 取 top-N 送 cross-encoder rerank
            # 有 path_map 的 chunk 前置证据链文本，帮助 reranker 理解关系型查询
            RERANK_LIMIT = max(k * 4, 15)
            rerank_cids = [cid for cid in merged_ids if cid in cand_map][:RERANK_LIMIT]
            rerank_inputs = []
            for cid in rerank_cids:
                raw_text, meta = cand_map[cid]
                if path_map and cid in path_map:
                    enriched = path_map[cid] + "\n\n" + raw_text
                else:
                    enriched = raw_text
                rerank_inputs.append((enriched, meta))
            ranked = rerank(query, rerank_inputs)

            be_cid_set = set(be_rank)
            for doc, meta, score in ranked[:k]:
                cid = meta.get("chunk_id", "")
                graph_only = cid in extra_map and cid not in be_cid_set
                knowledge.append({
                    "text":      meta.get("text", doc),
                    "source":    meta.get("source", ""),
                    "path":      meta.get("path", ""),
                    "chapter":   meta.get("chapter", meta.get("path", "")),
                    "chunk_id":  cid,
                    "question":  meta.get("question", ""),
                    "via_graph": graph_only,
                })
                retrieval_log.append({
                    "chunk_id":     cid,
                    "source":       meta.get("source", ""),
                    "bi_dist":      round(dist_map.get(cid, -1), 4),
                    "rrf_score":    round(rrf_scores.get(cid, 0.0), 6),
                    "rerank_score": round(score, 4),
                    "via_graph":    cid in extra_map,
                    "graph_only":   graph_only,
                    "question":     meta.get("question", "")[:60],
                })

            # 汇总信息供调用方打印
            retrieval_log.append({
                "_summary": True,
                "be_candidates":  be_only,
                "graph_extra":    len(graph_rank),
                "graph_new":      gph_only,
                "graph_overlap":  overlap,
                "rerank_input":   len(rerank_cids),
                "final_output":   len(knowledge),
            })
    except Exception:
        pass

    resume: list[dict] = []
    if session_id:
        try:
            col = _get_resume_col()
            existing = col.get(where={"session_id": session_id})
            count = len(existing["ids"])
            if count > 0:
                raw = _safe_query(col, query, min(RESUME_TOP_K, count),
                                  where={"session_id": session_id})
                resume = [{"text": doc, "source": "resume", "chapter": "",
                           "chunk_id": meta.get("chunk_id", ""), "question": ""}
                          for doc, meta in raw]
        except Exception:
            pass

    # ── 笔记检索 ──────────────────────────────────
    notes: list[dict] = []
    try:
        col = _get_notes_col()
        if col.count() > 0:
            raw = _safe_query(col, query, min(2, col.count()))
            seen_notes: set[str] = set()
            for doc, meta in raw:
                note_id = meta.get("note_id", doc[:30])
                if note_id in seen_notes:
                    continue
                seen_notes.add(note_id)
                notes.append({
                    "text":     meta.get("text", doc),  # 返回原文，非问题字符串
                    "source":   "笔记",
                    "chapter":  meta.get("title", ""),
                    "chunk_id": note_id,
                    "question": doc if doc != meta.get("text") else "",
                })
    except Exception:
        pass

    return {"knowledge": knowledge, "resume": resume, "notes": notes, "retrieval_log": retrieval_log}


def retrieve(query: str, session_id: str | None = None) -> dict:
    """
    根据 query 检索知识库和（可选的）简历，返回去重后的原文片段。

    返回:
        {
            "knowledge": ["原文片段1", ...],
            "resume":    ["原文片段1", ...],
        }
    """
    if not is_available():
        return {"knowledge": [], "resume": []}

    # ── 知识库检索：召回 → 去重 → 重排 ──────────────
    knowledge_chunks: list[str] = []
    try:
        col = _get_knowledge_col()
        if col.count() > 0:
            n_candidates = min(KNOWLEDGE_TOP_K * QA_PER_CHUNK * 2, col.count())
            raw        = _safe_query(col, query, n_candidates)
            candidates = _dedupe_chunks(raw, limit=KNOWLEDGE_TOP_K * 3)
            ranked     = rerank(query, candidates)
            knowledge_chunks = [doc for doc, _, _score in ranked[:KNOWLEDGE_TOP_K]]
    except Exception:
        pass

    # ── 简历检索 ──────────────────────────────────
    resume_chunks: list[str] = []
    if session_id:
        try:
            col = _get_resume_col()
            existing = col.get(where={"session_id": session_id})
            count = len(existing["ids"])
            if count > 0:
                raw = _safe_query(
                    col, query, min(RESUME_TOP_K, count),
                    where={"session_id": session_id},
                )
                resume_chunks = [doc for doc, _ in raw]
        except Exception:
            pass

    # ── 笔记检索 ──────────────────────────────────
    note_chunks: list[str] = []
    try:
        col = _get_notes_col()
        if col.count() > 0:
            raw = _safe_query(col, query, min(2, col.count()))
            seen_notes: set[str] = set()
            for doc, meta in raw:
                note_id = meta.get("note_id", doc[:30])
                if note_id not in seen_notes:
                    seen_notes.add(note_id)
                    note_chunks.append(meta.get("text", doc))
    except Exception:
        pass

    return {"knowledge": knowledge_chunks, "resume": resume_chunks, "notes": note_chunks}


def has_resume(session_id: str) -> bool:
    if not is_available():
        return False
    try:
        col = _get_resume_col()
        existing = col.get(where={"session_id": session_id}, limit=1)
        return len(existing["ids"]) > 0
    except Exception:
        return False


def knowledge_count() -> int:
    """返回知识库中的向量总数（含多问题扩展后的数量）。"""
    if not is_available():
        return 0
    try:
        return _get_knowledge_col().count()
    except Exception:
        return 0


# ── 笔记 CRUD ──────────────────────────────────────────────────────────────────

import datetime as _dt


def save_note_file(title: str, content: str, questions: list[str] | None = None) -> tuple[str, str]:
    """
    把笔记写到磁盘，立即返回 (note_id, full_text)。
    同时写 note_id.meta.json（包含 questions 和 indexed 状态）。
    不做 ChromaDB 索引——由调用方在后台线程执行 index_note()。
    """
    import json as _json
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    note_id = "note_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    text    = f"# {title}\n\n{content}"
    (NOTES_DIR / f"{note_id}.md").write_text(text, encoding="utf-8")
    (NOTES_DIR / f"{note_id}.meta.json").write_text(
        _json.dumps({"questions": questions or [], "indexed": False}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return note_id, text


def _read_meta(note_id: str) -> dict:
    """读取 .meta.json，兼容旧版 .qa.json + .indexed 格式。"""
    import json as _json
    meta_path = NOTES_DIR / f"{note_id}.meta.json"
    if meta_path.exists():
        try:
            return _json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {"questions": [], "indexed": False}
    # 旧格式兼容
    questions: list[str] = []
    qa_path = NOTES_DIR / f"{note_id}.qa.json"
    if qa_path.exists():
        try:
            questions = _json.loads(qa_path.read_text(encoding="utf-8")).get("questions", [])
        except Exception:
            pass
    indexed = (NOTES_DIR / f"{note_id}.indexed").exists()
    return {"questions": questions, "indexed": indexed}


def _write_meta(note_id: str, meta: dict) -> None:
    import json as _json
    (NOTES_DIR / f"{note_id}.meta.json").write_text(
        _json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def index_note(note_id: str, title: str, text: str) -> None:
    """
    在 ChromaDB 中索引一条笔记（同步，供后台线程调用）。
    若 meta.json 中有 questions，则以每个问题为 document 向量化；
    否则降级为直接向量化原文。
    """
    if not is_available():
        return
    try:
        col = _get_notes_col()
        # 先删除该笔记已有的所有向量
        try:
            existing = col.get(where={"note_id": note_id})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception:
            pass

        meta = _read_meta(note_id)
        questions = meta.get("questions", [])

        if questions:
            col.add(
                ids=[f"{note_id}_q{i}" for i in range(len(questions))],
                documents=questions,  # 嵌入问题向量
                metadatas=[{
                    "note_id": note_id,
                    "title":   title,
                    "text":    text,  # 检索时返回原文
                } for _ in questions],
            )
            print(f"[rag] 笔记已索引 {len(questions)} 个问题: {note_id}")
        else:
            col.add(
                ids=[note_id],
                documents=[text],
                metadatas=[{"note_id": note_id, "title": title, "text": text}],
            )
            print(f"[rag] 笔记已索引(原文): {note_id}")
        # 更新 meta.json 中的 indexed 状态
        meta["indexed"] = True
        _write_meta(note_id, meta)
    except Exception as e:
        print(f"[rag] 笔记索引失败: {e}")


def list_notes() -> list[dict]:
    """列出所有笔记，按时间倒序。"""
    if not NOTES_DIR.exists():
        return []
    notes = []
    for f in sorted(NOTES_DIR.glob("note_*.md"), reverse=True):
        lines = f.read_text(encoding="utf-8").split("\n", 2)
        title = lines[0].lstrip("# ").strip() if lines else f.stem
        notes.append({
            "note_id":    f.stem,
            "title":      title,
            "size":       f.stat().st_size,
            "created_at": f.stem[5:],  # "20240115_103045"
            "indexed":    _read_meta(f.stem).get("indexed", False),
        })
    return notes


def get_note(note_id: str) -> str | None:
    """返回笔记内容，不存在时返回 None。"""
    path = NOTES_DIR / f"{note_id}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def get_note_questions(note_id: str) -> list[str]:
    """返回笔记对应的问题列表，不存在则返回空列表。"""
    return _read_meta(note_id).get("questions", [])


def delete_note(note_id: str) -> bool:
    """删除笔记文件、meta.json 和向量索引，返回是否成功。"""
    path = NOTES_DIR / f"{note_id}.md"
    if not path.exists():
        return False
    path.unlink()
    for suffix in (".meta.json", ".qa.json", ".indexed"):
        p = NOTES_DIR / f"{note_id}{suffix}"
        if p.exists():
            p.unlink()
    if is_available():
        try:
            col = _get_notes_col()
            existing = col.get(where={"note_id": note_id})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception:
            pass
    return True


def notes_count() -> int:
    if not is_available():
        return 0
    try:
        return _get_notes_col().count()
    except Exception:
        return 0


def retrieve_graph(query: str) -> dict:
    """Graph RAG 查询包装 — 优雅降级，import 失败返回空结果。"""
    try:
        import graph_rag
        return graph_rag.retrieve_graph(query)
    except Exception:
        return {"entities": [], "relations": [], "source_chunk_ids": [], "graph_summary": ""}


# ── Profile（用户简历/自我介绍，全局持久）────────────────────────────────────────

PROFILE_DIR = Path(__file__).parent / "profile"
PROFILE_DIR.mkdir(exist_ok=True)


_PROFILE_FILE = "profile.md"


def save_profile(md_text: str) -> None:
    """保存 MD 格式的 profile 到文件（不建向量索引，全文直接用）。"""
    if not md_text.strip():
        raise ValueError("内容不能为空")
    (PROFILE_DIR / _PROFILE_FILE).write_text(md_text, encoding="utf-8")
    print(f"[profile] 已保存 {len(md_text)} 字")


def get_profile_text() -> str | None:
    """返回 profile 全文，未上传则返回 None。"""
    path = PROFILE_DIR / _PROFILE_FILE
    return path.read_text(encoding="utf-8") if path.exists() else None


def profile_status() -> dict:
    """返回 profile 状态。"""
    path = PROFILE_DIR / _PROFILE_FILE
    if not path.exists():
        return {"uploaded": False, "size": 0}
    text = path.read_text(encoding="utf-8")
    # 提取 ## 章节标题作为摘要
    import re
    sections = re.findall(r'^##\s+(.+)', text, re.MULTILINE)
    return {"uploaded": True, "size": len(text), "sections": sections}
