"""文本切块：Markdown、PDF"""
from __future__ import annotations

import re
from pathlib import Path


def _chunk_markdown(text: str, source: str) -> list[dict]:
    """
    按 ## 标题切割 Markdown，每块 ≤800 字符。
    返回 [{"text", "source", "path", "h1", "h2", "h3", "chapter"}, ...]
    """
    chunks: list[dict] = []
    lines = text.split('\n')
    sections: list[tuple[str, str, str, str]] = []
    current_lines: list[str] = []
    current_h1 = current_h2 = current_h3 = ""

    for line in lines:
        stripped = line.strip()
        if re.match(r'^# [^#]', stripped):
            if current_lines:
                sections.append(('\n'.join(current_lines), current_h1, current_h2, current_h3))
                current_lines = []
            current_h1 = stripped.lstrip('#').strip()
            current_h2 = current_h3 = ""
        elif re.match(r'^## [^#]', stripped):
            if current_lines:
                sections.append(('\n'.join(current_lines), current_h1, current_h2, current_h3))
            current_h2 = stripped.lstrip('#').strip()
            current_h3 = ""
            current_lines = [line]
        else:
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
                "chapter": s_h2,
            }

        if len(section_text) <= 800:
            for ln in section_text.split('\n'):
                if re.match(r'^### [^#]', ln.strip()):
                    running_h3 = ln.strip().lstrip('#').strip()
            chunks.append(_make_chunk(section_text, running_h3))
        else:
            sec_lines = section_text.split('\n', 1)
            header    = sec_lines[0] if len(sec_lines) > 1 else ''
            body      = sec_lines[1] if len(sec_lines) > 1 else sec_lines[0]
            paragraphs = re.split(r'\n{2,}', body)

            current    = header
            current_h3 = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                for ln in para.split('\n'):
                    if re.match(r'^### [^#]', ln.strip()):
                        current_h3 = ln.strip().lstrip('#').strip()
                if len(current) + len(para) + 2 > 800 and current != header:
                    chunks.append(_make_chunk(current.strip(), current_h3))
                    current = header + '\n\n' + para
                else:
                    current += '\n\n' + para
            if current.strip() and current.strip() != header.strip():
                chunks.append(_make_chunk(current.strip(), current_h3))

    return chunks


def _load_or_build_chunks(md_file: Path, source: str) -> list[dict]:
    """若 .chunks.json 存在则直接加载，否则切分并持久化。"""
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
    full_text  = "\n\n".join(pages_text)
    raw_paras  = [p.strip() for p in re.split(r'\n{2,}', full_text) if p.strip()]
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
