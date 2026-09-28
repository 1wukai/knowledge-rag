"""标题感知的 Markdown 切分：保留标题路径与源行号，跳过 frontmatter 与代码围栏。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    path: str
    source_hash: str
    heading: list[str]
    start_line: int
    end_line: int
    text: str

    @property
    def embed_text(self) -> str:
        parts = [self.title, *self.heading]
        return " > ".join(p for p in parts if p) + "\n" + self.text


@dataclass
class _Section:
    heading: tuple[str, ...]
    lines: list[tuple[int, str]] = field(default_factory=list)


def _sections(raw_lines: list[str]) -> list[_Section]:
    # 跳过 YAML frontmatter
    start = 0
    if raw_lines and raw_lines[0].strip() == "---":
        idx = 1
        while idx < len(raw_lines) and raw_lines[idx].strip() != "---":
            idx += 1
        start = idx + 1

    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current = _Section(tuple())
    in_fence = False

    for line_no, line in enumerate(raw_lines, start=1):
        if line_no < start:
            continue
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            current.lines.append((line_no, line))
            continue
        if not in_fence:
            match = _HEADING_RE.match(line)
            if match:
                if current.lines:
                    sections.append(current)
                    current = _Section(tuple())
                level = len(match.group(1))
                title = match.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current.heading = tuple(t for _, t in stack)
                continue
        current.lines.append((line_no, line))

    if current.lines:
        sections.append(current)
    return sections


def _chunk_section(section: _Section, max_chars: int, chunk_ctx: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[tuple[int, str]] = []
    buf_len = 0
    for line_no, line in section.lines:
        add = len(line) + 1
        if buf and buf_len + add > max_chars:
            chunks.append(_make_chunk(section.heading, buf, chunk_ctx))
            buf = []
            buf_len = 0
        buf.append((line_no, line))
        buf_len += add
    if buf:
        chunks.append(_make_chunk(section.heading, buf, chunk_ctx))
    return chunks


def _make_chunk(heading: tuple[str, ...], lines: list[tuple[int, str]], ctx: dict) -> Chunk:
    text = "\n".join(line for _, line in lines).strip()
    start_line = lines[0][0]
    end_line = lines[-1][0]
    heading_list = list(heading)
    raw = "\n".join(
        [
            ctx["doc_id"],
            ctx["source_hash"],
            " > ".join(heading_list),
            str(start_line),
            str(end_line),
            text,
        ]
    )
    chunk_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=ctx["doc_id"],
        title=ctx["title"],
        path=ctx["path"],
        source_hash=ctx["source_hash"],
        heading=heading_list,
        start_line=start_line,
        end_line=end_line,
        text=text,
    )


def chunk_text(
    text: str,
    *,
    doc_id: str,
    title: str,
    path: str,
    source_hash: str,
    max_chars: int = 1200,
) -> list[Chunk]:
    raw_lines = text.splitlines()
    ctx = {
        "doc_id": doc_id,
        "title": title,
        "path": path,
        "source_hash": source_hash,
    }
    chunks: list[Chunk] = []
    for section in _sections(raw_lines):
        chunks.extend(_chunk_section(section, max_chars, ctx))
    return [c for c in chunks if c.text]


def chunk_plain_text(
    text: str,
    *,
    doc_id: str,
    title: str,
    path: str,
    source_hash: str,
    heading: list[str] | None = None,
    max_chars: int = 1200,
) -> list[Chunk]:
    if max_chars < 1:
        raise ValueError("max_chars 必须大于零")
    context = {"doc_id": doc_id, "title": title, "path": path, "source_hash": source_hash}
    headings = tuple(heading or [])
    chunks: list[Chunk] = []
    buffer: list[tuple[int, str]] = []
    size = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        for offset in range(0, max(1, len(line)), max_chars):
            fragment = line[offset:offset + max_chars]
            cost = len(fragment) + (1 if buffer else 0)
            if buffer and size + cost > max_chars:
                chunks.append(_make_chunk(headings, buffer, context))
                buffer, size = [], 0
                cost = len(fragment)
            buffer.append((line_no, fragment))
            size += cost
    if buffer:
        chunks.append(_make_chunk(headings, buffer, context))
    for ordinal, chunk in enumerate(chunks):
        # 同一长行可能切出完全相同的片段，用序号保持 point ID 唯一。
        chunk.chunk_id = hashlib.sha256(f"{chunk.chunk_id}:plain:{ordinal}".encode()).hexdigest()[:32]
    return [chunk for chunk in chunks if chunk.text]
