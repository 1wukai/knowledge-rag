"""本地文档的文字提取，保留原始字节 hash 与可解释的位置标记。"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import BadZipFile, ZipFile

from interview_rag.chunking import Chunk, chunk_plain_text, chunk_text

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".doc"}
MAX_SOURCE_BYTES = 5 * 1024 * 1024
MAX_EXTRACTED_CHARS = 2_000_000
EXTRACTION_VERSION = "documents-v1"


class DocumentParseError(ValueError):
    pass


@dataclass
class TextSection:
    text: str
    heading: list[str]


def read_raw_document(path: Path) -> bytes:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError("仅支持 MD、TXT、PDF、DOCX 和 DOC 文档")
    with path.open("rb") as handle:
        raw = handle.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise DocumentParseError("文件超过 5 MiB 上限，请拆分后纳入")
    if not raw:
        raise DocumentParseError("文件为空")
    return raw


def decode_text(raw: bytes) -> str:
    try:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16")
        else:
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("gb18030")
    except UnicodeDecodeError as exc:
        raise DocumentParseError("文本编码无法识别，请另存为 UTF-8 后重试") from exc
    if any(ord(ch) < 32 and ch not in "\t\r\n\f" for ch in text):
        raise DocumentParseError("文件包含二进制控制字符，请确认格式或另存为 UTF-8")
    return text


def _pdf_sections(raw: bytes) -> list[TextSection]:
    from pypdf import PdfReader

    if not raw.startswith(b"%PDF-"):
        raise DocumentParseError("文件不是有效的 PDF，请勿只修改扩展名")
    reader = PdfReader(BytesIO(raw))
    if reader.is_encrypted:
        raise DocumentParseError("PDF 已加密，请先导出不带密码的副本")
    if len(reader.pages) > 500:
        raise DocumentParseError("PDF 超过 500 页，请拆分后纳入")
    sections = []
    total = 0
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            # 不静默丢弃可能包含扫描内容的页，以免只入库文档的一部分。
            raise DocumentParseError(f"PDF 第 {number} 页未提取到文字（可能是扫描页或空白页），请先 OCR 或移除空白页")
        total += len(text)
        if total > MAX_EXTRACTED_CHARS:
            raise DocumentParseError("PDF 提取文字过多，请拆分文档")
        sections.append(TextSection(text, [f"PDF 第 {number} 页（提取文本）"]))
    return sections


def _docx_sections(raw: bytes) -> list[TextSection]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        with ZipFile(BytesIO(raw)) as archive:
            if "word/document.xml" not in archive.namelist():
                raise DocumentParseError("文件不是有效的 DOCX")
            if len(archive.infolist()) > 10000 or sum(i.file_size for i in archive.infolist()) > 32 * 1024 * 1024:
                raise DocumentParseError("DOCX 解压体积过大，请拆分或移除图片后重试")
    except BadZipFile as exc:
        raise DocumentParseError("DOCX 文件已损坏或格式不正确") from exc
    document = Document(BytesIO(raw))
    lines = []
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            lines.append(block.text)
        elif isinstance(block, Table):
            lines.extend(" | ".join(cell.text.replace("\n", " / ") for cell in row.cells) for row in block.rows)
    return [TextSection("\n".join(lines), ["Word 正文（提取文本）"])]


def legacy_word_converter() -> list[str] | None:
    if sys.platform == "darwin" and (tool := shutil.which("textutil")):
        return [tool, "-convert", "txt", "-format", "doc", "-stdin", "-stdout", "-encoding", "UTF-8", "-noload", "-nostore"]
    if tool := shutil.which("antiword"):
        return [tool, "-m", "UTF-8.txt", "-"]
    return None


def _doc_sections(raw: bytes) -> list[TextSection]:
    if not raw.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        raise DocumentParseError("DOC 不是有效的旧版 Word 二进制文件，请用 Word 另存为 DOCX")
    command = legacy_word_converter()
    if command is None:
        raise DocumentParseError("当前系统缺少 DOC 转换器：macOS 需 textutil，其他系统需 antiword；也可将文件另存为 DOCX")
    try:
        result = subprocess.run(command, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
        text = result.stdout.decode("utf-8-sig")
    except (subprocess.SubprocessError, OSError, UnicodeDecodeError) as exc:
        raise DocumentParseError("DOC 转换失败或超时，请将文件另存为 DOCX 后重试") from exc
    return [TextSection(text, ["Word DOC 正文（转换文本）"])]


def extract_sections(raw: bytes, path: str) -> list[TextSection]:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError("仅支持 MD、TXT、PDF、DOCX 和 DOC 文档")
    if len(raw) > MAX_SOURCE_BYTES:
        raise DocumentParseError("文件超过 5 MiB 上限，请拆分后纳入")
    try:
        if suffix in {".md", ".txt"}:
            sections = [TextSection(decode_text(raw), [])]
        elif suffix == ".pdf":
            sections = _pdf_sections(raw)
        elif suffix == ".docx":
            sections = _docx_sections(raw)
        else:
            sections = _doc_sections(raw)
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"{suffix.upper().lstrip('.')} 解析失败，请检查文件是否损坏或加密") from exc
    if not sections or not any(s.text.strip() for s in sections):
        raise DocumentParseError("文档未提取到可用文字；图片或扫描文档需先执行 OCR")
    if sum(len(s.text) for s in sections) > MAX_EXTRACTED_CHARS:
        raise DocumentParseError("提取文字超过 200 万字符，请拆分文档")
    return sections


def chunk_document(raw: bytes, *, doc_id: str, title: str, path: str, source_hash: str, max_chars: int = 1200) -> list[Chunk]:
    sections = extract_sections(raw, path)
    context = dict(doc_id=doc_id, title=title, path=path, source_hash=source_hash, max_chars=max_chars)
    if Path(path).suffix.lower() == ".md":
        return chunk_text(sections[0].text, **context)
    chunks = []
    for section in sections:
        chunks.extend(chunk_plain_text(section.text, heading=section.heading, **context))
    return chunks
