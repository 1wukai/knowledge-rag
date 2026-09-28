"""显式语料清单的加载与校验。

manifest.json 记录纳入知识库的文档（路径、标题、hash、更新时间、启用状态）。
路径相对工作区根，doc_id 稳定不随路径变化。工具只计算摘要与差异，不自动改清单。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from interview_rag.document_processing import (
    DocumentParseError, SUPPORTED_EXTENSIONS, decode_text, legacy_word_converter, read_raw_document,
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ManifestDocument(BaseModel):
    doc_id: str
    path: str
    title: str
    hash: str
    updated_at: str = ""
    enabled: bool = True
    category: str = ""

    @field_validator("hash")
    @classmethod
    def _check_hash(cls, value: str) -> str:
        if not _HASH_RE.match(value):
            raise ValueError(f"invalid hash format: {value!r}")
        return value


class Manifest(BaseModel):
    schema_version: int = 1
    corpus_id: str
    documents: list[ManifestDocument] = Field(default_factory=list)


@dataclass
class ValidationProblem:
    severity: str  # "error" | "warning"
    doc_id: str
    message: str


@dataclass
class ValidationReport:
    problems: list[ValidationProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(p.severity == "error" for p in self.problems)

    def error(self, doc_id: str, message: str) -> None:
        self.problems.append(ValidationProblem("error", doc_id, message))

    def warning(self, doc_id: str, message: str) -> None:
        self.problems.append(ValidationProblem("warning", doc_id, message))


def load_manifest(path: Path) -> Manifest:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest 不是合法 JSON: {exc}") from exc
    return Manifest.model_validate(data)


def resolve_manifest_path(settings) -> Path:
    from interview_rag.config import resolve_path

    return resolve_path(settings.manifest_path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _resolve_doc_path(workspace_root: Path, rel: str) -> Path:
    candidate = (workspace_root / rel).resolve()
    try:
        candidate.relative_to(workspace_root.resolve())
    except ValueError:
        raise ValueError("路径越出工作区根")
    return candidate


def validate_manifest(manifest: Manifest, workspace_root: Path) -> ValidationReport:
    report = ValidationReport()
    seen_doc_ids: set[str] = set()
    seen_paths: set[str] = set()

    for doc in manifest.documents:
        if doc.doc_id in seen_doc_ids:
            report.error(doc.doc_id, "doc_id 重复")
        seen_doc_ids.add(doc.doc_id)

        normalized_path = doc.path.replace("\\", "/")
        if normalized_path != doc.path:
            report.error(doc.doc_id, "path 必须使用正斜杠")
        if doc.path.startswith("/") or ".." in Path(doc.path).parts:
            report.error(doc.doc_id, "path 不得为绝对路径或包含 .. 越界")

        key = normalized_path
        if key in seen_paths:
            report.error(doc.doc_id, "path 重复")
        seen_paths.add(key)

        try:
            target = _resolve_doc_path(workspace_root, doc.path)
        except ValueError:
            report.error(doc.doc_id, "path 解析后越出工作区根（含符号链接越界）")
            continue

        if not target.is_file():
            report.error(doc.doc_id, f"启用文件不存在: {doc.path}")
            continue

        try:
            raw = read_raw_document(target)
            actual_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
        except (OSError, DocumentParseError) as exc:
            report.error(doc.doc_id, f"无法读取文件: {exc}")
            continue

        if actual_hash != doc.hash:
            report.error(
                doc.doc_id,
                f"hash 不一致（清单 {doc.hash[:16]}… 实际 {actual_hash[:16]}…）",
            )
        if not doc.title.strip():
            report.warning(doc.doc_id, "title 为空")

    return report


def _print_report(report: ValidationReport) -> int:
    if not report.problems:
        print("manifest 校验通过，无问题。")
        return 0
    for problem in report.problems:
        print(f"[{problem.severity}] {problem.doc_id}: {problem.message}")
    return 0 if report.ok else 1


# 语料扫描允许的目录前缀（相对仓库根）；仓库自带 corpus/ 样例语料目录。
_ALLOWED_PREFIXES = ("corpus/",)


def _derive_category(rel_path: str) -> str:
    """按目录结构推导分类：corpus/<category>/... -> category。"""
    parts = rel_path.split("/")
    if len(parts) >= 3 and parts[0] == "corpus":
        return parts[1]
    return ""


def resolve_source_path(workspace_root: Path, rel: str) -> Path:
    if Path(rel).is_absolute() or ".." in Path(rel).parts or "\\" in rel:
        raise DocumentParseError("不允许的语料路径")
    target = _resolve_doc_path(workspace_root, rel)
    resolved = target.relative_to(workspace_root.resolve()).as_posix()
    if not rel.startswith(_ALLOWED_PREFIXES) or not resolved.startswith(_ALLOWED_PREFIXES):
        raise DocumentParseError("文件必须位于允许的 corpus 语料目录")
    return target


def read_source_bytes(workspace_root: Path, rel: str) -> bytes:
    return read_raw_document(resolve_source_path(workspace_root, rel))


def scan_candidates(manifest: Manifest, workspace_root: Path) -> list[dict]:
    existing_paths = {d.path for d in manifest.documents}
    candidates: list[dict] = []
    root = workspace_root.resolve()
    for prefix in _ALLOWED_PREFIXES:
        base = root / prefix.rstrip("/")
        if not base.is_dir() or not base.resolve().is_relative_to(root):
            continue
        for file in sorted(base.rglob("*")):
            if not file.is_file() or file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if file.stem.lower() in {"readme", "agent-job-readiness"}:
                continue
            rel = file.relative_to(root).as_posix()
            if rel in existing_paths:
                continue
            try:
                resolve_source_path(root, rel)
            except ValueError:
                continue
            item = {
                "path": rel, "title": file.stem, "category": _derive_category(rel),
                "format": file.suffix[1:].upper(), "hash": "", "error": None,
            }
            try:
                raw = read_raw_document(file)
                item["hash"] = "sha256:" + hashlib.sha256(raw).hexdigest()
                item["title"] = _extract_title(file, raw)
                item["size_bytes"] = len(raw)
                if file.suffix.lower() == ".doc" and not legacy_word_converter():
                    item["error"] = "当前系统缺少 DOC 转换器，请另存为 DOCX，或安装 antiword"
            except (OSError, DocumentParseError) as exc:
                item["error"] = str(exc) if isinstance(exc, DocumentParseError) else "文件无法读取"
            candidates.append(item)
    return candidates


def _extract_title(path: Path, raw: bytes) -> str:
    if path.suffix.lower() in {".md", ".txt"}:
        try:
            lines = decode_text(raw).splitlines()[:20]
            for line in lines:
                stripped = line.strip()
                if path.suffix.lower() == ".md":
                    if stripped.startswith("# "):
                        return stripped[2:].strip()[:256]
                elif stripped:
                    return stripped if len(stripped) <= 120 else path.stem
        except DocumentParseError:
            pass
    return path.stem


def add_documents(manifest: Manifest, entries: list[dict]) -> Manifest:
    """把扫描候选加入 manifest（写入 doc_id/title/hash/category），返回新 manifest。"""
    existing_ids = {d.doc_id for d in manifest.documents}
    existing_paths = {d.path for d in manifest.documents}
    for e in entries:
        if e.get("error"):
            raise DocumentParseError(f"{e['path']}: {e['error']}")
        if e["path"] in existing_paths:
            continue
        doc_id = re.sub(r"[^\w-]", "-", Path(e["path"]).with_suffix("").as_posix().replace("/", "-"))[:110]
        base = doc_id
        n = 1
        while doc_id in existing_ids:
            doc_id = f"{base}-{n}"
            n += 1
        manifest.documents.append(
            ManifestDocument(
                doc_id=doc_id,
                path=e["path"],
                title=e.get("title", ""),
                hash=e["hash"],
                category=e.get("category", ""),
            )
        )
        existing_ids.add(doc_id)
        existing_paths.add(e["path"])
    return manifest


def refresh_document(doc: ManifestDocument, workspace_root: Path) -> None:
    from datetime import datetime, timezone

    target = resolve_source_path(workspace_root, doc.path)
    raw = read_raw_document(target)
    if target.suffix.lower() in {".md", ".txt"}:
        decode_text(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if digest != doc.hash:
        doc.hash = digest
        doc.updated_at = datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat()


def save_manifest(path: Path, manifest: Manifest) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)



def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="interview-rag-manifest")
    sub = parser.add_subparsers(dest="command", required=True)
    validate_cmd = sub.add_parser("validate", help="校验 manifest 与源文件 hash")
    validate_cmd.add_argument("--manifest", type=Path, default=None)
    validate_cmd.add_argument("--workspace", type=Path, default=None)
    args = parser.parse_args(argv)

    from interview_rag.config import get_settings

    settings = get_settings()
    manifest_path = args.manifest or resolve_manifest_path(settings)
    workspace = args.workspace or settings.workspace_root

    try:
        manifest = load_manifest(manifest_path)
    except (ValueError, OSError) as exc:
        print(f"manifest 加载失败: {exc}", file=sys.stderr)
        return 1

    if args.command == "validate":
        return _print_report(validate_manifest(manifest, workspace))
    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
