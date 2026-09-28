"""冻结源快照：提交入库时校验并保存文档原始字节与 manifest 快照。

worker 只读 document_versions / corpus_snapshots，不再读取可能变化的工作区文件。
"""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from interview_rag.manifest import read_source_bytes
from interview_rag.storage.models import CorpusSnapshot, DocumentVersion


def compute_manifest_hash(manifest) -> str:
    """对按 doc_id 排序、固定字段序列化后的 manifest 求 SHA-256。"""
    docs = sorted(
        (
            {
                "doc_id": d.doc_id,
                "path": d.path,
                "title": d.title,
                "hash": d.hash,
                "updated_at": d.updated_at,
                "enabled": d.enabled,
                "category": d.category,
            }
            for d in manifest.documents
        ),
        key=lambda d: d["doc_id"],
    )
    canonical = json.dumps(
        {"schema_version": manifest.schema_version, "corpus_id": manifest.corpus_id, "documents": docs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def freeze_documents(
    session: AsyncSession,
    manifest,
    selected_docs: list,
    workspace_root,
) -> tuple[str, dict[str, bytes]]:
    """校验并冻结选中文档：写 document_versions（幂等），返回 manifest_hash 与字节映射。

    对每个 doc 重新计算 hash 并与清单比对，不一致即失败；同一 (doc_id, source_hash)
    已存在时复用现有快照，不重复写入。manifest 快照按 manifest_hash 幂等保存。
    """
    raw_by_doc: dict[str, bytes] = {}
    for doc in selected_docs:
        raw = read_source_bytes(workspace_root, doc.path)
        actual_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
        if actual_hash != doc.hash:
            raise ValueError(
                f"{doc.doc_id} hash 不一致（实际 {actual_hash[:16]}…），请刷新 manifest 后重试"
            )
        raw_by_doc[doc.doc_id] = raw

        exists = await session.scalar(
            select(DocumentVersion.id).where(
                DocumentVersion.doc_id == doc.doc_id,
                DocumentVersion.source_hash == doc.hash,
            )
        )
        if not exists:
            session.add(
                DocumentVersion(
                    doc_id=doc.doc_id,
                    source_hash=doc.hash,
                    title=doc.title,
                    path=doc.path,
                    raw=raw,
                )
            )

    manifest_hash = compute_manifest_hash(manifest)
    exists = await session.scalar(
        select(CorpusSnapshot.id).where(CorpusSnapshot.manifest_hash == manifest_hash)
    )
    if not exists:
        session.add(
            CorpusSnapshot(
                manifest_hash=manifest_hash,
                manifest_json=json.dumps(
                    manifest.model_dump(), ensure_ascii=False, sort_keys=True
                ),
            )
        )
    await session.flush()
    return manifest_hash, raw_by_doc
