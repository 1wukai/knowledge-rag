"""入库编排：读取 manifest 选定笔记 → 校验 hash → 增量切分与 embedding → 写入 Qdrant。

增量策略：对比 index_state 中已入库文档的 hash，只对新增或变化的文档重新切分与
Embedding，未变化的文档复用现有向量；本次未选中的已入库文档从索引移除。Embedding
模型或维度变化时退化为全量重建。Embedding 请求按批并发，受并发上限约束。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from qdrant_client.models import PointStruct

from interview_rag.application.index_state import load_state, save_state
from interview_rag.chunking import Chunk
from interview_rag.document_processing import EXTRACTION_VERSION, DocumentParseError, chunk_document
from interview_rag.config import Settings, resolve_path
from interview_rag.manifest import read_source_bytes, load_manifest, resolve_manifest_path
from interview_rag.providers import embed_texts
from interview_rag.runtime_settings import RuntimeSettings
from interview_rag.storage.vectorstore import store_for_settings

logger = logging.getLogger("interview_rag.ingest")

COLLECTION = "notes_v1"


class IngestError(RuntimeError):
    pass


@dataclass
class IngestResult:
    doc_ids: list[str]
    chunk_count: int
    embedding_model: str
    embedding_dim: int
    collection: str
    generated_at: str
    missing: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def _chunks_for_docs(selected: list, raw_by_doc: dict[str, bytes], max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in selected:
        try:
            chunks.extend(chunk_document(
                raw_by_doc[doc.doc_id], doc_id=doc.doc_id, title=doc.title,
                path=doc.path, source_hash=doc.hash, max_chars=max_chars,
            ))
        except DocumentParseError as exc:
            raise IngestError(f"{doc.path}: {exc}") from exc
    return chunks


def _points_for(chunks: list[Chunk], vectors: list[list[float]]) -> list[PointStruct]:
    return [
        PointStruct(
            id=uuid.UUID(hex=chunk.chunk_id),
            vector=vec,
            payload={
                "doc_id": chunk.doc_id,
                "path": chunk.path,
                "title": chunk.title,
                "hash": chunk.source_hash,
                "heading": chunk.heading,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
            },
        )
        for chunk, vec in zip(chunks, vectors)
    ]


async def _embed_batches(
    config, inputs: list[str], batch_size: int, concurrency: int
) -> list[list[float]]:
    """分批并发 Embedding；单批大小受供应商上限约束，并发批数受 RPM 上限约束。"""
    sem = asyncio.Semaphore(max(1, concurrency))

    async def run(batch: list[str]) -> list[list[float]]:
        async with sem:
            return await embed_texts(config, batch)

    batches = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
    results = await asyncio.gather(*(run(b) for b in batches))
    return [v for batch in results for v in batch]


async def ingest_documents(
    settings: Settings,
    runtime: RuntimeSettings,
    doc_ids: list[str],
    max_chars: int = 1200,
) -> IngestResult:
    manifest = load_manifest(resolve_manifest_path(settings))
    by_id = {doc.doc_id: doc for doc in manifest.documents}

    selected, missing = [], []
    for doc_id in doc_ids:
        doc = by_id.get(doc_id)
        if doc and doc.enabled:
            selected.append(doc)
        else:
            missing.append(doc_id)

    raw_by_doc = {}
    for doc in selected:
        try:
            raw = read_source_bytes(settings.workspace_root, doc.path)
        except (OSError, ValueError) as exc:
            raise IngestError(f"{doc.path}: {exc}") from exc
        actual = "sha256:" + hashlib.sha256(raw).hexdigest()
        if actual != doc.hash:
            raise IngestError(f"{doc.doc_id} hash 不一致，请刷新摘要后重试")
        raw_by_doc[doc.doc_id] = raw

    state = load_state(resolve_path(settings.index_state_path))
    store = store_for_settings(settings)
    pipeline = f"{EXTRACTION_VERSION}:{max_chars}:{runtime.embedding.base_url}:{runtime.embedding.model}"
    fingerprint = hashlib.sha256(pipeline.encode()).hexdigest()
    indexed = {d["doc_id"]: d for d in state.get("documents", [])} if state else {}
    model_changed = bool(state) and state.get("pipeline_fingerprint") != fingerprint

    if model_changed or not state:
        # 首次入库或模型变化：全量重建。
        if not selected:
            raise IngestError("所选笔记没有可入库的内容")
        chunks = await asyncio.to_thread(_chunks_for_docs, selected, raw_by_doc, max_chars)
        if not chunks:
            raise IngestError("所选笔记没有可入库的内容")
        vectors = await _embed_batches(
            runtime.embedding,
            [c.embed_text for c in chunks],
            settings.embedding_batch_size,
            settings.embedding_concurrency,
        )
        dim = len(vectors[0])
        store.recreate(COLLECTION, dim)
        store.upsert(COLLECTION, _points_for(chunks, vectors))
        reused: list[str] = []
        removed = sorted(set(indexed) - {d.doc_id for d in selected})
        chunk_count = store.count(COLLECTION)
    else:
        # 增量：按 hash 分类，只重算变化文档。
        to_embed = [d for d in selected if any(indexed.get(d.doc_id, {}).get(k) != getattr(d, k) for k in ("hash", "title", "path"))]
        changed_ids = {d.doc_id for d in to_embed}
        reused = [d.doc_id for d in selected if d.doc_id not in changed_ids]
        removed = sorted(set(indexed) - {d.doc_id for d in selected})

        dim = state.get("embedding_dim")
        if to_embed:
            chunks = await asyncio.to_thread(_chunks_for_docs, to_embed, raw_by_doc, max_chars)
            if not chunks:
                raise IngestError("所选文档没有可入库的文字")
            vectors = await _embed_batches(
                runtime.embedding,
                [c.embed_text for c in chunks],
                settings.embedding_batch_size,
                settings.embedding_concurrency,
            )
            if len(vectors) != len(chunks) or any(len(v) != dim for v in vectors):
                raise IngestError("Embedding 返回数量或维度异常，请检查模型配置")
            for doc_id in changed_ids:
                store.delete_by_doc_id(COLLECTION, doc_id)
            store.upsert(COLLECTION, _points_for(chunks, vectors))

        for doc_id in removed:
            store.delete_by_doc_id(COLLECTION, doc_id)

        chunk_count = store.count(COLLECTION)

    final_docs = [
        {"doc_id": d.doc_id, "title": d.title, "hash": d.hash, "path": d.path}
        for d in selected
    ]
    new_state = {
        "pipeline_fingerprint": fingerprint,
        "collection": COLLECTION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": runtime.embedding.model,
        "embedding_dim": dim,
        "chunk_count": chunk_count,
        "documents": final_docs,
    }
    save_state(resolve_path(settings.index_state_path), new_state)

    logger.info(
        "ingest done",
        extra={
            "chunks": chunk_count,
            "docs": len(selected),
            "reused": len(reused),
            "removed": len(removed),
            "full_rebuild": model_changed or not state,
        },
    )

    return IngestResult(
        doc_ids=[d.doc_id for d in selected],
        chunk_count=chunk_count,
        embedding_model=runtime.embedding.model,
        embedding_dim=dim,
        collection=COLLECTION,
        generated_at=new_state["generated_at"],
        missing=missing,
        reused=reused,
        removed=removed,
    )
