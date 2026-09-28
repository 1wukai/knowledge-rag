"""入库构建：worker 执行的核心——读快照 → 切分 → Embedding → 写 Qdrant。

构建目标 collection 按 generation 命名，与已有 collection 隔离；发布（切 active 指针）
由发布模块在构建校验通过后执行，不在此处处理。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from interview_rag.chunking import Chunk as ParsedChunk
from interview_rag.document_processing import chunk_document
from interview_rag.providers import embed_texts
from interview_rag.retrieval.sparse import (
    SparseStats,
    Vocabulary,
    compute_stats,
    encode_document,
    tokenize,
)
from interview_rag.storage.models import (
    Chunk,
    CorpusSnapshot,
    DocumentVersion,
    GenerationChunk,
    IndexGeneration,
)
from interview_rag.storage.vectorstore import VectorStore

logger = logging.getLogger("interview_rag.worker.build")


async def _embed_batches(config, inputs, batch_size, concurrency) -> list[list[float]]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def run(batch):
        async with sem:
            return await embed_texts(config, batch)

    batches = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
    results = await asyncio.gather(*(run(b) for b in batches))
    return [v for batch in results for v in batch]


def _points_for(chunks: list[ParsedChunk], vectors: list[list[float]]) -> list:
    from qdrant_client.models import PointStruct

    return [
        PointStruct(
            id=uuid.UUID(hex=chunk.chunk_id),
            vector=vec,
            payload={
                "doc_id": chunk.doc_id,
                "path": chunk.path,
                "title": chunk.title,
                "hash": chunk.source_hash,
                "heading": chunk.heading or [],
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
            },
        )
        for chunk, vec in zip(chunks, vectors)
    ]


async def build_generation(
    session: AsyncSession,
    settings,
    runtime,
    manifest_hash: str,
    doc_ids: list[str],
    corpus_id: str,
    max_chars: int = 1200,
) -> tuple[str, int]:
    """构建一个新的 index_generation 并写入 Qdrant，返回 (generation_id, chunk_count)。

    只读快照（document_versions / corpus_snapshots），不读工作区文件。
    """
    snapshot = await session.scalar(
        select(CorpusSnapshot).where(CorpusSnapshot.manifest_hash == manifest_hash)
    )
    if not snapshot:
        raise ValueError(f"未找到 manifest 快照 {manifest_hash}")

    manifest_data = json.loads(snapshot.manifest_json)
    meta_by_id = {d["doc_id"]: d for d in manifest_data.get("documents", [])}

    chunks: list[ParsedChunk] = []
    for doc_id in doc_ids:
        version = await session.scalar(
            select(DocumentVersion)
            .where(
                DocumentVersion.doc_id == doc_id,
                DocumentVersion.source_hash == meta_by_id.get(doc_id, {}).get("hash"),
            )
            .order_by(DocumentVersion.id.desc())
        )
        if not version:
            raise ValueError(f"文档 {doc_id} 的冻结版本不存在，不能发布不完整索引")
        meta = meta_by_id.get(doc_id, {})
        chunks.extend(
            await asyncio.to_thread(
                chunk_document,
                version.raw,
                doc_id=doc_id,
                title=meta.get("title", version.title),
                path=meta.get("path", version.path),
                source_hash=version.source_hash,
                max_chars=max_chars,
            )
        )

    if not chunks:
        raise ValueError("快照中没有可入库的内容")

    vectors = await _embed_batches(
        runtime.embedding,
        [c.embed_text for c in chunks],
        settings.embedding_batch_size,
        settings.embedding_concurrency,
    )

    # 稀疏索引：分词 → 统计 → 编码文档稀疏向量。
    sparse_stats, vocab, sparse_indices, sparse_values = _build_sparse(chunks)

    generation_id = uuid.uuid4().hex
    collection = f"gen_{generation_id}"
    store = _store(settings)
    store.recreate(collection, len(vectors[0]), sparse=bool(sparse_indices))
    points = _points_for(chunks, vectors)
    if sparse_indices:
        store.upsert_sparse(collection, points, sparse_indices, sparse_values)
    else:
        store.upsert(collection, points)

    row = IndexGeneration(
        generation_id=generation_id,
        collection=collection,
        corpus_id=corpus_id,
        status="validating",
        manifest_hash=manifest_hash,
        embedding_model=runtime.embedding.model,
        embedding_dim=len(vectors[0]),
        chunk_count=len(chunks),
        bm25_stats={
            "stats": sparse_stats.to_dict(),
            "vocabulary": vocab.to_dict(),
        },
    )
    session.add(row)
    await session.flush()

    # chunk 是内容级实体，跨 generation 复用；已存在的 chunk 不重复插入，只建关联。
    existing_chunk_ids = set(
        (
            await session.execute(
                select(Chunk.chunk_id).where(Chunk.chunk_id.in_([c.chunk_id for c in chunks]))
            )
        ).scalars().all()
    )
    for c in chunks:
        if c.chunk_id not in existing_chunk_ids:
            session.add(
                Chunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source_hash=c.source_hash,
                    title=c.title,
                    path=c.path,
                    heading=c.heading,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    text=c.text,
                    embed_input_hash="",
                )
            )
        session.add(GenerationChunk(generation_id=generation_id, chunk_id=c.chunk_id))
    await session.commit()

    logger.info(
        "generation built",
        extra={"generation_id": generation_id, "chunks": len(chunks)},
    )
    return generation_id, len(chunks)


def _store(settings) -> VectorStore:
    from interview_rag.storage.vectorstore import store_for_settings

    return store_for_settings(settings)


def _build_sparse(
    chunks: list[ParsedChunk],
) -> tuple[SparseStats, Vocabulary, dict, dict]:
    """对 chunks 分词并编码文档稀疏向量；返回 stats、vocab 与按 point id 的索引/值。"""
    doc_tokens = [tokenize(c.embed_text) for c in chunks]
    if not any(doc_tokens):
        return SparseStats(), Vocabulary(), {}, {}
    stats = compute_stats(doc_tokens)
    vocab = Vocabulary()
    indices_by_id: dict = {}
    values_by_id: dict = {}
    for c, tokens in zip(chunks, doc_tokens):
        idx, vals = encode_document(tokens, stats, vocab)
        if idx:
            pid = uuid.UUID(hex=c.chunk_id)
            indices_by_id[pid] = idx
            values_by_id[pid] = vals
    return stats, vocab, indices_by_id, values_by_id
