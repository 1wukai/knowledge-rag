"""旧 generation 的 Qdrant collection 清理。

保留 active generation 与最近 ``retain`` 个可回滚 generation 的 collection；
删除更早的 generation 的物理 collection、generation_chunks 关联与 index_generations
记录。跳过正在构建（building/validating）的 generation，不触碰 active。
chunks 表是内容级实体，删除旧向量不删除历史 chunk 证据。
"""

from __future__ import annotations

import logging

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from interview_rag.storage.models import GenerationChunk, IndexGeneration
from interview_rag.storage.vectorstore import VectorStore

logger = logging.getLogger("interview_rag.cleanup")

# 不在构建中的终态状态，可被清理（active 单独排除）。
_CLEANABLE_STATUSES = {"ready"}


async def cleanup_generations(
    session: AsyncSession,
    store: VectorStore,
    corpus_id: str,
    retain: int = 2,
) -> dict:
    """清理旧 generation；返回删除的数量与 collection 名列表。"""
    from interview_rag.storage.models import Corpus

    if retain < 0:
        raise ValueError("retain 不得小于 0")
    corpus = await session.scalar(select(Corpus).where(Corpus.corpus_id == corpus_id).with_for_update())
    if not corpus or not corpus.active_generation_id:
        return {"deleted_generations": [], "deleted_collections": []}
    rows = (
        await session.execute(
            select(IndexGeneration)
            .where(IndexGeneration.corpus_id == corpus_id)
            .order_by(IndexGeneration.id.desc())
            .with_for_update()
        )
    ).scalars().all()

    active_id = corpus.active_generation_id

    # 可清理候选：终态且非 active；保留最近 retain 个（按 id 倒序，即最新优先）。
    candidates = [g for g in rows if g.status in _CLEANABLE_STATUSES and g.generation_id != active_id]
    to_delete = candidates[retain:]

    deleted_collections: list[str] = []
    deleted_ids: list[str] = []
    for gen in to_delete:
        try:
            store.delete_collection(gen.collection)
        except Exception:
            logger.exception("delete collection failed", extra={"collection": gen.collection})
            continue
        deleted_collections.append(gen.collection)
        deleted_ids.append(gen.generation_id)

    if deleted_ids:
        await session.execute(
            delete(GenerationChunk).where(GenerationChunk.generation_id.in_(deleted_ids))
        )
        await session.execute(
            delete(IndexGeneration).where(IndexGeneration.generation_id.in_(deleted_ids))
        )
        await session.commit()

    logger.info(
        "cleanup done",
        extra={"deleted": len(deleted_ids), "retained": len(candidates) - len(deleted_ids)},
    )
    return {"deleted_generations": deleted_ids, "deleted_collections": deleted_collections}
