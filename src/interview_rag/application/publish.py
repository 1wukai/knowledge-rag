"""原子发布与回滚：切换 corpora 的 active generation 指针。

MySQL 的 corpora 表是唯一发布权威。发布使用 compare-and-swap：同时比较
预期 active_generation_id 与单调递增的 publication_revision，二者都匹配才切换。
Qdrant 写入成功但 MySQL 发布失败只留下未发布构建，旧服务不受影响。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from interview_rag.storage.models import Corpus, IndexGeneration

logger = logging.getLogger("interview_rag.publish")


class PublishConflict(RuntimeError):
    pass


async def _get_or_create_corpus(session: AsyncSession, corpus_id: str) -> Corpus:
    corpus = await session.scalar(
        select(Corpus).where(Corpus.corpus_id == corpus_id).with_for_update()
    )
    if not corpus:
        corpus = Corpus(corpus_id=corpus_id, publication_revision=0)
        session.add(corpus)
        await session.flush()
    return corpus


async def publish_generation(
    session: AsyncSession,
    corpus_id: str,
    target_generation_id: str,
    expected_generation_id: str | None,
    expected_revision: int,
) -> dict:
    """发布已验证 generation 为 active。返回发布结果摘要。

    若 active 指针或 revision 与预期不符（被其他任务抢先发布/回滚），抛 PublishConflict。
    """
    corpus = await _get_or_create_corpus(session, corpus_id)
    target = await session.scalar(
        select(IndexGeneration).where(
            IndexGeneration.generation_id == target_generation_id,
            IndexGeneration.corpus_id == corpus_id,
        ).with_for_update()
    )
    if not target or target.status not in {"validating", "ready"}:
        raise ValueError(f"generation {target_generation_id} 不存在或未就绪")

    if corpus.active_generation_id != expected_generation_id:
        raise PublishConflict(
            f"active generation 已变化（预期 {expected_generation_id}，实际 {corpus.active_generation_id}）"
        )
    if corpus.publication_revision != expected_revision:
        raise PublishConflict(
            f"publication_revision 已变化（预期 {expected_revision}，实际 {corpus.publication_revision}）"
        )

    new_revision = corpus.publication_revision + 1
    # 旧 active 降级为 ready，保留作回滚对象；目标成为唯一 active。
    if corpus.active_generation_id and corpus.active_generation_id != target_generation_id:
        await session.execute(
            update(IndexGeneration)
            .where(IndexGeneration.generation_id == corpus.active_generation_id)
            .values(status="ready")
        )
    await session.execute(
        update(Corpus)
        .where(Corpus.corpus_id == corpus_id)
        .values(active_generation_id=target_generation_id, publication_revision=new_revision)
    )
    target.status = "active"
    target.activated_at = datetime.now(timezone.utc)
    await session.commit()

    logger.info(
        "generation published",
        extra={"corpus_id": corpus_id, "generation_id": target_generation_id, "revision": new_revision},
    )
    return {"generation_id": target_generation_id, "revision": new_revision}


async def rollback(
    session: AsyncSession,
    corpus_id: str,
    target_generation_id: str,
    expected_revision: int,
) -> dict:
    """回滚 active 指针到指定 generation（同样走 CAS）。"""
    corpus = await _get_or_create_corpus(session, corpus_id)
    target = await session.scalar(
        select(IndexGeneration).where(
            IndexGeneration.generation_id == target_generation_id,
            IndexGeneration.corpus_id == corpus_id,
        ).with_for_update()
    )
    if not target or target.status not in {"ready", "active"}:
        raise ValueError(f"generation {target_generation_id} 不存在或不可回滚")

    if corpus.publication_revision != expected_revision:
        raise PublishConflict(
            f"publication_revision 已变化（预期 {expected_revision}，实际 {corpus.publication_revision}）"
        )

    new_revision = corpus.publication_revision + 1
    # 旧 active 降级为 ready，保留作回滚对象；目标成为唯一 active。
    if corpus.active_generation_id and corpus.active_generation_id != target_generation_id:
        await session.execute(
            update(IndexGeneration)
            .where(IndexGeneration.generation_id == corpus.active_generation_id)
            .values(status="ready")
        )
    await session.execute(
        update(Corpus)
        .where(Corpus.corpus_id == corpus_id)
        .values(active_generation_id=target_generation_id, publication_revision=new_revision)
    )
    target.status = "active"
    target.activated_at = datetime.now(timezone.utc)
    await session.commit()

    logger.info(
        "generation rolled back",
        extra={"corpus_id": corpus_id, "generation_id": target_generation_id, "revision": new_revision},
    )
    return {"generation_id": target_generation_id, "revision": new_revision}
