"""outbox dispatcher：轮询 outbox_events 的 pending 事件，投递到 arq 队列。

dispatcher 与 API 同进程或独立进程运行；投递成功后标记事件为 dispatched。
投递是至少一次的：Redis 丢失队列后仍可通过未投递事件恢复。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from arq.connections import RedisSettings, create_pool
from sqlalchemy import select, update

from interview_rag.application.ingestion_service import EVENT_INGEST_REQUESTED
from interview_rag.storage.database import get_database
from interview_rag.storage.models import OutboxEvent

logger = logging.getLogger("interview_rag.dispatcher")

QUEUE = "ingestion"


async def _enqueue(redis, event_type: str, payload: dict) -> None:
    if event_type == EVENT_INGEST_REQUESTED:
        await redis.enqueue_job("build_job", payload["job_id"], _queue_name=QUEUE)
    else:
        logger.warning("unknown outbox event type", extra={"event_type": event_type})


async def dispatch_pending(db, redis, limit: int = 50) -> int:
    """投递一批 pending 事件，返回投递数量。"""
    async with db.session_factory() as session:
        events = (
            (await session.execute(
                select(OutboxEvent)
                .where(OutboxEvent.status == "pending")
                .order_by(OutboxEvent.created_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            ))
            .scalars()
            .all()
        )
        count = 0
        for event in events:
            try:
                await _enqueue(redis, event.event_type, event.payload or {})
            except Exception:
                logger.exception("outbox enqueue failed", extra={"event_id": event.event_id})
                event.retry_count += 1
                continue
            event.status = "dispatched"
            event.dispatched_at = datetime.now(timezone.utc)
            count += 1
        await session.commit()
        return count


async def run_dispatcher(settings, poll_interval: float = 1.0) -> None:
    """独立 dispatcher 循环。"""
    db = get_database(settings.database_url)
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    logger.info("dispatcher started")
    try:
        while True:
            try:
                await dispatch_pending(db, redis)
            except Exception:
                logger.exception("dispatch round failed")
            await asyncio.sleep(poll_interval)
    finally:
        await redis.aclose()
