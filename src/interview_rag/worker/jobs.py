"""arq worker：消费入库任务，执行构建 → 校验 → 发布。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select

from interview_rag.application.build import build_generation
from interview_rag.application.publish import publish_generation
from interview_rag.config import get_settings, resolve_path
from interview_rag.providers import close_clients
from interview_rag.runtime_settings import load
from interview_rag.storage.database import get_database
from interview_rag.storage.models import Corpus, IngestionJob

logger = logging.getLogger("interview_rag.worker")

QUEUE = "ingestion"


def _mark(job, status: str, **fields) -> None:
    job.status = status
    job.updated_at = datetime.now(timezone.utc)
    for key, value in fields.items():
        setattr(job, key, value)
    if status in {"succeeded", "failed", "cancelled"}:
        job.finished_at = datetime.now(timezone.utc)


async def build_job(ctx, job_id: str) -> dict:
    """任务入口：读 job → 构建 generation → CAS 发布。"""
    settings = get_settings()
    runtime = load(resolve_path(settings.runtime_settings_path))
    db = get_database(settings.database_url)

    async with db.session_factory() as session:
        job = await session.scalar(select(IngestionJob).where(IngestionJob.job_id == job_id))
        if not job:
            logger.error("job not found", extra={"job_id": job_id})
            return {"ok": False, "error": "job not found"}
        if job.status == "cancelled":
            return {"ok": False, "error": "cancelled"}

        _mark(job, "running")
        await session.commit()

        try:
            corpus = await session.scalar(
                select(Corpus).where(Corpus.corpus_id == "agent-interview")
            )
            expected_generation_id = corpus.active_generation_id if corpus else None
            expected_revision = corpus.publication_revision if corpus else 0

            generation_id, chunk_count = await build_generation(
                session, settings, runtime, job.manifest_hash, job.doc_ids or [], "agent-interview"
            )
            _mark(job, "validating", target_generation_id=generation_id, progress={"chunks": chunk_count})
            await session.commit()

            # 发布：CAS 比较 active 指针与 revision。
            await publish_generation(
                session,
                "agent-interview",
                generation_id,
                expected_generation_id,
                expected_revision,
            )
            _mark(job, "succeeded")
            await session.commit()
            return {"ok": True, "generation_id": generation_id, "chunks": chunk_count}
        except Exception as exc:  # noqa: BLE001
            logger.exception("build job failed", extra={"job_id": job_id})
            await session.rollback()
            job = await session.scalar(select(IngestionJob).where(IngestionJob.job_id == job_id))
            if job:
                _mark(job, "failed", error=str(exc))
                await session.commit()
            return {"ok": False, "error": str(exc)}


class WorkerSettings:
    """arq worker 配置；functions 由 Worker 入口加载。"""

    functions = [build_job]
    max_tries = 3
    job_timeout = 600


async def on_startup(ctx) -> None:
    logger.info("worker started")


async def on_shutdown(ctx) -> None:
    await close_clients()
