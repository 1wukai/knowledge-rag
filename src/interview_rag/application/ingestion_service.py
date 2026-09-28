"""入库任务编排：提交入库任务（写 job + outbox），查询任务状态。

提交流程在一个数据库事务内完成：冻结源快照 → 创建 ingestion_jobs → 写 outbox_events。
事务提交后由 dispatcher 消费 outbox 投递到 Redis，arq worker 执行构建。
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from interview_rag.application.snapshot import freeze_documents
from interview_rag.manifest import load_manifest, resolve_manifest_path
from interview_rag.storage.models import IngestionJob, OutboxEvent

EVENT_INGEST_REQUESTED = "ingestion_requested"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def submit_ingestion(
    session: AsyncSession,
    settings,
    doc_ids: list[str],
) -> dict:
    """校验并提交入库任务；返回 job 摘要。同一 idempotency_key 幂等。"""
    manifest = load_manifest(resolve_manifest_path(settings))
    by_id = {d.doc_id: d for d in manifest.documents}

    selected, missing = [], []
    for doc_id in doc_ids:
        doc = by_id.get(doc_id)
        if doc and doc.enabled:
            selected.append(doc)
        else:
            missing.append(doc_id)
    if not selected:
        raise ValueError("所选笔记没有可入库的内容")

    manifest_hash, _raw = await freeze_documents(
        session, manifest, selected, settings.workspace_root
    )

    idempotency_key = "sha256:" + hashlib.sha256(
        f"{manifest_hash}:{','.join(sorted(d.doc_id for d in selected))}".encode("utf-8")
    ).hexdigest()

    # 幂等：相同 key 已存在成功/进行中的任务则直接返回，不重复入队。
    existing = await session.scalar(
        select(IngestionJob).where(IngestionJob.idempotency_key == idempotency_key)
    )
    if existing and existing.status in {"queued", "running", "validating"}:
        return _job_summary(existing)

    job = IngestionJob(
        job_id=uuid4().hex,
        status="queued",
        idempotency_key=idempotency_key,
        doc_ids=sorted(d.doc_id for d in selected),
        manifest_hash=manifest_hash,
    )
    session.add(job)
    await session.flush()

    session.add(
        OutboxEvent(
            event_id=uuid4().hex,
            aggregate_type="ingestion_job",
            aggregate_id=job.job_id,
            event_type=EVENT_INGEST_REQUESTED,
            payload={"job_id": job.job_id},
        )
    )
    await session.commit()
    return _job_summary(job)


async def get_job(session: AsyncSession, job_id: str) -> dict | None:
    job = await session.scalar(select(IngestionJob).where(IngestionJob.job_id == job_id))
    return _job_summary(job) if job else None


async def list_jobs(session: AsyncSession, limit: int = 50) -> list[dict]:
    jobs = (await session.scalars(select(IngestionJob).order_by(IngestionJob.id.desc()).limit(limit))).all()
    return [_job_summary(job) for job in jobs]


def _job_summary(job: IngestionJob) -> dict:
    return {
        "job_id": job.job_id,
        "doc_ids": job.doc_ids or [],
        "retry_count": job.retry_count,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "status": job.status,
        "manifest_hash": job.manifest_hash,
        "target_generation_id": job.target_generation_id,
        "error": job.error,
        "progress": job.progress,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }
