"""initial schema: documents, versions, generations, chunks, jobs, outbox

Revision ID: 0001
Revises:
Create Date: 2026-09-10
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("doc_id", sa.String(length=128), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doc_id"),
    )

    op.create_table(
        "document_versions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("doc_id", sa.String(length=128), nullable=False),
        sa.Column("source_hash", sa.String(length=72), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.Column("raw", mysql.MEDIUMBLOB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doc_id", "source_hash", name="uq_doc_version"),
    )
    op.create_index("ix_document_versions_doc_id", "document_versions", ["doc_id"])

    op.create_table(
        "corpus_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("manifest_hash", sa.String(length=72), nullable=False),
        sa.Column("manifest_json", mysql.LONGTEXT(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("manifest_hash"),
    )

    op.create_table(
        "corpora",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("corpus_id", sa.String(length=128), nullable=False),
        sa.Column("active_generation_id", sa.String(length=64), nullable=True),
        sa.Column("publication_revision", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("corpus_id"),
    )

    op.create_table(
        "index_generations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("generation_id", sa.String(length=64), nullable=False),
        sa.Column("collection", sa.String(length=128), nullable=False),
        sa.Column("corpus_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("manifest_hash", sa.String(length=72), nullable=False),
        sa.Column("pipeline_fingerprint", sa.String(length=72), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("bm25_stats", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("generation_id"),
    )
    op.create_index("ix_index_generations_corpus_id", "index_generations", ["corpus_id"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("doc_id", sa.String(length=128), nullable=False),
        sa.Column("source_hash", sa.String(length=72), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.Column("heading", sa.JSON(), nullable=True),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embed_input_hash", sa.String(length=72), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id"),
    )
    op.create_index("ix_chunks_doc_source", "chunks", ["doc_id", "source_hash"])

    op.create_table(
        "generation_chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("generation_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["generation_id"], ["index_generations.generation_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("generation_id", "chunk_id", name="uq_generation_chunk"),
    )
    op.create_index("ix_generation_chunks_generation_id", "generation_chunks", ["generation_id"])
    op.create_index("ix_generation_chunks_chunk_id", "generation_chunks", ["chunk_id"])

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("doc_ids", sa.JSON(), nullable=True),
        sa.Column("base_generation_id", sa.String(length=64), nullable=True),
        sa.Column("target_generation_id", sa.String(length=64), nullable=True),
        sa.Column("manifest_hash", sa.String(length=72), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("progress", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_ingestion_jobs_idempotency_key", "ingestion_jobs", ["idempotency_key"])

    op.create_table(
        "outbox_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("aggregate_type", sa.String(length=64), nullable=False),
        sa.Column("aggregate_id", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id"),
    )
    op.create_index("ix_outbox_events_aggregate_id", "outbox_events", ["aggregate_id"])
    op.create_index("ix_outbox_pending", "outbox_events", ["status", "created_at"])


def downgrade() -> None:
    op.drop_table("outbox_events")
    op.drop_table("ingestion_jobs")
    op.drop_table("generation_chunks")
    op.drop_table("chunks")
    op.drop_table("index_generations")
    op.drop_table("corpora")
    op.drop_table("corpus_snapshots")
    op.drop_table("document_versions")
    op.drop_table("documents")
