"""业务数据表模型（SQLAlchemy 2.0 declarative）。

P1 阶段涉及文档身份、源快照、generation 索引、chunk、入库任务与 outbox。
``corpora`` 表持有唯一 active generation 指针与单调递增的 ``publication_revision``，
是索引发布的权威状态；Qdrant 仅是从冻结快照可重建的检索索引。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import LONGTEXT, MEDIUMBLOB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from interview_rag.storage.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    """稳定的文档身份；doc_id 不随路径变化。"""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    enabled: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class DocumentVersion(Base):
    """冻结的源文件快照：已校验字节 + 元数据；worker 只读快照，不再读工作区。"""

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("doc_id", "source_hash", name="uq_doc_version"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_hash: Mapped[str] = mapped_column(String(72), nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    path: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    raw: Mapped[bytes] = mapped_column(MEDIUMBLOB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class CorpusSnapshot(Base):
    """manifest 提交时的原文与规范摘要快照，独立于当前工作区。"""

    __tablename__ = "corpus_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    manifest_hash: Mapped[str] = mapped_column(String(72), unique=True, nullable=False)
    manifest_json: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Corpus(Base):
    """语料发布指针：active generation 与 publication_revision 的唯一权威。"""

    __tablename__ = "corpora"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    corpus_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    active_generation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    publication_revision: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class IndexGeneration(Base):
    """一次成功构建的不可变检索快照；关联 manifest、pipeline 与物理 collection。"""

    __tablename__ = "index_generations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    generation_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    collection: Mapped[str] = mapped_column(String(128), nullable=False)
    corpus_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="building")
    manifest_hash: Mapped[str] = mapped_column(String(72), nullable=False, default="")
    pipeline_fingerprint: Mapped[str] = mapped_column(String(72), nullable=False, default="")
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bm25_stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Chunk(Base):
    """切分结果：文本、源跨度、Embedding 输入 hash。"""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_doc_source", "doc_id", "source_hash"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    doc_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(72), nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    path: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    heading: Mapped[list | None] = mapped_column(JSON, nullable=True)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embed_input_hash: Mapped[str] = mapped_column(String(72), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class GenerationChunk(Base):
    """generation 与 chunk 的对应关系（一个 chunk 可被多个 generation 引用）。"""

    __tablename__ = "generation_chunks"
    __table_args__ = (
        UniqueConstraint("generation_id", "chunk_id", name="uq_generation_chunk"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    generation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("index_generations.generation_id"), nullable=False, index=True
    )
    chunk_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)


class IngestionJob(Base):
    """入库任务状态机：queued → running → validating → succeeded，异常 retry_wait/failed，取消 cancelled。"""

    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, default=lambda: uuid4().hex)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    doc_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    base_generation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    target_generation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    manifest_hash: Mapped[str] = mapped_column(String(72), nullable=False, default="")
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class OutboxEvent(Base):
    """事务性 outbox 事件：任务状态落库后投递，dispatcher 消费并推入 Redis 队列。"""

    __tablename__ = "outbox_events"
    __table_args__ = (
        Index("ix_outbox_pending", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, default=lambda: uuid4().hex)
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False, default="ingestion_job")
    aggregate_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Conversation(Base):
    """多轮问答会话。"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, default=lambda: uuid4().hex)
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Message(Base):
    """会话中的一条消息；有序，按 token 预算截断进上下文。"""

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation", "conversation_id", "id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.conversation_id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user / assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Answer(Base):
    """一次问答的完整记录：检索与生成版本、最终答案、状态。"""

    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    answer_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, default=lambda: uuid4().hex)
    conversation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(LONGTEXT, nullable=True)
    generation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    generation_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="completed")
    refused: Mapped[bool] = mapped_column(default=False)
    reranked: Mapped[bool] = mapped_column(default=False)
    top_score: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AnswerCitation(Base):
    """答案引用的证据快照：doc_id、路径、标题、行号范围与证据文本。"""

    __tablename__ = "answer_citations"
    __table_args__ = (
        Index("ix_answer_citations_answer", "answer_id", "citation_index"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    answer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("answers.answer_id"), nullable=False, index=True
    )
    citation_index: Mapped[int] = mapped_column(Integer, nullable=False)
    doc_id: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    heading: Mapped[list | None] = mapped_column(JSON, nullable=True)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_hash: Mapped[str] = mapped_column(String(72), nullable=False, default="")
    snippet: Mapped[str] = mapped_column(Text, nullable=False, default="")
