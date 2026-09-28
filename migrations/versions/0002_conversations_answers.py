"""add category, conversations, messages, answers, answer_citations

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-10
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("category", sa.String(length=64), nullable=False, server_default=""))

    op.create_table(
        "conversations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id"),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.conversation_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_conversation", "messages", ["conversation_id", "id"])

    op.create_table(
        "answers",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("answer_id", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", mysql.LONGTEXT(), nullable=True),
        sa.Column("generation_id", sa.String(length=64), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("generation_model", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("refused", sa.Boolean(), nullable=False),
        sa.Column("reranked", sa.Boolean(), nullable=False),
        sa.Column("top_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("answer_id"),
    )
    op.create_index("ix_answers_conversation_id", "answers", ["conversation_id"])

    op.create_table(
        "answer_citations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("answer_id", sa.String(length=64), nullable=False),
        sa.Column("citation_index", sa.Integer(), nullable=False),
        sa.Column("doc_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("heading", sa.JSON(), nullable=True),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("source_hash", sa.String(length=72), nullable=False),
        sa.Column("snippet", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["answer_id"], ["answers.answer_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_answer_citations_answer", "answer_citations", ["answer_id", "citation_index"])


def downgrade() -> None:
    op.drop_table("answer_citations")
    op.drop_table("answers")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_column("documents", "category")
