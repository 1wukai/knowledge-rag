"""会话与答案持久化：会话创建/查询、消息追加、答案与引用落盘。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from interview_rag.storage.models import Answer, AnswerCitation, Conversation, Message


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def create_conversation(session: AsyncSession, title: str = "") -> dict:
    conv = Conversation(conversation_id=uuid4().hex, title=title or "新会话")
    session.add(conv)
    await session.commit()
    return {
        "conversation_id": conv.conversation_id, "title": conv.title,
        "created_at": conv.created_at.isoformat(), "updated_at": conv.updated_at.isoformat(),
    }


async def get_conversation(session: AsyncSession, conversation_id: str) -> Conversation | None:
    return await session.scalar(
        select(Conversation).where(Conversation.conversation_id == conversation_id)
    )


async def list_conversations(session: AsyncSession) -> list[dict]:
    rows = (
        await session.execute(
            select(Conversation).order_by(Conversation.updated_at.desc())
        )
    ).scalars().all()
    return [
        {
            "conversation_id": c.conversation_id,
            "title": c.title,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in rows
    ]


async def list_messages(session: AsyncSession, conversation_id: str) -> list[dict]:
    rows = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
    ).scalars().all()
    return [
        {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat() if m.created_at else None}
        for m in rows
    ]


async def append_message(session: AsyncSession, conversation_id: str, role: str, content: str) -> None:
    session.add(Message(conversation_id=conversation_id, role=role, content=content))
    conv = await session.scalar(
        select(Conversation).where(Conversation.conversation_id == conversation_id)
    )
    if conv:
        conv.updated_at = _utcnow()
        if role == "user" and conv.title == "新会话":
            conv.title = content.strip()[:48] or "新会话"
    await session.commit()


async def save_answer(
    session: AsyncSession,
    *,
    answer_id: str,
    conversation_id: str | None,
    question: str,
    answer: str | None,
    generation_id: str | None,
    embedding_model: str | None,
    generation_model: str | None,
    status: str = "completed",
    refused: bool = False,
    reranked: bool = False,
    top_score: float | None = None,
    citations: list[dict] | None = None,
) -> None:
    """落盘答案与引用快照。"""
    session.add(
        Answer(
            answer_id=answer_id,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            generation_id=generation_id,
            embedding_model=embedding_model,
            generation_model=generation_model,
            status=status,
            refused=refused,
            reranked=reranked,
            top_score=top_score,
        )
    )
    for c in citations or []:
        session.add(
            AnswerCitation(
                answer_id=answer_id,
                citation_index=c["index"],
                doc_id=c["doc_id"],
                title=c["title"],
                heading=c.get("heading") or [],
                start_line=c["start_line"],
                end_line=c["end_line"],
                source_hash=c["source_hash"],
                snippet=c.get("snippet", ""),
            )
        )
    await session.commit()


async def get_answer(session: AsyncSession, answer_id: str) -> dict | None:
    """查询答案及其版本化引用。"""
    answer = await session.scalar(select(Answer).where(Answer.answer_id == answer_id))
    if not answer:
        return None
    citations = (
        await session.execute(
            select(AnswerCitation)
            .where(AnswerCitation.answer_id == answer_id)
            .order_by(AnswerCitation.citation_index)
        )
    ).scalars().all()
    return {
        "answer_id": answer.answer_id,
        "conversation_id": answer.conversation_id,
        "question": answer.question,
        "answer": answer.answer,
        "generation_id": answer.generation_id,
        "embedding_model": answer.embedding_model,
        "generation_model": answer.generation_model,
        "status": answer.status,
        "refused": answer.refused,
        "reranked": answer.reranked,
        "top_score": answer.top_score,
        "created_at": answer.created_at.isoformat() if answer.created_at else None,
        "citations": [
            {
                "index": c.citation_index,
                "doc_id": c.doc_id,
                "title": c.title,
                "heading": c.heading or [],
                "start_line": c.start_line,
                "end_line": c.end_line,
                "source_hash": c.source_hash,
                "snippet": c.snippet,
            }
            for c in citations
        ],
    }


async def list_answers(session: AsyncSession, conversation_id: str | None = None) -> list[dict]:
    """按会话（或全部）列出答案摘要。"""
    query = select(Answer).order_by(Answer.id.desc())
    if conversation_id:
        query = query.where(Answer.conversation_id == conversation_id)
    rows = (await session.execute(query.limit(100))).scalars().all()
    return [
        {
            "answer_id": a.answer_id,
            "conversation_id": a.conversation_id,
            "question": a.question[:80],
            "status": a.status,
            "refused": a.refused,
            "top_score": a.top_score,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in rows
    ]
