"""按 active generation 查询：检索 → Rerank → 证据预算 → 生成（流式/非流式）。

server 模式（qdrant_mode=server）走本路径，支持多轮会话与答案/引用落盘。
P0 的 index_state.json 路径（query.py）保留作本地无 Docker 回退。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import uuid4

from sqlalchemy import select

from interview_rag.config import Settings
from interview_rag.providers import chat, chat_stream, embed_texts, rerank
from interview_rag.retrieval.budget import estimate_tokens, select_within_budget
from interview_rag.retrieval.hybrid import build_sparse_query, hybrid_search
from interview_rag.runtime_settings import RuntimeSettings
from interview_rag.storage.database import get_database
from interview_rag.storage.models import Conversation, Corpus, IndexGeneration, Message
from interview_rag.storage.vectorstore import store_for_settings

logger = logging.getLogger("interview_rag.query_generation")

_SYSTEM_PROMPT = (
    "你是面试知识库问答助手。只依据下面给出的资料回答问题，"
    "引用资料时用 [编号] 标注。资料不足以回答时，明确说明无法回答，不要编造。用中文回答。"
)

# 多轮历史进上下文的消息条数上限（按 token 预算从新到旧截断）。
_HISTORY_MAX_TOKENS = 2000


class NoActiveGenerationError(RuntimeError):
    pass


class IncompatibleIndexError(RuntimeError):
    pass


@dataclass
class RetrievalOutcome:
    """检索阶段结果：命中则带证据与引用，未命中则 refused。"""

    generation_id: str
    embedding_model: str
    chunk_count: int
    refused: bool
    top_score: float | None
    threshold: float
    reranked: bool = False
    hits: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)


async def _load_generation(session, corpus_id: str) -> IndexGeneration:
    corpus = await session.scalar(select(Corpus).where(Corpus.corpus_id == corpus_id))
    if not corpus or not corpus.active_generation_id:
        raise NoActiveGenerationError("尚无已发布的索引，请先提交入库")
    generation = await session.scalar(
        select(IndexGeneration).where(
            IndexGeneration.generation_id == corpus.active_generation_id
        )
    )
    if not generation:
        raise NoActiveGenerationError("active generation 不存在")
    return generation


async def _retrieve(
    settings: Settings,
    runtime: RuntimeSettings,
    session,
    generation: IndexGeneration,
    question: str,
    top_k: int,
    min_score: float | None,
) -> RetrievalOutcome:
    store = store_for_settings(settings)
    query_vector = (await embed_texts(runtime.embedding, [question]))[0]

    if generation.embedding_dim and len(query_vector) != generation.embedding_dim:
        raise IncompatibleIndexError(
            f"向量维度不匹配：索引用 {generation.embedding_model}"
            f"（{generation.embedding_dim} 维）建立，当前 embedding 输出 {len(query_vector)} 维。"
            "切换 embedding 模型后需重新入库。"
        )
    if generation.embedding_model and generation.embedding_model != runtime.embedding.model:
        raise IncompatibleIndexError(
            f"embedding 模型已变更：索引用 {generation.embedding_model} 建立，"
            f"当前为 {runtime.embedding.model}。请重新入库。"
        )

    sparse_query = build_sparse_query(generation, question)
    result = await hybrid_search(
        store,
        generation.collection,
        query_vector,
        sparse_query,
        dense_limit=30,
        sparse_limit=30,
        final_limit=20,
    )
    hits = result.hits

    threshold = settings.retrieval_min_score if min_score is None else min_score
    top_score = result.dense_top_score

    outcome = RetrievalOutcome(
        generation_id=generation.generation_id,
        embedding_model=generation.embedding_model,
        chunk_count=generation.chunk_count,
        refused=False,
        top_score=round(top_score, 4) if hits else None,
        threshold=threshold,
    )

    if not hits or top_score < threshold:
        outcome.refused = True
        return outcome

    # Rerank 重排候选；失败或未配置时降级为 RRF 排名。
    if runtime.rerank and runtime.rerank.base_url:
        try:
            rerank_results = await rerank(
                runtime.rerank, question, [h["text"] for h in hits], top_n=max(top_k, 8)
            )
            ordered_idx = [r["index"] for r in rerank_results if r.get("index") is not None]
            ordered_idx = [i for i in ordered_idx if 0 <= i < len(hits)]
            rest = [i for i in range(len(hits)) if i not in set(ordered_idx)]
            hits = [hits[i] for i in ordered_idx + rest]
            outcome.reranked = True
        except Exception:
            logger.exception("rerank failed, fallback to rrf order")

    # 证据预算：在 token 预算内选择进入上下文的证据。
    selected, _used = select_within_budget(hits[:top_k], settings.context_max_tokens, min_hits=1)
    outcome.hits = selected

    for i, hit in enumerate(selected, start=1):
        text = hit["text"]
        outcome.citations.append(
            {
                "index": i,
                "doc_id": hit["doc_id"],
                "title": hit["title"],
                "heading": hit.get("heading") or [],
                "start_line": hit["start_line"],
                "end_line": hit["end_line"],
                "source_hash": hit["hash"],
                "snippet": text[:200] + ("…" if len(text) > 200 else ""),
            }
        )
    return outcome


def _evidence_prompt(citations: list[dict], hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, start=1):
        heading = " > ".join(hit.get("heading") or [])
        source = hit["title"] + (f" / {heading}" if heading else "")
        parts.append(f"[{i}] 来源：{source}\n{hit['text']}")
    return "\n\n".join(parts)


def _history_messages(history: list[dict]) -> list[dict]:
    """按 token 预算从新到旧截断历史消息，保持时序。"""
    selected: list[dict] = []
    used = 0
    for msg in reversed(history):
        cost = estimate_tokens(msg["content"])
        if selected and used + cost > _HISTORY_MAX_TOKENS:
            break
        selected.append(msg)
        used += cost
    return list(reversed(selected))


async def _load_history(session, conversation_id: str | None) -> list[dict]:
    if not conversation_id:
        return []
    rows = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(20)
        )
    ).scalars().all()
    return [{"role": m.role, "content": m.content} for m in reversed(rows)]


def _build_messages(history: list[dict], question: str, evidence: str) -> list[dict]:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(_history_messages(history))
    messages.append({"role": "user", "content": f"资料：\n{evidence}\n\n问题：{question}"})
    return messages


async def answer_question(
    settings: Settings,
    runtime: RuntimeSettings,
    question: str,
    top_k: int = 5,
    min_score: float | None = None,
    conversation_id: str | None = None,
) -> dict:
    """非流式问答（保留给兼容与测试）。"""
    db = get_database(settings.database_url)
    async with db.session_factory() as session:
        generation = await _load_generation(session, "agent-interview")
        outcome = await _retrieve(settings, runtime, session, generation, question, top_k, min_score)

        if outcome.refused:
            return {
                "answer": None,
                "citations": [],
                "index": {
                    "generation_id": outcome.generation_id,
                    "embedding_model": outcome.embedding_model,
                    "chunk_count": outcome.chunk_count,
                },
                "refused": True,
                "top_score": outcome.top_score,
                "threshold": outcome.threshold,
            }

        history = await _load_history(session, conversation_id)
        evidence = _evidence_prompt(outcome.citations, outcome.hits)
        messages = _build_messages(history, question, evidence)
        answer = await chat(runtime.generation, messages)
        return {
            "answer": answer,
            "citations": outcome.citations,
            "index": {
                "generation_id": outcome.generation_id,
                "embedding_model": outcome.embedding_model,
                "chunk_count": outcome.chunk_count,
            },
            "refused": False,
            "top_score": outcome.top_score,
            "threshold": outcome.threshold,
            "reranked": outcome.reranked,
        }


async def answer_question_stream(
    settings: Settings,
    runtime: RuntimeSettings,
    question: str,
    conversation_id: str | None = None,
    top_k: int = 5,
    min_score: float | None = None,
):
    """流式问答：yield SSE 事件 dict（event/data）。

    事件序列：meta → delta* → citations → done；错误发 error。
    """
    db = get_database(settings.database_url)
    answer_id = uuid4().hex

    async with db.session_factory() as session:
        try:
            generation = await _load_generation(session, "agent-interview")
            outcome = await _retrieve(
                settings, runtime, session, generation, question, top_k, min_score
            )

            yield {
                "event": "meta",
                "data": {
                    "answer_id": answer_id,
                    "generation_id": outcome.generation_id,
                    "refused": outcome.refused,
                },
            }

            if outcome.refused:
                from interview_rag.application.conversation_service import (
                    append_message,
                    save_answer,
                )

                await save_answer(
                    session,
                    answer_id=answer_id,
                    conversation_id=conversation_id,
                    question=question,
                    answer=None,
                    generation_id=outcome.generation_id,
                    embedding_model=outcome.embedding_model,
                    generation_model=runtime.generation.model,
                    status="refused",
                    refused=True,
                    top_score=outcome.top_score,
                )
                if conversation_id:
                    await append_message(session, conversation_id, "user", question)
                    await append_message(session, conversation_id, "assistant", "")
                yield {
                    "event": "done",
                    "data": {
                        "answer_id": answer_id,
                        "refused": True,
                        "top_score": outcome.top_score,
                        "threshold": outcome.threshold,
                        "reranked": outcome.reranked,
                    },
                }
                return

            history = await _load_history(session, conversation_id)
            evidence = _evidence_prompt(outcome.citations, outcome.hits)
            messages = _build_messages(history, question, evidence)

            answer_parts: list[str] = []
            async for delta in chat_stream(runtime.generation, messages):
                answer_parts.append(delta)
                yield {"event": "delta", "data": {"content": delta}}

            answer = "".join(answer_parts)
            yield {
                "event": "citations",
                "data": {"citations": outcome.citations},
            }

            from interview_rag.application.conversation_service import (
                append_message,
                save_answer,
            )

            await save_answer(
                session,
                answer_id=answer_id,
                conversation_id=conversation_id,
                question=question,
                answer=answer,
                generation_id=outcome.generation_id,
                embedding_model=outcome.embedding_model,
                generation_model=runtime.generation.model,
                status="completed",
                refused=False,
                reranked=outcome.reranked,
                top_score=outcome.top_score,
                citations=outcome.citations,
            )
            if conversation_id:
                await append_message(session, conversation_id, "user", question)
                await append_message(session, conversation_id, "assistant", answer)

            yield {
                "event": "done",
                "data": {
                    "answer_id": answer_id,
                    "refused": False,
                    "top_score": outcome.top_score,
                    "threshold": outcome.threshold,
                    "reranked": outcome.reranked,
                },
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream answer failed")
            yield {"event": "error", "data": {"message": str(exc)}}
