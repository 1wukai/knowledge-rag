"""问答编排：检索 → 构建证据上下文 → 生成 → 返回带来源引用的答案。"""

from __future__ import annotations

import logging

from interview_rag.application.index_state import load_state
from interview_rag.config import Settings, resolve_path
from interview_rag.providers import chat, embed_texts
from interview_rag.runtime_settings import RuntimeSettings
from interview_rag.storage.vectorstore import store_for_settings

logger = logging.getLogger("interview_rag.query")

_SYSTEM_PROMPT = (
    "你是面试知识库问答助手。只依据下面给出的资料回答问题，"
    "引用资料时用 [编号] 标注。资料不足以回答时，明确说明无法回答，不要编造。用中文回答。"
)


class NoIndexError(RuntimeError):
    pass


class IncompatibleIndexError(RuntimeError):
    pass


async def answer_question(
    settings: Settings,
    runtime: RuntimeSettings,
    question: str,
    top_k: int = 5,
    min_score: float | None = None,
) -> dict:
    state = load_state(resolve_path(settings.index_state_path))
    if not state:
        raise NoIndexError("尚未入库：请先在网页选择笔记并入库")

    store = store_for_settings(settings)
    query_vector = (await embed_texts(runtime.embedding, [question]))[0]

    indexed_model = state.get("embedding_model")
    indexed_dim = state.get("embedding_dim")
    if indexed_dim and len(query_vector) != indexed_dim:
        raise IncompatibleIndexError(
            f"向量维度不匹配：索引用 {indexed_model}（{indexed_dim} 维）建立，"
            f"当前 embedding 输出 {len(query_vector)} 维。切换 embedding 模型后需重新入库。"
        )
    if indexed_model and indexed_model != runtime.embedding.model:
        raise IncompatibleIndexError(
            f"embedding 模型已变更：索引用 {indexed_model} 建立，当前为 {runtime.embedding.model}。请重新入库。"
        )

    hits = store.search(state["collection"], query_vector, top_k)

    threshold = settings.retrieval_min_score if min_score is None else min_score
    top_score = hits[0]["score"] if hits else 0.0

    if not hits or top_score < threshold:
        return {
            "answer": None,
            "citations": [],
            "index": state,
            "refused": True,
            "top_score": round(top_score, 4) if hits else None,
            "threshold": threshold,
        }

    context_parts: list[str] = []
    citations: list[dict] = []
    for i, hit in enumerate(hits, start=1):
        heading = " > ".join(hit.get("heading") or [])
        source = hit["title"] + (f" / {heading}" if heading else "")
        context_parts.append(f"[{i}] 来源：{source}\n{hit['text']}")
        text = hit["text"]
        citations.append(
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

    user_prompt = "资料：\n" + "\n\n".join(context_parts) + f"\n\n问题：{question}"
    answer = await chat(
        runtime.generation,
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return {
        "answer": answer,
        "citations": citations,
        "index": state,
        "refused": False,
        "top_score": round(hits[0]["score"], 4),
        "threshold": threshold,
    }
