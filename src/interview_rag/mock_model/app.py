"""OpenAI 兼容 mock 模型服务。

用于无云 Key 时联调 Embedding / Rerank / 流式生成。输出确定性强、可重复，
不得作为真实模型质量或性能证据。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="interview-rag-mock-model", version="0.1.0")

DEFAULT_DIM = 256


def _vector(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    # 字符 n-gram 特征哈希：文本词法重叠越高，余弦相似度越高，便于验证检索链路。
    vec = [0.0] * dim
    lowered = text.lower()
    grams = [lowered[i : i + 2] for i in range(len(lowered) - 1)] + [
        lowered[i : i + 3] for i in range(max(0, len(lowered) - 2))
    ]
    for gram in grams:
        digest = hashlib.md5(gram.encode("utf-8")).digest()
        h = int.from_bytes(digest[:8], "big")
        vec[h % dim] += 1.0 if (h & 1) == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def _rerank_score(query: str, text: str) -> float:
    # 关键词重叠的确定性伪分数，仅用于验证排序链路。
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q:
        return 0.0
    return round(len(q & t) / len(q), 6)


class EmbeddingRequest(BaseModel):
    model: str = "mock-embedding"
    input: str | list[str]


class RerankRequest(BaseModel):
    model: str = "mock-rerank"
    query: str
    documents: list[str]
    top_n: int | None = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "mock-chat"
    messages: list[Message]
    stream: bool = False


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest) -> dict[str, object]:
    inputs = req.input if isinstance(req.input, list) else [req.input]
    data = [
        {"object": "embedding", "index": i, "embedding": _vector(text)}
        for i, text in enumerate(inputs)
    ]
    return {
        "object": "list",
        "model": req.model,
        "data": data,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/rerank")
async def rerank(req: RerankRequest) -> dict[str, object]:
    scored = sorted(
        (
            {"index": i, "relevance_score": _rerank_score(req.query, doc)}
            for i, doc in enumerate(req.documents)
        ),
        key=lambda item: item["relevance_score"],
        reverse=True,
    )
    if req.top_n is not None:
        scored = scored[: req.top_n]
    return {"object": "list", "model": req.model, "results": scored}


def _last_user_content(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _answer_for(req: ChatRequest) -> str:
    question = _last_user_content(req.messages)[:60]
    return f"mock 回答：已收到你的问题「{question}」。"


def _chat_chunks(req: ChatRequest) -> Iterator[str]:
    answer = _answer_for(req)
    resp_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    for i in range(0, len(answer), 4):
        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"content": answer[i : i + 4]}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    finish = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.stream:
        return StreamingResponse(_chat_chunks(req), media_type="text/event-stream")
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _answer_for(req)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def main() -> None:
    import uvicorn

    uvicorn.run("interview_rag.mock_model.app:app", host="127.0.0.1", port=8001, log_config=None)


if __name__ == "__main__":
    main()
