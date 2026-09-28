"""OpenAI 兼容模型访问：Embedding、Chat 与 Rerank。"""

from __future__ import annotations

import httpx
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def _endpoint(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


# 按 base_url 复用连接池，避免每次调用重建 TCP + TLS 连接。
_clients: dict[str, httpx.AsyncClient] = {}


def _client_for(base_url: str) -> httpx.AsyncClient:
    client = _clients.get(base_url)
    if client is None or client.is_closed:
        client = httpx.AsyncClient()
        _clients[base_url] = client
    return client


async def close_clients() -> None:
    for client in _clients.values():
        await client.aclose()
    _clients.clear()


async def embed_texts(
    config: ProviderConfig, texts: list[str], timeout: float = 120.0
) -> list[list[float]]:
    if not texts:
        return []
    client = _client_for(config.base_url)
    response = await client.post(
        _endpoint(config.base_url, "/embeddings"),
        headers=_headers(config.api_key),
        json={"model": config.model, "input": texts},
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise ProviderError(
            f"embedding 调用失败（HTTP {response.status_code}）：{response.text[:300]}",
            status=response.status_code,
            body=response.text[:500],
        )
    data = response.json().get("data", [])
    data = sorted(data, key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in data]


async def chat(
    config: ProviderConfig,
    messages: list[dict[str, str]],
    timeout: float = 180.0,
    temperature: float = 0.2,
) -> str:
    client = _client_for(config.base_url)
    response = await client.post(
        _endpoint(config.base_url, "/chat/completions"),
        headers=_headers(config.api_key),
        json={
            "model": config.model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise ProviderError(
            f"chat 调用失败（HTTP {response.status_code}）：{response.text[:300]}",
            status=response.status_code,
            body=response.text[:500],
        )
    return response.json()["choices"][0]["message"]["content"]


async def chat_stream(
    config: ProviderConfig,
    messages: list[dict[str, str]],
    timeout: float = 180.0,
    temperature: float = 0.2,
):
    """流式 Chat：解析 OpenAI 兼容 SSE，逐段 yield 文本增量。"""
    client = _client_for(config.base_url)
    async with client.stream(
        "POST",
        _endpoint(config.base_url, "/chat/completions"),
        headers=_headers(config.api_key),
        json={
            "model": config.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        },
        timeout=timeout,
    ) as response:
        if response.status_code >= 400:
            body = await response.aread()
            raise ProviderError(
                f"chat 流式调用失败（HTTP {response.status_code}）：{body[:300].decode('utf-8', 'ignore')}",
                status=response.status_code,
                body=body[:500].decode("utf-8", "ignore"),
            )
        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                import json

                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


async def rerank(
    config: ProviderConfig,
    query: str,
    documents: list[str],
    top_n: int | None = None,
    timeout: float = 30.0,
) -> list[dict]:
    """Rerank：返回按相关性降序的 [{"index", "relevance_score"}]。

    Rerank 没有统一的 OpenAI 标准协议，本项目按 `results[index, relevance_score]`
    结构适配（mock、SiliconFlow、Cohere、Jina 等一致）。index 指向 documents 入参下标。
    """
    if not documents:
        return []
    client = _client_for(config.base_url)
    payload: dict = {"model": config.model, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n
    response = await client.post(
        _endpoint(config.base_url, "/rerank"),
        headers=_headers(config.api_key),
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise ProviderError(
            f"rerank 调用失败（HTTP {response.status_code}）：{response.text[:300]}",
            status=response.status_code,
            body=response.text[:500],
        )
    results = response.json().get("results", [])
    results.sort(key=lambda item: item.get("relevance_score", 0.0), reverse=True)
    return results
