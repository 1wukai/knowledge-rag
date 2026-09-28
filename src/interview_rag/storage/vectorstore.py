"""Qdrant 向量存储封装：支持本地内嵌模式与服务端 URL 模式，两者同 API。

本地内嵌模式用于无 Docker 调试；P1 起 API 与 worker 多进程共享时使用服务端模式
（Qdrant 服务），collection 按 generation 命名。
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


class VectorStore:
    def __init__(self, *, url: str | None = None, path: Path | None = None):
        if url:
            self._client = QdrantClient(url=url)
        elif path:
            self._client = QdrantClient(path=str(path))
        else:
            raise ValueError("VectorStore 需要 url 或 path 之一")

    def collection_exists(self, name: str) -> bool:
        return self._client.collection_exists(name)

    def recreate(self, name: str, dim: int, sparse: bool = False) -> None:
        if self._client.collection_exists(name):
            self._client.delete_collection(name)
        if sparse:
            self._client.create_collection(
                name,
                vectors_config={
                    "dense": VectorParams(size=dim, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                },
            )
        else:
            self._client.create_collection(
                name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def delete_collection(self, name: str) -> None:
        if self._client.collection_exists(name):
            self._client.delete_collection(name)

    def upsert(self, name: str, points: list[PointStruct]) -> None:
        self._client.upsert(collection_name=name, points=points, wait=True)

    def upsert_sparse(
        self,
        name: str,
        points: list[PointStruct],
        sparse_indices: dict,
        sparse_values: dict,
    ) -> None:
        """写入带 named sparse vector（"sparse"）的 points；dense 向量仍用 point.vector。"""
        full = [
            PointStruct(
                id=point.id,
                vector={
                    "dense": point.vector,
                    "sparse": SparseVector(
                        indices=sparse_indices[point.id], values=sparse_values[point.id]
                    ),
                },
                payload=point.payload,
            )
            for point in points
        ]
        self._client.upsert(collection_name=name, points=full, wait=True)

    def delete_by_doc_id(self, name: str, doc_id: str) -> None:
        self._client.delete(
            collection_name=name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            ),
            wait=True,
        )

    def search(self, name: str, vector: list[float], limit: int, using: str | None = None) -> list[dict]:
        kwargs: dict = {"collection_name": name, "query": vector, "limit": limit}
        if using:
            kwargs["using"] = using
        result = self._client.query_points(**kwargs)
        return self._to_hits(result)

    def search_sparse(
        self, name: str, sparse_indices: list[int], sparse_values: list[float], limit: int
    ) -> list[dict]:
        result = self._client.query_points(
            collection_name=name,
            query=SparseVector(indices=sparse_indices, values=sparse_values),
            using="sparse",
            limit=limit,
        )
        return self._to_hits(result)

    def count(self, name: str) -> int:
        return self._client.count(collection_name=name, exact=True).count

    @staticmethod
    def _to_hits(result) -> list[dict]:
        hits: list[dict] = []
        for point in result.points:
            item = dict(point.payload or {})
            item["id"] = str(point.id)
            item["score"] = point.score
            hits.append(item)
        return hits


# 按连接参数复用 VectorStore，避免每次请求重建客户端。
_stores: dict[str, VectorStore] = {}


def get_store(*, url: str | None = None, path: Path | None = None) -> VectorStore:
    key = url if url else str(path)
    store = _stores.get(key)
    if store is None:
        store = VectorStore(url=url, path=path)
        _stores[key] = store
    return store


def store_for_settings(settings) -> VectorStore:
    """根据 Settings 的 qdrant_mode 返回合适的 VectorStore。"""
    from interview_rag.config import resolve_path

    if settings.qdrant_mode == "server":
        return get_store(url=settings.qdrant_url)
    return get_store(path=resolve_path(settings.qdrant_local_path))
