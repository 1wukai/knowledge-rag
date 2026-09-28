"""混合检索：dense 与 sparse 并行召回 + RRF 融合。

RRF 用排名而非直接相加不同量纲的相似度分数。拒答判断仍以 dense top-1 余弦
相似度为准（与 embedding 空间阈值同量纲），RRF 分数只用于排序。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from interview_rag.retrieval.sparse import (
    SparseStats,
    Vocabulary,
    encode_query,
    tokenize,
)
from interview_rag.storage.vectorstore import VectorStore

logger = logging.getLogger("interview_rag.retrieval.hybrid")


@dataclass
class HybridResult:
    hits: list[dict]
    dense_top_score: float


def rrf_fuse(ranked_lists: list[list[dict]], k: int = 60) -> dict[str, float]:
    """RRF 融合多个已排序召回列表，返回 {point_id: fused_score}。"""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            pid = item["id"]
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return scores


def build_sparse_query(generation, question: str) -> tuple[list[int], list[float]] | None:
    """从 generation 的 bm25_stats 还原词表与统计，编码查询稀疏向量。"""
    stats_data = generation.bm25_stats
    if not stats_data:
        return None
    stats = SparseStats.from_dict(stats_data.get("stats", {}))
    vocab = Vocabulary.from_dict(stats_data.get("vocabulary", {}))
    indices, values = encode_query(tokenize(question), stats, vocab)
    return (indices, values) if indices else None


async def hybrid_search(
    store: VectorStore,
    collection: str,
    dense_vector: list[float],
    sparse_query: tuple[list[int], list[float]] | None,
    *,
    dense_limit: int = 30,
    sparse_limit: int = 30,
    final_limit: int = 20,
    rrf_k: int = 60,
) -> HybridResult:
    """并行召回 dense 与 sparse，RRF 融合后截断到 final_limit。"""
    # VectorStore 基于同步 QdrantClient，放到线程池并发执行，避免阻塞事件循环。
    if sparse_query:
        dense_hits, sparse_hits = await asyncio.gather(
            asyncio.to_thread(store.search, collection, dense_vector, dense_limit, "dense"),
            asyncio.to_thread(
                store.search_sparse, collection, sparse_query[0], sparse_query[1], sparse_limit
            ),
        )
    else:
        dense_hits = await asyncio.to_thread(
            store.search, collection, dense_vector, dense_limit, "dense"
        )
        sparse_hits: list[dict] = []

    dense_top_score = dense_hits[0]["score"] if dense_hits else 0.0

    fused = rrf_fuse([dense_hits, sparse_hits], k=rrf_k)
    by_id: dict[str, dict] = {}
    for hit in dense_hits + sparse_hits:
        by_id.setdefault(hit["id"], hit)

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:final_limit]
    hits = [dict(by_id[pid], rrf_score=round(score, 6)) for pid, score in ranked]
    return HybridResult(hits=hits, dense_top_score=dense_top_score)
