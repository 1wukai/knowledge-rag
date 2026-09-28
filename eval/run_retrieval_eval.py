"""P2 检索质量评估：对比 dense-only / 混合(RRF) / 混合+Rerank 三种模式。

用法（在 projects/interview-rag 目录）：
    uv run python eval/run_retrieval_eval.py [--dataset eval/datasets/p2_seed.json] [--k 10]

依赖 server 模式基础设施（MySQL + Qdrant 服务），读取 active generation 的
bm25_stats 与 collection。输出各模式的 Recall@k 与 MRR。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

from sqlalchemy import select

from interview_rag.config import Settings
from interview_rag.providers import embed_texts, rerank
from interview_rag.retrieval.hybrid import build_sparse_query, hybrid_search, rrf_fuse
from interview_rag.runtime_settings import load
from interview_rag.storage.database import get_database
from interview_rag.storage.models import Corpus, IndexGeneration
from interview_rag.storage.vectorstore import store_for_settings

MODE_DENSE = "dense"
MODE_HYBRID = "hybrid"
MODE_RERANK = "hybrid+rerank"
MODES = [MODE_DENSE, MODE_HYBRID, MODE_RERANK]


async def _load_generation(session) -> IndexGeneration:
    corpus = await session.scalar(select(Corpus).where(Corpus.corpus_id == "agent-interview"))
    if not corpus or not corpus.active_generation_id:
        raise RuntimeError("无 active generation，请先入库")
    generation = await session.scalar(
        select(IndexGeneration).where(IndexGeneration.generation_id == corpus.active_generation_id)
    )
    if not generation:
        raise RuntimeError("active generation 不存在")
    return generation


async def _retrieve(settings, runtime, store, generation, question, mode, k):
    qvec = (await embed_texts(runtime.embedding, [question]))[0]
    if mode == MODE_DENSE:
        return store.search(generation.collection, qvec, k, using="dense")
    sparse_query = build_sparse_query(generation, question)
    result = await hybrid_search(
        store, generation.collection, qvec, sparse_query,
        dense_limit=k, sparse_limit=k, final_limit=k,
    )
    hits = result.hits
    if mode == MODE_RERANK and runtime.rerank and runtime.rerank.base_url and hits:
        try:
            rr = await rerank(runtime.rerank, question, [h["text"] for h in hits], top_n=k)
            idx = [r["index"] for r in rr if 0 <= r.get("index", -1) < len(hits)]
            rest = [i for i in range(len(hits)) if i not in set(idx)]
            hits = [hits[i] for i in idx + rest]
        except Exception:
            pass
    return hits


def recall_at_k(hits: list[dict], expected_doc_id: str | None, k: int) -> bool:
    if expected_doc_id is None:
        return True  # out_of_corpus 不参与 recall 计算，单独看拒答
    top_docs = {h.get("doc_id") for h in hits[:k]}
    return expected_doc_id in top_docs


def mrr_at_k(hits: list[dict], expected_doc_id: str | None, k: int) -> float:
    if expected_doc_id is None:
        return 0.0
    for rank, h in enumerate(hits[:k], start=1):
        if h.get("doc_id") == expected_doc_id:
            return 1.0 / rank
    return 0.0


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("eval/datasets/p2_seed.json"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    settings = Settings(
        workspace_root=Path.cwd().parent.parent,
        manifest_path=Path("data/manifest.json"),
        database_url="mysql+aiomysql://rag:rag@127.0.0.1:3306/rag",
        qdrant_url="http://127.0.0.1:6333",
        qdrant_mode="server",
        runtime_settings_path=Path("data/runtime_settings.json"),
    )
    runtime = load(settings.runtime_settings_path)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    items = dataset["items"]

    db = get_database(settings.database_url)
    store = store_for_settings(settings)

    async with db.session_factory() as session:
        generation = await _load_generation(session)

        print(f"评估集: {dataset['name']} ({len(items)} 条)  k={args.k}")
        print(f"active generation: {generation.generation_id[:8]}… "
              f"embedding={generation.embedding_model} chunks={generation.chunk_count}")
        print()

        summary = {}
        for mode in MODES:
            recall_hits = 0
            recall_total = 0
            mrr_sum = 0.0
            mrr_total = 0
            out_of_corpus_refused = 0
            out_of_corpus_total = 0
            for item in items:
                q = item["question"]
                exp = item["expected_doc_id"]
                hits = await _retrieve(settings, runtime, store, generation, q, mode, args.k)
                if exp is None:
                    out_of_corpus_total += 1
                    # out_of_corpus：top-1 相似度低于阈值即视为正确拒答
                    if not hits or (hits[0].get("score", 0.0) < settings.retrieval_min_score):
                        out_of_corpus_refused += 1
                    continue
                recall_total += 1
                if recall_at_k(hits, exp, args.k):
                    recall_hits += 1
                mrr_total += 1
                mrr_sum += mrr_at_k(hits, exp, args.k)
            summary[mode] = {
                "recall@k": recall_hits / recall_total if recall_total else 0.0,
                "mrr": mrr_sum / mrr_total if mrr_total else 0.0,
                "recall_hits": recall_hits,
                "recall_total": recall_total,
                "out_of_corpus_refused": out_of_corpus_refused,
                "out_of_corpus_total": out_of_corpus_total,
            }

        print(f"{'模式':<16} {'Recall@{k}':<10} {'MRR':<8} {'拒答正确率':<10}")
        print("-" * 52)
        for mode in MODES:
            s = summary[mode]
            refused_rate = (
                s["out_of_corpus_refused"] / s["out_of_corpus_total"]
                if s["out_of_corpus_total"]
                else 0.0
            )
            print(
                f"{mode:<16} {s['recall@k']*100:>6.1f}%  {s['mrr']:>6.3f}  "
                f"{refused_rate*100:>6.1f}%  ({s['recall_hits']}/{s['recall_total']})"
            )
    await db.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
