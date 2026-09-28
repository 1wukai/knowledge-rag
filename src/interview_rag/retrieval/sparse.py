"""中文 BM25 稀疏编码：jieba 分词 + 文档/查询权重编码。

文档侧编码 BM25 的 TF 与长度归一化项，查询侧编码同一语料快照的 IDF，
通过 Qdrant 稀疏点积检索。Qdrant sparse vector 使用整数索引，故由 Vocabulary
维护词项到稳定整数 ID 的映射，随 generation 一并保存与恢复。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jieba


@dataclass
class Vocabulary:
    """词项到整数 ID 的映射；ID 按词项首次出现顺序分配，保证同代稳定。"""

    term_to_id: dict[str, int] = field(default_factory=dict)
    id_to_term: dict[int, str] = field(default_factory=dict)

    def get_or_add(self, term: str) -> int:
        idx = self.term_to_id.get(term)
        if idx is None:
            idx = len(self.term_to_id)
            self.term_to_id[term] = idx
            self.id_to_term[idx] = term
        return idx

    def to_dict(self) -> dict:
        return {"term_to_id": self.term_to_id}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        vocab = cls(term_to_id=dict(data.get("term_to_id", {})))
        vocab.id_to_term = {i: t for t, i in vocab.term_to_id.items()}
        return vocab


@dataclass
class SparseStats:
    """一个 generation 的稀疏检索统计量。"""

    df: dict[str, int] = field(default_factory=dict)  # term -> document frequency
    avg_len: float = 0.0
    doc_count: int = 0

    def to_dict(self) -> dict:
        return {"df": self.df, "avg_len": self.avg_len, "doc_count": self.doc_count}

    @classmethod
    def from_dict(cls, data: dict) -> "SparseStats":
        return cls(
            df=data.get("df", {}),
            avg_len=data.get("avg_len", 0.0),
            doc_count=data.get("doc_count", 0),
        )


def tokenize(text: str) -> list[str]:
    """分词并保留英文术语规则（小写、去空白）。"""
    return [t.strip().lower() for t in jieba.cut(text) if t.strip()]


def compute_stats(documents: list[list[str]]) -> SparseStats:
    """由文档词项列表计算 DF 与平均长度。"""
    df: dict[str, int] = {}
    total_len = 0
    for doc in documents:
        seen = set(doc)
        for term in seen:
            df[term] = df.get(term, 0) + 1
        total_len += len(doc)
    stats = SparseStats(df=df, doc_count=len(documents))
    stats.avg_len = total_len / len(documents) if documents else 0.0
    return stats


def encode_document(
    tokens: list[str], stats: SparseStats, vocab: Vocabulary, k1: float = 1.5, b: float = 0.75
) -> tuple[list[int], list[float]]:
    """文档稀疏向量：BM25 的 TF × 长度归一化项（不含 IDF，IDF 在查询侧编码）。"""
    if not tokens:
        return [], []
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    length_norm = 1 - b + b * (len(tokens) / stats.avg_len if stats.avg_len else 1.0)
    indices: list[int] = []
    values: list[float] = []
    for term, freq in tf.items():
        indices.append(vocab.get_or_add(term))
        values.append((freq * (k1 + 1)) / (freq + k1 * length_norm))
    return indices, values


def encode_query(
    tokens: list[str], stats: SparseStats, vocab: Vocabulary
) -> tuple[list[int], list[float]]:
    """查询稀疏向量：同一快照的 IDF。"""
    if not tokens:
        return [], []
    weights: dict[int, float] = {}
    for term in tokens:
        idx = vocab.term_to_id.get(term)
        if idx is None:
            continue
        n = stats.df.get(term, 0)
        idf = math.log((stats.doc_count - n + 0.5) / (n + 0.5) + 1.0)
        weights[idx] = weights.get(idx, 0.0) + idf
    return list(weights.keys()), list(weights.values())
