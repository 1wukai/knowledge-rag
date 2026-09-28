"""证据预算：在 token 预算内选择进入上下文的候选证据。

用字符数近似 token（中文约 1 字符 ≈ 1 token，英文按 4 字符 ≈ 1 token 折半估算），
累积拼接候选 chunk 直到接近上限，超出丢弃。预算只约束证据，不计入系统提示与问题。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("interview_rag.retrieval.budget")


def estimate_tokens(text: str) -> int:
    """字符数近似 token：中文 1 字符约 1 token，英文约 4 字符 1 token。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def select_within_budget(
    hits: list[dict],
    max_tokens: int,
    *,
    min_hits: int = 1,
) -> list[dict]:
    """按顺序累加 hits 的 text，直到超出 max_tokens；至少保留 min_hits 条。

    返回选中的 hits 与消耗的 token 数。
    """
    selected: list[dict] = []
    used = 0
    for hit in hits:
        cost = estimate_tokens(hit.get("text", ""))
        if selected and used + cost > max_tokens and len(selected) >= min_hits:
            break
        selected.append(hit)
        used += cost
    return selected, used
