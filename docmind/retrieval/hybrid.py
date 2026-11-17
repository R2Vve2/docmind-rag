"""Reciprocal Rank Fusion: merge ranked lists from dense and sparse retrievers.

RRF needs no score calibration between retrievers (cosine similarities and BM25 scores
live on different scales), which is why it is the standard choice for hybrid search.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float, list[int | None]]]:
    """Return ``(id, fused_score, [rank_in_list_0, rank_in_list_1, ...])`` sorted best-first."""
    fused: dict[str, float] = {}
    ranks: dict[str, list[int | None]] = {}
    for list_idx, ranked in enumerate(ranked_lists):
        for rank, (cid, _score) in enumerate(ranked, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(cid, [None] * len(ranked_lists))[list_idx] = rank
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [(cid, score, ranks[cid]) for cid, score in ordered]
