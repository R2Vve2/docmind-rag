"""Sparse lexical retrieval with BM25 (exact keyword matching: part numbers, names, codes)."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "be",
    "it", "this", "that", "with", "as", "by", "at", "from", "do", "does", "what", "how",
    "when", "which", "who", "i", "we", "you", "can", "my", "our",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._docs: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], texts: list[str]) -> None:
        self._ids.extend(ids)
        self._docs.extend(tokenize(t) for t in texts)
        self._rebuild()

    def remove(self, ids: set[str]) -> None:
        keep = [i for i, cid in enumerate(self._ids) if cid not in ids]
        self._ids = [self._ids[i] for i in keep]
        self._docs = [self._docs[i] for i in keep]
        self._rebuild()

    def _rebuild(self) -> None:
        self._bm25 = BM25Okapi(self._docs) if self._docs else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self._ids[i], float(scores[i])) for i in ranked if scores[i] > 0]
