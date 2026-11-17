"""A small, persistent cosine-similarity vector index built on NumPy.

For the corpus sizes a single service handles (tens of thousands of chunks) a brute-force
matrix product is faster than an ANN index and has zero recall loss. The class keeps the
same ``add / search / save / load`` shape you would use with FAISS or a vector database,
so swapping backends is a one-file change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[str] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if vectors.shape != (len(ids), self.dim):
            raise ValueError(f"expected shape {(len(ids), self.dim)}, got {vectors.shape}")
        self._ids.extend(ids)
        self._matrix = np.vstack([self._matrix, vectors.astype(np.float32)])

    def remove(self, ids: set[str]) -> None:
        keep = [i for i, cid in enumerate(self._ids) if cid not in ids]
        self._ids = [self._ids[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        scores = self._matrix @ query.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._matrix)
        (directory / "vector_ids.json").write_text(json.dumps(self._ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, dim: int) -> VectorStore:
        directory = Path(directory)
        store = cls(dim)
        if (directory / "vectors.npy").exists():
            store._matrix = np.load(directory / "vectors.npy")
            store._ids = json.loads((directory / "vector_ids.json").read_text(encoding="utf-8"))
        return store
