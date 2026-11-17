"""Embedding backends behind a single ``Embedder`` protocol.

* ``FastEmbedEmbedder`` - real neural sentence embeddings (BAAI/bge-small-en-v1.5 via ONNX).
* ``HashingEmbedder`` - dependency-free character n-gram hashing. Deterministic and fast,
  used for unit tests and CI so the suite never downloads a model.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 512, ngram: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.ngram = ngram

    def _features(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text.lower()).strip()
        feats = text.split(" ")
        padded = f" {text} "
        for n in range(self.ngram[0], self.ngram[1] + 1):
            feats.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = int(hashlib.blake2b(f.encode("utf-8"), digest_size=8).hexdigest(), 16)
                sign = 1.0 if (h >> 1) & 1 else -1.0
                out[row, h % self.dim] += sign
        return _normalize(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class FastEmbedEmbedder:
    name = "fastembed"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  # lazy: heavy import, optional dependency

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.dim = int(self.embed(["dimension probe"]).shape[1])

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return _normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        # bge models were trained with this instruction prefix for retrieval queries.
        return self.embed([f"Represent this sentence for searching relevant passages: {text}"])[0]


def build_embedder(kind: str, model_name: str = "BAAI/bge-small-en-v1.5") -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder(model_name)
    raise ValueError(f"Unknown embedder {kind!r} (expected 'fastembed' or 'hashing')")
