"""Core data types shared across ingestion, retrieval and generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    title: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    text: str
    index: int  # position of this chunk inside its document

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # fused score (higher is better)
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass
class Citation:
    marker: int  # the [n] number used in the answer
    chunk_id: str
    doc_id: str
    doc_title: str
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    llm_backend: str
