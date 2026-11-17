"""Request/response models for the HTTP API (validated by pydantic, documented by OpenAPI)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    doc_title: str
    snippet: str


class RetrievedOut(BaseModel):
    chunk_id: str
    doc_title: str
    score: float
    dense_rank: int | None
    sparse_rank: int | None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    llm_backend: str
    citations: list[CitationOut]
    retrieved: list[RetrievedOut]


class DocumentOut(BaseModel):
    id: str
    title: str
    source: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    embedder: str
    llm_backend: str
