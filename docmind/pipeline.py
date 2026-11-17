"""The RAG pipeline: ingest -> chunk -> index (dense + sparse) -> retrieve -> fuse -> generate."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from docmind.config import Settings
from docmind.generation.llm import LLM, build_llm
from docmind.ingest.chunking import chunk_document
from docmind.ingest.loaders import load_file
from docmind.models import Answer, Chunk, Citation, Document, RetrievedChunk
from docmind.retrieval.bm25 import BM25Index
from docmind.retrieval.embeddings import Embedder, build_embedder
from docmind.retrieval.hybrid import reciprocal_rank_fusion
from docmind.retrieval.vector_store import VectorStore

log = logging.getLogger(__name__)
_CITATION = re.compile(r"\[(\d+)\]")


class RAGPipeline:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, llm: LLM | None = None):
        self.settings = settings
        self.embedder = embedder or build_embedder(settings.embedder, settings.embedding_model)
        self.llm = llm or build_llm(settings.resolved_llm(), settings.claude_model, settings.max_answer_tokens)
        self.vectors = VectorStore(self.embedder.dim)
        self.bm25 = BM25Index()
        self.chunks: dict[str, Chunk] = {}
        self.documents: dict[str, Document] = {}

    # ------------------------------------------------------------------ ingestion
    def ingest_document(self, doc: Document) -> int:
        if doc.id in self.documents:
            log.info("document %s already indexed; skipping", doc.id)
            return 0
        chunks = chunk_document(doc, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            return 0
        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        self.vectors.add(ids, self.embedder.embed(texts))
        self.bm25.add(ids, texts)
        self.chunks.update({c.id: c for c in chunks})
        self.documents[doc.id] = doc
        log.info("indexed %s (%d chunks)", doc.title, len(chunks))
        return len(chunks)

    def ingest_path(self, path: str | Path) -> int:
        path = Path(path)
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        total = 0
        for f in files:
            try:
                total += self.ingest_document(load_file(f))
            except ValueError as exc:
                log.warning("skipping %s: %s", f, exc)
        return total

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        chunk_ids = {cid for cid, c in self.chunks.items() if c.doc_id == doc_id}
        self.vectors.remove(chunk_ids)
        self.bm25.remove(chunk_ids)
        for cid in chunk_ids:
            del self.chunks[cid]
        del self.documents[doc_id]
        return True

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.settings.top_k
        if not self.chunks:
            return []
        candidates = max(k * 4, 20)  # over-fetch from each retriever before fusing
        dense = self.vectors.search(self.embedder.embed_query(question), candidates)
        sparse = self.bm25.search(question, candidates)
        fused = reciprocal_rank_fusion([dense, sparse], k=self.settings.rrf_k)
        return [
            RetrievedChunk(chunk=self.chunks[cid], score=score, dense_rank=ranks[0], sparse_rank=ranks[1])
            for cid, score, ranks in fused[:k]
        ]

    # ------------------------------------------------------------------ generation
    def query(self, question: str, top_k: int | None = None) -> Answer:
        retrieved = self.retrieve(question, top_k)
        if not retrieved:
            return Answer(question, "No documents have been indexed yet.", [], [], self.llm.name)
        text = self.llm.generate(question, [r.chunk.text for r in retrieved])
        citations: list[Citation] = []
        seen: set[int] = set()
        for marker in (int(m) for m in _CITATION.findall(text)):
            if marker in seen or not 1 <= marker <= len(retrieved):
                continue
            seen.add(marker)
            c = retrieved[marker - 1].chunk
            citations.append(Citation(marker, c.id, c.doc_id, c.doc_title, c.text[:200]))
        return Answer(question, text, citations, retrieved, self.llm.name)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path | None = None) -> None:
        directory = Path(directory or self.settings.index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.vectors.save(directory)
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks.values()]), encoding="utf-8"
        )
        docs = [{"id": d.id, "title": d.title, "source": d.source, "text": d.text} for d in self.documents.values()]
        (directory / "documents.json").write_text(json.dumps(docs), encoding="utf-8")
        meta = {"embedder": self.embedder.name, "dim": self.embedder.dim}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, directory: str | Path | None = None) -> bool:
        directory = Path(directory or self.settings.index_dir)
        if not (directory / "chunks.json").exists():
            return False
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta.get("embedder") != self.embedder.name or meta.get("dim") != self.embedder.dim:
            raise RuntimeError(
                f"Index built with embedder {meta.get('embedder')} (dim {meta.get('dim')}) but the "
                f"running embedder is {self.embedder.name} (dim {self.embedder.dim}). Re-index."
            )
        self.vectors = VectorStore.load(directory, self.embedder.dim)
        chunk_list = [Chunk.from_dict(c) for c in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        self.chunks = {c.id: c for c in chunk_list}
        self.bm25 = BM25Index()
        self.bm25.add([c.id for c in chunk_list], [c.text for c in chunk_list])
        self.documents = {
            d["id"]: Document(**d) for d in json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        }
        return True
