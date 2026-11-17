"""FastAPI service exposing the RAG pipeline.

Run locally:  uvicorn docmind.api.app:app --reload
OpenAPI docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from docmind import __version__
from docmind.api.schemas import (
    CitationOut,
    DocumentOut,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RetrievedOut,
)
from docmind.config import Settings, get_settings
from docmind.ingest.loaders import load_bytes, load_text
from docmind.pipeline import RAGPipeline

log = logging.getLogger("docmind.api")
UI_PATH = Path(__file__).resolve().parents[2] / "ui" / "index.html"


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or RAGPipeline(settings)
        if app.state.pipeline.load():
            log.info("loaded persisted index from %s", settings.index_dir)
        yield
        app.state.pipeline.save()

    app = FastAPI(title="DocMind RAG API", version=__version__, lifespan=lifespan)

    def rag() -> RAGPipeline:
        return app.state.pipeline

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        if UI_PATH.exists():
            return UI_PATH.read_text(encoding="utf-8")
        return "<h1>DocMind</h1><p>See <a href='/docs'>/docs</a>.</p>"

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        p = rag()
        return HealthResponse(
            status="ok", documents=len(p.documents), chunks=len(p.chunks), embedder=p.embedder.name, llm_backend=p.llm.name
        )

    @app.post("/ingest/text", response_model=IngestResponse, status_code=201)
    def ingest_text(req: IngestTextRequest) -> IngestResponse:
        chunks = rag().ingest_document(load_text(req.text, title=req.title))
        rag().save()
        return IngestResponse(documents_indexed=1 if chunks else 0, chunks_indexed=chunks)

    @app.post("/ingest/files", response_model=IngestResponse, status_code=201)
    async def ingest_files(files: list[UploadFile] = File(...)) -> IngestResponse:  # noqa: B008 (FastAPI idiom)
        docs = chunks = 0
        for f in files:
            try:
                doc = load_bytes(await f.read(), f.filename or "upload.txt")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            n = rag().ingest_document(doc)
            docs += 1 if n else 0
            chunks += n
        rag().save()
        return IngestResponse(documents_indexed=docs, chunks_indexed=chunks)

    @app.get("/documents", response_model=list[DocumentOut])
    def list_documents() -> list[DocumentOut]:
        p = rag()
        counts: dict[str, int] = {}
        for c in p.chunks.values():
            counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
        return [DocumentOut(id=d.id, title=d.title, source=d.source, chunks=counts.get(d.id, 0)) for d in p.documents.values()]

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str) -> None:
        if not rag().delete_document(doc_id):
            raise HTTPException(status_code=404, detail="document not found")
        rag().save()

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        ans = rag().query(req.question, req.top_k)
        return QueryResponse(
            question=ans.question,
            answer=ans.answer,
            llm_backend=ans.llm_backend,
            citations=[CitationOut(**c.__dict__) for c in ans.citations],
            retrieved=[
                RetrievedOut(
                    chunk_id=r.chunk.id,
                    doc_title=r.chunk.doc_title,
                    score=round(r.score, 5),
                    dense_rank=r.dense_rank,
                    sparse_rank=r.sparse_rank,
                    text=r.chunk.text,
                )
                for r in ans.retrieved
            ],
        )

    return app


app = create_app()
