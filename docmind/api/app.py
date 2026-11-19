"""FastAPI service exposing the RAG pipeline.

Run locally:  uvicorn docmind.api.app:app --reload
OpenAPI docs: http://localhost:8000/docs

Public demo mode (see README "Deployment"): ``DOCMIND_READ_ONLY=true`` disables the ingest and
delete endpoints, ``DOCMIND_SEED_DIR`` indexes a folder at start-up when the index is empty, and
``DOCMIND_RATE_LIMIT_PER_MINUTE`` caps queries per client IP.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from docmind import __version__
from docmind.api.ratelimit import RateLimiter
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


def client_ip(request: Request) -> str:
    # Behind a hosting proxy the socket peer is the proxy, so prefer the first forwarded hop.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or RAGPipeline(settings)
        if app.state.pipeline.load():
            log.info("loaded persisted index from %s", settings.index_dir)
        if settings.seed_dir and not app.state.pipeline.documents:
            n = app.state.pipeline.ingest_path(settings.seed_dir)
            log.info("seeded index with %d chunks from %s", n, settings.seed_dir)
        yield
        if not settings.read_only:
            app.state.pipeline.save()

    app = FastAPI(title="DocMind RAG API", version=__version__, lifespan=lifespan)

    def rag() -> RAGPipeline:
        return app.state.pipeline

    def require_writable() -> None:
        if settings.read_only:
            raise HTTPException(status_code=403, detail="This is a read-only public demo; ingest and delete are disabled.")

    def rate_limited(request: Request) -> None:
        if not limiter.allow(client_ip(request)):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit: at most {settings.rate_limit_per_minute} questions per minute.",
                headers={"Retry-After": "60"},
            )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        if UI_PATH.exists():
            return UI_PATH.read_text(encoding="utf-8")
        return "<h1>DocMind</h1><p>See <a href='/docs'>/docs</a>.</p>"

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        p = rag()
        return HealthResponse(
            status="ok", documents=len(p.documents), chunks=len(p.chunks), embedder=p.embedder.name,
            llm_backend=p.llm.name, read_only=settings.read_only,
        )

    @app.post("/ingest/text", response_model=IngestResponse, status_code=201, dependencies=[Depends(require_writable)])
    def ingest_text(req: IngestTextRequest) -> IngestResponse:
        chunks = rag().ingest_document(load_text(req.text, title=req.title))
        rag().save()
        return IngestResponse(documents_indexed=1 if chunks else 0, chunks_indexed=chunks)

    @app.post("/ingest/files", response_model=IngestResponse, status_code=201, dependencies=[Depends(require_writable)])
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
        return [DocumentOut(id=d.id, title=d.title, source=Path(d.source).name, chunks=counts.get(d.id, 0)) for d in p.documents.values()]

    @app.delete("/documents/{doc_id}", status_code=204, dependencies=[Depends(require_writable)])
    def delete_document(doc_id: str) -> None:
        if not rag().delete_document(doc_id):
            raise HTTPException(status_code=404, detail="document not found")
        rag().save()

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limited)])
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
