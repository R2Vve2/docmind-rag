import os
from pathlib import Path

import pytest

# Force the offline backends before any docmind module reads settings.
os.environ.setdefault("DOCMIND_EMBEDDER", "hashing")
os.environ.setdefault("DOCMIND_LLM", "extractive")

from docmind.config import Settings  # noqa: E402
from docmind.pipeline import RAGPipeline  # noqa: E402

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(embedder="hashing", llm="extractive", index_dir=str(tmp_path / "index"))


@pytest.fixture
def pipeline(settings) -> RAGPipeline:
    p = RAGPipeline(settings)
    p.ingest_path(SAMPLE_DOCS)
    return p
