"""Public-demo hardening: read-only mode, start-up seeding and per-IP rate limiting."""

import pytest
from fastapi.testclient import TestClient

from docmind.api.app import create_app
from docmind.api.ratelimit import RateLimiter
from docmind.config import Settings
from docmind.pipeline import RAGPipeline
from tests.conftest import SAMPLE_DOCS


@pytest.fixture
def demo_client(tmp_path):
    settings = Settings(
        embedder="hashing", llm="extractive", index_dir=str(tmp_path / "index"),
        read_only=True, seed_dir=str(SAMPLE_DOCS), rate_limit_per_minute=3,
    )
    with TestClient(create_app(settings=settings, pipeline=RAGPipeline(settings))) as c:
        yield c


def test_seeded_at_startup_and_reports_read_only(demo_client):
    h = demo_client.get("/health").json()
    assert h["read_only"] is True and h["documents"] == 4 and h["chunks"] > 0
    sources = [d["source"] for d in demo_client.get("/documents").json()]
    assert all("/" not in s and "\\" not in s for s in sources)  # no server paths leaked


def test_write_endpoints_are_forbidden(demo_client):
    assert demo_client.post("/ingest/text", json={"title": "x", "text": "y"}).status_code == 403
    files = [("files", ("a.md", b"hello there", "text/markdown"))]
    assert demo_client.post("/ingest/files", files=files).status_code == 403
    doc_id = demo_client.get("/documents").json()[0]["id"]
    assert demo_client.delete(f"/documents/{doc_id}").status_code == 403
    assert demo_client.get("/health").json()["documents"] == 4


def test_query_rate_limit_per_ip(demo_client):
    q = {"question": "How many days of annual leave do employees get?"}
    a = {"X-Forwarded-For": "203.0.113.7"}
    assert [demo_client.post("/query", json=q, headers=a).status_code for _ in range(3)] == [200, 200, 200]
    blocked = demo_client.post("/query", json=q, headers=a)
    assert blocked.status_code == 429 and blocked.headers["retry-after"] == "60"
    assert demo_client.post("/query", json=q, headers={"X-Forwarded-For": "198.51.100.2"}).status_code == 200


def test_rate_limiter_window_slides():
    rl = RateLimiter(limit=2, window_seconds=60)
    assert rl.allow("ip", now=0) and rl.allow("ip", now=1)
    assert not rl.allow("ip", now=2)
    assert rl.allow("ip", now=61)  # the first hit has expired
    assert RateLimiter(limit=0).allow("anyone")


def test_seed_skipped_when_index_already_populated(tmp_path):
    settings = Settings(embedder="hashing", llm="extractive", index_dir=str(tmp_path / "index"), seed_dir=str(SAMPLE_DOCS))
    p = RAGPipeline(settings)
    p.ingest_path(SAMPLE_DOCS / "employee_handbook.md")
    with TestClient(create_app(settings=settings, pipeline=p)) as c:
        assert c.get("/health").json()["documents"] == 1
