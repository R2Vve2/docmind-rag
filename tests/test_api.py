import pytest
from fastapi.testclient import TestClient

from docmind.api.app import create_app
from docmind.pipeline import RAGPipeline


@pytest.fixture
def client(settings):
    app = create_app(settings=settings, pipeline=RAGPipeline(settings))
    with TestClient(app) as c:
        yield c


def test_health_empty(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["documents"] == 0
    assert body["embedder"] == "hashing" and body["llm_backend"] == "extractive"


def test_ingest_text_then_query(client):
    r = client.post("/ingest/text", json={"title": "Policy", "text": "Refunds are processed within 14 days of the request."})
    assert r.status_code == 201
    assert r.json()["chunks_indexed"] == 1
    r = client.post("/query", json={"question": "How long do refunds take?"})
    assert r.status_code == 200
    body = r.json()
    assert "14 days" in body["answer"]
    assert body["citations"][0]["doc_title"] == "Policy"
    assert body["retrieved"][0]["dense_rank"] == 1


def test_ingest_files_and_list_delete(client):
    files = [("files", ("notes.md", b"# Notes\n\nThe meeting is on Monday at noon.", "text/markdown"))]
    r = client.post("/ingest/files", files=files)
    assert r.status_code == 201 and r.json()["documents_indexed"] == 1
    docs = client.get("/documents").json()
    assert len(docs) == 1 and docs[0]["chunks"] >= 1
    assert client.delete(f"/documents/{docs[0]['id']}").status_code == 204
    assert client.get("/documents").json() == []
    assert client.delete("/documents/nope").status_code == 404


def test_unsupported_upload_rejected(client):
    r = client.post("/ingest/files", files=[("files", ("image.png", b"\x89PNG", "image/png"))])
    assert r.status_code == 400


def test_validation_errors(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 422
    assert client.post("/query", json={"question": "long enough", "top_k": 99}).status_code == 422


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "DocMind" in r.text
