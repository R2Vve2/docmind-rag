import pytest

from docmind.ingest.chunking import chunk_document, chunk_text
from docmind.ingest.loaders import load_text


def test_short_text_is_single_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size_and_cover_text():
    text = "\n\n".join(f"Paragraph {i}. " + ("word " * 40).strip() for i in range(10))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 50 + 1 for c in chunks)
    for i in range(10):
        assert any(f"Paragraph {i}." in c for c in chunks)


def test_overlap_repeats_previous_tail():
    text = "A" * 10 + " first block ends here.\n\n" + "second block starts here " + "B" * 10
    chunks = chunk_text(text, size=40, overlap=15)
    assert len(chunks) == 2
    assert chunks[1].startswith("ends here.") or "ends here" in chunks[1]


def test_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("x", size=0)
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)


def test_chunk_document_ids_are_stable_and_ordered():
    doc = load_text("para one.\n\n" * 50, title="T")
    chunks = chunk_document(doc, size=100, overlap=10)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.id == f"{doc.id}-{c.index}" for c in chunks)
    assert chunk_document(doc, size=100, overlap=10)[0].id == chunks[0].id
