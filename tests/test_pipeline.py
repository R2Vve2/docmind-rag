from docmind.generation.llm import NOT_FOUND, ExtractiveLLM
from docmind.ingest.loaders import load_text
from docmind.pipeline import RAGPipeline


def test_ingest_indexes_all_sample_docs(pipeline):
    assert len(pipeline.documents) == 4
    assert len(pipeline.chunks) == len(pipeline.vectors) == len(pipeline.bm25)


def test_retrieval_finds_right_document(pipeline):
    hits = pipeline.retrieve("What is the maximum payload of the NX-200?", top_k=3)
    assert hits[0].chunk.doc_title == "Nx200 Product Manual"
    assert all(h.score > 0 for h in hits)


def test_query_returns_cited_answer(pipeline):
    ans = pipeline.query("How many days of annual leave do full-time employees get?")
    assert "24" in ans.answer
    assert ans.citations, "extractive answers must carry citations"
    assert ans.citations[0].doc_title == "Employee Handbook"
    assert ans.llm_backend == "extractive"


def test_citation_markers_map_to_retrieved_chunks(pipeline):
    ans = pipeline.query("Where must API keys be stored?", top_k=4)
    for c in ans.citations:
        assert 1 <= c.marker <= len(ans.retrieved)
        assert ans.retrieved[c.marker - 1].chunk.id == c.chunk_id


def test_extractive_llm_returns_not_found_when_no_overlap():
    assert ExtractiveLLM().generate("zzqx", ["completely unrelated passage text here."]) == NOT_FOUND


def test_duplicate_ingest_is_skipped(pipeline):
    doc = load_text("Some unique content about volcanoes.", title="Volcanoes")
    assert pipeline.ingest_document(doc) == 1
    assert pipeline.ingest_document(doc) == 0


def test_save_and_load_round_trip(pipeline, settings):
    pipeline.save()
    fresh = RAGPipeline(settings)
    assert fresh.load() is True
    assert set(fresh.chunks) == set(pipeline.chunks)
    assert fresh.retrieve("joint lubrication", top_k=1)[0].chunk.doc_title == "Nx200 Product Manual"


def test_delete_document_removes_chunks(pipeline):
    doc_id = next(d.id for d in pipeline.documents.values() if d.title == "Employee Handbook")
    before = len(pipeline.chunks)
    assert pipeline.delete_document(doc_id)
    assert len(pipeline.chunks) < before
    assert all(c.doc_id != doc_id for c in pipeline.chunks.values())
    assert not pipeline.delete_document(doc_id)


def test_empty_index_query(settings):
    ans = RAGPipeline(settings).query("anything?")
    assert "No documents" in ans.answer
