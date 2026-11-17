import numpy as np

from docmind.retrieval.bm25 import BM25Index, tokenize
from docmind.retrieval.embeddings import HashingEmbedder
from docmind.retrieval.hybrid import reciprocal_rank_fusion
from docmind.retrieval.vector_store import VectorStore


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dim=128)
    a = e.embed(["robot arm payload"])
    b = e.embed(["robot arm payload"])
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5
    assert e.dim == 128


def test_hashing_embedder_similarity_is_meaningful():
    e = HashingEmbedder()
    q = e.embed_query("maximum payload of the robot")
    docs = e.embed(["The robot has a maximum payload of 12 kg.", "Annual leave is 24 days per year."])
    assert docs[0] @ q > docs[1] @ q


def test_vector_store_search_and_remove(tmp_path):
    e = HashingEmbedder(dim=64)
    store = VectorStore(64)
    texts = ["alpha beta", "gamma delta", "alpha gamma"]
    store.add(["a", "b", "c"], e.embed(texts))
    top = store.search(e.embed_query("alpha beta"), k=2)
    assert top[0][0] == "a"
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, 64)
    assert len(loaded) == 3
    loaded.remove({"a"})
    assert [cid for cid, _ in loaded.search(e.embed_query("alpha beta"), k=5)] == ["c", "b"]


def test_bm25_exact_term_match():
    idx = BM25Index()
    idx.add(
        ["x", "y", "z", "w"],
        [
            "The controller is the NXC-4 model.",
            "Leave accrues at two days per month.",
            "Passwords must be at least 14 characters.",
            "Deployments go through Argo CD.",
        ],
    )
    results = idx.search("NXC-4 controller", k=2)
    assert results and results[0][0] == "x"
    assert "nxc-4" in tokenize("NXC-4 controller")


def test_rrf_prefers_items_ranked_by_both_lists():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("c", 10.0), ("a", 5.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    order = [cid for cid, _, _ in fused]
    assert order[0] == "a"  # rank 1 + rank 2 beats rank 3 + rank 1
    assert set(order) == {"a", "b", "c"}
    ranks = {cid: r for cid, _, r in fused}
    assert ranks["b"] == [2, None]
