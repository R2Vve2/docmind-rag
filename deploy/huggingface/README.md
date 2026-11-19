---
title: DocMind RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Hybrid-search RAG with cited answers (BM25 + embeddings)
---

# DocMind: Retrieval-Augmented Generation demo

Ask questions about four fictional company documents and get answers that cite the passages they came from.

Retrieval combines BM25 keyword search with `BAAI/bge-small-en-v1.5` embeddings, merged by Reciprocal Rank Fusion.
This public demo runs in read-only mode with the offline extractive answer backend and a per-visitor rate limit.

Source code, tests, evaluation harness and CI: https://github.com/R2Vve2/docmind-rag
