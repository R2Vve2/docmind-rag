"""Retrieval and answer-quality metrics for the golden evaluation set.

Retrieval:
  * hit@k        - fraction of questions whose expected document appears in the top-k chunks
  * MRR          - mean reciprocal rank of the first chunk from the expected document
Answer:
  * keyword recall   - fraction of expected keywords that appear in the generated answer
  * citation precision - fraction of citations that point at the expected document
"""

from __future__ import annotations

from dataclasses import dataclass

from docmind.models import Answer


@dataclass
class ExampleResult:
    question: str
    expected_doc_title: str
    hit: bool
    reciprocal_rank: float
    keyword_recall: float
    citation_precision: float | None  # None when the answer has no citations
    answer: str


def evaluate_example(answer: Answer, expected_doc_title: str, expected_keywords: list[str]) -> ExampleResult:
    titles = [r.chunk.doc_title.lower() for r in answer.retrieved]
    expected = expected_doc_title.lower()
    rr = 0.0
    for rank, title in enumerate(titles, start=1):
        if title == expected:
            rr = 1.0 / rank
            break
    text = answer.answer.lower()
    kw_recall = sum(k.lower() in text for k in expected_keywords) / len(expected_keywords) if expected_keywords else 1.0
    if answer.citations:
        cit_prec: float | None = sum(c.doc_title.lower() == expected for c in answer.citations) / len(answer.citations)
    else:
        cit_prec = None
    return ExampleResult(answer.question, expected_doc_title, rr > 0, rr, kw_recall, cit_prec, answer.answer)


def aggregate(results: list[ExampleResult]) -> dict[str, float]:
    n = len(results) or 1
    with_cits = [r.citation_precision for r in results if r.citation_precision is not None]
    return {
        "examples": len(results),
        "hit_rate": sum(r.hit for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "keyword_recall": sum(r.keyword_recall for r in results) / n,
        "citation_precision": (sum(with_cits) / len(with_cits)) if with_cits else 0.0,
        "answers_with_citations": len(with_cits) / n,
    }
