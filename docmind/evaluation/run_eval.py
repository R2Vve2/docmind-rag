"""Run the golden evaluation set end-to-end and print a scorecard.

    python -m docmind.evaluation.run_eval --docs data/sample_docs --golden eval/golden.jsonl

Exit code is non-zero when hit@k falls below ``--min-hit-rate`` so it can gate CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from docmind.config import Settings
from docmind.evaluation.metrics import aggregate, evaluate_example
from docmind.pipeline import RAGPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--golden", default="eval/golden.jsonl")
    parser.add_argument("--out", default="eval/results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    args = parser.parse_args(argv)

    settings = Settings()
    pipeline = RAGPipeline(settings)
    t0 = time.perf_counter()
    n_chunks = pipeline.ingest_path(args.docs)
    index_s = time.perf_counter() - t0

    golden = [json.loads(line) for line in Path(args.golden).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    t0 = time.perf_counter()
    for ex in golden:
        ans = pipeline.query(ex["question"], top_k=args.top_k)
        results.append(evaluate_example(ans, ex["expected_doc"], ex.get("expected_keywords", [])))
    query_s = (time.perf_counter() - t0) / max(len(golden), 1)

    summary = aggregate(results)
    summary.update({"embedder": pipeline.embedder.name, "llm": pipeline.llm.name, "chunks": n_chunks,
                    "index_seconds": round(index_s, 2), "avg_query_seconds": round(query_s, 3)})

    print(f"\nDocMind evaluation  (embedder={summary['embedder']}, llm={summary['llm']}, top_k={args.top_k})")
    print("-" * 72)
    for r in results:
        flag = "OK " if r.hit else "MISS"
        cp = "-" if r.citation_precision is None else f"{r.citation_precision:.2f}"
        print(f"{flag}  rr={r.reciprocal_rank:.2f}  kw={r.keyword_recall:.2f}  cit={cp}  {r.question}")
    print("-" * 72)
    for key in ("hit_rate", "mrr", "keyword_recall", "citation_precision", "answers_with_citations"):
        print(f"{key:>24}: {summary[key]:.3f}")
    print(f"{'index_seconds':>24}: {summary['index_seconds']}")
    print(f"{'avg_query_seconds':>24}: {summary['avg_query_seconds']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "results": [r.__dict__ for r in results]}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    if summary["hit_rate"] < args.min_hit_rate:
        print(f"FAIL: hit_rate {summary['hit_rate']:.3f} < {args.min_hit_rate}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
