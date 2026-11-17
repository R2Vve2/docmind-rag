"""Command-line interface.

    python -m docmind ingest data/sample_docs
    python -m docmind ask "How many days of annual leave do employees get?"
    python -m docmind serve --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys

from docmind.config import get_settings
from docmind.pipeline import RAGPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docmind", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="index a file or a folder of .txt/.md/.pdf files")
    p_ingest.add_argument("path")
    p_ask = sub.add_parser("ask", help="ask a question against the persisted index")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_serve = sub.add_parser("serve", help="start the HTTP API")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    if args.command == "serve":
        import uvicorn

        uvicorn.run("docmind.api.app:app", host=args.host, port=args.port)
        return 0

    pipeline = RAGPipeline(settings)
    pipeline.load()
    if args.command == "ingest":
        n = pipeline.ingest_path(args.path)
        pipeline.save()
        print(f"indexed {n} chunks from {len(pipeline.documents)} documents -> {settings.index_dir}")
        return 0

    ans = pipeline.query(args.question, args.top_k)
    print(f"\n{ans.answer}\n")
    if ans.citations:
        print("Sources:")
        for c in ans.citations:
            print(f"  [{c.marker}] {c.doc_title}: {c.snippet[:100]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
