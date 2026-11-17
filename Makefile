.PHONY: install test lint eval serve docker-build docker-run

install:
	pip install -r requirements-dev.txt

test:
	DOCMIND_EMBEDDER=hashing DOCMIND_LLM=extractive pytest

lint:
	ruff check .

eval:
	python -m docmind.evaluation.run_eval --docs data/sample_docs --golden eval/golden.jsonl --min-hit-rate 0.9

serve:
	uvicorn docmind.api.app:app --reload --port 8000

docker-build:
	docker build -t docmind-rag:latest .

docker-run:
	docker compose up --build
