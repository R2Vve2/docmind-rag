# ---- build stage: install dependencies into a virtualenv -------------------
FROM python:3.11-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt
# Bake the embedding model into the image so containers start without a download.
ENV FASTEMBED_CACHE_PATH=/opt/models
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# ---- runtime stage: slim image, non-root user -----------------------------
FROM python:3.11-slim
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DOCMIND_INDEX_DIR=/data/index FASTEMBED_CACHE_PATH=/opt/models PORT=8000
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models
COPY docmind ./docmind
COPY ui ./ui
COPY data/sample_docs ./data/sample_docs
# uid 1000 matches the user Hugging Face Spaces runs containers as.
RUN useradd --create-home --uid 1000 appuser && mkdir -p /data && chown -R appuser /data /app /opt/models
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health').status==200 else 1)"
CMD ["sh", "-c", "uvicorn docmind.api.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
