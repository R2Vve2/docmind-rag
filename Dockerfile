# ---- build stage: install dependencies into a virtualenv -------------------
FROM python:3.11-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- runtime stage: slim image, non-root user -----------------------------
FROM python:3.11-slim
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DOCMIND_INDEX_DIR=/data/index FASTEMBED_CACHE_PATH=/data/models
COPY --from=builder /opt/venv /opt/venv
COPY docmind ./docmind
COPY ui ./ui
COPY data/sample_docs ./data/sample_docs
RUN useradd --create-home appuser && mkdir -p /data && chown -R appuser /data /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "docmind.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
