"""Runtime configuration, loaded from environment variables or a .env file.

Every option is prefixed with ``DOCMIND_`` (e.g. ``DOCMIND_EMBEDDER=hashing``).
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCMIND_", env_file=".env", extra="ignore")

    # Retrieval
    embedder: str = "fastembed"  # "fastembed" (real neural embeddings) or "hashing" (offline/tests)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 800  # characters
    chunk_overlap: int = 120
    top_k: int = 5
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    index_dir: str = "data/index"

    # Generation
    llm: str = "auto"  # "claude", "extractive", or "auto" (claude if an API key is present)
    claude_model: str = "claude-opus-5-5"
    max_answer_tokens: int = 4096

    # Public demo hardening
    read_only: bool = False  # disable ingest/delete endpoints (for public deployments)
    seed_dir: str | None = None  # folder indexed at start-up when the index is empty
    rate_limit_per_minute: int = 0  # max /query calls per client IP per minute; 0 disables

    def resolved_llm(self) -> str:
        if self.llm != "auto":
            return self.llm
        return "claude" if os.environ.get("ANTHROPIC_API_KEY") else "extractive"


@lru_cache
def get_settings() -> Settings:
    return Settings()
