"""Answer generation backends.

``ClaudeLLM`` calls the Claude API. ``ExtractiveLLM`` is an offline fallback that selects
the most relevant sentences from the retrieved passages and cites them; it keeps the whole
pipeline runnable (and testable in CI) without an API key.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from docmind.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from docmind.retrieval.bm25 import tokenize

log = logging.getLogger(__name__)

NOT_FOUND = "I could not find this in the provided documents."


class LLM(Protocol):
    name: str

    def generate(self, question: str, contexts: list[str]) -> str: ...


class ExtractiveLLM:
    """Keyword-overlap sentence selection. Cheap, deterministic, always cites."""

    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_overlap: int = 1):
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, question: str, contexts: list[str]) -> str:
        q_terms = set(tokenize(question))
        scored: list[tuple[int, int, str]] = []
        for idx, ctx in enumerate(contexts, start=1):
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", ctx):
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                overlap = len(q_terms & set(tokenize(sentence)))
                if overlap >= self.min_overlap:
                    scored.append((overlap, -idx, f"{sentence} [{idx}]"))
        if not scored:
            return NOT_FOUND
        scored.sort(reverse=True)
        # Keep only sentences that are at least half as relevant as the best one, so a single
        # shared term (e.g. the product name) does not drag in unrelated sentences.
        floor = max(self.min_overlap, (scored[0][0] + 1) // 2)
        seen: set[str] = set()
        picked: list[str] = []
        for overlap, _neg_idx, line in scored:
            if overlap < floor:
                break
            if line not in seen:
                picked.append(line)
                seen.add(line)
            if len(picked) == self.max_sentences:
                break
        return " ".join(picked)


class ClaudeLLM:
    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 4096):
        import anthropic  # lazy: only required when this backend is selected

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, question: str, contexts: list[str]) -> str:
        # Server-side refusal fallback ("default" mode) re-runs a declined request on a
        # fallback model inside the same call, so a safety-classifier refusal does not
        # surface as an empty answer to the user.
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": build_user_prompt(question, contexts)}],
        )
        if response.stop_reason == "refusal":
            log.warning("Claude refused the request: %s", response.stop_details)
            return NOT_FOUND
        return "".join(block.text for block in response.content if block.type == "text").strip()


def build_llm(kind: str, model: str = "claude-opus-5-5", max_tokens: int = 4096) -> LLM:
    if kind == "extractive":
        return ExtractiveLLM()
    if kind == "claude":
        return ClaudeLLM(model=model, max_tokens=max_tokens)
    raise ValueError(f"Unknown LLM backend {kind!r} (expected 'claude' or 'extractive')")
