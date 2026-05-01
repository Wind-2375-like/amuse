"""Abstract evaluator interface.

A `SentenceEvaluator` takes a (document, sentence) pair and returns a
`SentenceScore`. Implementations are responsible for any backend-specific
details (LLM call, retries, parsing). Caching is layered on top of an
evaluator via `CachedEvaluator` (see `evaluators.cache`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field


class SentenceScore(BaseModel):
    faithful: int = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""
    raw_response: str = ""
    # Set when the model output failed to parse and we fell back to a default.
    parse_failed: bool = False


class SentenceEvaluator(ABC):
    name: str = "base"

    @abstractmethod
    def score_sentence(self, document: str, sentence: str) -> SentenceScore:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def prompt_version(self) -> str: ...
