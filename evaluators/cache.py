"""Append-only JSONL cache, keyed by (doc_hash, summary_hash, sent_idx,
model_name, prompt_version).

Layout: results/cache/<model_slug>/<dataset>.jsonl

Each line is one JSON object containing both the cache key and the value
(SentenceScore fields + raw_response). On startup we load the file into an
in-memory dict; on cache miss we score and append a single line. The file is
never rewritten, which makes it safe for concurrent appends (POSIX guarantees
atomic writes under PIPE_BUF) and easy to share / inspect / git-diff.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import SentenceEvaluator, SentenceScore


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def text_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CacheKey:
    doc_hash: str
    summary_hash: str
    sent_idx: int
    model_name: str
    prompt_version: str

    def to_tuple(self) -> tuple:
        return (
            self.doc_hash,
            self.summary_hash,
            self.sent_idx,
            self.model_name,
            self.prompt_version,
        )


class JSONLCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[tuple, dict] = {}
        if self.path.exists():
            self._load()
        self._fh = open(self.path, "a", buffering=1)  # line-buffered

    def _load(self) -> None:
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (
                    obj["doc_hash"],
                    obj["summary_hash"],
                    obj["sent_idx"],
                    obj["model_name"],
                    obj["prompt_version"],
                )
                self._mem[key] = obj

    def __contains__(self, key: CacheKey) -> bool:
        return key.to_tuple() in self._mem

    def get(self, key: CacheKey) -> Optional[dict]:
        return self._mem.get(key.to_tuple())

    def put(self, key: CacheKey, score: SentenceScore) -> None:
        obj = {
            "doc_hash": key.doc_hash,
            "summary_hash": key.summary_hash,
            "sent_idx": key.sent_idx,
            "model_name": key.model_name,
            "prompt_version": key.prompt_version,
            "faithful": score.faithful,
            "confidence": score.confidence,
            "reason": score.reason,
            "raw_response": score.raw_response,
            "parse_failed": score.parse_failed,
        }
        self._mem[key.to_tuple()] = obj
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def __len__(self) -> int:
        return len(self._mem)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class CachedEvaluator:
    """Wraps a SentenceEvaluator with a JSONL cache."""

    def __init__(
        self,
        evaluator: SentenceEvaluator,
        cache_dir: Path,
        dataset: str,
    ):
        self.evaluator = evaluator
        model_slug = _slugify(evaluator.model_name)
        cache_path = Path(cache_dir) / model_slug / f"{dataset}.jsonl"
        self.cache = JSONLCache(cache_path)
        self.cache_path = cache_path

    def score(
        self,
        document: str,
        summary: str,
        sent_idx: int,
        sentence: str,
    ) -> tuple[SentenceScore, bool]:
        """Returns (score, was_cache_hit)."""
        key = CacheKey(
            doc_hash=text_hash(document),
            summary_hash=text_hash(summary),
            sent_idx=sent_idx,
            model_name=self.evaluator.model_name,
            prompt_version=self.evaluator.prompt_version,
        )
        hit = self.cache.get(key)
        if hit is not None and not hit.get("parse_failed", False):
            return (
                SentenceScore(
                    faithful=hit["faithful"],
                    confidence=hit["confidence"],
                    reason=hit.get("reason", ""),
                    raw_response=hit.get("raw_response", ""),
                    parse_failed=hit.get("parse_failed", False),
                ),
                True,
            )
        # cache miss OR previously-failed entry — re-score.
        score = self.evaluator.score_sentence(document, sentence)
        self.cache.put(key, score)
        return score, False

    def close(self) -> None:
        self.cache.close()
