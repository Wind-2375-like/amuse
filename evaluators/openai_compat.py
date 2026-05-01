"""OpenAI-compatible evaluator (talks to vLLM, OpenAI, anything compatible)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .base import SentenceEvaluator, SentenceScore


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_response(text: str) -> Optional[dict]:
    """Try strict JSON first, then a regex fallback to grab the first {...}."""
    text = text.strip()
    # Strip code fences if the model wrapped output.
    if text.startswith("```"):
        text = text.strip("`")
        # remove a leading "json\n" if present
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _coerce_score(parsed: dict) -> Optional[tuple[int, float, str]]:
    if not isinstance(parsed, dict):
        return None
    if "faithful" not in parsed:
        return None
    try:
        faithful = int(parsed["faithful"])
        if faithful not in (0, 1):
            faithful = 1 if faithful >= 1 else 0
    except (TypeError, ValueError):
        return None
    try:
        conf = float(parsed.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = 0.5
    reason = str(parsed.get("reason", ""))[:500]
    return faithful, conf, reason


class OpenAICompatEvaluator(SentenceEvaluator):
    """Sends one prompt per (doc, sentence). Uses /v1/chat/completions."""

    name = "openai_compat"

    def __init__(
        self,
        model_name: str,
        endpoint: str,
        api_key: str,
        prompt_template: str,
        prompt_version: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        request_timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self._model_name = model_name
        self._prompt_version = prompt_version
        self._template = prompt_template
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.client = OpenAI(
            base_url=endpoint,
            api_key=api_key or "EMPTY",
            timeout=request_timeout,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "OpenAICompatEvaluator":
        prompt_path = Path(cfg["prompt_path"])
        template = prompt_path.read_text()
        return cls(
            model_name=cfg["model_name"],
            endpoint=cfg["endpoint"],
            api_key=cfg.get("api_key", "EMPTY"),
            prompt_template=template,
            prompt_version=cfg["prompt_version"],
            max_tokens=cfg.get("max_tokens", 256),
            temperature=cfg.get("temperature", 0.0),
            top_p=cfg.get("top_p", 1.0),
            request_timeout=cfg.get("request_timeout", 120.0),
            max_retries=cfg.get("max_retries", 3),
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def _build_prompt(self, document: str, sentence: str) -> str:
        return self._template.replace("{document}", document).replace(
            "{sentence}", sentence
        )

    def score_sentence(self, document: str, sentence: str) -> SentenceScore:
        prompt = self._build_prompt(document, sentence)
        last_raw = ""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                last_raw = resp.choices[0].message.content or ""
                parsed = _parse_json_response(last_raw)
                coerced = _coerce_score(parsed) if parsed is not None else None
                if coerced is not None:
                    f, c, r = coerced
                    return SentenceScore(
                        faithful=f,
                        confidence=c,
                        reason=r,
                        raw_response=last_raw,
                        parse_failed=False,
                    )
            except Exception as e:  # network / 5xx / rate limit
                last_err = e
                time.sleep(0.5 * (2**attempt))
                continue
            # parse failed: retry with same prompt
            time.sleep(0.2)
        # Fallback: conservative not-faithful with low confidence + flag.
        return SentenceScore(
            faithful=0,
            confidence=0.0,
            reason=f"parse_failed (last_err={last_err})",
            raw_response=last_raw,
            parse_failed=True,
        )


class MockEvaluator(SentenceEvaluator):
    """Deterministic stub for offline smoke tests (no GPU required)."""

    name = "mock"

    def __init__(self, model_name: str = "mock-model", prompt_version: str = "mock-v1"):
        self._model_name = model_name
        self._prompt_version = prompt_version

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def score_sentence(self, document: str, sentence: str) -> SentenceScore:
        # Heuristic: faithful iff every >=4-char alphabetic token in sentence
        # appears (case-insensitive) in the document. Pure stub.
        doc_low = document.lower()
        toks = [t for t in re.findall(r"[A-Za-z]+", sentence) if len(t) >= 4]
        if not toks:
            return SentenceScore(
                faithful=1, confidence=0.3, reason="no content tokens",
                raw_response='{"faithful":1,"confidence":0.3,"reason":"no content tokens"}',
            )
        hits = sum(1 for t in toks if t.lower() in doc_low)
        ratio = hits / len(toks)
        faithful = 1 if ratio >= 0.6 else 0
        return SentenceScore(
            faithful=faithful,
            confidence=round(abs(ratio - 0.5) * 2, 3),
            reason=f"mock token-overlap ratio={ratio:.2f}",
            raw_response=(
                f'{{"faithful":{faithful},"confidence":{round(abs(ratio-0.5)*2,3)},'
                f'"reason":"mock ratio={ratio:.2f}"}}'
            ),
        )
