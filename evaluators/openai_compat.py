"""OpenAI-compatible evaluator (talks to vLLM, OpenAI, anything compatible).

Prompt v2 contract: the model returns a single-line JSON object
``{"faithful": 0 or 1}``. The soft label P(faithful=1) is recovered from
token-level logprobs at the position where the 0/1 answer token is emitted:

    P(faithful=1) = exp(lp1) / (exp(lp0) + exp(lp1))

This value is stored in the `confidence` field of `SentenceScore` (the field
name is kept for schema/cache backwards-compat, but the semantics is now
"posterior probability of being faithful").
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .base import SentenceEvaluator, SentenceScore


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse_json_response(text: str) -> Optional[dict]:
    """Try strict JSON first, then a regex fallback to grab the first {...}."""
    text = text.strip()
    # Strip <think>...</think> blocks that some reasoning models always emit.
    text = _THINK_RE.sub("", text).strip()
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


def _coerce_score(parsed: dict) -> Optional[int]:
    """v2 contract: only the `faithful` 0/1 field is required."""
    if not isinstance(parsed, dict) or "faithful" not in parsed:
        return None
    try:
        faithful = int(parsed["faithful"])
    except (TypeError, ValueError):
        return None
    return 1 if faithful >= 1 else 0


def _soft_label_from_logprobs(
    logprobs_content,
) -> tuple[Optional[float], Optional[int]]:
    """Walk the response token stream and find the first token whose stripped
    value is "0" or "1" (the answer digit in our JSON). At that position,
    compute P(faithful=1) by renormalizing over {"0", "1"} from the
    top_logprobs distribution.

    Returns (p_faithful, faithful_int) or (None, None) if not recoverable.
    """
    if not logprobs_content:
        return None, None
    for tok in logprobs_content:
        chosen = (tok.token or "").strip()
        if chosen not in {"0", "1"}:
            continue
        # collect logprobs for both "0" and "1" at this position
        lp0: Optional[float] = None
        lp1: Optional[float] = None
        # the chosen token always exposes its own logprob
        if chosen == "0":
            lp0 = float(tok.logprob)
        else:
            lp1 = float(tok.logprob)
        for alt in (tok.top_logprobs or []):
            t = (alt.token or "").strip()
            if t == "0" and lp0 is None:
                lp0 = float(alt.logprob)
            elif t == "1" and lp1 is None:
                lp1 = float(alt.logprob)
        if lp0 is None and lp1 is None:
            return None, None
        # If only one side present in top_logprobs, treat the missing side as
        # negligible (P ~= 0 within numeric tolerance of top_logprobs cutoff).
        e0 = math.exp(lp0) if lp0 is not None else 0.0
        e1 = math.exp(lp1) if lp1 is not None else 0.0
        denom = e0 + e1
        if denom <= 0.0:
            return None, None
        p1 = e1 / denom
        return p1, (1 if chosen == "1" else 0)
    return None, None


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
        enable_thinking: bool = False,
    ):
        self._model_name = model_name
        self._prompt_version = prompt_version
        self._template = prompt_template
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
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
            enable_thinking=cfg.get("enable_thinking", False),
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
        # Qwen3 / DeepSeek-R1 are reasoning models that emit <think>...</think>
        # by default and burn through max_tokens before producing JSON. We
        # disable that via Qwen's `enable_thinking` chat-template flag, which
        # vLLM forwards from `extra_body.chat_template_kwargs`.
        extra_body = (
            {} if self.enable_thinking
            else {"chat_template_kwargs": {"enable_thinking": False}}
        )
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    logprobs=True,
                    top_logprobs=20,
                    extra_body=extra_body,
                )
                choice = resp.choices[0]
                last_raw = choice.message.content or ""
                parsed = _parse_json_response(last_raw)
                faithful = _coerce_score(parsed) if parsed is not None else None
                if faithful is None:
                    time.sleep(0.2)
                    continue
                # Soft label from token-level logprobs at the answer digit.
                logprobs_content = (
                    choice.logprobs.content if choice.logprobs is not None else None
                )
                p1, lp_faithful = _soft_label_from_logprobs(logprobs_content)
                if p1 is None:
                    # logprobs unavailable: fall back to hard 0/1 as soft label.
                    p1 = 1.0 if faithful == 1 else 0.0
                # Cross-check: hard parse and argmax-from-logprobs should agree
                # at temperature=0. If they disagree (rare, can happen when
                # the JSON answer digit isn't the top-1 due to surrounding
                # tokens), trust the parsed JSON for the hard label but keep
                # the renormalized soft label.
                return SentenceScore(
                    faithful=faithful,
                    confidence=float(p1),
                    reason="",
                    raw_response=last_raw,
                    parse_failed=False,
                )
            except Exception as e:  # network / 5xx / rate limit
                last_err = e
                time.sleep(0.5 * (2**attempt))
                continue
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
        # Soft label = token-overlap ratio directly (interpreted as P(faithful=1)).
        doc_low = document.lower()
        toks = [t for t in re.findall(r"[A-Za-z]+", sentence) if len(t) >= 4]
        if not toks:
            return SentenceScore(
                faithful=1, confidence=0.5, reason="",
                raw_response='{"faithful":1}',
            )
        hits = sum(1 for t in toks if t.lower() in doc_low)
        ratio = hits / len(toks)
        faithful = 1 if ratio >= 0.6 else 0
        return SentenceScore(
            faithful=faithful,
            confidence=round(ratio, 3),
            reason="",
            raw_response=f'{{"faithful":{faithful}}}',
        )
