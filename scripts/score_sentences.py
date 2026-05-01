#!/usr/bin/env python
"""Phase-1 main pipeline: AggreFact -> sentence split -> LLM score (cached) -> parquet.

Examples:
    # Real run against vLLM at localhost:8000
    python scripts/score_sentences.py --limit 20

    # Offline smoke test (no GPU, deterministic mock model)
    python scripts/score_sentences.py --limit 20 --mock

    # Override config
    python scripts/score_sentences.py --config configs/default.yaml --model Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

# Make repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.sentences import split_sentences
from evaluators import (
    CachedEvaluator,
    MockEvaluator,
    OpenAICompatEvaluator,
)


def _load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _maybe_load_dataset(cfg: dict, fallback_inline: bool) -> pd.DataFrame:
    """Try to load the AggreFact parquet. If missing AND --mock with
    `fallback_inline=True`, return a tiny inline frame so we can prove the
    pipeline end-to-end without HF access."""
    path = Path(cfg["dataset_path"])
    if path.exists():
        return pd.read_parquet(path)
    if not fallback_inline:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run scripts/load_aggrefact.py first."
        )
    print(f"[score_sentences] {path} missing — using inline fallback dataset.", flush=True)
    return _inline_fallback()


def _inline_fallback() -> pd.DataFrame:
    docs = [
        (
            "doc1",
            "Marie Curie was a Polish-born physicist and chemist who conducted "
            "pioneering research on radioactivity. She was the first woman to "
            "win a Nobel Prize, winning the 1903 Nobel Prize in Physics with "
            "her husband Pierre and Henri Becquerel. In 1911 she won a second "
            "Nobel Prize, this time in Chemistry.",
            "Marie Curie won two Nobel Prizes. She was born in France. Her "
            "first Nobel Prize was awarded in 1903 in Physics.",
            0,  # not faithful overall (born in France is wrong)
        ),
        (
            "doc2",
            "The James Webb Space Telescope launched on December 25, 2021, "
            "from French Guiana aboard an Ariane 5 rocket. It is the largest "
            "optical telescope in space and is operated by NASA, ESA and CSA. "
            "Its primary mirror is 6.5 meters across.",
            "The James Webb Space Telescope launched in December 2021. It has "
            "a primary mirror that is 6.5 meters wide.",
            1,
        ),
        (
            "doc3",
            "The 2022 FIFA World Cup was held in Qatar. Argentina won the "
            "tournament, defeating France in the final on penalties after a "
            "3-3 draw in extra time. Lionel Messi was named the Golden Ball "
            "winner.",
            "Argentina won the 2022 World Cup in Brazil. They beat Germany in "
            "the final.",
            0,
        ),
    ]
    rows = []
    for doc_id, document, summary, label in docs:
        rows.append({
            "doc_id": doc_id,
            "document": document,
            "summary": summary,
            "human_label": label,
            "score": float("nan"),
            "origin": "inline_fallback",
            "split": "smoke",
            "model": "n/a",
            "cut": "inline",
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N (doc, summary) pairs.")
    p.add_argument("--out", default="results/sentence_scores.parquet")
    p.add_argument("--mock", action="store_true",
                   help="Use offline MockEvaluator instead of an HTTP LLM.")
    p.add_argument("--inline-fallback", action="store_true",
                   help="If dataset parquet is missing, use a tiny inline dataset (smoke test only).")
    p.add_argument("--model", default=None, help="Override model_name from config.")
    p.add_argument("--endpoint", default=None, help="Override endpoint from config.")
    args = p.parse_args()

    cfg = _load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model
    if args.endpoint:
        cfg["endpoint"] = args.endpoint
    # env-var overrides (helpful for collaborators on different boxes)
    cfg["model_name"] = os.environ.get("MODEL_NAME", cfg["model_name"])
    cfg["endpoint"] = os.environ.get("ENDPOINT", cfg["endpoint"])

    fallback_inline = args.mock or args.inline_fallback
    df = _maybe_load_dataset(cfg, fallback_inline=fallback_inline)
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)

    # Build evaluator
    if args.mock:
        evaluator = MockEvaluator(model_name=cfg["model_name"] + "+mock",
                                  prompt_version=cfg["prompt_version"] + "+mock")
    else:
        evaluator = OpenAICompatEvaluator.from_config(cfg)

    cached = CachedEvaluator(
        evaluator=evaluator,
        cache_dir=Path(cfg["cache_dir"]),
        dataset=cfg.get("dataset_name", "aggrefact"),
    )
    print(f"[score_sentences] cache file: {cached.cache_path} (loaded {len(cached.cache)} entries)")

    rows = []
    n_calls = 0
    n_hits = 0
    t0 = time.time()
    for i, r in tqdm(df.iterrows(), total=len(df), desc="docs"):
        document = r["document"]
        summary = r["summary"]
        doc_id = r["doc_id"]
        # summary_id: stable per (doc_id, summary)
        summary_id = f"{doc_id}::{r.get('model', 'm')}"
        sents = split_sentences(summary, model=cfg.get("spacy_model", "en_core_web_sm"))
        for sent in sents:
            score, hit = cached.score(document, summary, sent.idx, sent.text)
            n_hits += int(hit)
            n_calls += int(not hit)
            rows.append({
                "doc_id": doc_id,
                "summary_id": summary_id,
                "sent_idx": sent.idx,
                "sent_text": sent.text,
                "faithful": int(score.faithful),
                "confidence": float(score.confidence),
                "reason": score.reason,
                "raw_response": score.raw_response,
                "parse_failed": bool(score.parse_failed),
                "model_name": evaluator.model_name,
                "prompt_version": evaluator.prompt_version,
                "origin": r.get("origin", ""),
                "split": r.get("split", ""),
                "human_label": r.get("human_label", None),
            })

    cached.close()
    elapsed = time.time() - t0
    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(
        f"[score_sentences] done. sentences={len(out_df)} "
        f"cache_hits={n_hits} new_calls={n_calls} elapsed={elapsed:.1f}s -> {out_path}"
    )
    if len(out_df):
        cols = ["doc_id", "sent_idx", "faithful", "confidence", "sent_text"]
        print(out_df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
