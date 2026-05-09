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
import concurrent.futures
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer
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


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def _default_output_path(model_name: str, dataset_name: str) -> Path:
    return (
        Path("results")
        / f"sentence_scores.{_slugify(model_name)}.{_slugify(dataset_name)}.parquet"
    )


def _maybe_load_dataset(cfg: dict, fallback_inline: bool) -> pd.DataFrame:
    """Try to load the configured dataset parquet. If missing AND --mock with
    `fallback_inline=True`, return a tiny inline frame so we can prove the
    pipeline end-to-end without HF access."""
    path = Path(cfg["dataset_path"])
    if path.exists():
        return pd.read_parquet(path)
    if not fallback_inline:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Build or download the matching dataset parquet first."
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


def _count_chat_tokens(tokenizer: Any, prompt: str, enable_thinking: bool) -> int:
    messages = [{"role": "user", "content": prompt}]
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    return len(token_ids)


def _iter_sentence_tasks(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    evaluator: Any,
    tokenizer: Any,
    prompt_token_budget: int | None,
    stats: dict[str, int],
):
    for doc_idx, row in df.iterrows():
        document = row["document"]
        summary = row["summary"]
        doc_id = row["doc_id"]
        summary_id = f"{doc_id}::{row.get('model', 'm')}"
        yielded_for_doc = False
        sents = split_sentences(summary, model=cfg.get("spacy_model", "en_core_web_sm"))
        for sent in sents:
            if tokenizer is not None and prompt_token_budget is not None:
                prompt = evaluator.build_prompt(document, sent.text)
                prompt_tokens = _count_chat_tokens(
                    tokenizer,
                    prompt,
                    enable_thinking=getattr(evaluator, "enable_thinking", False),
                )
                if prompt_tokens > prompt_token_budget:
                    stats["skipped_too_long"] += 1
                    continue
            yielded_for_doc = True
            yield {
                "doc_idx": int(doc_idx),
                "doc_id": doc_id,
                "summary_id": summary_id,
                "document": document,
                "summary": summary,
                "sent_idx": sent.idx,
                "sent_text": sent.text,
                "origin": row.get("origin", ""),
                "split": row.get("split", ""),
                "human_label": row.get("human_label", None),
            }
        if not yielded_for_doc:
            stats.setdefault("empty_docs", 0)
            stats["empty_docs"] += 1


def _score_task(task: dict[str, Any], cached: CachedEvaluator, evaluator: Any) -> tuple[dict[str, Any], bool]:
    score, hit = cached.score(
        task["document"],
        task["summary"],
        task["sent_idx"],
        task["sent_text"],
    )
    row = {
        "doc_id": task["doc_id"],
        "summary_id": task["summary_id"],
        "sent_idx": task["sent_idx"],
        "sent_text": task["sent_text"],
        "faithful": int(score.faithful),
        "confidence": float(score.confidence),
        "reason": score.reason,
        "raw_response": score.raw_response,
        "parse_failed": bool(score.parse_failed),
        "model_name": evaluator.model_name,
        "prompt_version": evaluator.prompt_version,
        "origin": task["origin"],
        "split": task["split"],
        "human_label": task["human_label"],
    }
    return row, hit


def _score_task_with_task(
    task: dict[str, Any],
    cached: CachedEvaluator,
    evaluator: Any,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    row, hit = _score_task(task, cached, evaluator)
    return task, row, hit


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
    p.add_argument("--dataset-path", default=None,
                   help="Override `dataset_path` from config (parquet path).")
    p.add_argument("--dataset-name", default=None,
                   help="Override `dataset_name` from config (used for cache filename).")
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override `max_model_len` from config for local prompt-length filtering.",
    )
    p.add_argument(
        "--origin",
        nargs="+",
        default=["all"],
        help="Keep only rows whose `origin` is in this list. "
             "Default 'all' = no filter. Examples: "
             "'--origin AggreFact-CNN' for the CNN-only benchmark; "
             "'--origin all' for diversumm or aggrefact_other_multi parquets "
             "(which are already pre-filtered).",
    )
    p.add_argument(
        "--split",
        nargs="+",
        default=None,
        help="Keep only these splits (e.g. test). Default: all splits in the parquet.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SCORE_WORKERS", "1")),
        help="Concurrent in-flight sentence requests for OpenAI-compatible backends.",
    )
    args = p.parse_args()

    cfg = _load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model
    if args.endpoint:
        cfg["endpoint"] = args.endpoint
    if args.dataset_path:
        cfg["dataset_path"] = args.dataset_path
    if args.dataset_name:
        cfg["dataset_name"] = args.dataset_name
    if args.max_model_len is not None:
        cfg["max_model_len"] = args.max_model_len
    # env-var overrides (helpful for collaborators on different boxes)
    cfg["model_name"] = os.environ.get("MODEL_NAME", cfg["model_name"])
    cfg["endpoint"] = os.environ.get("ENDPOINT", cfg["endpoint"])
    cfg["max_model_len"] = int(
        os.environ.get("MAX_MODEL_LEN", cfg.get("max_model_len", 0))
    )

    fallback_inline = args.mock or args.inline_fallback
    df = _maybe_load_dataset(cfg, fallback_inline=fallback_inline)

    # --- subset filtering ---
    if args.origin and not (len(args.origin) == 1 and args.origin[0].lower() == "all"):
        before = len(df)
        df = df[df["origin"].isin(args.origin)].reset_index(drop=True)
        print(f"[score_sentences] origin filter {args.origin}: {before} -> {len(df)} rows")
    if args.split:
        before = len(df)
        df = df[df["split"].isin(args.split)].reset_index(drop=True)
        print(f"[score_sentences] split filter {args.split}: {before} -> {len(df)} rows")

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
    print(f"[score_sentences] workers: {args.workers}")

    prompt_token_budget = None
    tokenizer = None
    if isinstance(evaluator, OpenAICompatEvaluator):
        max_model_len = int(cfg.get("max_model_len", 0))
        if max_model_len <= 0:
            raise ValueError("cfg.max_model_len must be set for OpenAI-compatible runs.")
        prompt_token_budget = max_model_len - int(cfg.get("max_tokens", 0))
        if prompt_token_budget <= 0:
            raise ValueError(
                f"max_model_len={max_model_len} must exceed max_tokens={cfg.get('max_tokens', 0)}"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            evaluator.model_name,
            trust_remote_code=True,
        )
        print(
            "[score_sentences] prompt budget: "
            f"input<={prompt_token_budget} tokens, output<={cfg.get('max_tokens', 0)} tokens"
        )

    rows = []
    n_calls = 0
    n_hits = 0
    n_skipped_too_long = 0
    t0 = time.time()
    task_stats = {"skipped_too_long": 0}
    task_iter = _iter_sentence_tasks(
        df,
        cfg,
        evaluator,
        tokenizer,
        prompt_token_budget,
        task_stats,
    )
    progress = tqdm(total=len(df), desc="docs")
    try:
        if args.workers <= 1 or args.mock:
            last_doc_idx = None
            for task in task_iter:
                row, hit = _score_task(task, cached, evaluator)
                n_hits += int(hit)
                n_calls += int(not hit)
                rows.append(row)
                if task["doc_idx"] != last_doc_idx:
                    progress.update(1)
                    last_doc_idx = task["doc_idx"]
        else:
            max_pending = max(args.workers * 4, args.workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                pending: set[concurrent.futures.Future] = set()
                doc_pending_counts: dict[int, int] = {}
                completed_docs: set[int] = set()

                def drain(done_futures):
                    nonlocal n_hits, n_calls
                    for future in done_futures:
                        task, row, hit = future.result()
                        n_hits += int(hit)
                        n_calls += int(not hit)
                        rows.append(row)
                        doc_idx = task["doc_idx"]
                        remaining = doc_pending_counts[doc_idx] - 1
                        doc_pending_counts[doc_idx] = remaining
                        if remaining == 0 and doc_idx not in completed_docs:
                            completed_docs.add(doc_idx)
                            progress.update(1)

                for task in task_iter:
                    doc_idx = task["doc_idx"]
                    doc_pending_counts[doc_idx] = doc_pending_counts.get(doc_idx, 0) + 1
                    pending.add(executor.submit(_score_task_with_task, task, cached, evaluator))
                    if len(pending) >= max_pending:
                        done, pending = concurrent.futures.wait(
                            pending,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        drain(done)
                if pending:
                    done, _ = concurrent.futures.wait(pending)
                    drain(done)
    finally:
        empty_docs = task_stats.get("empty_docs", 0)
        if empty_docs:
            progress.update(empty_docs)
        progress.close()

    n_skipped_too_long = task_stats["skipped_too_long"]

    cached.close()
    elapsed = time.time() - t0
    out_df = pd.DataFrame(rows)
    out_path = (
        _default_output_path(evaluator.model_name, cfg.get("dataset_name", "aggrefact"))
        if args.out == "results/sentence_scores.parquet"
        else Path(args.out)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(
        f"[score_sentences] done. sentences={len(out_df)} "
        f"cache_hits={n_hits} new_calls={n_calls} skipped_too_long={n_skipped_too_long} "
        f"elapsed={elapsed:.1f}s -> {out_path}"
    )
    if len(out_df):
        cols = ["doc_id", "sent_idx", "faithful", "confidence", "sent_text"]
        print(out_df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
