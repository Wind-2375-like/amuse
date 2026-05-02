#!/usr/bin/env python
"""Sanity scatter: summary-level aggregated score vs human label.

Overlays mean-aggregation (soft) and min-aggregation (soft) on the same axes.
Reads ``results/summary_scores.parquet`` produced by ``eval/run_meta_eval.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_summary_path(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)

    candidates = sorted(
        Path("results").glob("summary_scores*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No summary-score parquet found under results/. Run python -m eval.run_meta_eval first."
        )
    if len(candidates) > 1:
        listed = "\n".join(f"  - {path}" for path in candidates)
        raise ValueError(
            "Multiple summary-score parquet files found. Pass --summary-parquet explicitly.\n"
            f"Candidates:\n{listed}"
        )
    return candidates[0]


def _default_plot_path(summary_parquet: Path) -> Path:
    stem = summary_parquet.stem
    if stem == "summary_scores":
        return Path("results/figs/mean_vs_min_scatter.png")
    suffix = stem.removeprefix("summary_scores")
    return Path("results/figs") / f"mean_vs_min_scatter{suffix}.png"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-parquet", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    summary_parquet = _resolve_summary_path(args.summary_parquet)
    out = Path(args.out) if args.out else _default_plot_path(summary_parquet)

    df = pd.read_parquet(summary_parquet)
    rng = np.random.default_rng(0)
    # Jitter human_label (binary 0/1) on y for legibility.
    y = df["human_label"].astype(float).to_numpy()
    y_j = y + rng.uniform(-0.07, 0.07, size=len(y))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["mean__soft"], y_j, s=10, alpha=0.35,
               label="mean (soft)", color="tab:blue")
    ax.scatter(df["min__soft"], y_j, s=10, alpha=0.35,
               label="min (soft)", color="tab:red", marker="x")

    ax.set_xlabel("aggregated summary-level score")
    ax.set_ylabel("human label (0=unfaithful, 1=faithful)  [jittered]")
    ax.set_title("Aggregated LLM score vs human label\n"
                 "(AggreFact-CNN + AggreFact-XSum, soft semantic)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks([0, 1])
    ax.legend(loc="center left")
    ax.grid(alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
