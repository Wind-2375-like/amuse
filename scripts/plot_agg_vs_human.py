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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-parquet", default="results/summary_scores.parquet")
    p.add_argument("--out", default="results/figs/mean_vs_min_scatter.png")
    args = p.parse_args()

    df = pd.read_parquet(args.summary_parquet)
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
