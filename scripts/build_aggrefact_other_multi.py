#!/usr/bin/env python
"""Build the 'AggreFact-other (multi-sentence)' benchmark parquet.

Takes the full LLM-AggreFact parquet, drops AggreFact-CNN (kept as its own
benchmark), and keeps only summaries with >= 2 spaCy-detected sentences.
The `origin` column is preserved verbatim so each row records its original
sub-benchmark name.

Usage:
    python scripts/build_aggrefact_other_multi.py \\
        --in data/aggrefact/aggrefact.parquet \\
        --out data/aggrefact_other_multi/aggrefact_other_multi.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.sentences import split_sentences


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/aggrefact/aggrefact.parquet")
    p.add_argument("--out", default="data/aggrefact_other_multi/aggrefact_other_multi.parquet")
    p.add_argument("--exclude", nargs="+", default=["AggreFact-CNN"],
                   help="Origins to exclude (kept as separate benchmarks).")
    p.add_argument("--min-sents", type=int, default=2)
    p.add_argument("--spacy-model", default="en_core_web_sm")
    args = p.parse_args()

    df = pd.read_parquet(args.inp)
    print(f"[build] loaded {len(df)} rows from {args.inp}")
    print(f"[build] origin breakdown:\n{df['origin'].value_counts().to_string()}")

    df = df[~df["origin"].isin(args.exclude)].reset_index(drop=True)
    print(f"[build] after dropping {args.exclude}: {len(df)} rows")

    # Count sentences per summary.
    n_sents = []
    for s in tqdm(df["summary"].tolist(), desc="splitting"):
        n_sents.append(len(split_sentences(str(s), model=args.spacy_model)))
    df["n_sents"] = n_sents

    keep = df["n_sents"] >= args.min_sents
    out_df = df[keep].reset_index(drop=True)
    print(
        f"[build] keep n_sents>={args.min_sents}: "
        f"{int(keep.sum())} / {len(df)} ({keep.mean():.1%})"
    )
    print(f"[build] origin breakdown after filter:\n{out_df['origin'].value_counts().to_string()}")
    print(
        "[build] sentence-count after filter: "
        f"mean={out_df['n_sents'].mean():.2f}  "
        f"median={out_df['n_sents'].median():.0f}  "
        f"max={out_df['n_sents'].max()}"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Drop helper column to keep schema aligned with aggrefact.parquet.
    out_df.drop(columns=["n_sents"]).to_parquet(out, index=False)
    print(f"[build] wrote {len(out_df)} rows -> {out}")


if __name__ == "__main__":
    main()
