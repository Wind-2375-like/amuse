#!/usr/bin/env python
"""Pull AggreFact from HuggingFace and write a normalized parquet.

Usage:
    python scripts/load_aggrefact.py --out data/aggrefact/aggrefact.parquet
    python scripts/load_aggrefact.py --max-rows 100   # quick check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.aggrefact import load_aggrefact


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/aggrefact/aggrefact.parquet")
    p.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="If omitted, use all splits the dataset actually exposes.",
    )
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    print(f"[load_aggrefact] pulling splits={args.splits or 'auto'} ...", flush=True)
    df = load_aggrefact(
        splits=tuple(args.splits) if args.splits else None,
        max_rows=args.max_rows,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[load_aggrefact] wrote {len(df)} rows -> {out}")
    print(df.head(3).to_string())
    print("origin counts:")
    print(df["origin"].value_counts().to_string())


if __name__ == "__main__":
    main()
