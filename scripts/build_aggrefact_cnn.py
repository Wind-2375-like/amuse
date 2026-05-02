#!/usr/bin/env python
"""Build the AggreFact-CNN benchmark parquet.

Just filters the full LLM-AggreFact parquet to `origin == "AggreFact-CNN"`
and writes it out, so all three benchmarks (CNN / DiverSumm / other-multi)
have a pre-built parquet with the same CLI form.

Usage:
    python scripts/build_aggrefact_cnn.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/aggrefact/aggrefact.parquet")
    p.add_argument("--out", default="data/aggrefact_cnn/aggrefact_cnn.parquet")
    p.add_argument("--origin", default="AggreFact-CNN")
    args = p.parse_args()

    df = pd.read_parquet(args.inp)
    out_df = df[df["origin"] == args.origin].reset_index(drop=True)
    print(f"[build] kept {len(out_df)} / {len(df)} rows where origin == {args.origin!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"[build] wrote {len(out_df)} rows -> {out}")


if __name__ == "__main__":
    main()
