#!/usr/bin/env python
"""Pull DiverSumm CSV and write a normalized parquet matching aggrefact schema.

Usage:
    python scripts/load_diversumm.py --out data/diversumm/diversumm.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.diversumm import load_diversumm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/diversumm/diversumm.parquet")
    p.add_argument("--csv", default=None,
                   help="Local CSV path (default: download to /tmp/DiverSumm.csv)")
    args = p.parse_args()

    df = load_diversumm(csv_path=args.csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[load_diversumm] wrote {len(df)} rows -> {out}")
    print("origin counts:")
    print(df["origin"].value_counts().to_string())
    print("model counts:")
    print(df["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
