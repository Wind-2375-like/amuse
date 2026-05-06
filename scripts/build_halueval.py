#!/usr/bin/env python
"""Build the HaluEval summarization benchmark parquet.

Usage:
	python scripts/build_halueval.py
	python scripts/build_halueval.py --max-rows 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.halueval import load_halueval


def main() -> None:
	p = argparse.ArgumentParser()
	p.add_argument("--out", default="data/halueval/halueval.parquet")
	p.add_argument(
		"--json",
		default=None,
		help="Local HaluEval summarization JSON path. Default: download to /tmp.",
	)
	p.add_argument(
		"--max-rows",
		type=int,
		default=None,
		help="Limit source documents before expanding to faithful/hallucinated rows.",
	)
	args = p.parse_args()

	df = load_halueval(json_path=args.json, max_rows=args.max_rows)
	out = Path(args.out)
	out.parent.mkdir(parents=True, exist_ok=True)
	df.to_parquet(out, index=False)
	print(f"[build_halueval] wrote {len(df)} rows -> {out}")
	print("label counts:")
	print(df["human_label"].value_counts(dropna=False).sort_index().to_string())
	print("model counts:")
	print(df["model"].value_counts().to_string())


if __name__ == "__main__":
	main()
