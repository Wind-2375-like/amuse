#!/usr/bin/env python
"""Phase-2 main script: aggregate sentence-level scores -> summary-level
scores, then compute correlations with human labels (with bootstrap CIs).

Inputs (defaults):
    results/sentence_scores.parquet   (Phase 1 output)
    data/aggrefact/aggrefact.parquet  (source dataset, for human_label / origin)

Outputs:
    results/meta_eval_summary.csv     long format, one row per
                                      (aggregation, semantic, origin, metric)
    results/meta_eval_table.md        human-readable markdown tables

Run:
    python -m eval.run_meta_eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Make repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregation.methods import AGGREGATIONS, SOFT_ONLY
from eval.metrics import METRICS, bootstrap_ci


def _resolve_sentence_scores_path(arg_value: Optional[str]) -> Path:
    if arg_value:
        return Path(arg_value)

    candidates = sorted(
        Path("results").glob("sentence_scores*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No sentence-score parquet found under results/. Run scripts/score_sentences.py first."
        )
    if len(candidates) > 1:
        listed = "\n".join(f"  - {path}" for path in candidates)
        raise ValueError(
            "Multiple sentence-score parquet files found. Pass --sentence-scores explicitly.\n"
            f"Candidates:\n{listed}"
        )
    return candidates[0]


def _default_output_path(sentence_scores: Path, prefix: str, suffix: str) -> Path:
    stem = sentence_scores.stem
    if stem == "sentence_scores":
        return Path("results") / f"{prefix}{suffix}"
    derived = stem.replace("sentence_scores", prefix, 1)
    return Path("results") / f"{derived}{suffix}"


# ---------------------------------------------------------------------------
# Data loading + aggregation
# ---------------------------------------------------------------------------

def load_sentence_scores(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    needed = {"doc_id", "sent_idx", "faithful", "confidence"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"sentence_scores parquet missing columns: {missing}")
    # P(faithful = 1) under the LLM's self-reported confidence.
    f = df["faithful"].astype(float).to_numpy()
    c = df["confidence"].astype(float).to_numpy()
    df = df.copy()
    df["p_faithful"] = f * c + (1.0 - f) * (1.0 - c)
    return df


def load_dataset_meta(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    keep = ["doc_id", "human_label", "origin", "split"]
    return df[[c for c in keep if c in df.columns]].copy()


def aggregate_per_summary(sent_df: pd.DataFrame) -> pd.DataFrame:
    """For each (doc_id) group, compute every aggregation under both
    semantics. Returns one row per doc_id."""
    rows = []
    for doc_id, g in sent_df.groupby("doc_id", sort=False):
        # Sort by sent_idx for determinism (doesn't affect aggregations).
        g = g.sort_values("sent_idx")
        hard = g["faithful"].to_numpy(dtype=float)
        soft = g["p_faithful"].to_numpy(dtype=float)
        out = {"doc_id": doc_id, "n_sent": len(g)}
        for name, fn in AGGREGATIONS.items():
            if name in SOFT_ONLY:
                out[f"{name}__soft"] = fn(soft)
            else:
                out[f"{name}__hard"] = fn(hard)
                out[f"{name}__soft"] = fn(soft)
        rows.append(out)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Meta-evaluation
# ---------------------------------------------------------------------------

def evaluate(
    summary_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """For each (aggregation, semantic, origin, metric), compute point
    estimate + bootstrap CI. ``summary_df`` must contain 'human_label',
    'origin', and one column per aggregation named ``<agg>__<sem>``."""
    origins = sorted(summary_df["origin"].dropna().unique().tolist())
    origins_with_overall = origins + ["__overall__"]

    agg_cols = [c for c in summary_df.columns if "__" in c and c.split("__")[-1] in {"hard", "soft"}]

    rows = []
    for origin in origins_with_overall:
        sub = (
            summary_df if origin == "__overall__"
            else summary_df[summary_df["origin"] == origin]
        )
        if len(sub) == 0:
            continue
        y = sub["human_label"].to_numpy(dtype=float)

        for col in agg_cols:
            agg_name, sem = col.rsplit("__", 1)
            x = sub[col].to_numpy(dtype=float)
            for metric_name, metric_fn in METRICS.items():
                pt, lo, hi = bootstrap_ci(
                    metric_fn, x, y, n=n_bootstrap, seed=seed,
                )
                rows.append({
                    "aggregation": agg_name,
                    "semantic": sem,
                    "origin": origin,
                    "metric": metric_name,
                    "value": pt,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n": int(len(sub)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_cell(v: float, lo: float, hi: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    star = "" if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else " ⁰"
    if not np.isfinite(lo) or not np.isfinite(hi):
        return f"{v:+.3f}{star}"
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def to_markdown(df: pd.DataFrame) -> str:
    """Render one markdown pivot table per origin. ``⁰`` flags CIs that
    cross zero (i.e. correlation not significant at alpha=0.05)."""
    metrics_order = ["pearson", "spearman", "kendall", "roc_auc"]
    out = []
    for origin in sorted(df["origin"].unique()):
        sub = df[df["origin"] == origin]
        n = int(sub["n"].iloc[0])
        title = "Overall (AggreFact-CNN + AggreFact-XSum)" if origin == "__overall__" else origin
        out.append(f"### {title}  (n_summaries = {n})")
        # rows: aggregation x semantic; cols: metric
        for sem in ["hard", "soft"]:
            ssub = sub[sub["semantic"] == sem]
            if ssub.empty:
                continue
            out.append(f"\n**Input semantic: `{sem}`**\n")
            header = "| aggregation | " + " | ".join(metrics_order) + " |"
            sep = "|---" * (len(metrics_order) + 1) + "|"
            out.append(header)
            out.append(sep)
            # preserve aggregation order from registry
            agg_order = [a for a in AGGREGATIONS if not (sem == "hard" and a in SOFT_ONLY)]
            for agg in agg_order:
                row_cells = [agg]
                for m in metrics_order:
                    r = ssub[(ssub["aggregation"] == agg) & (ssub["metric"] == m)]
                    if r.empty:
                        row_cells.append("n/a")
                    else:
                        row_cells.append(_fmt_cell(
                            float(r["value"].iloc[0]),
                            float(r["ci_lo"].iloc[0]),
                            float(r["ci_hi"].iloc[0]),
                        ))
                out.append("| " + " | ".join(row_cells) + " |")
            out.append("")
        out.append("")
    out.append("Legend: `⁰` = bootstrap 95% CI includes 0 (correlation not "
               "significantly different from 0).")
    return "\n".join(out)


def takeaway(df: pd.DataFrame) -> str:
    overall = df[(df["origin"] == "__overall__") & (df["metric"] == "spearman")]
    if overall.empty:
        return ""
    best = overall.loc[overall["value"].idxmax()]
    return (
        f"Best Spearman (overall): **{best['aggregation']}** on `{best['semantic']}` "
        f"inputs: rho={best['value']:+.3f} (95% CI [{best['ci_lo']:+.3f}, "
        f"{best['ci_hi']:+.3f}], n={int(best['n'])})."
    )


# ---------------------------------------------------------------------------
# Sanity checks (logged, not asserted hard)
# ---------------------------------------------------------------------------

def sanity_checks(summary_df: pd.DataFrame) -> list[str]:
    """min<=mean<=max under hard semantic, on every summary."""
    msgs = []
    h_min = summary_df["min__hard"].to_numpy()
    h_mean = summary_df["mean__hard"].to_numpy()
    h_max = summary_df["max__hard"].to_numpy()
    bad1 = int(np.sum(h_min > h_mean + 1e-9))
    bad2 = int(np.sum(h_mean > h_max + 1e-9))
    msgs.append(f"sanity: min<=mean violated on {bad1} / {len(summary_df)} summaries (hard)")
    msgs.append(f"sanity: mean<=max violated on {bad2} / {len(summary_df)} summaries (hard)")
    return msgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sentence-scores",
        default=None,
        help="Sentence-level parquet from scripts/score_sentences.py. Required when multiple model-specific files exist.",
    )
    p.add_argument("--dataset", default="data/aggrefact_cnn/aggrefact_cnn.parquet")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-md", default=None)
    p.add_argument("--out-summary-parquet",
                   default=None,
                   help="Per-summary aggregated scores (used by plot script).")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sentence_scores_path = _resolve_sentence_scores_path(args.sentence_scores)
    out_csv = Path(args.out_csv) if args.out_csv else _default_output_path(
        sentence_scores_path, "meta_eval_summary", ".csv"
    )
    out_md = Path(args.out_md) if args.out_md else _default_output_path(
        sentence_scores_path, "meta_eval_table", ".md"
    )
    out_summary_parquet = (
        Path(args.out_summary_parquet)
        if args.out_summary_parquet
        else _default_output_path(sentence_scores_path, "summary_scores", ".parquet")
    )

    sent = load_sentence_scores(sentence_scores_path)
    meta = load_dataset_meta(Path(args.dataset))
    print(f"[meta_eval] sentence scores: {sentence_scores_path}")
    print(f"[meta_eval] sentences: {len(sent)} | summaries (sent_df): "
          f"{sent['doc_id'].nunique()} | dataset rows: {len(meta)}")

    summary = aggregate_per_summary(sent)

    # Re-attach origin / human_label from the canonical dataset (sentence_scores
    # already carried them, but the dataset is ground truth).
    summary = summary.merge(meta, on="doc_id", how="left")
    n_unlabeled = int(summary["human_label"].isna().sum())
    if n_unlabeled:
        print(f"[meta_eval] dropping {n_unlabeled} summaries without human_label")
        summary = summary[summary["human_label"].notna()].reset_index(drop=True)
    summary["human_label"] = summary["human_label"].astype(float)

    print(f"[meta_eval] aggregated summaries: {len(summary)}")
    for msg in sanity_checks(summary):
        print(f"[meta_eval] {msg}")
    print("[meta_eval] origin counts:", summary["origin"].value_counts().to_dict())

    out_summary_parquet.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out_summary_parquet, index=False)

    eval_df = evaluate(summary, n_bootstrap=args.n_bootstrap, seed=args.seed)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(out_csv, index=False)

    md = to_markdown(eval_df)
    out_md.write_text(md + "\n")

    print()
    print(md)
    print()
    print(takeaway(eval_df))
    print()
    print(f"[meta_eval] wrote {out_csv}, {out_md}, {out_summary_parquet}")


if __name__ == "__main__":
    main()
