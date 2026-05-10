#!/usr/bin/env python
"""Generate LaTeX longtables from meta-eval summary CSV files.

Each output table is grouped by dataset and contains one hard block and one
soft block, with rows grouped by model and aggregation.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_AGGREGATIONS = ["min", "mean", "max"]
DEFAULT_METRICS = ["pearson", "spearman", "kendall", "roc_auc"]
DEFAULT_MODEL_ORDER = [
    "Qwen_Qwen3-4B",
    "Qwen_Qwen3-8B",
    "Qwen_Qwen3-32B",
    "meta-llama_Llama-3.1-8B-Instruct",
    "allenai_Olmo-3-7B-Instruct",
]

MODEL_LABELS = {
    "Qwen_Qwen3-4B": "Qwen3-4B",
    "Qwen_Qwen3-8B": "Qwen3-8B",
    "Qwen_Qwen3-32B": "Qwen3-32B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "allenai_Olmo-3-7B-Instruct": "OLMo-3-7B",
}

DATASET_LABELS = {
    "aggrefact_cnn": "AggreFact-CNN",
    "aggrefact_other_multi": "AggreFact-Other-Multi",
    "diversumm": "Diversumm",
    "halueval": "HaluEval-summarization",
}

SEMANTIC_TITLES = {
    "hard": "Hard sentence labels",
    "soft": "Soft sentence probabilities",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset slugs to render. Default: all datasets with summary CSVs.",
    )
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=DEFAULT_AGGREGATIONS,
        help="Aggregations to include in each block.",
    )
    parser.add_argument(
        "--model-order",
        nargs="+",
        default=DEFAULT_MODEL_ORDER,
        help="Model slug order in the rendered table.",
    )
    parser.add_argument(
        "--out-prefix",
        default="meta_eval_longtable",
        help="Output filename prefix under results/.",
    )
    return parser.parse_args()


def _discover_summary_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("meta_eval_summary.*.csv"))


def _split_summary_filename(path: Path) -> tuple[str, str]:
    name = path.name
    prefix = "meta_eval_summary."
    if not name.startswith(prefix) or not name.endswith(".csv"):
        raise ValueError(f"unexpected summary filename: {path}")
    stem = name[len(prefix):-4]
    model_slug, dataset_slug = stem.rsplit(".", 1)
    return model_slug, dataset_slug


def _display_model(model_slug: str) -> str:
    return MODEL_LABELS.get(model_slug, model_slug)


def _display_dataset(dataset_slug: str, frame: pd.DataFrame) -> str:
    if dataset_slug in DATASET_LABELS:
        return DATASET_LABELS[dataset_slug]
    non_overall = frame[frame["origin"] != "__overall__"]
    if not non_overall.empty:
        return str(non_overall["origin"].iloc[0])
    return dataset_slug


def _label_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _is_non_significant(metric: str, lo: float, hi: float) -> bool:
    null_value = 0.5 if metric == "roc_auc" else 0.0
    return lo <= null_value <= hi


def _fmt_value(row: pd.Series | None, metric: str) -> str:
    if row is None:
        return "PH"
    value = float(row["value"])
    lo = float(row["ci_lo"])
    hi = float(row["ci_hi"])
    suffix = r"\ns" if _is_non_significant(metric, lo, hi) else ""
    return f"{value:+.3f}{suffix}"


def _select_overall(frame: pd.DataFrame) -> pd.DataFrame:
    overall = frame[frame["origin"] == "__overall__"]
    return overall if not overall.empty else frame


def _multirow_lines(
    frame: pd.DataFrame,
    model_slug: str,
    semantic: str,
    aggregations: list[str],
) -> list[str]:
    lines = []
    model_label = _display_model(model_slug)
    semantic_frame = frame[frame["semantic"] == semantic]
    for index, aggregation in enumerate(aggregations):
        prefix = (
            rf"\multirow{{{len(aggregations)}}}{{*}}{{{model_label}}}"
            if index == 0
            else ""
        )
        metric_cells = []
        for metric in DEFAULT_METRICS:
            row = semantic_frame[
                (semantic_frame["aggregation"] == aggregation)
                & (semantic_frame["metric"] == metric)
            ]
            metric_cells.append(_fmt_value(None if row.empty else row.iloc[0], metric))
        lines.append(
            f"{prefix}"
            + ("\n  " if prefix else "  ")
            + f"& {aggregation} & "
            + " & ".join(metric_cells)
            + r" \\"
        )
    return lines


def _render_semantic_block(
    dataset_frames: dict[str, pd.DataFrame],
    model_order: list[str],
    semantic: str,
    aggregations: list[str],
) -> list[str]:
    lines = [
        f"% ============== {semantic.upper()} BLOCK ==============",
        rf"\multicolumn{{6}}{{c}}{{\textbf{{{SEMANTIC_TITLES[semantic]}}}}} \\",
        r"\midrule",
        "",
    ]
    available_models = [m for m in model_order if m in dataset_frames]
    for index, model_slug in enumerate(available_models):
        frame = _select_overall(dataset_frames[model_slug])
        lines.extend(_multirow_lines(frame, model_slug, semantic, aggregations))
        is_last = index == len(available_models) - 1
        lines.append(r"\midrule" if is_last else r"\cmidrule(l){2-6}")
    lines.append("")
    return lines


def _render_table(
    dataset_slug: str,
    dataset_frames: dict[str, pd.DataFrame],
    aggregations: list[str],
    model_order: list[str],
) -> str:
    sample_frame = next(iter(dataset_frames.values()))
    dataset_title = _display_dataset(dataset_slug, sample_frame)
    caption = (
        f"Full results on {dataset_title}. Per-sentence scoring is either hard "
        "binary or soft logit-derived probability. Cells marked with "
        r"\ns{} have a 95\% bootstrap CI that includes the null value "
        "(0 for Pearson, Spearman, Kendall; 0.5 for ROC-AUC)."
    )
    label = f"tab:full-{_label_slug(dataset_slug)}"

    lines = [
        r"\begin{longtable}{@{}llcccc@{}}",
        r"\toprule",
        r"Model & Aggregation & Pearson $r$ & Spearman $\rho$ & Kendall $\tau$ & ROC-AUC \\",
        r"\midrule",
        r"\endfirsthead",
        "",
        r"\multicolumn{6}{l}{\small\textit{Table~\thetable\ continued from previous page.}}\\",
        r"\toprule",
        r"Model & Aggregation & Pearson $r$ & Spearman $\rho$ & Kendall $\tau$ & ROC-AUC \\",
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule",
        r"\multicolumn{6}{r}{\small\textit{(continued on next page)}}\\",
        r"\endfoot",
        "",
        r"\bottomrule",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}\\",
        r"\endlastfoot",
        "",
    ]
    lines.extend(_render_semantic_block(dataset_frames, model_order, "hard", aggregations))
    lines.extend(_render_semantic_block(dataset_frames, model_order, "soft", aggregations))
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    summary_files = _discover_summary_files(results_dir)
    if not summary_files:
        raise FileNotFoundError(f"no summary CSVs found under {results_dir}")

    dataset_to_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for path in summary_files:
        model_slug, dataset_slug = _split_summary_filename(path)
        if args.datasets and dataset_slug not in args.datasets:
            continue
        dataset_to_frames.setdefault(dataset_slug, {})[model_slug] = pd.read_csv(path)

    if not dataset_to_frames:
        wanted = ", ".join(args.datasets or [])
        raise ValueError(f"no matching datasets found for: {wanted}")

    for dataset_slug, dataset_frames in sorted(dataset_to_frames.items()):
        tex = _render_table(
            dataset_slug=dataset_slug,
            dataset_frames=dataset_frames,
            aggregations=args.aggregations,
            model_order=args.model_order,
        )
        out_path = results_dir / f"{args.out_prefix}.{dataset_slug}.tex"
        out_path.write_text(tex)
        print(f"[latex_tables] wrote {out_path}")


if __name__ == "__main__":
    main()