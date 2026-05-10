#!/usr/bin/env python
"""Plot grouped bar charts from meta-eval summary CSV files.

Default figure: one wide row of dataset panels showing Pearson correlation for
the `min` and `mean` aggregations under the hard sentence-label semantic.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_DATASETS = ["aggrefact_cnn", "aggrefact_other_multi", "diversumm", "halueval"]
DEFAULT_MODELS = [
    "Qwen_Qwen3-4B",
    "Qwen_Qwen3-8B",
    "Qwen_Qwen3-32B",
    "meta-llama_Llama-3.1-8B-Instruct",
    "allenai_Olmo-3-7B-Instruct",
]
DEFAULT_AGGREGATIONS = ["min", "mean"]

MODEL_LABELS = {
    "Qwen_Qwen3-4B": "Qwen3-4B",
    "Qwen_Qwen3-8B": "Qwen3-8B",
    "Qwen_Qwen3-32B": "Qwen3-32B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "allenai_Olmo-3-7B-Instruct": "OLMo-3-7B",
}

DATASET_TITLES = {
    "aggrefact_cnn": "AggreFact-CNN",
    "aggrefact_other_multi": "AggreFact-other",
    "diversumm": "DiverSumm",
    "halueval": "HaluEval",
}

AGGREGATION_COLORS = {
    "min": "#4C72B0",
    "mean": "#DD8452",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--aggregations", nargs="+", default=DEFAULT_AGGREGATIONS)
    parser.add_argument("--metric", default="pearson")
    parser.add_argument("--semantic", default="hard")
    parser.add_argument("--origin", default="__overall__")
    parser.add_argument("--ncols", type=int, default=None,
                        help="Number of subplot columns. Default: all datasets in one row.")
    parser.add_argument("--fig-width", type=float, default=13.5)
    parser.add_argument("--fig-height", type=float, default=3.6)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--out",
        default=f"results/figs/meta_eval_bars_pearson_{parser.get_default('semantic')}.png",
        help="Output image path.",
    )
    args = parser.parse_args()
    if args.out == f"results/figs/meta_eval_bars_pearson_{parser.get_default('semantic')}.png":
        args.out = f"results/figs/meta_eval_bars_pearson_{args.semantic}.png"
    return args


def _summary_path(results_dir: Path, model_slug: str, dataset_slug: str) -> Path:
    return results_dir / f"meta_eval_summary.{model_slug}.{dataset_slug}.csv"


def _display_model(model_slug: str) -> str:
    return MODEL_LABELS.get(model_slug, model_slug)


def _display_dataset(dataset_slug: str) -> str:
    return DATASET_TITLES.get(dataset_slug, dataset_slug)


def _load_summary_row(
    results_dir: Path,
    model_slug: str,
    dataset_slug: str,
    semantic: str,
    aggregation: str,
    metric: str,
    origin: str,
) -> pd.Series | None:
    path = _summary_path(results_dir, model_slug, dataset_slug)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    row = frame[
        (frame["origin"] == origin)
        & (frame["semantic"] == semantic)
        & (frame["aggregation"] == aggregation)
        & (frame["metric"] == metric)
    ]
    if row.empty:
        return None
    return row.iloc[0]


def _plot_dataset_panel(
    ax,
    results_dir: Path,
    dataset_slug: str,
    model_slugs: list[str],
    aggregations: list[str],
    semantic: str,
    metric: str,
    origin: str,
) -> None:
    width = 0.35 if len(aggregations) == 2 else 0.8 / max(len(aggregations), 1)
    x = list(range(len(model_slugs)))
    offsets = [
        (idx - (len(aggregations) - 1) / 2.0) * width
        for idx in range(len(aggregations))
    ]

    ymax = 0.0
    for offset, aggregation in zip(offsets, aggregations):
        rows = [
            _load_summary_row(results_dir, model_slug, dataset_slug, semantic, aggregation, metric, origin)
            for model_slug in model_slugs
        ]
        values = [None if row is None else float(row["value"]) for row in rows]
        plot_values = [float("nan") if value is None else value for value in values]
        lower_errors = [
            0.0 if row is None else max(0.0, float(row["value"]) - float(row["ci_lo"]))
            for row in rows
        ]
        upper_errors = [
            0.0 if row is None else max(0.0, float(row["ci_hi"]) - float(row["value"]))
            for row in rows
        ]
        bars = ax.bar(
            [xi + offset for xi in x],
            plot_values,
            width=width,
            label=aggregation,
            color=AGGREGATION_COLORS.get(aggregation),
            yerr=[lower_errors, upper_errors],
            ecolor="#2F2F2F",
            capsize=3,
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
        )
        for bar, value, upper_error in zip(bars, values, upper_errors):
            if value is None:
                continue
            ymax = max(ymax, value + upper_error)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + upper_error + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_title(f"{metric.capitalize()} Correlation on {_display_dataset(dataset_slug)}", fontsize=16)
    ax.set_ylabel(metric.capitalize(), fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([_display_model(model_slug) for model_slug in model_slugs], rotation=25, ha="right", fontsize=14)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_ylim(0.0, max(0.1, ymax * 1.18))


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    out = Path(args.out)
    datasets = args.datasets
    ncols = args.ncols or len(datasets)
    nrows = math.ceil(len(datasets) / ncols)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(args.fig_width, args.fig_width * 0.618), squeeze=False)
    flat_axes = list(axes.flat)

    for ax, dataset_slug in zip(flat_axes, datasets):
        _plot_dataset_panel(
            ax=ax,
            results_dir=results_dir,
            dataset_slug=dataset_slug,
            model_slugs=args.models,
            aggregations=args.aggregations,
            semantic=args.semantic,
            metric=args.metric,
            origin=args.origin,
        )

    for ax in flat_axes[len(datasets):]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=16)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"[plot_meta_eval_bars] wrote {out}")


if __name__ == "__main__":
    main()