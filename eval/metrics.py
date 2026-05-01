"""Correlation / classification metrics + bootstrap CI for meta-evaluation.

All metric functions take two equal-length 1-D arrays and return a single
float. NaNs in either array are dropped pairwise. If the inputs are constant
(zero variance) the correlation metrics return ``np.nan`` rather than raising.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _clean(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson(x, y) -> float:
    x, y = _clean(x, y)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.pearsonr(x, y).statistic)


def spearman(x, y) -> float:
    x, y = _clean(x, y)
    if x.size < 2:
        return float("nan")
    r = stats.spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def kendall(x, y) -> float:
    x, y = _clean(x, y)
    if x.size < 2:
        return float("nan")
    r = stats.kendalltau(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def roc_auc(scores, labels) -> float:
    """ROC-AUC with ``scores`` as the predictor and binary ``labels`` as truth.

    Returns NaN if labels are not binary or only one class is present.
    """
    s, y = _clean(scores, labels)
    uniq = np.unique(y)
    if uniq.size != 2 or not set(uniq.tolist()) <= {0.0, 1.0}:
        return float("nan")
    return float(roc_auc_score(y, s))


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    x: np.ndarray,
    y: np.ndarray,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Returns (point_estimate, lo, hi) for a (1-alpha)*100% percentile CI.

    Resamples paired ``(x, y)`` with replacement ``n`` times. NaN bootstrap
    replicates (e.g. constant resample) are dropped before computing
    percentiles. If everything is NaN, returns (nan, nan, nan).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    point = metric_fn(x, y)
    rng = np.random.default_rng(seed)
    nsamp = len(x)
    if nsamp == 0:
        return point, float("nan"), float("nan")
    reps = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.choice(nsamp, size=nsamp, replace=True)
        reps[i] = metric_fn(x[idx], y[idx])
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(reps, alpha / 2))
    hi = float(np.quantile(reps, 1 - alpha / 2))
    return float(point), lo, hi


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "pearson": pearson,
    "spearman": spearman,
    "kendall": kendall,
    "roc_auc": roc_auc,
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = x + 0.1 * rng.normal(size=100)
    assert pearson(x, y) > 0.95
    assert spearman(x, y) > 0.9
    assert kendall(x, y) > 0.7
    labels = (x > 0).astype(float)
    assert roc_auc(x, labels) > 0.95
    pt, lo, hi = bootstrap_ci(pearson, x, y, n=200, seed=1)
    assert lo <= pt <= hi
    print("eval/metrics.py: all self-tests passed.")
