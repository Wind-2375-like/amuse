"""Sentence-level -> summary-level aggregation functions.

Each function has signature
    f(scores: np.ndarray) -> float
where ``scores`` is the 1-D vector of sentence-level scores for a single
summary. Inputs may be either:

  (a) hard labels in {0, 1} (the ``faithful`` column), or
  (b) soft probabilities P(faithful=1) in [0, 1]
      (e.g. ``faithful * confidence + (1 - faithful) * (1 - confidence)``).

``softmin_agg`` and ``prob_all_faithful`` only make sense for (b).
The dispatch table ``AGGREGATIONS`` lists the canonical (name, callable)
pairs used by ``eval/run_meta_eval.py``.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Core aggregation functions
# ---------------------------------------------------------------------------

def _as_array(s) -> np.ndarray:
    a = np.asarray(s, dtype=float).ravel()
    if a.size == 0:
        # An empty summary is an edge case (e.g. spaCy split returned nothing).
        # Treat as fully faithful: vacuously no unfaithful sentence found.
        return np.array([1.0])
    return a


def min_agg(s) -> float:
    """Worst-sentence rule: a summary is only as faithful as its weakest sentence."""
    return float(np.min(_as_array(s)))


def mean_agg(s) -> float:
    """Average sentence-level faithfulness."""
    return float(np.mean(_as_array(s)))


def max_agg(s) -> float:
    """Best-sentence rule. Mostly a sanity baseline (we expect it to be bad)."""
    return float(np.max(_as_array(s)))


def softmin_agg(s, tau: float = 0.5) -> float:
    """Smooth approximation to ``min``.

        softmin_tau(s) = -tau * logsumexp(-s / tau)

    As tau -> 0 this approaches ``min``; as tau -> inf it approaches the mean
    (shifted by a log-N constant). We use ``scipy.special.logsumexp`` for
    numerical stability.
    """
    a = _as_array(s)
    return float(-tau * logsumexp(-a / tau))


def trimmed_mean_agg(s, k: float = 0.2) -> float:
    """Mean after dropping the lowest ``k`` fraction of sentences.

    This is *not* a symmetric trimmed mean — we only trim the bottom, since
    in faithfulness the worst sentences are the informative ones we may want
    to discount as label noise.
    """
    a = _as_array(s)
    n = a.size
    drop = int(np.floor(k * n))
    if drop >= n:
        # Asked to drop everything; fall back to the single highest score.
        return float(np.max(a))
    a_sorted = np.sort(a)              # ascending
    return float(np.mean(a_sorted[drop:]))


def prob_all_faithful(s) -> float:
    """Joint probability that *every* sentence is faithful, assuming
    independence:  prod_i p_i = exp(sum_i log p_i).

    Inputs are clipped to [1e-6, 1] so a single hard 0 doesn't permanently
    zero out the score (it would also kill any gradient information).
    """
    a = np.clip(_as_array(s), 1e-6, 1.0)
    return float(np.exp(np.sum(np.log(a))))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Each entry is (display_name, callable, semantics) where semantics is:
#   "any"  -> works on hard or soft inputs
#   "soft" -> only meaningful with soft probabilities
AGGREGATIONS: Dict[str, Callable[[np.ndarray], float]] = {
    "min":            min_agg,
    "mean":           mean_agg,
    "max":            max_agg,
    "trimmed_mean@0.2": lambda s: trimmed_mean_agg(s, k=0.2),
    "softmin@tau=0.1":  lambda s: softmin_agg(s, tau=0.1),
    "softmin@tau=0.5":  lambda s: softmin_agg(s, tau=0.5),
    "softmin@tau=1.0":  lambda s: softmin_agg(s, tau=1.0),
    "prob_all_faithful": prob_all_faithful,
}

# Which aggregations require *soft* (probability) inputs. The driver script
# will only pair the others with hard inputs.
SOFT_ONLY = {"softmin@tau=0.1", "softmin@tau=0.5", "softmin@tau=1.0", "prob_all_faithful"}


# ---------------------------------------------------------------------------
# Tiny self-tests (run this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Hand-checked numbers.
    s_hard = np.array([1, 1, 0, 1])
    assert min_agg(s_hard) == 0.0
    assert max_agg(s_hard) == 1.0
    assert abs(mean_agg(s_hard) - 0.75) < 1e-12

    s_soft = np.array([0.9, 0.8, 0.2, 0.95])
    # trimmed_mean@0.2 on n=4 drops floor(0.8) = 0 sentences -> equals mean.
    assert abs(trimmed_mean_agg(s_soft, k=0.2) - s_soft.mean()) < 1e-12
    # On n=5 we drop the single lowest one.
    s5 = np.array([0.1, 0.5, 0.5, 0.9, 1.0])
    assert abs(trimmed_mean_agg(s5, k=0.2) - np.mean([0.5, 0.5, 0.9, 1.0])) < 1e-12

    # softmin sanity: as tau -> 0 it -> min.
    assert abs(softmin_agg(s_soft, tau=1e-3) - s_soft.min()) < 1e-2
    # And softmin is always <= min (LogSumExp >= max -> -tau*LSE(-s/tau) <= min(s)).
    for tau in (0.1, 0.5, 1.0):
        assert softmin_agg(s_soft, tau=tau) <= s_soft.min() + 1e-12, tau

    # prob_all_faithful: 0.9 * 0.8 * 0.2 * 0.95
    expected = 0.9 * 0.8 * 0.2 * 0.95
    assert abs(prob_all_faithful(s_soft) - expected) < 1e-12

    # Empty input -> 1.0 by convention.
    assert mean_agg(np.array([])) == 1.0

    print("aggregation/methods.py: all self-tests passed.")
