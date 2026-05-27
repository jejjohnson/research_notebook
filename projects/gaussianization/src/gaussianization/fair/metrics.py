"""Plain-numpy fairness diagnostics used at evaluation time.

These are *evaluation* statistics — they do not back-prop and never see
the training loop. The training-time penalty lives in
:mod:`gaussianization.fair.losses`.
"""

from __future__ import annotations

import numpy as np


def demographic_parity_difference(
    y_pred: np.ndarray, q: np.ndarray, threshold: float = 0.5
) -> float:
    r"""``|P(ŷ = 1 | q = 1) - P(ŷ = 1 | q = 0)|`` for binary ``q``.

    Args:
        y_pred: Predicted scores or labels of shape ``(n,)``. Scores
            are thresholded at ``threshold``; integer labels in
            ``{0, 1}`` pass through unchanged.
        q: Binary sensitive attribute of shape ``(n,)``.
        threshold: Decision threshold for soft scores.
    """
    yhat = _to_binary(y_pred, threshold)
    q = np.asarray(q).astype(int).ravel()
    mask1 = q == 1
    mask0 = q == 0
    if mask1.sum() == 0 or mask0.sum() == 0:
        return float("nan")
    return float(abs(yhat[mask1].mean() - yhat[mask0].mean()))


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Max of TPR-gap and FPR-gap across the two values of ``q``.

    Following the standard definition (Hardt et al. 2016), this is

        max( |TPR_{q=1} - TPR_{q=0}|, |FPR_{q=1} - FPR_{q=0}| ).
    """
    yhat = _to_binary(y_pred, threshold)
    y = np.asarray(y_true).astype(int).ravel()
    q = np.asarray(q).astype(int).ravel()

    def rate(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return float("nan")
        return float(yhat[mask].mean())

    tpr1 = rate((q == 1) & (y == 1))
    tpr0 = rate((q == 0) & (y == 1))
    fpr1 = rate((q == 1) & (y == 0))
    fpr0 = rate((q == 0) & (y == 0))
    return float(max(abs(tpr1 - tpr0), abs(fpr1 - fpr0)))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation between two 1-D arrays."""
    a = np.asarray(a).ravel().astype(float)
    b = np.asarray(b).ravel().astype(float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def _to_binary(scores: np.ndarray, threshold: float) -> np.ndarray:
    s = np.asarray(scores).ravel()
    if s.dtype.kind in "iu" and set(np.unique(s)).issubset({0, 1}):
        return s.astype(int)
    return (s >= threshold).astype(int)
