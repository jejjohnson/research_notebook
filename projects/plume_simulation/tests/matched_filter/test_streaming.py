"""Tests for ``plume_simulation.matched_filter.streaming``.

Ensures Welford online estimates match batch numpy estimates exactly, and
that merging two accumulators on disjoint chunks is associative.
"""

from __future__ import annotations

import numpy as np
import pytest
from plume_simulation.matched_filter.streaming import (
    WelfordAccumulator,
    streaming_background,
)


def test_welford_matches_numpy(rng):
    X = rng.standard_normal((500, 8))
    acc = WelfordAccumulator(n_bands=8)
    for chunk in np.array_split(X, 7):
        acc.update(chunk)
    np.testing.assert_allclose(acc.mean(), X.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(
        acc.covariance(ddof=1), np.cov(X, rowvar=False), atol=1e-12
    )


def test_welford_merge_matches_single_pass(rng):
    X = rng.standard_normal((500, 5))
    n_split = 200
    a = WelfordAccumulator(n_bands=5).update(X[:n_split])
    b = WelfordAccumulator(n_bands=5).update(X[n_split:])
    merged = WelfordAccumulator(n_bands=5)
    merged.merge(a).merge(b)
    ref = WelfordAccumulator(n_bands=5).update(X)
    np.testing.assert_allclose(merged.mean(), ref.mean(), atol=1e-12)
    np.testing.assert_allclose(merged.covariance(), ref.covariance(), atol=1e-12)


def test_welford_accepts_cube_shape(rng):
    cube = rng.standard_normal((7, 5, 4))
    acc = WelfordAccumulator(n_bands=4).update(cube)
    flat = cube.reshape(-1, 4)
    np.testing.assert_allclose(acc.mean(), flat.mean(axis=0), atol=1e-12)


def test_welford_rejects_band_mismatch():
    with pytest.raises(ValueError, match="expected shape"):
        WelfordAccumulator(n_bands=4).update(np.zeros((10, 3)))


def test_welford_empty_stats_raise():
    acc = WelfordAccumulator(n_bands=3)
    with pytest.raises(ValueError, match="no data"):
        acc.mean()
    with pytest.raises(ValueError, match="count"):
        acc.covariance()


def test_streaming_background_returns_operator(rng):
    X = rng.standard_normal((400, 5))
    batches = [X[:200], X[200:]]
    mu, cov_op = streaming_background(iter(batches), n_bands=5)
    np.testing.assert_allclose(mu, X.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(cov_op.as_matrix()), np.cov(X, rowvar=False), atol=1e-12
    )


def test_streaming_background_ridge(rng):
    X = rng.standard_normal((400, 5))
    _, cov_op = streaming_background([X], n_bands=5, ridge=0.5)
    M = np.asarray(cov_op.as_matrix())
    np.testing.assert_allclose(M - 0.5 * np.eye(5), np.cov(X, rowvar=False), atol=1e-12)
