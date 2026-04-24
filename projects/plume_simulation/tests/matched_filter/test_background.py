"""Tests for ``plume_simulation.matched_filter.background``.

Covers mean robustness, covariance estimator identities, and the low-rank +
Tikhonov Woodbury operator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.matched_filter.background import (
    estimate_cov_empirical,
    estimate_cov_lowrank,
    estimate_cov_shrunk,
    estimate_mean,
)


jax.config.update("jax_enable_x64", True)


def _noisy_cube(rng, H=20, W=20, B=12):
    return 1.0 + rng.standard_normal((H, W, B)) * 0.05


def test_estimate_mean_methods_agree_without_outliers(rng):
    """Without outliers, mean/median/trimmed/huber all land near the true mean."""
    cube = _noisy_cube(rng)
    for method in ("mean", "median", "trimmed", "huber"):
        mu = estimate_mean(cube, method=method)
        np.testing.assert_allclose(mu, 1.0, atol=1e-2)


def test_estimate_mean_robust_to_outliers(rng):
    """Heavy contamination: robust methods stay close, plain mean drifts."""
    cube = _noisy_cube(rng)
    H, W, B = cube.shape
    # 10% of pixels get a huge spike in band 0.
    mask = rng.random((H, W)) < 0.1
    cube[mask, 0] += 10.0
    mu_plain = estimate_mean(cube, method="mean")
    mu_med = estimate_mean(cube, method="median")
    mu_trim = estimate_mean(cube, method="trimmed", trim_proportion=0.15)
    mu_hub = estimate_mean(cube, method="huber")
    assert abs(mu_plain[0] - 1.0) > 0.5, "plain mean should be biased by outliers"
    for mu in (mu_med, mu_trim, mu_hub):
        assert abs(mu[0] - 1.0) < 0.2, f"robust method drifted: {mu[0]}"


def test_estimate_mean_rejects_bad_shape():
    with pytest.raises(ValueError, match="must be"):
        estimate_mean(np.zeros((3, 4)), method="mean")  # 2-D: missing band axis


def test_empirical_covariance_matches_numpy(rng):
    cube = _noisy_cube(rng)
    mu = estimate_mean(cube, method="mean")
    cov_op = estimate_cov_empirical(cube, mean=mu)
    X = cube.reshape(-1, cube.shape[-1])
    Xc = X - mu
    cov_ref = (Xc.T @ Xc) / X.shape[0]
    cov_mat = np.asarray(cov_op.as_matrix())
    np.testing.assert_allclose(cov_mat, cov_ref, atol=1e-10)


def test_shrunk_covariance_is_pd(rng):
    """LedoitWolf with ``n_samples > n_bands`` → Σ̂ PD."""
    cube = _noisy_cube(rng)
    cov_op = estimate_cov_shrunk(cube, method="ledoit_wolf")
    cov_mat = np.asarray(cov_op.as_matrix())
    eigs = np.linalg.eigvalsh(cov_mat)
    assert eigs.min() > 0.0, f"shrunk covariance not PD (min eig {eigs.min()})"


def test_shrunk_covariance_handles_small_n(rng):
    """n_samples < n_bands → empirical cov is singular, shrinkage still PD."""
    X = rng.standard_normal((4, 20))  # 4 samples, 20 bands — rank 4 empirical
    cube = X.reshape(4, 1, 20)
    cov_op = estimate_cov_shrunk(cube, method="ledoit_wolf")
    cov_mat = np.asarray(cov_op.as_matrix())
    eigs = np.linalg.eigvalsh(cov_mat)
    assert eigs.min() > 0.0


def test_lowrank_covariance_recovers_structure(rng):
    """Injected rank-k signal + isotropic noise → Truncated-SVD recovers top k."""
    B, k = 20, 3
    n = 200
    V_true = np.linalg.qr(rng.standard_normal((B, k)))[0]
    s_true = np.array([3.0, 2.0, 1.0])
    Xc = rng.standard_normal((n, k)) * s_true @ V_true.T + 0.05 * rng.standard_normal(
        (n, B)
    )
    cube = Xc.reshape(n, 1, B)
    cov_op = estimate_cov_lowrank(cube, rank=k, tikhonov=1e-3, random_state=0)
    # Woodbury solve against an arbitrary RHS must match the dense equivalent.
    import gaussx as gx

    rhs = jnp.asarray(rng.standard_normal(B))
    x_woodbury = np.asarray(gx.solve(cov_op, rhs))
    dense_mat = np.asarray(cov_op.as_matrix())
    x_dense = np.linalg.solve(dense_mat, np.asarray(rhs))
    np.testing.assert_allclose(x_woodbury, x_dense, atol=1e-6)


def test_lowrank_covariance_rejects_nonpositive_tikhonov(rng):
    cube = _noisy_cube(rng)
    with pytest.raises(ValueError, match="tikhonov"):
        estimate_cov_lowrank(cube, rank=2, tikhonov=0.0)


def test_lowrank_covariance_is_symmetric(rng):
    cube = _noisy_cube(rng)
    cov_op = estimate_cov_lowrank(cube, rank=3, tikhonov=1e-3)
    M = np.asarray(cov_op.as_matrix())
    np.testing.assert_allclose(M, M.T, atol=1e-12)
