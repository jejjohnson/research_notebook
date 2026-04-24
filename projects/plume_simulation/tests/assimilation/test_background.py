"""Background covariance constructor tests.

Each ``B`` operator must:
1. Be PD (gaussx.solve returns finite values).
2. Round-trip via Cholesky: ``U Uᵀ x ≈ B x``.
3. Match a dense reference for small problems (sanity check on structure).
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from plume_simulation.assimilation.background import (
    build_diagonal_background,
    build_kronecker_background,
    build_lowrank_background,
)


jax.config.update("jax_enable_x64", True)


# ── diagonal ─────────────────────────────────────────────────────────────────


def test_diagonal_solve_is_per_element_division():
    B = build_diagonal_background(0.5, n_pixels=4)
    r = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    x = gx.solve(B, r)
    np.testing.assert_allclose(np.asarray(x), np.asarray(r) / 0.5, atol=1e-12)


def test_diagonal_rejects_negative_variance():
    with pytest.raises(ValueError, match="> 0"):
        build_diagonal_background(-1.0, n_pixels=3)


# ── Kronecker ────────────────────────────────────────────────────────────────


def test_kronecker_round_trip_via_cholesky():
    B = build_kronecker_background(
        ny=3, nx=3, variance=0.04, length_scale_y=2.0, length_scale_x=2.0,
    )
    U = gx.cholesky(B)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(9))
    # B @ x  vs  U (Uᵀ x)
    Bx = B.mv(x)
    UUtx = U.mv(U.transpose().mv(x))
    np.testing.assert_allclose(np.asarray(Bx), np.asarray(UUtx), atol=1e-9)


def test_kronecker_solve_matches_dense():
    """For a tiny problem, gaussx.solve should match np.linalg.solve."""
    ny, nx = 2, 3
    var, ly, lx_scale = 0.01, 1.5, 2.0
    B = build_kronecker_background(
        ny=ny, nx=nx, variance=var, length_scale_y=ly, length_scale_x=lx_scale,
    )
    # Reconstruct dense B by applying to identity columns.
    n = ny * nx
    Bdense = np.stack([np.asarray(B.mv(jnp.eye(n)[i])) for i in range(n)], axis=1)
    rng = np.random.default_rng(1)
    r = rng.standard_normal(n)
    expected = np.linalg.solve(Bdense, r)
    actual = np.asarray(gx.solve(B, jnp.asarray(r)))
    np.testing.assert_allclose(actual, expected, atol=1e-8)


# ── low-rank update ──────────────────────────────────────────────────────────


def test_lowrank_solve_runs():
    rng = np.random.default_rng(2)
    samples = rng.standard_normal((20, 8))
    B = build_lowrank_background(samples=samples, rank=3, regularization=1e-3)
    r = jnp.asarray(rng.standard_normal(8))
    x = gx.solve(B, r)
    np.testing.assert_allclose(np.asarray(B.mv(x)), np.asarray(r), atol=1e-7)


def test_lowrank_rejects_zero_regularization():
    samples = np.random.default_rng(3).standard_normal((10, 5))
    with pytest.raises(ValueError, match="regularization"):
        build_lowrank_background(samples=samples, rank=2, regularization=0.0)
