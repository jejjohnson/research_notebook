"""Tests for ``plume_simulation.matched_filter.core``.

Covers:
- Unbiasedness of the linear estimator when observations follow the model.
- Equivalence of ``apply_pixel`` and the ``(i,j)``-th entry of ``apply_image``.
- Equivalence of dense and low-rank covariance operators on the same Σ.
- SNR formula consistency with the score variance under N(0, Σ) noise.
- Detection-threshold monotonicity and identity-covariance shortcut.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from plume_simulation.matched_filter.background import (
    estimate_cov_empirical,
)
from plume_simulation.matched_filter.core import (
    apply_image,
    apply_pixel,
    detection_threshold,
    matched_filter_snr,
)


jax.config.update("jax_enable_x64", True)


def test_apply_image_unbiased_noiseless(hyperspectral_scene, rng):
    """With the true (μ, Σ, t) and no noise, the MF recovers α exactly."""
    _, amp_map, target = hyperspectral_scene
    H, W, B = amp_map.shape + (target.size,)
    mu = np.ones(B)
    cube = mu + amp_map[..., None] * target
    cube_jax = jnp.asarray(cube)
    cov_op = lx.DiagonalLinearOperator(jnp.ones(B))  # identity Σ
    scores = apply_image(
        cube_jax, mean=jnp.asarray(mu), cov_op=cov_op, target=jnp.asarray(target)
    )
    np.testing.assert_allclose(np.asarray(scores), amp_map, atol=1e-10)


def test_apply_pixel_matches_apply_image(hyperspectral_scene):
    """``apply_pixel(cube[i,j])`` == ``apply_image(cube)[i, j]``."""
    cube, _, target = hyperspectral_scene
    cube_jax = jnp.asarray(cube)
    cov = np.cov(cube.reshape(-1, cube.shape[-1]), rowvar=False)
    cov_op = lx.MatrixLinearOperator(
        jnp.asarray(cov),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    mu = jnp.asarray(cube.reshape(-1, cube.shape[-1]).mean(axis=0))
    tgt = jnp.asarray(target)
    scores = apply_image(cube_jax, mean=mu, cov_op=cov_op, target=tgt)
    for i, j in [(0, 0), (5, 7), (10, 11), (19, 23)]:
        s_pixel = apply_pixel(cube_jax[i, j], mean=mu, cov_op=cov_op, target=tgt)
        np.testing.assert_allclose(float(s_pixel), float(scores[i, j]), rtol=1e-10)


def test_dense_and_lowrank_agree_on_same_cov(rng):
    """Explicit ``Σ = λI + VDVᵀ`` → dense and LowRankUpdate give the same score."""
    B, k = 12, 3
    V = rng.standard_normal((B, k))
    d = np.array([2.0, 1.0, 0.5])
    tikhonov = 0.1
    cov_dense_mat = tikhonov * np.eye(B) + V @ np.diag(d) @ V.T
    cov_dense = lx.MatrixLinearOperator(
        jnp.asarray(cov_dense_mat),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    import gaussx as gx

    cov_lr = gx.LowRankUpdate(
        lx.DiagonalLinearOperator(jnp.asarray(tikhonov * np.ones(B))),
        jnp.asarray(V),
        jnp.asarray(d),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    target = jnp.asarray(rng.standard_normal(B))
    pixel = jnp.asarray(rng.standard_normal(B))
    mean = jnp.zeros(B)
    s_d = apply_pixel(pixel, mean=mean, cov_op=cov_dense, target=target)
    s_l = apply_pixel(pixel, mean=mean, cov_op=cov_lr, target=target)
    np.testing.assert_allclose(float(s_d), float(s_l), rtol=1e-8)


def test_snr_matches_score_variance(rng):
    """Empirical std of the score under N(0, Σ) noise matches theory."""
    B = 8
    Sigma = rng.standard_normal((B, B))
    Sigma = Sigma @ Sigma.T + 0.1 * np.eye(B)
    target = rng.standard_normal(B)
    cov_op = lx.MatrixLinearOperator(
        jnp.asarray(Sigma),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    L = np.linalg.cholesky(Sigma)
    n_draws = 40_000
    eps = rng.standard_normal((n_draws, B)) @ L.T
    scores = np.asarray(
        jax.vmap(
            lambda x: apply_pixel(
                x, mean=jnp.zeros(B), cov_op=cov_op, target=jnp.asarray(target)
            )
        )(jnp.asarray(eps))
    )
    theory_std = 1.0 / float(matched_filter_snr(1.0, cov_op, jnp.asarray(target)))
    np.testing.assert_allclose(scores.std(), theory_std, rtol=0.03)
    # SNR at α = theory_std should be exactly 1.
    snr_unit = matched_filter_snr(theory_std, cov_op, jnp.asarray(target))
    np.testing.assert_allclose(float(snr_unit), 1.0, rtol=1e-10)


def test_detection_threshold_identity_cov():
    """Σ = I, ‖t‖=1 → threshold(FAR) == Φ⁻¹(1−FAR)."""
    B = 5
    target = jnp.zeros(B).at[0].set(1.0)  # unit norm
    cov_op = lx.DiagonalLinearOperator(jnp.ones(B))
    import jax.scipy.special as jsp

    for far in (1e-2, 1e-4, 1e-6):
        thr = detection_threshold(far, cov_op, target)
        expected = float(jsp.ndtri(1.0 - far))
        np.testing.assert_allclose(float(thr), expected, rtol=1e-10)


def test_detection_threshold_is_monotone():
    """Lower FAR → higher detection threshold."""
    B = 5
    target = jnp.ones(B)
    cov_op = lx.DiagonalLinearOperator(jnp.ones(B))
    fars = [1e-2, 1e-4, 1e-6, 1e-8]
    thrs = [float(detection_threshold(f, cov_op, target)) for f in fars]
    assert all(earlier < later for earlier, later in zip(thrs, thrs[1:], strict=False))


def test_detection_threshold_rejects_out_of_range():
    target = jnp.ones(3)
    cov_op = lx.DiagonalLinearOperator(jnp.ones(3))
    with pytest.raises(ValueError, match="false_alarm_rate"):
        detection_threshold(0.0, cov_op, target)
    with pytest.raises(ValueError, match="false_alarm_rate"):
        detection_threshold(1.0, cov_op, target)


def test_apply_image_runs_under_jit(hyperspectral_scene):
    """The MF kernel should be jit-compilable (core promise of the redesign)."""
    cube, _, target = hyperspectral_scene
    cube_jax = jnp.asarray(cube)
    cov_op = estimate_cov_empirical(cube, ridge=1e-6)
    mu = jnp.asarray(cube.reshape(-1, cube.shape[-1]).mean(axis=0))
    tgt = jnp.asarray(target)
    f = jax.jit(lambda c: apply_image(c, mean=mu, cov_op=cov_op, target=tgt))
    out = f(cube_jax)
    out2 = f(cube_jax)  # second call shouldn't re-trace
    np.testing.assert_allclose(np.asarray(out), np.asarray(out2))
