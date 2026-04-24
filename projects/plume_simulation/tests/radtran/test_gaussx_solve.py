"""Gaussx-backed operator matched filter — functional + agreement with numpy."""

from __future__ import annotations

import numpy as np
import pytest

from plume_simulation.radtran.background import robust_lowrank_covariance
from plume_simulation.radtran.gaussx_solve import (
    build_lowrank_covariance_operator,
    matched_filter_image_op,
    matched_filter_pixel_op,
    matched_filter_snr_op,
)
from plume_simulation.radtran.matched_filter import (
    matched_filter_image,
    matched_filter_pixel,
    matched_filter_snr,
)


def _synthetic_scene(
    n_bands: int = 4,
    shape: tuple[int, int] = (16, 16),
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    loc = np.linspace(0.3, 0.5, n_bands)
    base = rng.normal(loc=loc, scale=0.02, size=(*shape, n_bands))
    return np.moveaxis(base, -1, 0)


def test_build_lowrank_covariance_operator_has_correct_shape_and_mu():
    scene = _synthetic_scene(n_bands=4, shape=(16, 16))
    cov, mu = build_lowrank_covariance_operator(scene, rank=3)
    assert cov.in_size() == 4 and cov.out_size() == 4
    assert mu.shape == (4,)


def test_operator_matches_dense_covariance_on_a_test_vector():
    """``cov · v`` should match ``Σ · v`` from the dense estimator."""
    scene = _synthetic_scene(n_bands=4, shape=(16, 16))
    cov, _ = build_lowrank_covariance_operator(
        scene, rank=3, regularization=1e-6,
    )
    Sigma, _ = robust_lowrank_covariance(
        scene, rank=3, regularization=1e-6,
    )
    v = np.arange(4, dtype=float)
    import jax.numpy as jnp
    mv = np.asarray(cov.mv(jnp.asarray(v)))
    np.testing.assert_allclose(mv, Sigma @ v, atol=1e-10)


def test_pixel_op_agrees_with_dense_pixel_path():
    """Operator-backed and dense matched filter agree pixel-for-pixel."""
    scene = _synthetic_scene(n_bands=5, shape=(16, 16))
    cov, mu = build_lowrank_covariance_operator(scene, rank=3)
    Sigma, Sigma_inv = robust_lowrank_covariance(scene, rank=3)
    # Build μ identically so the two paths use the same background.
    # (robust_lowrank_covariance also uses `trimmed_mean_spectrum` internally.)
    target = np.array([-0.05, -0.02, -0.1, -0.03, -0.08])
    rng = np.random.default_rng(1)
    for _ in range(5):
        pixel = mu + 0.3 * target + rng.normal(scale=0.01, size=mu.shape)
        a_dense = matched_filter_pixel(pixel, mu, Sigma_inv, target)
        a_op = matched_filter_pixel_op(pixel, mu, cov, target)
        assert a_op == pytest.approx(a_dense, rel=1e-3, abs=1e-4)


def test_image_op_agrees_with_dense_image_path():
    scene = _synthetic_scene(n_bands=5, shape=(12, 12))
    cov, mu = build_lowrank_covariance_operator(scene, rank=3)
    _, Sigma_inv = robust_lowrank_covariance(scene, rank=3)

    # Plant a half-strength target at one pixel.
    target = np.array([-0.05, -0.02, -0.1, -0.03, -0.08])
    scene[:, 6, 6] = mu + 0.5 * target

    eps_dense = matched_filter_image(scene, mu, Sigma_inv, target)
    eps_op = matched_filter_image_op(scene, mu, cov, target)
    assert eps_op.shape == eps_dense.shape
    np.testing.assert_allclose(eps_op, eps_dense, rtol=1e-3, atol=1e-4)


def test_snr_op_matches_dense_snr():
    scene = _synthetic_scene(n_bands=5, shape=(10, 10))
    cov, _ = build_lowrank_covariance_operator(scene, rank=3)
    _, Sigma_inv = robust_lowrank_covariance(scene, rank=3)
    target = np.array([-0.05, -0.02, -0.1, -0.03, -0.08])
    abundance = np.array([[0.1, 0.5], [1.0, 2.0]])
    snr_op = matched_filter_snr_op(abundance, cov, target)
    snr_dense = matched_filter_snr(abundance, Sigma_inv, target)
    np.testing.assert_allclose(snr_op, snr_dense, rtol=1e-4, atol=1e-6)


def test_image_op_rejects_band_mismatch():
    scene = _synthetic_scene(n_bands=4)
    cov, mu = build_lowrank_covariance_operator(scene, rank=2)
    target = np.zeros(5)  # wrong size
    with pytest.raises(ValueError, match="bands but target"):
        matched_filter_image_op(scene, mu, cov, target)


def test_pixel_op_rejects_degenerate_target():
    scene = _synthetic_scene(n_bands=4)
    cov, mu = build_lowrank_covariance_operator(scene, rank=2)
    with pytest.raises(ValueError, match="tᵀ Σ⁻¹ t"):
        matched_filter_pixel_op(mu, mu, cov, np.zeros(4))


def test_tiny_scene_rejected():
    with pytest.raises(ValueError, match="≥ 2 pixels"):
        build_lowrank_covariance_operator(np.array([[[0.3]]], dtype=float))


def test_tight_trim_frac_rejected_when_bands_exceed_kept_pixels():
    scene = _synthetic_scene(n_bands=40, shape=(8, 8))  # 64 pixels, 40 bands
    with pytest.raises(ValueError, match="fewer pixels than bands"):
        build_lowrank_covariance_operator(scene, rank=8, trim_frac=0.45)


def test_rank_default_is_reasonable():
    scene = _synthetic_scene(n_bands=6, shape=(16, 16))
    cov, _ = build_lowrank_covariance_operator(scene)
    # Operator has a rank-5 update plus λI → the product Σv should still be
    # full-rank expressible (λ > 0). Check with a non-zero vector.
    import jax.numpy as jnp
    v = jnp.ones(6)
    mv = np.asarray(cov.mv(v))
    assert np.all(np.isfinite(mv))
