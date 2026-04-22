"""Tests for the per-dim mixture-of-Gaussians primitives."""

from __future__ import annotations

import numpy as np
from keras import ops

from gaussianization.gauss_keras.mixtures import MixtureOfGaussians


def _to_numpy(x):
    return np.asarray(ops.convert_to_numpy(x))


def _make_mog(seed=0, d=2, k=4):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((d, k)).astype(np.float32)
    means = rng.standard_normal((d, k)).astype(np.float32)
    log_scales = (0.2 * rng.standard_normal((d, k)) - 0.5).astype(np.float32)
    return MixtureOfGaussians(
        logits=ops.convert_to_tensor(logits),
        means=ops.convert_to_tensor(means),
        log_scales=ops.convert_to_tensor(log_scales),
    )


def test_cdf_monotone_per_dim():
    mog = _make_mog()
    x = np.tile(np.linspace(-6.0, 6.0, 201)[:, None], (1, 2)).astype(np.float32)
    cdf = _to_numpy(mog.cdf(ops.convert_to_tensor(x)))
    # Each dim's CDF, as a function of x, must be monotone non-decreasing.
    assert np.all(np.diff(cdf[:, 0]) >= -1e-6)
    assert np.all(np.diff(cdf[:, 1]) >= -1e-6)


def test_cdf_tails_approach_limits():
    mog = _make_mog()
    x_low = np.full((1, 2), -30.0, dtype=np.float32)
    x_high = np.full((1, 2), 30.0, dtype=np.float32)
    np.testing.assert_allclose(
        _to_numpy(mog.cdf(ops.convert_to_tensor(x_low))), 0.0, atol=1e-5
    )
    np.testing.assert_allclose(
        _to_numpy(mog.cdf(ops.convert_to_tensor(x_high))), 1.0, atol=1e-5
    )


def test_pdf_matches_numerical_derivative_of_cdf():
    mog = _make_mog()
    x = np.linspace(-3.0, 3.0, 31).astype(np.float32)
    x_pair = np.stack([x, x], axis=-1)
    eps = 1e-3
    cdf_plus = _to_numpy(mog.cdf(ops.convert_to_tensor(x_pair + eps)))
    cdf_minus = _to_numpy(mog.cdf(ops.convert_to_tensor(x_pair - eps)))
    numerical = (cdf_plus - cdf_minus) / (2.0 * eps)
    analytic = _to_numpy(mog.pdf(ops.convert_to_tensor(x_pair)))
    np.testing.assert_allclose(analytic, numerical, atol=5e-4)


def test_log_pdf_matches_log_of_pdf_away_from_tails():
    mog = _make_mog()
    x = np.linspace(-2.0, 2.0, 21).astype(np.float32)
    x_pair = np.stack([x, x], axis=-1)
    log_pdf = _to_numpy(mog.log_pdf(ops.convert_to_tensor(x_pair)))
    pdf = _to_numpy(mog.pdf(ops.convert_to_tensor(x_pair)))
    np.testing.assert_allclose(log_pdf, np.log(pdf), atol=1e-5)


def test_log_pdf_stable_in_tails():
    # logsumexp path must stay finite where direct pdf would underflow.
    mog = _make_mog()
    x = np.full((1, 2), 15.0, dtype=np.float32)
    log_pdf = _to_numpy(mog.log_pdf(ops.convert_to_tensor(x)))
    assert np.all(np.isfinite(log_pdf))


def test_pdf_integrates_to_one():
    mog = _make_mog(seed=1, d=1, k=3)
    xs = np.linspace(-15.0, 15.0, 4001).astype(np.float32)
    # Shape (N, 1) for the d=1 mixture.
    pdf = _to_numpy(mog.pdf(ops.convert_to_tensor(xs[:, None])))[:, 0]
    integral = np.trapezoid(pdf, xs)
    np.testing.assert_allclose(integral, 1.0, atol=1e-3)
