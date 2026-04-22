"""Tests for the standard-normal helpers built on keras.ops."""

from __future__ import annotations

import math

import numpy as np
from keras import ops

from gaussianization.gauss_keras._math import norm_cdf, norm_icdf, norm_log_pdf


def _to_numpy(x):
    return np.asarray(ops.convert_to_numpy(x))


def test_norm_cdf_matches_known_values():
    # Φ(0) = 0.5, Φ(±1.959964) ≈ {0.025, 0.975}, Φ(±∞) = {0, 1}.
    x = ops.convert_to_tensor(np.array([-10.0, -1.959964, 0.0, 1.959964, 10.0]))
    cdf = _to_numpy(norm_cdf(x))
    np.testing.assert_allclose(
        cdf, [0.0, 0.025, 0.5, 0.975, 1.0], atol=1e-4
    )


def test_norm_cdf_monotone():
    x = ops.convert_to_tensor(np.linspace(-5.0, 5.0, 257))
    cdf = _to_numpy(norm_cdf(x))
    assert np.all(np.diff(cdf) >= 0.0)


def test_norm_icdf_inverts_norm_cdf():
    x = np.linspace(-3.0, 3.0, 31)
    roundtrip = _to_numpy(norm_icdf(norm_cdf(ops.convert_to_tensor(x))))
    np.testing.assert_allclose(roundtrip, x, atol=1e-5)


def test_norm_log_pdf_matches_formula():
    x = np.linspace(-3.0, 3.0, 31)
    expected = -0.5 * (x**2 + math.log(2.0 * math.pi))
    got = _to_numpy(norm_log_pdf(ops.convert_to_tensor(x)))
    np.testing.assert_allclose(got, expected, atol=1e-6)
