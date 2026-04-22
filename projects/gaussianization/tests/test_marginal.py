"""Tests for the mixture-CDF marginal Gaussianization bijector."""

from __future__ import annotations

import numpy as np
from keras import ops

from gaussianization.gauss_keras.bijectors import MixtureCDFGaussianization


def _to_numpy(x):
    return np.asarray(ops.convert_to_numpy(x))


def _make_layer(d=2, k=4, seed=0):
    layer = MixtureCDFGaussianization(num_components=k)
    layer.build(input_shape=(None, d))
    rng = np.random.default_rng(seed)
    layer.logits.assign(rng.standard_normal((d, k)).astype(np.float32))
    layer.means.assign((2.0 * rng.standard_normal((d, k))).astype(np.float32))
    layer.log_scales.assign((0.1 * rng.standard_normal((d, k))).astype(np.float32))
    return layer


def test_forward_inverse_roundtrip():
    layer = _make_layer(d=2, k=6, seed=1)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 2)).astype(np.float32) * 1.5
    z, _ = layer.forward_and_log_det(ops.convert_to_tensor(x))
    x_rt, _ = layer.inverse_and_log_det(z)
    np.testing.assert_allclose(_to_numpy(x_rt), x, atol=1e-3)


def test_inverse_forward_roundtrip():
    layer = _make_layer(d=2, k=6, seed=2)
    rng = np.random.default_rng(3)
    z = (0.8 * rng.standard_normal((16, 2))).astype(np.float32)
    x, _ = layer.inverse_and_log_det(ops.convert_to_tensor(z))
    z_rt, _ = layer.forward_and_log_det(x)
    np.testing.assert_allclose(_to_numpy(z_rt), z, atol=1e-3)


def test_forward_log_det_matches_numerical():
    layer = _make_layer(d=2, k=5, seed=4)
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((4, 2))).astype(np.float32)
    x_t = ops.convert_to_tensor(x)
    _, ldj_analytic = layer.forward_and_log_det(x_t)
    ldj_analytic = _to_numpy(ldj_analytic)

    # Numerical log|det dz/dx|: since the Jacobian is diagonal, it is
    # the sum over dims of log|dz_i/dx_i|.
    eps = 1e-3
    ldj_num = np.zeros(x.shape[0], dtype=np.float64)
    for i in range(x.shape[1]):
        x_plus = x.copy()
        x_plus[:, i] += eps
        x_minus = x.copy()
        x_minus[:, i] -= eps
        z_plus, _ = layer.forward_and_log_det(ops.convert_to_tensor(x_plus))
        z_minus, _ = layer.forward_and_log_det(ops.convert_to_tensor(x_minus))
        dz_dx = (_to_numpy(z_plus)[:, i] - _to_numpy(z_minus)[:, i]) / (2.0 * eps)
        ldj_num += np.log(np.abs(dz_dx))
    np.testing.assert_allclose(ldj_analytic, ldj_num, atol=1e-3, rtol=1e-3)


def test_inverse_log_det_is_negative_of_forward():
    # For a diagonal transform, log|dx/dz|(z) = -log|dz/dx|(x) when x = f^{-1}(z).
    layer = _make_layer(d=2, k=5, seed=6)
    rng = np.random.default_rng(7)
    z = (0.6 * rng.standard_normal((8, 2))).astype(np.float32)
    z_t = ops.convert_to_tensor(z)
    x, ldj_inv = layer.inverse_and_log_det(z_t)
    _, ldj_fwd = layer.forward_and_log_det(x)
    np.testing.assert_allclose(_to_numpy(ldj_inv), -_to_numpy(ldj_fwd), atol=1e-4)


def test_call_adds_negative_mean_ldj_to_losses():
    layer = _make_layer(d=2, k=4, seed=8)
    rng = np.random.default_rng(9)
    x = (rng.standard_normal((5, 2))).astype(np.float32)
    x_t = ops.convert_to_tensor(x)
    _, ldj = layer.forward_and_log_det(x_t)
    expected = -float(_to_numpy(ops.mean(ldj)))
    _ = layer(x_t)
    assert len(layer.losses) == 1
    np.testing.assert_allclose(
        float(_to_numpy(layer.losses[0])), expected, atol=1e-6
    )


def test_inverse_call_does_not_add_loss():
    layer = _make_layer(d=2, k=4, seed=10)
    rng = np.random.default_rng(11)
    z = (0.5 * rng.standard_normal((5, 2))).astype(np.float32)
    _ = layer(ops.convert_to_tensor(z), inverse=True)
    assert len(layer.losses) == 0


def test_adapt_means_from_quantiles_rejects_non_2d():
    import pytest

    layer = MixtureCDFGaussianization(num_components=4)
    with pytest.raises(ValueError, match="shape \\(n, d\\)"):
        layer.adapt_means_from_quantiles(np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError, match="shape \\(n, d\\)"):
        layer.adapt_means_from_quantiles(np.zeros((2, 3, 4), dtype=np.float32))


def test_adapt_means_from_quantiles_handles_single_component():
    """With num_components=1 the inter-quantile diff is empty; the
    bandwidth should fall back to the data std rather than NaN."""
    layer = MixtureCDFGaussianization(num_components=1)
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((500, 2)) * 2.0 + 1.0).astype(np.float32)
    layer.adapt_means_from_quantiles(data)
    means = _to_numpy(layer.means)
    log_scales = _to_numpy(layer.log_scales)
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(log_scales))
    # Sanity: with d=2, k=1 the single mean per dim is the 0.5 quantile.
    np.testing.assert_allclose(
        means[:, 0],
        [np.quantile(data[:, 0], 0.5), np.quantile(data[:, 1], 0.5)],
        atol=1e-6,
    )


def test_adapt_means_from_quantiles_sets_means():
    layer = MixtureCDFGaussianization(num_components=5)
    rng = np.random.default_rng(12)
    data = rng.standard_normal((1000, 2)).astype(np.float32) * 2.0 + 1.0
    layer.adapt_means_from_quantiles(data)
    means = _to_numpy(layer.means)
    # Means per dim should be within the empirical range of that dim.
    for i in range(2):
        assert means[i].min() >= data[:, i].min() - 1e-3
        assert means[i].max() <= data[:, i].max() + 1e-3
