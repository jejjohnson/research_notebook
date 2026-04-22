"""Tests for the MixtureCDFCoupling bijector + conditioner builders."""

from __future__ import annotations

import numpy as np
import keras
import pytest
from keras import ops

from gaussianization.gauss_keras import (
    MixtureCDFCoupling,
    default_half_mask,
    make_coupling_flow,
    make_mlp_conditioner,
    make_shared_mlp_conditioner,
    sigmoid_log_scale_clamp,
    tanh_log_scale_clamp,
)


def _to_numpy(x):
    return np.asarray(ops.convert_to_numpy(x))


def _build_layer(d=4, mask=None, shared=False, num_components=5):
    if mask is None:
        mask = default_half_mask(d)
    mask = np.asarray(mask, dtype=bool)
    d_b = int((~mask).sum())
    factory = make_shared_mlp_conditioner if shared else make_mlp_conditioner
    cond = factory(d_b=d_b, num_components=num_components, hidden=(16, 16))
    layer = MixtureCDFCoupling(
        mask=mask, conditioner=cond, num_components=num_components
    )
    layer.build(input_shape=(None, d))
    return layer


def test_default_half_mask():
    m = default_half_mask(4)
    np.testing.assert_array_equal(m, [True, True, False, False])
    m = default_half_mask(5)
    # First d//2 = 2 True, remaining 3 False.
    np.testing.assert_array_equal(m, [True, True, False, False, False])


def test_tanh_clamp_bounds():
    clamp = tanh_log_scale_clamp(bound=3.0)
    out = _to_numpy(clamp(ops.convert_to_tensor([-100.0, 0.0, 100.0])))
    np.testing.assert_allclose(out, [-3.0, 0.0, 3.0], atol=1e-5)


def test_sigmoid_clamp_bounds():
    clamp = sigmoid_log_scale_clamp(min_log_scale=-2.0, max_log_scale=2.0)
    out = _to_numpy(clamp(ops.convert_to_tensor([-1e3, 0.0, 1e3])))
    np.testing.assert_allclose(out, [-2.0, 0.0, 2.0], atol=1e-5)


def test_forward_identity_at_zero_init():
    """Zero-init conditioner => mixture = N(0,1) => F(x) = Phi(x) => z = x, ldj = 0."""
    layer = _build_layer(d=4)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 4)).astype(np.float32) * 0.8
    z, ldj = layer.forward_and_log_det(ops.convert_to_tensor(x))
    np.testing.assert_allclose(_to_numpy(z), x, atol=1e-5)
    np.testing.assert_allclose(_to_numpy(ldj), 0.0, atol=1e-5)


def test_a_dims_unchanged_by_forward():
    mask = np.array([True, False, True, False])
    layer = _build_layer(d=4, mask=mask)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 4)).astype(np.float32)
    z, _ = layer.forward_and_log_det(ops.convert_to_tensor(x))
    z_np = _to_numpy(z)
    # a-dims (mask True) must be passed through unchanged.
    np.testing.assert_allclose(z_np[:, mask], x[:, mask], atol=1e-6)


def test_forward_inverse_roundtrip():
    layer = _build_layer(d=4)
    # Perturb conditioner weights so we're not testing identity.
    for w in layer.conditioner.trainable_weights:
        w.assign(0.1 * np.random.default_rng(0).standard_normal(w.shape).astype("float32"))
    rng = np.random.default_rng(2)
    x = rng.standard_normal((32, 4)).astype(np.float32) * 0.7
    z, _ = layer.forward_and_log_det(ops.convert_to_tensor(x))
    x_rt, _ = layer.inverse_and_log_det(z)
    err = np.abs(_to_numpy(x_rt) - x)
    assert np.median(err) < 1e-4
    assert np.max(err) < 1e-2


def test_log_det_matches_numerical():
    layer = _build_layer(d=3, mask=[True, False, False], num_components=4)
    for w in layer.conditioner.trainable_weights:
        w.assign(0.1 * np.random.default_rng(3).standard_normal(w.shape).astype("float32"))
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 3)).astype(np.float32) * 0.5
    _, ldj_a = layer.forward_and_log_det(ops.convert_to_tensor(x))
    ldj_a = _to_numpy(ldj_a)
    # Numerical Jacobian via central differences — per-sample scalar log|det|.
    eps = 1e-3
    n, d = x.shape
    ldj_n = np.zeros(n, dtype=np.float64)
    for b in range(n):
        jac = np.zeros((d, d), dtype=np.float64)
        for j in range(d):
            xp = x[b : b + 1].copy()
            xm = x[b : b + 1].copy()
            xp[0, j] += eps
            xm[0, j] -= eps
            zp, _ = layer.forward_and_log_det(ops.convert_to_tensor(xp.astype("float32")))
            zm, _ = layer.forward_and_log_det(ops.convert_to_tensor(xm.astype("float32")))
            jac[:, j] = (_to_numpy(zp)[0] - _to_numpy(zm)[0]) / (2.0 * eps)
        sign, logdet = np.linalg.slogdet(jac)
        ldj_n[b] = logdet
    np.testing.assert_allclose(ldj_a, ldj_n, atol=2e-3, rtol=2e-3)


def test_call_adds_loss_forward_only():
    layer = _build_layer(d=4)
    for w in layer.conditioner.trainable_weights:
        w.assign(0.1 * np.random.default_rng(5).standard_normal(w.shape).astype("float32"))
    x = ops.convert_to_tensor(
        np.random.default_rng(6).standard_normal((4, 4)).astype("float32") * 0.5
    )
    _ = layer(x)
    assert len(layer.losses) == 1
    _ = layer(x, inverse=True)
    # Keras clears layer.losses at the start of each __call__, and the
    # inverse path must not contribute a new entry, so losses is empty.
    assert len(layer.losses) == 0


def test_conditioner_wrong_output_raises():
    # Build a Dense with the wrong output width.
    bad_conditioner = keras.Sequential(
        [keras.layers.Dense(7, kernel_initializer="zeros", bias_initializer="zeros")]
    )
    with pytest.raises(ValueError, match="conditioner output width"):
        layer = MixtureCDFCoupling(
            mask=[True, True, False, False],
            conditioner=bad_conditioner,
            num_components=5,
        )
        layer.build(input_shape=(None, 4))


def test_shared_mixture_broadcasts_params():
    """With a shared conditioner, every b-dim sees the same mixture at init."""
    d = 4
    mask = np.array([True, True, False, False])
    cond = make_shared_mlp_conditioner(d_b=2, num_components=5, hidden=(8,))
    layer = MixtureCDFCoupling(mask=mask, conditioner=cond, num_components=5)
    layer.build(input_shape=(None, d))
    # Set non-zero conditioner weights so there's something to compare.
    rng = np.random.default_rng(7)
    for w in cond.trainable_weights:
        w.assign(0.3 * rng.standard_normal(w.shape).astype("float32"))
    # Evaluate params for a single input; the two b-dims must share them.
    x_a = ops.convert_to_tensor(rng.standard_normal((1, 2)).astype("float32"))
    flat = _to_numpy(cond(x_a))  # (1, 3*d_b*K) = (1, 30)
    params = flat.reshape(1, 3, 2, 5)
    np.testing.assert_allclose(params[:, :, 0, :], params[:, :, 1, :], atol=1e-6)


def test_make_coupling_flow_roundtrip_and_log_prob():
    flow = make_coupling_flow(
        input_dim=2, num_blocks=2, num_components=5, hidden=(16, 16)
    )
    rng = np.random.default_rng(8)
    x = rng.standard_normal((64, 2)).astype("float32") * 0.8
    z = flow(ops.convert_to_tensor(x))
    x_rt = _to_numpy(flow.invert(z))
    err = np.abs(x_rt - x)
    assert np.median(err) < 1e-4
    assert np.max(err) < 1e-2
    lp = _to_numpy(flow.log_prob(ops.convert_to_tensor(x)))
    assert lp.shape == (64,)
    assert np.all(np.isfinite(lp))


def test_make_coupling_flow_shared_mixture_smoke():
    flow = make_coupling_flow(
        input_dim=4, num_blocks=1, num_components=4, hidden=(8,), shared_mixture=True
    )
    x = ops.convert_to_tensor(
        np.random.default_rng(9).standard_normal((8, 4)).astype("float32")
    )
    _ = flow(x)
    lp = _to_numpy(flow.log_prob(x))
    assert lp.shape == (8,)
    assert np.all(np.isfinite(lp))
