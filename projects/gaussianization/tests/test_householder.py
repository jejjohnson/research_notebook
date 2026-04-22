"""Tests for the Householder rotation bijector."""

from __future__ import annotations

import numpy as np
from keras import ops

from gaussianization.gauss_keras.bijectors import Householder


def _to_numpy(x):
    return np.asarray(ops.convert_to_numpy(x))


def _make_layer(d=4, k=3, seed=0):
    layer = Householder(num_reflectors=k)
    layer.build(input_shape=(None, d))
    # Overwrite random init with a reproducible RNG so tests are deterministic.
    rng = np.random.default_rng(seed)
    layer.v.assign(rng.standard_normal((k, d)).astype(np.float32))
    return layer


def test_forward_produces_orthogonal_transform():
    d = 4
    layer = _make_layer(d=d, k=3)
    identity = np.eye(d, dtype=np.float32)
    q, _ = layer.forward_and_log_det(ops.convert_to_tensor(identity))
    q_np = _to_numpy(q)
    np.testing.assert_allclose(q_np @ q_np.T, identity, atol=1e-5)
    np.testing.assert_allclose(q_np.T @ q_np, identity, atol=1e-5)


def test_forward_inverse_roundtrip():
    d = 5
    layer = _make_layer(d=d, k=4)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((8, d)).astype(np.float32)
    z, _ = layer.forward_and_log_det(ops.convert_to_tensor(x))
    x_roundtrip, _ = layer.inverse_and_log_det(z)
    np.testing.assert_allclose(_to_numpy(x_roundtrip), x, atol=1e-5)


def test_log_det_is_zero():
    layer = _make_layer(d=4, k=3)
    x = ops.convert_to_tensor(np.random.default_rng(1).standard_normal((6, 4)).astype(np.float32))
    _, ldj_f = layer.forward_and_log_det(x)
    _, ldj_i = layer.inverse_and_log_det(x)
    np.testing.assert_allclose(_to_numpy(ldj_f), 0.0, atol=1e-6)
    np.testing.assert_allclose(_to_numpy(ldj_i), 0.0, atol=1e-6)


def test_call_dispatches_to_forward_and_inverse():
    layer = _make_layer(d=3, k=2)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3)).astype(np.float32)
    x_t = ops.convert_to_tensor(x)
    # call(x) and forward_and_log_det(x) must agree.
    z_call = _to_numpy(layer(x_t))
    z_direct, _ = layer.forward_and_log_det(x_t)
    np.testing.assert_allclose(z_call, _to_numpy(z_direct), atol=1e-6)
    # call(x, inverse=True) and inverse_and_log_det(x) must agree.
    x_call = _to_numpy(layer(x_t, inverse=True))
    x_direct, _ = layer.inverse_and_log_det(x_t)
    np.testing.assert_allclose(x_call, _to_numpy(x_direct), atol=1e-6)


def test_forward_call_contributes_zero_loss():
    # Householder log |det| = 0 => add_loss(-mean(0)) = 0.
    layer = _make_layer(d=3, k=2)
    x = ops.convert_to_tensor(np.ones((2, 3), dtype=np.float32))
    _ = layer(x)
    loss_values = [float(_to_numpy(v)) for v in layer.losses]
    np.testing.assert_allclose(loss_values, 0.0, atol=1e-6)
