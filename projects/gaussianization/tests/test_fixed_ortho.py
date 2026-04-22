"""Tests for the FixedOrtho bijector (random Haar + PCA factories)."""

from __future__ import annotations

import numpy as np
from keras import ops

from gaussianization.gauss_keras.bijectors import FixedOrtho


def _to_numpy(x):
    return np.asarray(ops.convert_to_numpy(x))


def test_random_haar_is_orthogonal():
    d = 5
    layer = FixedOrtho.random_haar(dim=d, seed=0)
    layer.build(input_shape=(None, d))
    q = _to_numpy(layer.q)
    np.testing.assert_allclose(q @ q.T, np.eye(d), atol=1e-5)
    np.testing.assert_allclose(q.T @ q, np.eye(d), atol=1e-5)


def test_random_haar_is_deterministic():
    a = FixedOrtho.random_haar(dim=4, seed=7)
    b = FixedOrtho.random_haar(dim=4, seed=7)
    a.build(input_shape=(None, 4))
    b.build(input_shape=(None, 4))
    np.testing.assert_allclose(_to_numpy(a.q), _to_numpy(b.q), atol=0.0)


def test_forward_inverse_roundtrip():
    d = 4
    layer = FixedOrtho.random_haar(dim=d, seed=1)
    layer.build(input_shape=(None, d))
    rng = np.random.default_rng(2)
    x = rng.standard_normal((6, d)).astype(np.float32)
    z, _ = layer.forward_and_log_det(ops.convert_to_tensor(x))
    x_rt, _ = layer.inverse_and_log_det(z)
    np.testing.assert_allclose(_to_numpy(x_rt), x, atol=1e-5)


def test_q_is_non_trainable():
    layer = FixedOrtho.random_haar(dim=3, seed=0)
    layer.build(input_shape=(None, 3))
    assert layer.q.trainable is False
    # trainable_weights must not expose q.
    assert not any(w is layer.q for w in layer.trainable_weights)


def test_from_pca_recovers_known_eigenvectors():
    # Build data whose sample covariance has a known eigenbasis: draw from
    # N(0, diag(9, 1)) rotated by 30°. PCA should recover the rotation up
    # to sign and column order.
    rng = np.random.default_rng(0)
    n = 20000
    x_raw = rng.standard_normal((n, 2)) * np.array([3.0, 1.0])
    theta = np.deg2rad(30.0)
    r = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    x = x_raw @ r.T
    layer = FixedOrtho.from_pca(x)
    layer.build(input_shape=(None, 2))
    q = _to_numpy(layer.q)
    # Rotating x by q^T should decorrelate it; sample covariance ~ diag.
    decorrelated = x @ q
    cov = np.cov(decorrelated, rowvar=False)
    off_diag = abs(cov[0, 1])
    assert off_diag < 0.05
    # And the variances should be close to (9, 1).
    np.testing.assert_allclose(np.sort(np.diag(cov))[::-1], [9.0, 1.0], atol=0.3)


def test_from_pca_rejects_non_2d_input():
    try:
        FixedOrtho.from_pca(np.zeros((2, 3, 4)))
    except ValueError:
        return
    raise AssertionError("FixedOrtho.from_pca should reject 3-D input")


def test_forward_call_contributes_zero_loss():
    layer = FixedOrtho.random_haar(dim=3, seed=0)
    layer.build(input_shape=(None, 3))
    x = ops.convert_to_tensor(np.ones((2, 3), dtype=np.float32))
    _ = layer(x)
    loss_values = [float(_to_numpy(v)) for v in layer.losses]
    np.testing.assert_allclose(loss_values, 0.0, atol=1e-6)
