"""Tests for the stacked GaussianizationFlow model."""

from __future__ import annotations

import numpy as np
from keras import ops

from gaussianization.gauss_keras import make_gaussianization_flow


def _to_numpy(x):
    return np.asarray(ops.convert_to_numpy(x))


def _toy_data(n=1024, seed=0):
    rng = np.random.default_rng(seed)
    x = np.stack(
        [rng.standard_normal(n) * 1.5, rng.standard_normal(n) * 0.5 + 0.3],
        axis=-1,
    ).astype(np.float32)
    return x


def test_forward_inverse_roundtrip():
    import keras

    keras.utils.set_random_seed(0)
    x = _toy_data()
    flow = make_gaussianization_flow(
        input_dim=2, num_blocks=3, num_reflectors=2, num_components=4
    )
    z = flow(ops.convert_to_tensor(x))
    x_rt = _to_numpy(flow.invert(z))
    # Clipping F(x) to [eps, 1-eps] introduces small per-sample error in
    # the deep tails; gate the test on the bulk via median, and a looser
    # tail tolerance to absorb occasional outliers where a sample lands
    # beyond the bisection range.
    err = np.abs(x_rt - x)
    assert np.median(err) < 1e-4
    assert np.percentile(err, 99) < 1e-2


def test_log_prob_shape_and_finite():
    x = _toy_data(n=64)
    flow = make_gaussianization_flow(
        input_dim=2, num_blocks=2, num_reflectors=2, num_components=4
    )
    lp = _to_numpy(flow.log_prob(ops.convert_to_tensor(x)))
    assert lp.shape == (64,)
    assert np.all(np.isfinite(lp))


def test_log_prob_matches_manual_computation():
    from gaussianization.gauss_keras._math import norm_log_pdf

    x = _toy_data(n=16)
    flow = make_gaussianization_flow(
        input_dim=2, num_blocks=2, num_reflectors=2, num_components=3
    )
    lp = _to_numpy(flow.log_prob(ops.convert_to_tensor(x)))

    # Manual: accumulate ldj while threading forward.
    y = ops.convert_to_tensor(x)
    ldj_total = 0.0
    for b in flow.bijector_layers:
        y, ldj = b.forward_and_log_det(y)
        ldj_total = ldj_total + _to_numpy(ldj)
    base = _to_numpy(ops.sum(norm_log_pdf(y), axis=-1))
    np.testing.assert_allclose(lp, base + ldj_total, atol=1e-5)


def test_sample_shape_and_finite():
    flow = make_gaussianization_flow(
        input_dim=2, num_blocks=2, num_reflectors=2, num_components=4
    )
    # Build by a dummy forward pass.
    _ = flow(ops.convert_to_tensor(_toy_data(n=4)))
    samples = _to_numpy(flow.sample(num_samples=100, seed=0))
    assert samples.shape == (100, 2)
    assert np.all(np.isfinite(samples))


def test_flow_trainable_weights_nonzero():
    flow = make_gaussianization_flow(
        input_dim=2, num_blocks=2, num_reflectors=2, num_components=3
    )
    _ = flow(ops.convert_to_tensor(_toy_data(n=4)))
    # Should include Householder v's and mixture params; at least 4 tensors.
    assert len(flow.trainable_weights) >= 4
