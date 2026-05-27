"""Tests for the ``gaussianization.fair`` subpackage."""

from __future__ import annotations

import os


os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np
import pytest
from gaussianization.fair import (
    GaussianizedMutualInfoLoss,
    GaussianizedTotalCorrelationLoss,
    GaussianizedXCovLoss,
    demographic_parity_difference,
    equalized_odds_difference,
    fit_and_freeze,
    fit_and_freeze_joint,
    freeze_flow,
    is_fully_frozen,
)
from gaussianization.gauss_keras.training import make_gaussianization_flow


@pytest.fixture(scope="module")
def tiny_flow():
    """A 1-D Gaussianization flow trained for a handful of epochs."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(512, 1)).astype("float32")
    flow, _ = fit_and_freeze(
        data,
        num_blocks=2,
        num_components=4,
        epochs=3,
        batch_size=128,
        validation_split=0.2,
        patience=10,
        seed=0,
        verbose=0,
    )
    return flow


@pytest.fixture(scope="module")
def tiny_joint_flow():
    """A 2-D joint flow trained on the product distribution of (z, q)."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=(512,)).astype("float32")
    q = rng.normal(size=(512,)).astype("float32")
    flow, _ = fit_and_freeze_joint(
        z,
        q,
        num_blocks=2,
        num_components=4,
        epochs=3,
        batch_size=128,
        validation_split=0.2,
        patience=10,
        seed=0,
        n_shuffles=1,
        verbose=0,
    )
    return flow


def test_freeze_marks_all_weights_non_trainable():
    flow = make_gaussianization_flow(input_dim=2, num_blocks=2, num_components=4)
    flow(keras.ops.zeros((4, 2)))
    assert len(flow.trainable_weights) > 0
    freeze_flow(flow)
    assert is_fully_frozen(flow)


def test_fit_and_freeze_returns_frozen_flow():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(256, 2)).astype("float32")
    flow, history = fit_and_freeze(
        data,
        num_blocks=2,
        num_components=4,
        epochs=2,
        batch_size=64,
        validation_split=0.2,
        patience=10,
        seed=1,
        verbose=0,
    )
    assert is_fully_frozen(flow)
    assert "loss" in history.history


def test_fit_and_freeze_joint_returns_frozen_flow(tiny_joint_flow):
    assert is_fully_frozen(tiny_joint_flow)
    # The joint flow operates on (z, q) concatenated -- it should be 2-D.
    out = tiny_joint_flow(keras.ops.zeros((4, 2)))
    assert tuple(keras.ops.shape(out)) == (4, 2)


def test_xcov_loss_shape_and_non_negative(tiny_flow):
    rng = np.random.default_rng(2)
    z = rng.normal(size=(64, 1)).astype("float32")
    q = rng.binomial(1, 0.5, size=(64,)).astype("float32")
    loss_fn = GaussianizedXCovLoss(flow_z=tiny_flow, flow_q=tiny_flow)
    val = float(loss_fn(q, z))
    assert val >= 0.0
    assert np.isfinite(val)


def test_xcov_loss_independent_vs_dependent(tiny_flow):
    rng = np.random.default_rng(3)
    n = 1024
    z = rng.normal(size=(n, 1)).astype("float32")
    q_indep = rng.normal(size=(n, 1)).astype("float32")
    q_dep = z + 0.01 * rng.normal(size=(n, 1)).astype("float32")
    loss_fn = GaussianizedXCovLoss(flow_z=tiny_flow, flow_q=tiny_flow)
    val_indep = float(loss_fn(q_indep, z))
    val_dep = float(loss_fn(q_dep, z))
    assert val_dep > val_indep, f"dep={val_dep} should exceed indep={val_indep}"


def test_xcov_loss_unnormalized_equals_hsic_linear(tiny_flow):
    """G-XCOV(normalize=False) = HSIC with linear kernels (= ||C||_F^2).

    We check the identity by computing both expressions directly on the
    same Gaussianised batch.
    """
    rng = np.random.default_rng(99)
    n = 256
    z = rng.normal(size=(n, 1)).astype("float32")
    q = rng.normal(size=(n, 1)).astype("float32")
    loss_fn = GaussianizedXCovLoss(flow_z=tiny_flow, flow_q=tiny_flow, normalize=False)
    g_xcov_unnorm = float(loss_fn(q, z))

    zg = np.asarray(tiny_flow(z))
    qg = np.asarray(tiny_flow(q))
    zg = zg - zg.mean(0, keepdims=True)
    qg = qg - qg.mean(0, keepdims=True)
    C = zg.T @ qg / (n - 1)
    np.testing.assert_allclose(g_xcov_unnorm, float((C * C).sum()), rtol=1e-5)


def test_xcov_loss_normalized_bounded_in_unit_interval(tiny_flow):
    """With normalize=True, G-XCOV \\in [0, 1] for 1-D z, q."""
    rng = np.random.default_rng(7)
    n = 1024
    loss_fn = GaussianizedXCovLoss(flow_z=tiny_flow, flow_q=tiny_flow, normalize=True)
    # Independent: should be small
    z = rng.normal(size=(n, 1)).astype("float32")
    q = rng.normal(size=(n, 1)).astype("float32")
    val_indep = float(loss_fn(q, z))
    # Perfectly dependent: should be close to 1
    q_dep = z.copy()
    val_dep = float(loss_fn(q_dep, z))
    assert 0.0 <= val_indep <= 1.0 + 1e-3
    assert 0.0 <= val_dep <= 1.0 + 1e-3
    assert val_dep > 0.8, f"perfect-dep G-XCOV should be near 1, got {val_dep}"


def test_mi_loss_shape_and_non_negative(tiny_flow):
    rng = np.random.default_rng(4)
    z = rng.normal(size=(64, 1)).astype("float32")
    q = rng.binomial(1, 0.5, size=(64,)).astype("float32")
    loss_fn = GaussianizedMutualInfoLoss(flow_z=tiny_flow, flow_q=tiny_flow)
    val = float(loss_fn(q, z))
    assert val >= -1e-6
    assert np.isfinite(val)


def test_mi_loss_diverges_for_perfect_dependence(tiny_flow):
    """For z = q exactly, G-MI -> -0.5 log(eps) -- the clipped maximum."""
    rng = np.random.default_rng(8)
    n = 1024
    z = rng.normal(size=(n, 1)).astype("float32")
    q_indep = rng.normal(size=(n, 1)).astype("float32")
    loss_fn = GaussianizedMutualInfoLoss(flow_z=tiny_flow, flow_q=tiny_flow, eps=1e-4)
    val_indep = float(loss_fn(q_indep, z))
    val_dep = float(loss_fn(z, z))  # z = q
    # Upper bound at the eps clip: -0.5 log(1e-4) ~= 4.6
    expected_max = -0.5 * np.log(1e-4)
    assert val_dep > val_indep
    assert val_dep <= expected_max + 1e-3, (
        f"G-MI exceeded eps-clipped maximum {expected_max}: got {val_dep}"
    )


def test_mi_loss_matches_closed_form_1d(tiny_flow):
    """G-MI in 1-D == -0.5 log(1 - rho^2) of Gaussianised features."""
    rng = np.random.default_rng(11)
    n = 1024
    z = rng.normal(size=(n, 1)).astype("float32")
    q = z + 0.5 * rng.normal(size=(n, 1)).astype("float32")
    loss_fn = GaussianizedMutualInfoLoss(flow_z=tiny_flow, flow_q=tiny_flow, eps=1e-8)
    val = float(loss_fn(q, z))

    zg = np.asarray(tiny_flow(z)).ravel()
    qg = np.asarray(tiny_flow(q)).ravel()
    zg = (zg - zg.mean()) / (zg.std() + 1e-8)
    qg = (qg - qg.mean()) / (qg.std() + 1e-8)
    rho = float(np.mean(zg * qg))
    expected = -0.5 * np.log(max(1.0 - rho * rho, 1e-8))
    np.testing.assert_allclose(val, expected, rtol=5e-3, atol=5e-3)


def test_tc_loss_finite_and_responds_to_dependence(tiny_joint_flow):
    """G-TC is large when (z, q) is dependent, smaller when independent."""
    rng = np.random.default_rng(13)
    n = 1024
    z = rng.normal(size=(n, 1)).astype("float32")
    q_indep = rng.normal(size=(n, 1)).astype("float32")
    q_dep = z + 0.1 * rng.normal(size=(n, 1)).astype("float32")
    loss_fn = GaussianizedTotalCorrelationLoss(joint_flow=tiny_joint_flow)
    val_indep = float(loss_fn(q_indep, z))
    val_dep = float(loss_fn(q_dep, z))
    assert np.isfinite(val_indep) and np.isfinite(val_dep)
    assert val_dep > val_indep, (
        f"G-TC should rise with dependence: indep={val_indep}, dep={val_dep}"
    )


@pytest.mark.parametrize(
    "loss_factory",
    [
        lambda f: GaussianizedXCovLoss(flow_z=f, flow_q=f),
        lambda f: GaussianizedMutualInfoLoss(flow_z=f, flow_q=f),
    ],
    ids=["g_xcov", "g_mi"],
)
def test_loss_gradients_flow_to_predictor(tiny_flow, loss_factory):
    """Gradients of every fairness loss reach the upstream model's weights."""
    import jax

    rng = np.random.default_rng(5)
    x = rng.normal(size=(64, 3)).astype("float32")
    q = rng.binomial(1, 0.5, size=(64, 1)).astype("float32")
    mlp = keras.Sequential(
        [keras.layers.Input(shape=(3,)), keras.layers.Dense(1, use_bias=False)]
    )
    mlp(keras.ops.zeros((1, 3)))
    w = mlp.trainable_weights[0]
    loss_fn = loss_factory(tiny_flow)

    def fairness(w_value):
        mlp.trainable_weights[0].assign(w_value)
        z = mlp(x)
        return loss_fn(q, z)

    grad = jax.grad(fairness)(w.value)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.linalg.norm(np.asarray(grad)) > 0, (
        f"{loss_fn.name} produced a zero gradient"
    )


def test_demographic_parity_difference_known_case():
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    q = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert demographic_parity_difference(y_pred, q) == pytest.approx(0.25)


def test_equalized_odds_difference_known_case():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    q = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    eod = equalized_odds_difference(y_true, y_pred, q)
    assert eod == pytest.approx(0.5)
