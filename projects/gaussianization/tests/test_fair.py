"""Tests for the ``gaussianization.fair`` subpackage."""

from __future__ import annotations

import os


os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np
import pytest
from gaussianization.fair import (
    GaussianizedHSICLoss,
    GaussianizedXCovLoss,
    demographic_parity_difference,
    equalized_odds_difference,
    fit_and_freeze,
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


def test_xcov_loss_shape_and_non_negative(tiny_flow):
    rng = np.random.default_rng(2)
    z = rng.normal(size=(64, 1)).astype("float32")
    q = rng.binomial(1, 0.5, size=(64,)).astype("float32")
    loss_fn = GaussianizedXCovLoss(flow_z=tiny_flow, flow_q=tiny_flow)
    val = float(loss_fn(q, z))
    assert val >= 0.0
    assert np.isfinite(val)


def test_xcov_loss_independent_inputs_small(tiny_flow):
    rng = np.random.default_rng(3)
    n = 1024
    z = rng.normal(size=(n, 1)).astype("float32")
    q = rng.normal(size=(n, 1)).astype("float32")
    loss_fn = GaussianizedXCovLoss(flow_z=tiny_flow, flow_q=tiny_flow)
    val_indep = float(loss_fn(q, z))

    q_dep = z + 0.01 * rng.normal(size=(n, 1)).astype("float32")
    val_dep = float(loss_fn(q_dep, z))
    assert val_dep > val_indep, f"dep={val_dep} should exceed indep={val_indep}"


def test_hsic_loss_runs(tiny_flow):
    rng = np.random.default_rng(4)
    z = rng.normal(size=(64, 1)).astype("float32")
    q = rng.binomial(1, 0.5, size=(64,)).astype("float32")
    loss_fn = GaussianizedHSICLoss(flow_z=tiny_flow, flow_q=tiny_flow)
    val = float(loss_fn(q, z))
    assert val >= 0.0
    assert np.isfinite(val)


def test_loss_gradients_flow_to_predictor(tiny_flow):
    """Gradients of the fairness loss reach the upstream model's weights."""
    import jax

    rng = np.random.default_rng(5)
    x = rng.normal(size=(32, 3)).astype("float32")
    q = rng.binomial(1, 0.5, size=(32, 1)).astype("float32")

    mlp = keras.Sequential(
        [keras.layers.Input(shape=(3,)), keras.layers.Dense(1, use_bias=False)]
    )
    mlp(keras.ops.zeros((1, 3)))
    loss_fn = GaussianizedXCovLoss(flow_z=tiny_flow, flow_q=tiny_flow)
    w = mlp.trainable_weights[0]

    def fairness(w_value):
        mlp.trainable_weights[0].assign(w_value)
        z = mlp(x)
        return loss_fn(q, z)

    grad = jax.grad(fairness)(w.value)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.linalg.norm(np.asarray(grad)) > 0


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
