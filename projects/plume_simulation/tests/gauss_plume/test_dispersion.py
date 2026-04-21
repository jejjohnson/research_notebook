"""Tests for Briggs dispersion coefficients."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.gauss_plume.dispersion import (
    BRIGGS_DISPERSION_PARAMS,
    STABILITY_CLASSES,
    calculate_briggs_dispersion,
    get_dispersion_params,
)


def test_stability_classes_registered():
    assert set(BRIGGS_DISPERSION_PARAMS.keys()) == set(STABILITY_CLASSES)
    assert STABILITY_CLASSES == ("A", "B", "C", "D", "E", "F")


@pytest.mark.parametrize("stab", STABILITY_CLASSES)
def test_params_shape(stab):
    params = BRIGGS_DISPERSION_PARAMS[stab]
    assert params.shape == (6,)


def test_get_dispersion_params_returns_registered_array():
    params = get_dispersion_params("C")
    np.testing.assert_array_equal(
        np.asarray(params), np.asarray(BRIGGS_DISPERSION_PARAMS["C"])
    )


def test_get_dispersion_params_rejects_unknown_class():
    with pytest.raises(ValueError, match=r"stability_class must be one of"):
        get_dispersion_params("Z")
    with pytest.raises(ValueError, match=r"stability_class must be one of"):
        get_dispersion_params("a")  # case-sensitive


@pytest.mark.parametrize("stab", STABILITY_CLASSES)
def test_sigmas_positive_and_monotone(stab):
    distance = jnp.linspace(10.0, 5000.0, 100)
    sigma_y, sigma_z = calculate_briggs_dispersion(
        distance, BRIGGS_DISPERSION_PARAMS[stab]
    )
    sigma_y, sigma_z = np.asarray(sigma_y), np.asarray(sigma_z)
    assert np.all(sigma_y > 0.0)
    assert np.all(sigma_z > 0.0)
    # Both coefficients are strictly increasing in downwind distance for all
    # Briggs stability classes over this regime.
    assert np.all(np.diff(sigma_y) > 0)
    assert np.all(np.diff(sigma_z) > 0)


def test_sigmas_clamp_negative_distance():
    # Negative distances should not produce NaN / Inf — the function clamps
    # internally before raising to a fractional power.
    sigma_y, sigma_z = calculate_briggs_dispersion(
        jnp.array([-100.0, 0.0, 1.0, 100.0]), BRIGGS_DISPERSION_PARAMS["C"]
    )
    assert np.all(np.isfinite(np.asarray(sigma_y)))
    assert np.all(np.isfinite(np.asarray(sigma_z)))


def test_class_ordering_unstable_disperses_faster():
    # Class A (unstable) → larger σ_y/σ_z than class F (stable) at same x.
    x = jnp.array([500.0])
    sigma_y_A, sigma_z_A = calculate_briggs_dispersion(
        x, BRIGGS_DISPERSION_PARAMS["A"]
    )
    sigma_y_F, sigma_z_F = calculate_briggs_dispersion(
        x, BRIGGS_DISPERSION_PARAMS["F"]
    )
    assert np.asarray(sigma_y_A).item() > np.asarray(sigma_y_F).item()
    assert np.asarray(sigma_z_A).item() > np.asarray(sigma_z_F).item()
