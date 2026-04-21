"""Tests for NumPyro Bayesian inference of Gaussian plume emission rate."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.gauss_plume.dispersion import BRIGGS_DISPERSION_PARAMS
from plume_simulation.gauss_plume.inference import (
    gaussian_plume_model,
    infer_emission_rate,
)
from plume_simulation.gauss_plume.plume import plume_concentration


def test_model_importable():
    # Smoke — confirms the numpyro import path stays working.
    assert callable(gaussian_plume_model)


def test_infer_emission_rate_rejects_bad_inputs():
    obs = np.array([1e-6, 2e-6, 3e-6])
    coords = (np.array([500.0, 600.0, 700.0]),
              np.array([0.0, 0.0, 0.0]),
              np.array([1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match=r"`wind_speed` must be > 0"):
        infer_emission_rate(
            obs, coords, (0, 0, 2), wind_speed=0.0, wind_direction=270.0,
            num_warmup=1, num_samples=1,
        )
    with pytest.raises(ValueError, match=r"`stability_class` must be one of"):
        infer_emission_rate(
            obs, coords, (0, 0, 2), wind_speed=5.0, wind_direction=270.0,
            stability_class="Z",
            num_warmup=1, num_samples=1,
        )
    with pytest.raises(ValueError, match=r"must contain ≥ 1 point"):
        infer_emission_rate(
            np.array([]), (np.array([]), np.array([]), np.array([])),
            (0, 0, 2), wind_speed=5.0, wind_direction=270.0,
            num_warmup=1, num_samples=1,
        )
    with pytest.raises(ValueError, match=r"observation_coords.*must be \(x, y, z\)"):
        infer_emission_rate(
            obs, (coords[0], coords[1]), (0, 0, 2),
            wind_speed=5.0, wind_direction=270.0,
            num_warmup=1, num_samples=1,
        )
    with pytest.raises(ValueError, match=r"`prior_mean` must be > 0"):
        infer_emission_rate(
            obs, coords, (0, 0, 2), wind_speed=5.0, wind_direction=270.0,
            prior_mean=0.0, num_warmup=1, num_samples=1,
        )


def test_infer_shape_mismatch_rejected():
    obs = np.array([1e-6, 2e-6, 3e-6])
    bad_coords = (np.array([500.0, 600.0]),
                  np.array([0.0, 0.0, 0.0]),
                  np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match=r"axis 'x' has shape"):
        infer_emission_rate(
            obs, bad_coords, (0, 0, 2), wind_speed=5.0, wind_direction=270.0,
            num_warmup=1, num_samples=1,
        )


@pytest.mark.slow
def test_infer_emission_rate_recovers_synthetic_Q():
    """NUTS should recover a synthetic emission rate from noisy transect data."""
    stab = "D"
    params = BRIGGS_DISPERSION_PARAMS[stab]
    source = (0.0, 0.0, 2.0)
    wind_u, wind_v = 5.0, 0.0

    # Downwind transect: y = 0, z = 1 m.
    x_obs = jnp.linspace(200.0, 2000.0, 24)
    y_obs = jnp.zeros_like(x_obs)
    z_obs = jnp.ones_like(x_obs)

    true_Q = 0.15
    clean = plume_concentration(
        x_obs, y_obs, z_obs, *source, wind_u, wind_v, true_Q, params
    )
    rng = np.random.default_rng(0)
    noise_std = 2e-8
    noisy = np.asarray(clean) + rng.normal(0.0, noise_std, size=clean.shape[0])

    samples = infer_emission_rate(
        noisy,
        (np.asarray(x_obs), np.asarray(y_obs), np.asarray(z_obs)),
        source_location=source,
        wind_speed=float(jnp.sqrt(wind_u**2 + wind_v**2)),
        wind_direction=270.0,  # from west → u > 0
        stability_class=stab,
        prior_mean=0.1,
        prior_std=0.08,
        num_warmup=300,
        num_samples=500,
        num_chains=1,
        seed=0,
    )
    assert set(samples.keys()) == {"emission_rate", "background"}
    q_mean = float(samples["emission_rate"].mean())
    q_std = float(samples["emission_rate"].std())
    # Smoke-grade: posterior should be in the ballpark of the truth.
    assert 0.05 < q_mean < 0.30
    assert q_std < 0.5 * q_mean
