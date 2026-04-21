"""Tests for NumPyro Bayesian inference of Gaussian plume emission rate."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer import Predictive
from plume_simulation.gauss_plume.dispersion import BRIGGS_DISPERSION_PARAMS
from plume_simulation.gauss_plume.inference import (
    _lognormal_from_moments,
    gaussian_plume_model,
    infer_emission_rate,
)
from plume_simulation.gauss_plume.plume import plume_concentration


def test_model_importable():
    # Smoke — confirms the numpyro import path stays working.
    assert callable(gaussian_plume_model)


# ── LogNormal prior parameterisation ─────────────────────────────────────────


@pytest.mark.parametrize(
    "mean,std",
    [(0.1, 0.05), (0.1, 0.08), (1.0, 0.2), (5.0, 1.0)],
)
def test_lognormal_from_moments_analytical(mean, std):
    """μ_log, σ_log chosen so the LogNormal has exactly the requested moments."""
    mu_log, sigma_log = _lognormal_from_moments(mean, std)
    # Analytical moments of LogNormal(μ, σ):
    #   E[X]   = exp(μ + σ²/2)
    #   Var[X] = (exp(σ²) - 1) · exp(2μ + σ²)
    mean_back = float(jnp.exp(mu_log + 0.5 * sigma_log**2))
    var_back = float((jnp.exp(sigma_log**2) - 1.0) * jnp.exp(2 * mu_log + sigma_log**2))
    np.testing.assert_allclose(mean_back, mean, rtol=1e-6)
    np.testing.assert_allclose(np.sqrt(var_back), std, rtol=1e-6)


def test_gaussian_plume_prior_matches_requested_moments():
    """Sampled prior over ``emission_rate`` has ~requested mean/std."""
    predictive = Predictive(gaussian_plume_model, num_samples=20000)
    samples = predictive(
        jax.random.PRNGKey(0),
        observations=None,
        receptor_coords=None,
        source_location=None,
        wind_u=None,
        wind_v=None,
        stability_class="D",
        prior_emission_rate_mean=0.15,
        prior_emission_rate_std=0.05,
    )
    q = np.asarray(samples["emission_rate"])
    np.testing.assert_allclose(q.mean(), 0.15, rtol=0.02)
    np.testing.assert_allclose(q.std(), 0.05, rtol=0.05)


# ── Forward-inputs guard ─────────────────────────────────────────────────────


def test_gaussian_plume_model_rejects_missing_forward_inputs():
    """Providing observations without the full forward-model inputs must raise."""
    obs = jnp.array([1e-7, 2e-7, 3e-7])

    with pytest.raises(ValueError, match=r"missing inputs: receptor_coords"):
        gaussian_plume_model(
            observations=obs,
            receptor_coords=None,
            source_location=(0.0, 0.0, 2.0),
            wind_u=5.0,
            wind_v=0.0,
        )

    with pytest.raises(ValueError, match=r"missing inputs: source_location, wind_v"):
        gaussian_plume_model(
            observations=obs,
            receptor_coords=(jnp.array([500.0]),) * 3,
            source_location=None,
            wind_u=5.0,
            wind_v=None,
        )


def test_gaussian_plume_model_allows_prior_predictive_without_inputs():
    """With observations=None the model must be callable without forward inputs."""
    predictive = Predictive(gaussian_plume_model, num_samples=5)
    samples = predictive(
        jax.random.PRNGKey(0),
        observations=None,
        receptor_coords=None,
        source_location=None,
        wind_u=None,
        wind_v=None,
    )
    # Prior-predictive samples "obs" from the observation-noise distribution
    # around the background — just check the call succeeded and shapes match.
    assert samples["emission_rate"].shape == (5,)
    assert samples["obs"].shape == (5,)


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
