"""Tests for the Gaussian-puff NumPyro inference helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer import Predictive
from plume_simulation.gauss_puff.inference import (
    gaussian_puff_model,
    infer_emission_rate,
)


def test_model_importable():
    assert callable(gaussian_puff_model)


def test_gaussian_puff_model_rejects_missing_forward_inputs():
    obs = jnp.array([1e-7, 2e-7])
    with pytest.raises(ValueError, match=r"missing inputs: .*schedule"):
        gaussian_puff_model(
            observations=obs,
            receptor_coords=(jnp.zeros(2), jnp.zeros(2), jnp.zeros(2)),
            observation_times=jnp.array([1.0, 2.0]),
            source_location=(0.0, 0.0, 2.0),
            schedule=None,
        )


def test_gaussian_puff_prior_matches_requested_moments():
    predictive = Predictive(gaussian_puff_model, num_samples=20000)
    samples = predictive(
        jax.random.PRNGKey(0),
        observations=None,
        receptor_coords=None,
        observation_times=None,
        source_location=None,
        schedule=None,
        prior_emission_rate_mean=0.15,
        prior_emission_rate_std=0.05,
    )
    q = np.asarray(samples["emission_rate"])
    np.testing.assert_allclose(q.mean(), 0.15, rtol=0.02)
    np.testing.assert_allclose(q.std(), 0.05, rtol=0.05)


def test_infer_emission_rate_rejects_empty_observations():
    with pytest.raises(ValueError, match=r"must contain ≥ 1 point"):
        infer_emission_rate(
            observations=np.array([]),
            observation_coords=(np.array([]), np.array([]), np.array([])),
            observation_times=np.array([]),
            source_location=(0, 0, 2),
            wind_times=np.linspace(0, 10, 3),
            wind_speed=np.full(3, 5.0),
            wind_direction=np.full(3, 270.0),
            release_frequency=1.0,
            t_start=0.0,
            t_end=10.0,
            num_warmup=1, num_samples=1,
        )


def test_infer_emission_rate_rejects_shape_mismatch():
    obs = np.array([1e-6, 2e-6, 3e-6])
    coords = (
        np.array([200., 400., 600.]),
        np.zeros(3),
        np.ones(3),
    )
    times = np.array([10.0, 20.0])  # wrong length
    with pytest.raises(ValueError, match=r"`observation_times`"):
        infer_emission_rate(
            observations=obs,
            observation_coords=coords,
            observation_times=times,
            source_location=(0, 0, 2),
            wind_times=np.linspace(0, 30, 4),
            wind_speed=np.full(4, 5.0),
            wind_direction=np.full(4, 270.0),
            release_frequency=1.0,
            t_start=0.0,
            t_end=30.0,
            num_warmup=1, num_samples=1,
        )


def test_infer_emission_rate_rejects_bad_prior_mean():
    obs = np.array([1e-6, 2e-6])
    coords = (np.array([200., 400.]), np.zeros(2), np.ones(2))
    times = np.array([10.0, 20.0])
    with pytest.raises(ValueError, match=r"`prior_mean` must be > 0"):
        infer_emission_rate(
            observations=obs, observation_coords=coords,
            observation_times=times,
            source_location=(0, 0, 2),
            wind_times=np.linspace(0, 30, 4),
            wind_speed=np.full(4, 5.0),
            wind_direction=np.full(4, 270.0),
            release_frequency=1.0, t_start=0.0, t_end=30.0,
            prior_mean=0.0,
            num_warmup=1, num_samples=1,
        )


@pytest.mark.slow
def test_infer_emission_rate_recovers_synthetic_Q():
    """NUTS should recover a synthetic emission rate from time-series data."""
    import numpyro
    numpyro.enable_x64(False)

    from plume_simulation.gauss_puff.puff import simulate_puff

    # Simulate a true field.
    t_end = 120.0
    time_array = np.linspace(0.0, t_end, 61, dtype=np.float32)
    true_Q = 0.12
    ds = simulate_puff(
        emission_rate=true_Q,
        source_location=(0.0, 0.0, 2.0),
        wind_speed=np.full(61, 5.0, dtype=np.float32),
        wind_direction=np.full(61, 270.0, dtype=np.float32),
        stability_class="C",
        domain_x=(100, 700, 31),
        domain_y=(-50, 50, 11),
        domain_z=(0, 20, 5),
        time_array=time_array,
        release_frequency=1.0,
    )
    # Sample at a downwind transect (y=0, z=1m), every 10s from t=60.
    # Use np.interp against model coords for synthetic obs.
    transect_x = np.array([200., 300., 400., 500.], dtype=np.float32)
    transect_y = np.zeros_like(transect_x)
    transect_z = np.full_like(transect_x, 1.0)
    obs_times = np.array([60., 80., 100., 120.], dtype=np.float32)

    # For each (x, t), pull the grid value at that x, y=0, z=0 (closest).
    x_grid = ds["x"].values
    y_grid = ds["y"].values
    z_grid = ds["z"].values
    t_grid = ds["time"].values
    obs_list = []
    obs_coords = ([], [], [])
    obs_time_list = []
    rng = np.random.default_rng(0)
    noise_std = 2e-8
    for t_k in obs_times:
        i_t = int(np.argmin(np.abs(t_grid - t_k)))
        for xi in transect_x:
            i_x = int(np.argmin(np.abs(x_grid - xi)))
            i_y = int(np.argmin(np.abs(y_grid - 0.0)))
            i_z = int(np.argmin(np.abs(z_grid - 1.0)))
            clean = float(ds["concentration"].values[i_t, i_x, i_y, i_z])
            obs_list.append(clean + float(rng.normal(0.0, noise_std)))
            obs_coords[0].append(float(xi))
            obs_coords[1].append(0.0)
            obs_coords[2].append(1.0)
            obs_time_list.append(float(t_k))
    obs = np.asarray(obs_list)
    coords = tuple(np.asarray(c, dtype=np.float32) for c in obs_coords)
    obs_times_full = np.asarray(obs_time_list, dtype=np.float32)

    samples = infer_emission_rate(
        observations=obs,
        observation_coords=coords,
        observation_times=obs_times_full,
        source_location=(0.0, 0.0, 2.0),
        wind_times=time_array,
        wind_speed=np.full(61, 5.0, dtype=np.float32),
        wind_direction=np.full(61, 270.0, dtype=np.float32),
        release_frequency=1.0,
        t_start=0.0,
        t_end=float(t_end),
        stability_class="C",
        prior_mean=0.1,
        prior_std=0.08,
        obs_noise_std=3e-8,
        num_warmup=200,
        num_samples=400,
        num_chains=1,
        seed=0,
    )
    q_mean = float(samples["emission_rate"].mean())
    q_std = float(samples["emission_rate"].std())
    assert 0.03 < q_mean < 0.30
    assert q_std < 1.2 * q_mean
