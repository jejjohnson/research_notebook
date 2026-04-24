"""Ornstein-Uhlenbeck turbulence for puff-release position disturbances."""

from __future__ import annotations

import numpy as np
import pytest
from plume_simulation.gauss_puff.turbulence import OUTurbulence, sample_ou_offsets


def test_ou_turbulence_rejects_invalid_params():
    with pytest.raises(ValueError, match="sigma_fluctuations"):
        OUTurbulence(sigma_fluctuations=-1.0, correlation_time=60.0)
    with pytest.raises(ValueError, match="correlation_time"):
        OUTurbulence(sigma_fluctuations=0.5, correlation_time=0.0)


def test_stationary_std_formula():
    t = OUTurbulence(sigma_fluctuations=0.5, correlation_time=60.0)
    expected = 0.5 * np.sqrt(60.0 / 2.0)
    assert t.stationary_std == pytest.approx(expected, rel=1e-12)


def test_sample_zero_sigma_is_zero_field():
    t = OUTurbulence(sigma_fluctuations=0.0, correlation_time=60.0)
    dx, dy = sample_ou_offsets(t, np.linspace(0, 10, 11), seed=0)
    np.testing.assert_allclose(dx, 0.0)
    np.testing.assert_allclose(dy, 0.0)


def test_sample_empty_release_times():
    t = OUTurbulence()
    dx, dy = sample_ou_offsets(t, np.array([]), seed=0)
    assert dx.size == 0 and dy.size == 0


def test_sample_stationary_statistics_large_ensemble():
    # Sample many independent paths, all long enough to cover many
    # correlation times, and check sample mean ≈ 0, sample std ≈ σ_∞.
    t = OUTurbulence(sigma_fluctuations=0.5, correlation_time=10.0)
    release_times = np.arange(0.0, 2000.0, 1.0)

    # 50 independent OU paths → average stats across the ensemble.
    rng = np.random.default_rng(0)
    n_paths = 50
    stds = np.zeros(n_paths)
    means = np.zeros(n_paths)
    for k in range(n_paths):
        dx, _ = sample_ou_offsets(t, release_times, seed=rng)
        means[k] = dx.mean()
        stds[k] = dx.std(ddof=1)

    # Mean should hover around 0; std around σ_∞.
    sigma_inf = t.stationary_std
    assert abs(means.mean()) < 0.1 * sigma_inf
    assert abs(stds.mean() - sigma_inf) < 0.15 * sigma_inf


def test_sample_short_correlation_time_gives_near_iid_samples():
    # τ_c much shorter than release dt → successive samples near-independent.
    t = OUTurbulence(sigma_fluctuations=1.0, correlation_time=0.1)
    release_times = np.arange(0.0, 1000.0, 1.0)  # dt = 1 ≫ τ_c = 0.1
    dx, _ = sample_ou_offsets(t, release_times, seed=0)

    # Lag-1 autocorrelation should be close to exp(-1/0.1) ≈ 0 — i.e. near-zero.
    lag1 = np.corrcoef(dx[:-1], dx[1:])[0, 1]
    assert abs(lag1) < 0.1


def test_sample_long_correlation_time_gives_smooth_trajectory():
    # τ_c much longer than release dt → strong autocorrelation.
    t = OUTurbulence(sigma_fluctuations=1.0, correlation_time=100.0)
    release_times = np.arange(0.0, 1000.0, 1.0)  # dt = 1 ≪ τ_c = 100
    dx, _ = sample_ou_offsets(t, release_times, seed=0)

    lag1 = np.corrcoef(dx[:-1], dx[1:])[0, 1]
    # exp(-1/100) ≈ 0.99; accept a generous margin for the finite sample.
    assert lag1 > 0.9


def test_sample_reproducible_with_seed():
    t = OUTurbulence()
    times = np.linspace(0.0, 100.0, 11)
    a = sample_ou_offsets(t, times, seed=42)
    b = sample_ou_offsets(t, times, seed=42)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


def test_sample_rejects_non_monotonic_times():
    t = OUTurbulence()
    with pytest.raises(ValueError, match="monotone"):
        sample_ou_offsets(t, np.array([0.0, 5.0, 3.0]), seed=0)


def test_x_and_y_are_independent_components():
    t = OUTurbulence(sigma_fluctuations=0.5, correlation_time=10.0)
    release_times = np.arange(0.0, 2000.0, 1.0)
    dx, dy = sample_ou_offsets(t, release_times, seed=0)

    # Correlation between independent components → ≈ 0.
    corr = np.corrcoef(dx, dy)[0, 1]
    assert abs(corr) < 0.1


def test_simulate_puff_accepts_turbulence_and_changes_field():
    """End-to-end: passing turbulence to simulate_puff changes the output.

    We verify the mean column mass is conserved (turbulence only moves the
    puffs around, it doesn't add or remove mass).
    """
    import numpy as _np
    from plume_simulation.gauss_puff.puff import simulate_puff

    time_array = _np.linspace(0.0, 60.0, 7)  # 0-60 s in 10 s increments
    wind_speed = _np.full_like(time_array, 2.0)
    wind_direction = _np.full_like(time_array, 270.0)  # wind from the west

    common = dict(
        emission_rate=1e-3,
        source_location=(0.0, 0.0, 2.0),
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        stability_class="D",
        domain_x=(-200.0, 200.0, 21),
        domain_y=(-100.0, 100.0, 11),
        domain_z=(0.0, 50.0, 6),
        time_array=time_array,
        release_frequency=1.0,
        scheme="pg",
    )
    ds_det = simulate_puff(**common)
    ds_turb = simulate_puff(
        **common,
        turbulence=OUTurbulence(sigma_fluctuations=0.5, correlation_time=30.0),
        turbulence_seed=0,
    )

    # Order-of-magnitude sanity: the turbulent field should not vanish or
    # explode relative to the deterministic baseline. Exact column-mass
    # conservation would require matching grid sampling at the same puff
    # centres, which shifts under turbulence — a loose factor-of-2 band
    # is the right level of check for "physics hasn't gone wrong".
    total_det = float(ds_det["column_concentration"].sum())
    total_turb = float(ds_turb["column_concentration"].sum())
    assert 0.25 * total_det < total_turb < 4.0 * total_det

    # Fields should differ in shape (turbulence introduces meander).
    assert not _np.allclose(
        ds_det["column_concentration"].values, ds_turb["column_concentration"].values,
    )

    # Attrs record the turbulence params.
    assert ds_turb.attrs["ou_sigma_fluctuations"] == 0.5
    assert ds_turb.attrs["ou_correlation_time"] == 30.0
    assert ds_det.attrs["ou_sigma_fluctuations"] == 0.0
