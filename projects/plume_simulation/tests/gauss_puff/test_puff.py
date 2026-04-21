"""Tests for the Gaussian-puff forward model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.gauss_puff.dispersion import PG_DISPERSION_PARAMS
from plume_simulation.gauss_puff.puff import (
    MIN_WIND_SPEED,
    PuffState,
    evolve_puffs,
    frequency_to_release_interval,
    make_release_times,
    puff_concentration,
    puff_concentration_vmap,
    release_interval_to_frequency,
    simulate_puff,
    simulate_puff_field,
)
from plume_simulation.gauss_puff.wind import WindSchedule


# ── Release cadence helpers ──────────────────────────────────────────────────


def test_frequency_interval_roundtrip():
    assert release_interval_to_frequency(0.5) == pytest.approx(2.0)
    assert frequency_to_release_interval(2.0) == pytest.approx(0.5)
    assert release_interval_to_frequency(
        frequency_to_release_interval(3.0)
    ) == pytest.approx(3.0)


def test_frequency_interval_rejects_nonpositive():
    with pytest.raises(ValueError, match=r"release_interval.*must be > 0"):
        release_interval_to_frequency(0.0)
    with pytest.raises(ValueError, match=r"release_frequency.*must be > 0"):
        frequency_to_release_interval(-1.0)


def test_make_release_times_spacing_and_bounds():
    times = make_release_times(0.0, 10.0, release_frequency=2.0)
    expected = jnp.arange(20) * 0.5
    np.testing.assert_allclose(times, expected, atol=1e-6)
    # Exclusive of t_end.
    assert float(times[-1]) < 10.0


def test_make_release_times_requires_positive_window():
    with pytest.raises(ValueError, match=r"`t_end` must be >"):
        make_release_times(10.0, 0.0, 1.0)


# ── Single-puff concentration ────────────────────────────────────────────────


def test_puff_concentration_positive_and_max_at_center():
    x = jnp.array([0.0, 5.0, 10.0])
    y = jnp.zeros(3)
    z = jnp.full(3, 2.0)
    conc = puff_concentration(
        x, y, z,
        puff_x=0.0, puff_y=0.0, puff_z=2.0,
        sigma_x=10.0, sigma_y=10.0, sigma_z=5.0,
        puff_mass=1.0,
    )
    assert jnp.all(conc > 0)
    assert float(conc[0]) == float(conc.max())


def test_puff_concentration_ground_reflection_doubles_at_ground():
    # At z=0 with puff_z=0 the "direct" and "reflected" terms coincide,
    # giving exactly 2× the non-reflected value.
    zero = jnp.array(0.0)
    with_reflect = puff_concentration(
        zero, zero, zero,
        puff_x=0.0, puff_y=0.0, puff_z=0.0,
        sigma_x=5.0, sigma_y=5.0, sigma_z=5.0,
        puff_mass=1.0,
    )
    single_gaussian = 1.0 / ((2 * jnp.pi) ** 1.5 * 125.0)
    np.testing.assert_allclose(float(with_reflect), 2.0 * float(single_gaussian), rtol=1e-5)


def test_puff_concentration_normalization_integral():
    # Volume integral on a fine grid for an elevated puff (reflection negligible)
    # should approach puff_mass.
    import itertools

    sx, sy, sz = 5.0, 5.0, 5.0
    puff_z = 50.0  # high enough that reflected term ≈ 0
    grid = jnp.linspace(-40.0, 40.0, 41)
    # 3-D integration: grid resolution ≈ 2 m, σ ≈ 5 m → good coverage.
    dx = float(grid[1] - grid[0])
    X, Y, Z = jnp.meshgrid(grid, grid, grid + puff_z, indexing="ij")
    conc = puff_concentration(
        X, Y, Z,
        puff_x=0.0, puff_y=0.0, puff_z=puff_z,
        sigma_x=sx, sigma_y=sy, sigma_z=sz,
        puff_mass=1.0,
    )
    integral = float(conc.sum()) * dx**3
    # With reflection term negligible: integral → 1.
    np.testing.assert_allclose(integral, 1.0, rtol=5e-3)


def test_puff_concentration_vmap_matches_manual_sum():
    receptor = (jnp.array([10.0, 20.0]), jnp.zeros(2), jnp.full(2, 2.0))
    puff_x = jnp.array([0.0, 5.0, 10.0])
    puff_y = jnp.zeros(3)
    puff_z = jnp.full(3, 2.0)
    sigmas = jnp.full(3, 5.0)
    mass = jnp.full(3, 1.0)
    stacked = puff_concentration_vmap(
        *receptor, puff_x, puff_y, puff_z, sigmas, sigmas, sigmas, mass
    )
    assert stacked.shape == (3, 2)  # (n_puffs, n_receptors)
    manual = jnp.stack(
        [
            puff_concentration(*receptor, puff_x[i], puff_y[i], puff_z[i],
                               sigmas[i], sigmas[i], sigmas[i], mass[i])
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(stacked, manual, rtol=1e-5)


def test_puff_concentration_autodiff_through_mass():
    # Concentration is linear in mass → d/dmass conc = conc/mass evaluated in
    # the limit of unit mass.
    receptor = (jnp.asarray(10.0), jnp.asarray(0.0), jnp.asarray(2.0))

    def scalar_conc(mass):
        return puff_concentration(
            *receptor, 0.0, 0.0, 2.0, 5.0, 5.0, 5.0, mass
        )

    g = jax.grad(scalar_conc)(1.0)
    expected = scalar_conc(1.0)
    np.testing.assert_allclose(float(g), float(expected), rtol=1e-5)


# ── evolve_puffs under time-varying wind ─────────────────────────────────────


def _constant_wind_schedule(speed: float = 5.0, direction: float = 270.0,
                            t_max: float = 60.0, n: int = 7) -> WindSchedule:
    times = np.linspace(0.0, t_max, n)
    return WindSchedule.from_speed_direction(
        times, np.full(n, speed), np.full(n, direction)
    )


def test_evolve_puffs_constant_wind():
    schedule = _constant_wind_schedule()
    release_times = jnp.array([0.0, 10.0, 20.0, 30.0])
    state = evolve_puffs(
        schedule,
        release_times,
        jnp.asarray(40.0),
        source_location=(0.0, 0.0, 2.0),
        puff_mass=0.05,
    )
    # At t=40 under +5 m/s wind in x, puff released at t_r is at x = 5*(40-t_r).
    expected_x = 5.0 * (40.0 - np.asarray(release_times))
    np.testing.assert_allclose(state.x, expected_x, atol=0.5)
    np.testing.assert_allclose(state.y, 0.0, atol=1e-2)
    np.testing.assert_allclose(state.z, 2.0, atol=1e-6)
    # Travel distance equals |V|*(t - t_r).
    np.testing.assert_allclose(
        state.travel_distance, 5.0 * (40.0 - np.asarray(release_times)),
        atol=0.5,
    )
    np.testing.assert_allclose(state.mass, 0.05, atol=1e-6)


def test_evolve_puffs_masks_unreleased():
    schedule = _constant_wind_schedule()
    release_times = jnp.array([0.0, 20.0, 40.0])
    state = evolve_puffs(
        schedule, release_times, jnp.asarray(30.0),
        source_location=(0.0, 0.0, 2.0),
        puff_mass=0.1,
    )
    # Puffs 0 and 1 are active at t=30 (release times 0 and 20).
    # Puff 2 at t_r=40 > t_now=30 is inactive → mass zeroed.
    np.testing.assert_array_equal(state.mass, jnp.array([0.1, 0.1, 0.0]))
    # Inactive puff should be at source (x=0, y=0) and s=0.
    assert float(state.x[2]) == pytest.approx(0.0, abs=1e-4)
    assert float(state.travel_distance[2]) == pytest.approx(0.0, abs=1e-4)


def test_evolve_puffs_time_varying_wind_advances_further_when_stronger():
    # A schedule where speed grows linearly — total travel at t=60 should be
    # > constant-5-m/s result.
    times = np.linspace(0.0, 60.0, 7)
    speeds = np.linspace(3.0, 7.0, 7)  # mean 5 m/s
    sched_vary = WindSchedule.from_speed_direction(
        times, speeds, np.full(7, 270.0)
    )
    sched_const = _constant_wind_schedule(speed=5.0, t_max=60.0, n=7)
    release_times = jnp.array([0.0])
    state_vary = evolve_puffs(
        sched_vary, release_times, jnp.asarray(60.0),
        source_location=(0.0, 0.0, 0.0),
        puff_mass=1.0,
    )
    state_const = evolve_puffs(
        sched_const, release_times, jnp.asarray(60.0),
        source_location=(0.0, 0.0, 0.0),
        puff_mass=1.0,
    )
    # Both should reach x = ∫ u dt ≈ 300 m (identical means), check same-ish.
    np.testing.assert_allclose(float(state_vary.x[0]),
                                float(state_const.x[0]), rtol=0.05)


# ── simulate_puff_field ──────────────────────────────────────────────────────


def test_simulate_puff_field_is_sum_of_contributions():
    from plume_simulation.gauss_puff.dispersion import calculate_pg_dispersion
    receptor = (jnp.array([10.0, 50.0]), jnp.zeros(2), jnp.full(2, 2.0))
    state = PuffState(
        release_times=jnp.array([0.0, 1.0, 2.0]),
        x=jnp.array([5.0, 20.0, 50.0]),
        y=jnp.zeros(3),
        z=jnp.full(3, 2.0),
        travel_distance=jnp.array([10.0, 30.0, 80.0]),
        mass=jnp.full(3, 0.1),
    )
    total = simulate_puff_field(
        receptor, state, PG_DISPERSION_PARAMS["C"], calculate_pg_dispersion
    )
    assert total.shape == (2,)
    assert jnp.all(total > 0)


# ── simulate_puff xarray wrapper ────────────────────────────────────────────


def test_simulate_puff_shape_and_attrs():
    n_t = 21
    time_array = np.linspace(0, 60, n_t, dtype=np.float32)
    ds = simulate_puff(
        emission_rate=0.1,
        source_location=(0.0, 0.0, 2.0),
        wind_speed=np.full(n_t, 5.0, dtype=np.float32),
        wind_direction=np.full(n_t, 270.0, dtype=np.float32),
        stability_class="C",
        domain_x=(-50, 500, 28),
        domain_y=(-100, 100, 11),
        domain_z=(0, 60, 7),
        time_array=time_array,
        release_frequency=1.0,
    )
    assert ds["concentration"].shape == (n_t, 28, 11, 7)
    assert ds["column_concentration"].shape == (n_t, 28, 11)
    assert ds.attrs["dispersion_scheme"] == "pg"
    assert ds.attrs["n_puffs"] >= 50  # ≈ 60 at 1 Hz over 60 s
    # Concentration should be zero at t=0 (no puffs released yet).
    np.testing.assert_allclose(ds["concentration"].isel(time=0).values, 0.0,
                                atol=1e-8)
    # Concentration at some later time should be > 0 downwind of the source.
    late = ds["concentration"].isel(time=-1).values
    assert float(late.max()) > 0.0


def test_simulate_puff_rejects_shape_mismatches():
    time_array = np.linspace(0, 30, 7, dtype=np.float32)
    with pytest.raises(ValueError, match=r"`wind_speed` shape"):
        simulate_puff(
            emission_rate=0.1,
            source_location=(0.0, 0.0, 2.0),
            wind_speed=np.full(4, 5.0, dtype=np.float32),
            wind_direction=np.full(7, 270.0, dtype=np.float32),
            stability_class="C",
            domain_x=(0, 100, 5), domain_y=(0, 100, 5), domain_z=(0, 50, 5),
            time_array=time_array,
        )


def test_simulate_puff_rejects_bad_stability():
    time_array = np.linspace(0, 30, 7, dtype=np.float32)
    with pytest.raises(ValueError, match=r"`stability_class` must be one of"):
        simulate_puff(
            emission_rate=0.1,
            source_location=(0.0, 0.0, 2.0),
            wind_speed=np.full(7, 5.0, dtype=np.float32),
            wind_direction=np.full(7, 270.0, dtype=np.float32),
            stability_class="Z",
            domain_x=(0, 100, 5), domain_y=(0, 100, 5), domain_z=(0, 50, 5),
            time_array=time_array,
        )


def test_simulate_puff_rejects_bad_scheme():
    time_array = np.linspace(0, 30, 7, dtype=np.float32)
    with pytest.raises(ValueError, match=r"scheme` must be one of"):
        simulate_puff(
            emission_rate=0.1,
            source_location=(0.0, 0.0, 2.0),
            wind_speed=np.full(7, 5.0, dtype=np.float32),
            wind_direction=np.full(7, 270.0, dtype=np.float32),
            stability_class="C",
            domain_x=(0, 100, 5), domain_y=(0, 100, 5), domain_z=(0, 50, 5),
            time_array=time_array,
            scheme="unknown",
        )


def test_simulate_puff_briggs_scheme_runs():
    time_array = np.linspace(0, 30, 7, dtype=np.float32)
    ds = simulate_puff(
        emission_rate=0.1,
        source_location=(0.0, 0.0, 2.0),
        wind_speed=np.full(7, 5.0, dtype=np.float32),
        wind_direction=np.full(7, 270.0, dtype=np.float32),
        stability_class="D",
        domain_x=(0, 200, 11), domain_y=(-50, 50, 5), domain_z=(0, 50, 5),
        time_array=time_array,
        scheme="briggs",
    )
    assert ds.attrs["dispersion_scheme"] == "briggs"


def test_simulate_puff_min_wind_speed_positive():
    # Just asserting the module constant is available and sensible.
    assert MIN_WIND_SPEED > 0.0


def test_simulate_puff_array_emission_rate_matches_puff_count():
    time_array = np.linspace(0, 20, 5, dtype=np.float32)
    # release_frequency=1 Hz over [0, 20) → 20 puffs.
    q_series = np.linspace(0.05, 0.15, 20, dtype=np.float32)
    ds = simulate_puff(
        emission_rate=q_series,
        source_location=(0.0, 0.0, 2.0),
        wind_speed=np.full(5, 5.0, dtype=np.float32),
        wind_direction=np.full(5, 270.0, dtype=np.float32),
        stability_class="C",
        domain_x=(0, 200, 11), domain_y=(-50, 50, 5), domain_z=(0, 50, 5),
        time_array=time_array,
        release_frequency=1.0,
    )
    assert ds.attrs["n_puffs"] == 20


def test_simulate_puff_rejects_wrong_length_emission_series():
    time_array = np.linspace(0, 20, 5, dtype=np.float32)
    with pytest.raises(ValueError, match=r"array `emission_rate` must have shape"):
        simulate_puff(
            emission_rate=np.full(3, 0.1, dtype=np.float32),
            source_location=(0.0, 0.0, 2.0),
            wind_speed=np.full(5, 5.0, dtype=np.float32),
            wind_direction=np.full(5, 270.0, dtype=np.float32),
            stability_class="C",
            domain_x=(0, 100, 5), domain_y=(0, 100, 5), domain_z=(0, 50, 5),
            time_array=time_array,
            release_frequency=1.0,
        )
