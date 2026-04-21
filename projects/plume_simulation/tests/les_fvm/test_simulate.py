"""Integration tests for the high-level ``simulate_eulerian_dispersion`` runner."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from plume_simulation.gauss_puff.wind import WindSchedule
from plume_simulation.les_fvm import (
    pg_eddy_diffusivity,
    simulate_eulerian_dispersion,
    uniform_wind_field,
)
from plume_simulation.les_fvm.grid import make_grid


def _common_kwargs(**overrides):
    base = dict(
        domain_x=(0.0, 400.0, 16),
        domain_y=(0.0, 200.0, 8),
        domain_z=(0.0, 80.0, 8),
        t_start=0.0,
        t_end=20.0,
        save_interval=10.0,
        emission_rate=0.1,
        source_location=(50.0, 100.0, 20.0),
        uniform_wind=(5.0, 0.0, 0.0),
        eddy_diffusivity="pg",
        stability_class="C",
        solver="tsit5",
        dt0=0.5,
    )
    base.update(overrides)
    return base


def test_simulate_returns_xarray_dataset_with_expected_vars():
    ds = simulate_eulerian_dispersion(**_common_kwargs())
    assert isinstance(ds, xr.Dataset)
    assert "concentration" in ds.data_vars
    assert "column_concentration" in ds.data_vars
    # Coords match the requested domain.
    assert ds.sizes == {"time": 3, "z": 8, "y": 8, "x": 16}


def test_simulate_zero_emission_yields_zero_concentration():
    ds = simulate_eulerian_dispersion(**_common_kwargs(emission_rate=0.0))
    np.testing.assert_allclose(ds["concentration"].values, 0.0, atol=1e-10)


def test_simulate_positive_emission_produces_downwind_plume():
    ds = simulate_eulerian_dispersion(**_common_kwargs())
    c = ds["concentration"].values  # (time, z, y, x)
    # At the last save, there must be non-trivial concentration somewhere
    # downwind of the source (x > 50.0).
    last = c[-1]
    x_coord = ds["x"].values
    downwind_mask = x_coord > 60.0
    max_downwind = last[..., downwind_mask].max()
    assert max_downwind > 0.0


def test_simulate_mass_grows_with_time_under_constant_emission():
    # Total mass under a constant emission rate ``q`` should grow roughly as
    # ``q * t`` minus outflow.  With an open downstream BC and a weak
    # transient, the monotonicity in ``t`` is what we really care about here.
    ds = simulate_eulerian_dispersion(**_common_kwargs(t_end=30.0, save_interval=5.0))
    mass = (
        ds["concentration"].sum(dim=("x", "y", "z"))
        * (400.0 / 16) * (200.0 / 8) * (80.0 / 8)
    ).values
    # Mass is non-decreasing across the early saves (before much outflow).
    assert np.all(np.diff(mass[:4]) >= -1e-6)


def test_simulate_rejects_missing_wind_spec():
    with pytest.raises(ValueError, match=r"exactly one of"):
        simulate_eulerian_dispersion(
            domain_x=(0.0, 100.0, 8),
            domain_y=(0.0, 100.0, 8),
            domain_z=(0.0, 40.0, 4),
            t_start=0.0,
            t_end=5.0,
            save_interval=1.0,
            emission_rate=0.01,
            source_location=(20.0, 50.0, 10.0),
            eddy_diffusivity=1.0,
            # (No wind supplied.)
        )


def test_simulate_rejects_multiple_wind_specs():
    schedule = WindSchedule.from_speed_direction(
        times=jnp.linspace(0.0, 5.0, 3),
        wind_speed=jnp.full(3, 5.0),
        wind_direction=jnp.full(3, 270.0),
    )
    with pytest.raises(ValueError, match=r"exactly one of"):
        simulate_eulerian_dispersion(
            domain_x=(0.0, 100.0, 8),
            domain_y=(0.0, 100.0, 8),
            domain_z=(0.0, 40.0, 4),
            t_start=0.0,
            t_end=5.0,
            save_interval=1.0,
            emission_rate=0.01,
            source_location=(20.0, 50.0, 10.0),
            eddy_diffusivity=1.0,
            uniform_wind=(5.0, 0.0, 0.0),
            wind_schedule=schedule,
        )


def test_simulate_pg_diffusivity_requires_nonzero_wind():
    # 'pg' + uniform_wind=(0,0,0) → no meaningful K-theory calibration.
    with pytest.raises(ValueError, match=r"non-zero mean wind"):
        simulate_eulerian_dispersion(
            domain_x=(0.0, 100.0, 8),
            domain_y=(0.0, 100.0, 8),
            domain_z=(0.0, 40.0, 4),
            t_start=0.0,
            t_end=5.0,
            save_interval=1.0,
            emission_rate=0.01,
            source_location=(20.0, 50.0, 10.0),
            uniform_wind=(0.0, 0.0, 0.0),
            eddy_diffusivity="pg",
            stability_class="C",
        )


def test_simulate_pg_accepts_calm_start_then_windy_schedule():
    # Regression test for PR #16: the PG calibration used to take the
    # first schedule knot only, which rejected a calm-start-then-windy
    # schedule.  The time-mean speed across all knots must be used
    # instead, so this schedule now runs successfully.
    n = 7
    times = jnp.linspace(0.0, 60.0, n)
    speed = jnp.concatenate(
        [jnp.zeros(2), jnp.linspace(2.0, 7.0, n - 2)]  # calm then ramp
    )
    schedule = WindSchedule.from_speed_direction(
        times=times,
        wind_speed=speed,
        wind_direction=jnp.full(n, 270.0),
    )
    ds = simulate_eulerian_dispersion(
        domain_x=(0.0, 400.0, 16),
        domain_y=(0.0, 200.0, 8),
        domain_z=(0.0, 80.0, 8),
        t_start=0.0, t_end=30.0, save_interval=15.0,
        emission_rate=0.05,
        source_location=(50.0, 100.0, 20.0),
        wind_schedule=schedule,
        eddy_diffusivity="pg",
        stability_class="C",
        solver="tsit5", dt0=0.5,
    )
    assert float(ds["concentration"].max()) >= 0.0  # no NaNs + valid dataset


def test_simulate_save_interval_not_dividing_window_does_not_overshoot():
    # Regression test for PR #16 (second round): save_interval=6 and
    # window 0..10 used to build save_times=[0, 6, 12], and 12 > t_end
    # was rejected by diffrax.SaveAt.  The clipped / endpoint-snapped
    # build must keep every save time inside [t_start, t_end].
    ds = simulate_eulerian_dispersion(
        domain_x=(0.0, 200.0, 8),
        domain_y=(0.0, 100.0, 4),
        domain_z=(0.0, 40.0, 4),
        t_start=0.0, t_end=10.0, save_interval=6.0,
        emission_rate=0.01,
        source_location=(20.0, 50.0, 10.0),
        uniform_wind=(5.0, 0.0, 0.0),
        eddy_diffusivity=1.0,
        solver="tsit5", dt0=0.5,
    )
    times = ds["time"].values
    assert times[0] == 0.0
    assert times[-1] <= 10.0 + 1e-6
    # Also verify we still got at least two save points (start and end).
    assert times.size >= 2


def test_simulate_pg_calibration_uses_simulation_window_only():
    # Regression test for PR #16 (second round): the PG calibration mean
    # speed must depend only on the simulated window, not on schedule knot
    # density nor on knots outside [t_start, t_end].  Two schedules that
    # agree over the run window but differ outside it must produce the
    # same concentration field.
    times_short = jnp.asarray([0.0, 5.0, 10.0])
    speed_short = jnp.asarray([5.0, 5.0, 5.0])
    direction = jnp.asarray([270.0, 270.0, 270.0])
    schedule_short = WindSchedule.from_speed_direction(
        times=times_short, wind_speed=speed_short, wind_direction=direction
    )

    # A longer schedule that agrees with `schedule_short` on [0, 10] but
    # has a wild calm period before and after — the calibration *must*
    # ignore those out-of-window knots.
    times_long = jnp.asarray([-20.0, -10.0, 0.0, 5.0, 10.0, 20.0, 30.0])
    speed_long = jnp.asarray([0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0])
    direction_long = jnp.full(7, 270.0)
    schedule_long = WindSchedule.from_speed_direction(
        times=times_long, wind_speed=speed_long, wind_direction=direction_long
    )

    common_kwargs = dict(
        domain_x=(0.0, 400.0, 16),
        domain_y=(0.0, 200.0, 8),
        domain_z=(0.0, 80.0, 8),
        t_start=0.0, t_end=10.0, save_interval=5.0,
        emission_rate=0.05,
        source_location=(50.0, 100.0, 20.0),
        eddy_diffusivity="pg",
        stability_class="C",
        solver="tsit5", dt0=0.5,
    )
    ds_short = simulate_eulerian_dispersion(**common_kwargs, wind_schedule=schedule_short)
    ds_long = simulate_eulerian_dispersion(**common_kwargs, wind_schedule=schedule_long)
    np.testing.assert_allclose(
        ds_short["concentration"].values,
        ds_long["concentration"].values,
        rtol=1e-4,
        atol=1e-10,
    )


def test_simulate_pg_rejects_all_zero_schedule():
    # The guard must still fire when the entire schedule is calm — no
    # physically meaningful PG calibration exists in that case.
    n = 5
    times = jnp.linspace(0.0, 20.0, n)
    schedule = WindSchedule.from_speed_direction(
        times=times,
        wind_speed=jnp.zeros(n),
        wind_direction=jnp.full(n, 270.0),
    )
    with pytest.raises(ValueError, match=r"non-zero mean wind"):
        simulate_eulerian_dispersion(
            domain_x=(0.0, 100.0, 8),
            domain_y=(0.0, 100.0, 8),
            domain_z=(0.0, 40.0, 4),
            t_start=0.0, t_end=5.0, save_interval=1.0,
            emission_rate=0.01,
            source_location=(20.0, 50.0, 10.0),
            wind_schedule=schedule,
            eddy_diffusivity="pg",
            stability_class="C",
        )


def test_simulate_rejects_mismatched_initial_concentration():
    # Initial tracer shape must match the interior grid shape (no ghost ring).
    with pytest.raises(ValueError, match=r"does not match interior shape"):
        simulate_eulerian_dispersion(
            **_common_kwargs(initial_concentration=jnp.zeros((4, 4, 4)))
        )


def test_simulate_accepts_explicit_wind_field():
    # Passing a pre-built PrescribedWindField should skip both the uniform
    # and schedule branches and work identically to uniform_wind=(5, 0, 0).
    g = make_grid(
        domain_x=(0.0, 400.0, 16),
        domain_y=(0.0, 200.0, 8),
        domain_z=(0.0, 80.0, 8),
    )
    wf = uniform_wind_field(g, u=5.0, v=0.0, w=0.0)
    eddy = pg_eddy_diffusivity(stability_class="C", wind_speed=5.0)
    ds = simulate_eulerian_dispersion(
        domain_x=(0.0, 400.0, 16),
        domain_y=(0.0, 200.0, 8),
        domain_z=(0.0, 80.0, 8),
        t_start=0.0, t_end=10.0, save_interval=5.0,
        emission_rate=0.1,
        source_location=(50.0, 100.0, 20.0),
        wind_field=wf,
        eddy_diffusivity=eddy,
        solver="tsit5", dt0=0.5,
    )
    assert float(ds["concentration"].max()) > 0.0


def test_simulate_with_time_varying_wind_schedule():
    schedule = WindSchedule.from_speed_direction(
        times=jnp.linspace(0.0, 30.0, 7),
        wind_speed=jnp.linspace(3.0, 7.0, 7),
        wind_direction=jnp.full(7, 270.0),
    )
    ds = simulate_eulerian_dispersion(
        domain_x=(0.0, 400.0, 16),
        domain_y=(0.0, 200.0, 8),
        domain_z=(0.0, 80.0, 8),
        t_start=0.0, t_end=20.0, save_interval=10.0,
        emission_rate=0.05,
        source_location=(50.0, 100.0, 20.0),
        wind_schedule=schedule,
        eddy_diffusivity="pg",
        stability_class="C",
        solver="tsit5", dt0=0.5,
    )
    assert float(ds["concentration"].max()) > 0.0
