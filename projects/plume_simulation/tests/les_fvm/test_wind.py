"""Tests for les_fvm.wind — prescribed wind fields."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from plume_simulation.gauss_puff.wind import WindSchedule
from plume_simulation.les_fvm.grid import make_grid
from plume_simulation.les_fvm.wind import (
    PrescribedWindField,
    uniform_wind_field,
    wind_field_from_callable,
    wind_field_from_schedule,
)


def _build_grid():
    return make_grid(
        domain_x=(0.0, 200.0, 8),
        domain_y=(0.0, 100.0, 4),
        domain_z=(0.0, 40.0, 4),
    )


def test_uniform_wind_constant_in_space_and_time():
    g = _build_grid()
    wf = uniform_wind_field(g, u=5.0, v=1.0, w=0.3)
    for t in (0.0, 100.0):
        u, v, w = wf(jnp.asarray(t, dtype=jnp.float32))
        assert u.shape == g.shape
        np.testing.assert_allclose(np.asarray(u), 5.0)
        np.testing.assert_allclose(np.asarray(v), 1.0)
        np.testing.assert_allclose(np.asarray(w), 0.3)


def test_wind_field_from_schedule_time_varying():
    g = _build_grid()
    schedule = WindSchedule.from_speed_direction(
        times=jnp.linspace(0.0, 100.0, 11),
        wind_speed=jnp.linspace(2.0, 7.0, 11),
        wind_direction=jnp.full(11, 270.0),  # wind from the west → u > 0
    )
    wf = wind_field_from_schedule(g, schedule)
    # At t=0 the speed is 2; at t=100 it's 7; at t=50 it's 4.5 (linear).
    for t_query, expected in [(0.0, 2.0), (50.0, 4.5), (100.0, 7.0)]:
        u, v, w = wf(jnp.asarray(t_query, dtype=jnp.float32))
        np.testing.assert_allclose(np.asarray(u).mean(), expected, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(v), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.asarray(w), 0.0)


def test_wind_field_from_callable_sees_coordinate_arrays():
    g = _build_grid()

    def shear_flow(t, X, Y, Z):
        del t, X, Y
        # u = 0.1 * z, constant v = 0, constant w = 0
        return 0.1 * Z, jnp.zeros_like(Z), jnp.zeros_like(Z)

    wf = wind_field_from_callable(g, shear_flow)
    u, v, w = wf(jnp.asarray(0.0))
    # u is z-proportional; interior slice values should match 0.1 * z_interior.
    u_int = np.asarray(u)[1:-1, 1:-1, 1:-1]
    # Check that each z-level has a constant u equal to 0.1 * g.z[k].
    for k, z_val in enumerate(np.asarray(g.z)):
        np.testing.assert_allclose(u_int[k].mean(), 0.1 * z_val, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(v), 0.0)
    np.testing.assert_allclose(np.asarray(w), 0.0)


def test_prescribed_wind_field_returns_pytree():
    g = _build_grid()
    wf = uniform_wind_field(g, u=1.0, v=0.0, w=0.0)
    assert isinstance(wf, PrescribedWindField)
    # Callable when sampled.
    u, v, w = wf(jnp.asarray(0.0))
    assert u.shape == w.shape == v.shape == g.shape


def test_wind_field_from_callable_is_reexported_from_package_root():
    # Regression test for PR #16: wind_field_from_callable was reachable
    # via plume_simulation.les_fvm.wind but not via the package root.
    import plume_simulation.les_fvm as L

    assert L.wind_field_from_callable is wind_field_from_callable
    assert "wind_field_from_callable" in L.__all__
