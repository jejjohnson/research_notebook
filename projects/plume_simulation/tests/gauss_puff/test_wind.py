"""Tests for the wind schedule and diffrax cumulative integrals."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from plume_simulation.gauss_puff.wind import (
    WindSchedule,
    cumulative_wind_integrals,
)


def test_from_speed_direction_west_wind():
    # Wind "from west" (270°) should give +u, ≈0 v.
    times = np.linspace(0, 10, 5)
    speed = np.full(5, 5.0)
    direction = np.full(5, 270.0)
    sched = WindSchedule.from_speed_direction(times, speed, direction)
    np.testing.assert_allclose(sched.u_wind, 5.0, atol=1e-5)
    np.testing.assert_allclose(sched.v_wind, 0.0, atol=1e-5)


def test_from_speed_direction_north_wind():
    # Wind "from north" (0°): flows south → v negative.
    times = np.linspace(0, 10, 5)
    sched = WindSchedule.from_speed_direction(
        times, np.full(5, 3.0), np.full(5, 0.0)
    )
    np.testing.assert_allclose(sched.u_wind, 0.0, atol=1e-5)
    np.testing.assert_allclose(sched.v_wind, -3.0, atol=1e-5)


def test_constant_wind_cumulative_integrals():
    # Constant 5 m/s from west → I_u(t) = 5t, I_v(t) = 0, S(t) = 5t.
    times = np.linspace(0.0, 60.0, 7)
    sched = WindSchedule.from_speed_direction(
        times, np.full(7, 5.0), np.full(7, 270.0)
    )
    save_at = jnp.array([0.0, 15.0, 30.0, 60.0])
    I_u, I_v, S = cumulative_wind_integrals(sched, save_at)
    np.testing.assert_allclose(I_u, 5.0 * save_at, atol=1e-3)
    np.testing.assert_allclose(I_v, jnp.zeros_like(save_at), atol=1e-3)
    np.testing.assert_allclose(S, 5.0 * save_at, atol=1e-3)


def test_piecewise_linear_wind_integrals():
    # u(t) is 5 for [0,10], linearly 5→3 for [10,20], 3 for [20,30].
    # Analytical cumulative:
    #   F(10) = 50, F(20) = 50 + (5+3)/2 * 10 = 90, F(30) = 120
    # At t=15: 50 + trapezoid(5, (5+3)/2 ≈ 4) * 5 = 50 + (5+4)/2*5 = 72.5
    times = jnp.array([0., 10., 20., 30.])
    u_vals = jnp.array([5., 5., 3., 3.])
    sched = WindSchedule(times=times, u_wind=u_vals, v_wind=jnp.zeros(4))
    save_at = jnp.array([0.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    I_u, I_v, S = cumulative_wind_integrals(sched, save_at)
    expected_I_u = jnp.array([0.0, 50.0, 72.5, 90.0, 105.0, 120.0])
    np.testing.assert_allclose(I_u, expected_I_u, atol=0.05)
    # Since v = 0, S = |u|·integral = I_u here.
    np.testing.assert_allclose(S, expected_I_u, atol=0.05)
    np.testing.assert_allclose(I_v, jnp.zeros_like(save_at), atol=1e-4)


def test_wind_speed_accumulation_tracks_vector_magnitude():
    # Wind blowing diagonally at 5 m/s (45° from west → u=v=5/√2 each).
    # Not a meteorological "from" convention case — direct (u, v) construction.
    s = 5.0 / np.sqrt(2.0)
    times = jnp.array([0., 30., 60.])
    sched = WindSchedule(
        times=times,
        u_wind=jnp.full(3, s),
        v_wind=jnp.full(3, s),
    )
    save_at = jnp.array([0.0, 30.0, 60.0])
    I_u, I_v, S = cumulative_wind_integrals(sched, save_at)
    # |V| = 5, so S(t) = 5t.
    np.testing.assert_allclose(S, 5.0 * save_at, atol=1e-3)
    np.testing.assert_allclose(I_u, s * save_at, atol=1e-3)
    np.testing.assert_allclose(I_v, s * save_at, atol=1e-3)


def test_wind_at_interpolates_linearly():
    times = jnp.array([0., 10., 20.])
    u_vals = jnp.array([5., 10., 5.])
    sched = WindSchedule(times=times, u_wind=u_vals, v_wind=jnp.zeros(3))
    u, v = sched.wind_at(jnp.asarray(5.0))
    np.testing.assert_allclose(float(u), 7.5, atol=1e-5)
    np.testing.assert_allclose(float(v), 0.0, atol=1e-5)
