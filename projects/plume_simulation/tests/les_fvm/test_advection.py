"""Tests for les_fvm.advection."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.les_fvm.advection import advection_tendency
from plume_simulation.les_fvm.grid import make_grid


def _build_grid():
    return make_grid(
        domain_x=(0.0, 200.0, 16),
        domain_y=(0.0, 200.0, 16),
        domain_z=(0.0, 40.0, 8),
    )


def test_uniform_concentration_no_flow_tendency():
    # For uniform C and zero flow, the advective tendency vanishes identically.
    g = _build_grid()
    c = jnp.ones(g.shape)
    zero = jnp.zeros(g.shape)
    t = advection_tendency(c, zero, zero, zero, g, method="weno5")
    np.testing.assert_allclose(np.asarray(t), 0.0, atol=1e-6)


def test_uniform_concentration_with_uniform_flow_has_zero_tendency_interior():
    # ∇·(u C) = C ∇·u + u·∇C = 0 when C is constant and u is constant.
    g = _build_grid()
    c = jnp.full(g.shape, 3.0)
    u = jnp.full(g.shape, 4.0)
    v = jnp.full(g.shape, -1.0)
    w = jnp.full(g.shape, 0.5)
    t = advection_tendency(c, u, v, w, g, method="weno5")
    # WENO writes only to the [1:-1, 2:-2, 2:-2] horizontal interior; the
    # outer ring is zero by construction.  The core interior must be ~0.
    np.testing.assert_allclose(
        np.asarray(t)[1:-1, 2:-2, 2:-2], 0.0, atol=1e-5
    )


def test_advection_reduces_to_vertical_when_horizontal_uniform():
    # With ∇_h C = 0 the total advective tendency equals the vertical term.
    g = _build_grid()
    # C depends only on z.
    z = jnp.arange(g.shape[0], dtype=jnp.float32)
    c = jnp.broadcast_to(z[:, None, None], g.shape)
    u = jnp.zeros(g.shape)
    v = jnp.zeros(g.shape)
    w = jnp.full(g.shape, 1.5)  # positive w
    t = advection_tendency(c, u, v, w, g, method="weno5")
    # Vertical upwind with w>0 and linear C gives tendency = -w/dz.
    # Interior z-range is [1:-1]; take a representative slice in the
    # *core* horizontal interior to ignore WENO's outer-ring zero mask.
    t_core = np.asarray(t)[1:-1, 2:-2, 2:-2]
    expected = -1.5 / g.dz
    np.testing.assert_allclose(t_core, expected, atol=1e-4)


def test_advection_jit_compiles_and_is_deterministic():
    # Calling twice with the same inputs should produce bit-identical outputs
    # — catches subtle tracing bugs where arrays are captured from globals.
    g = _build_grid()
    c = jnp.zeros(g.shape).at[3, 6, 6].set(1.0)
    u = jnp.full(g.shape, 2.0)
    v = jnp.zeros(g.shape)
    w = jnp.zeros(g.shape)
    t1 = advection_tendency(c, u, v, w, g, method="weno5")
    t2 = advection_tendency(c, u, v, w, g, method="weno5")
    np.testing.assert_array_equal(np.asarray(t1), np.asarray(t2))


def test_unknown_advection_scheme_raises():
    g = _build_grid()
    c = jnp.ones(g.shape)
    u = jnp.zeros(g.shape)
    with pytest.raises(ValueError, match=r"Unknown method"):
        advection_tendency(c, u, u, u, g, method="not_a_scheme")
