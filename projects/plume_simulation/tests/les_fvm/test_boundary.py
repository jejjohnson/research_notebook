"""Tests for les_fvm.boundary."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.les_fvm.boundary import (
    HorizontalBC,
    VerticalBC,
    apply_boundary_conditions,
    build_default_concentration_bc,
)
from plume_simulation.les_fvm.grid import make_grid


def _build_grid():
    return make_grid(
        domain_x=(0.0, 200.0, 8),
        domain_y=(0.0, 200.0, 8),
        domain_z=(0.0, 40.0, 4),
    )


def test_default_periodic_y_horizontal_bc_wraps_edges():
    g = _build_grid()
    hbc, vbc = build_default_concentration_bc(
        bc_x=("dirichlet", "outflow"),
        bc_y="periodic",
        bc_z=("neumann", "neumann"),
    )
    field = jnp.zeros(g.shape)
    # Place a marker at the southernmost interior row.
    field = field.at[2, 1, 2].set(7.0)
    out = hbc(field, dx=g.dx, dy=g.dy)
    # Periodic y: the north ghost row copies the southernmost interior row.
    np.testing.assert_allclose(float(out[2, -1, 2]), 7.0)


def test_default_dirichlet_west_sets_ghost_to_enforce_boundary_value():
    g = _build_grid()
    hbc, _ = build_default_concentration_bc(
        bc_x=(("dirichlet", 3.0), ("outflow", 0.0)),
        bc_y="periodic",
        bc_z=("neumann", "neumann"),
    )
    field = jnp.full(g.shape, 5.0)
    out = hbc(field, dx=g.dx, dy=g.dy)
    # Dirichlet ghost is ``2 * value - interior`` so the half-cell average
    # of (ghost, interior) equals ``value``.
    np.testing.assert_allclose(float(out[2, 2, 0]), 2.0 * 3.0 - 5.0)


def test_vertical_neumann_zero_gradient_mirrors_interior():
    g = _build_grid()
    _, vbc = build_default_concentration_bc(
        bc_z=("neumann", "neumann"),
    )
    field = jnp.zeros(g.shape).at[1, 3, 3].set(2.0).at[-2, 3, 3].set(-1.0)
    out = vbc(field, dz=g.dz)
    # Ghost slices mirror nearest interior rows when gradient = 0.
    np.testing.assert_allclose(float(out[0, 3, 3]), 2.0)
    np.testing.assert_allclose(float(out[-1, 3, 3]), -1.0)


def test_vertical_neumann_nonzero_gradient_applies_offset():
    # Regression test for PR #16: non-zero Neumann gradients must be
    # propagated into the ghost row via ghost = interior + sign·grad·dz.
    g = _build_grid()
    _, vbc = build_default_concentration_bc(
        bc_z=(("neumann", 0.25), ("neumann", -0.1)),
    )
    field = jnp.ones(g.shape) * 5.0
    out = vbc(field, dz=g.dz)
    # Bottom: outward sign is -1, so ghost = 5 + (-1) * 0.25 * dz = 5 - 0.25·dz.
    np.testing.assert_allclose(
        float(out[0, 3, 3]), 5.0 - 0.25 * g.dz, rtol=1e-6
    )
    # Top: outward sign is +1, ghost = 5 + (+1) * (-0.1) * dz.
    np.testing.assert_allclose(
        float(out[-1, 3, 3]), 5.0 - 0.1 * g.dz, rtol=1e-6
    )


def test_vertical_dirichlet_sets_correct_ghost():
    g = _build_grid()
    _, vbc = build_default_concentration_bc(
        bc_z=(("dirichlet", 0.5), ("dirichlet", 1.5)),
    )
    field = jnp.ones(g.shape) * 2.0
    out = vbc(field, dz=g.dz)
    # Dirichlet ghost at bottom/top: ghost = 2 * value - interior.
    np.testing.assert_allclose(float(out[0, 3, 3]), 2.0 * 0.5 - 2.0)
    np.testing.assert_allclose(float(out[-1, 3, 3]), 2.0 * 1.5 - 2.0)


def test_apply_boundary_conditions_composes_horizontal_and_vertical():
    g = _build_grid()
    hbc, vbc = build_default_concentration_bc(
        bc_x=("dirichlet", "outflow"),
        bc_y="periodic",
        bc_z=("neumann", "neumann"),
    )
    field = jnp.ones(g.shape)
    out = apply_boundary_conditions(field, hbc, vbc, g)
    # No NaN / inf leaks, shape preserved, ghost slices touched.
    assert out.shape == g.shape
    assert jnp.all(jnp.isfinite(out))


def test_unknown_horizontal_kind_rejected():
    with pytest.raises(ValueError, match=r"horizontal BC kind must be one of"):
        build_default_concentration_bc(bc_x=("mystery", "outflow"))


def test_unknown_vertical_kind_rejected():
    with pytest.raises(ValueError, match=r"vertical BC kind must be one of"):
        build_default_concentration_bc(bc_z=("mystery", "neumann"))


def test_bc_set_types_returned():
    hbc, vbc = build_default_concentration_bc()
    assert isinstance(hbc, HorizontalBC)
    assert isinstance(vbc, VerticalBC)
