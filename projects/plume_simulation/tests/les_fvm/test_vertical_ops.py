"""Tests for les_fvm._vertical_ops — vertical flux + diffusion helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from plume_simulation.les_fvm._vertical_ops import (
    vertical_advection_tendency,
    vertical_diffusion_tendency,
    zero_horizontal_ghosts,
    zero_vertical_ghosts,
)


def _linear_z_field(nz: int, ny: int, nx: int, slope: float = 1.0):
    """Return a field C[k, j, i] = slope * k (linear in z)."""
    z = jnp.arange(nz, dtype=jnp.float32)
    return jnp.broadcast_to(z[:, None, None] * slope, (nz, ny, nx))


def test_vertical_advection_zero_w_gives_zero_tendency():
    # With w = 0 everywhere, the vertical advection tendency must vanish.
    nz, ny, nx = 6, 4, 4
    c = _linear_z_field(nz, ny, nx)
    w = jnp.zeros((nz, ny, nx))
    t = vertical_advection_tendency(c, w, dz=1.0)
    np.testing.assert_allclose(np.asarray(t), 0.0, atol=1e-6)


def test_vertical_advection_matches_upwind_analytic_positive_w():
    # Linear profile C = z, constant w > 0 everywhere.  The upwind flux at
    # k-face with w>0 uses C from the cell below; tendency at T[k] is
    # -(F[k+1/2] - F[k-1/2])/dz = -(w·(k) - w·(k-1))/dz = -w.
    nz, ny, nx = 6, 1, 1
    c = _linear_z_field(nz, ny, nx, slope=1.0)  # C[k,...] = k
    w = jnp.full((nz, ny, nx), 2.0)
    dz = 1.0
    t = vertical_advection_tendency(c, w, dz=dz)
    # Ghost cells: k=0 and k=-1 remain zero.
    t_np = np.asarray(t)
    np.testing.assert_allclose(t_np[0], 0.0)
    np.testing.assert_allclose(t_np[-1], 0.0)
    # Interior cells: tendency = -w = -2.0 at every interior k.
    np.testing.assert_allclose(t_np[1:-1], -2.0, atol=1e-5)


def test_vertical_advection_matches_upwind_analytic_negative_w():
    # For w < 0 the upwind reconstruction pulls from above.  Tendency is
    # -(F[k+1/2] - F[k-1/2])/dz = -w * (C[k+1] - C[k])/dz = -w for C = z.
    nz, ny, nx = 6, 1, 1
    c = _linear_z_field(nz, ny, nx, slope=1.0)
    w = jnp.full((nz, ny, nx), -3.0)
    t = vertical_advection_tendency(c, w, dz=1.0)
    t_np = np.asarray(t)
    np.testing.assert_allclose(t_np[1:-1], 3.0, atol=1e-5)  # = -w


def test_vertical_diffusion_constant_profile_zero_tendency():
    # Uniform C → Laplacian is zero.
    nz, ny, nx = 6, 4, 4
    c = jnp.ones((nz, ny, nx))
    t = vertical_diffusion_tendency(c, kappa_z=1.5, dz=0.5)
    np.testing.assert_allclose(np.asarray(t), 0.0, atol=1e-6)


def test_vertical_diffusion_quadratic_profile():
    # C = z² → ∂²C/∂z² = 2 everywhere.  Tendency at T[k] should equal
    # 2 * kappa_z to within the discretisation error of a 2nd-order
    # central difference (which is exact for quadratics on a uniform grid).
    nz, ny, nx = 6, 1, 1
    dz = 1.0
    z = jnp.arange(nz, dtype=jnp.float32) * dz
    c = jnp.broadcast_to((z**2)[:, None, None], (nz, ny, nx))
    kappa = 2.0
    t = vertical_diffusion_tendency(c, kappa_z=kappa, dz=dz)
    t_np = np.asarray(t)
    np.testing.assert_allclose(t_np[1:-1], 2.0 * kappa, atol=1e-5)
    # Ghost rows untouched.
    np.testing.assert_allclose(t_np[0], 0.0)
    np.testing.assert_allclose(t_np[-1], 0.0)


def test_vertical_diffusion_handles_field_kappa():
    # Field-valued kappa should be averaged at k-faces before flux
    # computation.  On a quadratic profile with constant kappa supplied
    # as a 3-D field, the result must match the scalar-kappa case.
    nz, ny, nx = 6, 2, 2
    dz = 0.5
    z = jnp.arange(nz, dtype=jnp.float32) * dz
    c = jnp.broadcast_to((z**2)[:, None, None], (nz, ny, nx))
    kappa_scalar = 1.7
    kappa_field = jnp.full((nz, ny, nx), kappa_scalar)
    t_scalar = vertical_diffusion_tendency(c, kappa_z=kappa_scalar, dz=dz)
    t_field = vertical_diffusion_tendency(c, kappa_z=kappa_field, dz=dz)
    np.testing.assert_allclose(np.asarray(t_scalar), np.asarray(t_field), atol=1e-6)


def test_zero_ghost_helpers_do_not_touch_interior():
    field = jnp.arange(6 * 4 * 4, dtype=jnp.float32).reshape(6, 4, 4)
    interior = np.asarray(field)[1:-1, 1:-1, 1:-1]

    zeroed_h = zero_horizontal_ghosts(field)
    np.testing.assert_allclose(
        np.asarray(zeroed_h)[1:-1, 1:-1, 1:-1], interior
    )
    # Horizontal ghost rings are zero.
    assert float(jnp.abs(zeroed_h[:, 0, :]).max()) == 0.0
    assert float(jnp.abs(zeroed_h[:, -1, :]).max()) == 0.0
    assert float(jnp.abs(zeroed_h[:, :, 0]).max()) == 0.0
    assert float(jnp.abs(zeroed_h[:, :, -1]).max()) == 0.0

    zeroed_v = zero_vertical_ghosts(field)
    np.testing.assert_allclose(
        np.asarray(zeroed_v)[1:-1, 1:-1, 1:-1], interior
    )
    assert float(jnp.abs(zeroed_v[0]).max()) == 0.0
    assert float(jnp.abs(zeroed_v[-1]).max()) == 0.0
