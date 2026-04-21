"""Tests for les_fvm.diffusion — K-theory eddy diffusivity."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.les_fvm.diffusion import (
    EddyDiffusivity,
    diffusion_tendency,
    make_eddy_diffusivity,
    pg_eddy_diffusivity,
)
from plume_simulation.les_fvm.grid import make_grid


def _build_grid():
    return make_grid(
        domain_x=(0.0, 100.0, 8),
        domain_y=(0.0, 100.0, 8),
        domain_z=(0.0, 40.0, 4),
    )


def test_constant_field_has_zero_tendency():
    g = _build_grid()
    c = jnp.ones(g.shape)
    kappa = EddyDiffusivity(horizontal=1.5, vertical=0.4)
    t = diffusion_tendency(c, kappa, g)
    np.testing.assert_allclose(np.asarray(t), 0.0, atol=1e-6)


def test_quadratic_z_field_matches_expected_tendency():
    # For C(z) = z² and uniform K_z, vertical Laplacian = 2 * K_z.
    g = _build_grid()
    z = jnp.arange(g.shape[0], dtype=jnp.float32) * g.dz
    c = jnp.broadcast_to(z[:, None, None] ** 2, g.shape)
    kappa = EddyDiffusivity(horizontal=0.0, vertical=2.0)
    t = diffusion_tendency(c, kappa, g)
    # Interior cells carry 2 * K_z = 4.0; ghost rings are zero.
    t_int = np.asarray(t)[1:-1, 1:-1, 1:-1]
    np.testing.assert_allclose(t_int, 4.0, atol=1e-3)


def test_pg_eddy_diffusivity_positive_values():
    eddy = pg_eddy_diffusivity(
        stability_class="C", wind_speed=5.0, reference_distance=200.0
    )
    assert eddy.horizontal > 0.0
    assert eddy.vertical > 0.0
    # Under neutral class C the vertical diffusivity is smaller than the
    # lateral, consistent with σ_z < σ_y.
    assert eddy.vertical < eddy.horizontal


def test_pg_eddy_diffusivity_rejects_zero_wind_speed():
    with pytest.raises(ValueError, match=r"wind_speed must be > 0"):
        pg_eddy_diffusivity(stability_class="C", wind_speed=0.0)


def test_pg_eddy_diffusivity_rejects_zero_reference_distance():
    with pytest.raises(ValueError, match=r"reference_distance must be > 0"):
        pg_eddy_diffusivity(
            stability_class="C", wind_speed=5.0, reference_distance=0.0
        )


def test_make_eddy_diffusivity_from_scalar():
    e = make_eddy_diffusivity(0.7)
    assert isinstance(e, EddyDiffusivity)
    assert e.horizontal == 0.7
    assert e.vertical == 0.7


def test_make_eddy_diffusivity_from_tuple():
    e = make_eddy_diffusivity((1.5, 0.3))
    assert e.horizontal == 1.5
    assert e.vertical == 0.3


def test_make_eddy_diffusivity_pg_requires_class_and_speed():
    with pytest.raises(ValueError, match=r"requires both"):
        make_eddy_diffusivity("pg")


def test_make_eddy_diffusivity_unknown_string_rejected():
    with pytest.raises(ValueError, match=r"unknown string spec"):
        make_eddy_diffusivity("smagorinsky")
