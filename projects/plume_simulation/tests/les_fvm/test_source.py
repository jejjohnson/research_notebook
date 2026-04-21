"""Tests for les_fvm.source — Gaussian point-source emission term."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.les_fvm.grid import make_grid
from plume_simulation.les_fvm.source import (
    GaussianSource,
    make_gaussian_source,
)


def _build_grid():
    return make_grid(
        domain_x=(0.0, 400.0, 16),
        domain_y=(0.0, 200.0, 8),
        domain_z=(0.0, 80.0, 8),
    )


def test_constant_source_integrates_to_emission_rate():
    g = _build_grid()
    q = 0.07
    source = make_gaussian_source(
        plume_grid=g,
        emission_rate=q,
        source_location=(50.0, 100.0, 20.0),
    )
    # Integrate the source tendency ∫ S dV.  Only interior cells carry mass,
    # so we sum the interior slice and multiply by the cell volume.
    s = source(jnp.asarray(0.0))
    interior = s[1:-1, 1:-1, 1:-1]
    total = float(interior.sum()) * (g.dx * g.dy * g.dz)
    # The Gaussian density is normalised to integrate to 1, so the result
    # must be exactly ``q`` up to floating-point noise.
    np.testing.assert_allclose(total, q, rtol=1e-5)


def test_time_varying_emission_reflects_rate_function():
    g = _build_grid()
    # q(t) = 0.01 * t → linearly ramping emission.
    rate = lambda t: 0.01 * t
    source = make_gaussian_source(
        plume_grid=g,
        emission_rate=rate,
        source_location=(100.0, 100.0, 20.0),
    )
    for t_query in (0.0, 5.0, 10.0):
        s = source(jnp.asarray(t_query, dtype=jnp.float32))
        interior = s[1:-1, 1:-1, 1:-1]
        total = float(interior.sum()) * (g.dx * g.dy * g.dz)
        np.testing.assert_allclose(total, rate(t_query), rtol=1e-5)


def test_constant_negative_rate_rejected():
    g = _build_grid()
    with pytest.raises(ValueError, match=r"`emission_rate` must be"):
        make_gaussian_source(
            plume_grid=g,
            emission_rate=-0.1,
            source_location=(100.0, 100.0, 20.0),
        )


def test_zero_rate_accepted_and_yields_zero_source():
    g = _build_grid()
    source = make_gaussian_source(
        plume_grid=g,
        emission_rate=0.0,
        source_location=(100.0, 100.0, 20.0),
    )
    s = source(jnp.asarray(0.0))
    np.testing.assert_allclose(np.asarray(s), 0.0, atol=1e-10)


def test_source_outside_domain_rejected():
    g = _build_grid()
    with pytest.raises(ValueError, match=r"outside the interior domain"):
        make_gaussian_source(
            plume_grid=g,
            emission_rate=0.1,
            source_location=(-10.0, 100.0, 20.0),
        )


def test_negative_radius_rejected():
    g = _build_grid()
    with pytest.raises(ValueError, match=r"`source_radius` must be > 0"):
        make_gaussian_source(
            plume_grid=g,
            emission_rate=0.1,
            source_location=(100.0, 100.0, 20.0),
            source_radius=-1.0,
        )


def test_source_pytree_roundtrips_via_equinox():
    # Smoke test: GaussianSource must be a pytree leaf compatible with
    # JAX transformations.  A scalar eqx.field + a density array.
    import equinox as eqx

    g = _build_grid()
    source = make_gaussian_source(
        plume_grid=g,
        emission_rate=0.1,
        source_location=(100.0, 100.0, 20.0),
    )
    # The density is a JAX array; emission_fn is a Python closure.
    leaves, _ = eqx.tree_flatten_one_level(source)
    assert any(isinstance(leaf, jnp.ndarray) for leaf in leaves)
    assert isinstance(source, GaussianSource)
