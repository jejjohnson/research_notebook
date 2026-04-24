"""Control variable transform tests.

Whitening identity: ``δx = U ξ`` then ``U⁻¹ δx == ξ`` to ~1e-9.

Plus the cost-shape invariant: in ξ-space the prior is
``½ ξᵀ ξ`` (no B⁻¹ involved), so for ξ = e₀ the cost should equal 0.5 + obs term.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from plume_simulation.assimilation.background import (
    build_diagonal_background,
    build_kronecker_background,
)
from plume_simulation.assimilation.control import (
    IdentityTransform,
    WhiteningTransform,
)


jax.config.update("jax_enable_x64", True)


def test_identity_round_trip():
    T = IdentityTransform()
    x = jnp.arange(5, dtype=jnp.float64)
    np.testing.assert_array_equal(np.asarray(T.apply(x)), np.asarray(x))
    np.testing.assert_array_equal(np.asarray(T.apply_inverse(x)), np.asarray(x))


def test_whitening_round_trip_diagonal():
    B = build_diagonal_background(0.25, n_pixels=6)
    T = WhiteningTransform.from_background(B)
    rng = np.random.default_rng(0)
    xi = jnp.asarray(rng.standard_normal(6))
    delta_x = T.apply(xi)
    xi_back = T.apply_inverse(delta_x)
    np.testing.assert_allclose(np.asarray(xi_back), np.asarray(xi), atol=1e-9)


def test_whitening_round_trip_kronecker():
    B = build_kronecker_background(
        ny=3, nx=3, variance=0.04, length_scale_y=2.0, length_scale_x=2.0,
    )
    T = WhiteningTransform.from_background(B)
    rng = np.random.default_rng(1)
    xi = jnp.asarray(rng.standard_normal(9))
    delta_x = T.apply(xi)
    xi_back = T.apply_inverse(delta_x)
    np.testing.assert_allclose(np.asarray(xi_back), np.asarray(xi), atol=1e-8)


def test_whitening_recovers_B_via_UUt():
    """U Uᵀ should equal B as a matvec — the defining property of Cholesky."""
    B = build_kronecker_background(
        ny=3, nx=3, variance=0.04, length_scale_y=1.5, length_scale_x=1.5,
    )
    T = WhiteningTransform.from_background(B)
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal(9))
    Bx = B.mv(x)
    UUtx = T.apply(T.cholesky_op.transpose().mv(x))
    np.testing.assert_allclose(np.asarray(Bx), np.asarray(UUtx), atol=1e-9)
