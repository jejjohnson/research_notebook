"""Tests for the JAX observation operator.

Three invariants:

1. **Forward shape + zero-VMR sanity** — at ``ΔVMR = 0`` the radiance equals
   the L1-normalised SRF acting on a flat unit spectrum, i.e. ``1.0`` per band.
2. **Linear == nonlinear at small ΔVMR** — Maclaurin-1 differs from
   ``exp(−Δτ)`` by O(Δτ²); for tiny ΔVMR they should match to ~1e-6.
3. **Adjoint identity via JAX VJP** — for the *linear* forward,
   ``⟨H δx, u⟩ = ⟨δx, Hᵀ u⟩`` to machine precision via :func:`jax.vjp`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)


def test_forward_shape_no_optics(obs_model_no_optics):
    model = obs_model_no_optics
    x = jnp.zeros((6, 6))
    y = model.forward(x)
    assert y.shape == (6, 6, model.n_bands)


def test_forward_zero_vmr_is_unit_radiance(obs_model_no_optics):
    """At ΔVMR = 0, exp(0) = 1 and the SRF integrates to 1 per band."""
    model = obs_model_no_optics
    y = model.forward(jnp.zeros((4, 4)))
    np.testing.assert_allclose(np.asarray(y), 1.0, atol=1e-12)


def test_linear_approximates_nonlinear_at_small_vmr(obs_model_no_optics):
    model = obs_model_no_optics
    x = jnp.full((3, 3), 1e-9)  # small VMR perturbation
    y_nl = model.forward(x, linear=False)
    y_lin = model.forward(x, linear=True)
    # Δτ ≈ a · 1e-9 ≈ 1e-9 · O(1e-19 cross-section · 1e19 number density · 1e6 path)
    # → very small; linear and nonlinear agree to 1e-9 absolute.
    np.testing.assert_allclose(np.asarray(y_nl), np.asarray(y_lin), atol=1e-8)


def test_with_psf_gsd_shape(obs_model_with_gsd):
    model = obs_model_with_gsd
    y = model.forward(jnp.zeros((8, 8)))
    assert y.shape == (4, 4, model.n_bands)


def test_jax_vjp_adjoint_identity_linear(obs_model_no_optics):
    """For the linear forward, ⟨H δx, u⟩ = ⟨δx, Hᵀ u⟩ via jax.vjp."""
    model = obs_model_no_optics

    def H(delta_x_field):
        # Linear forward, shifted to remove the constant '1' so we get a pure
        # linear map: H(δx) = -a(ν) · δx (band-integrated).
        return model.forward(delta_x_field, linear=True) - model.forward(
            jnp.zeros_like(delta_x_field), linear=True
        )

    rng = np.random.default_rng(0)
    dx = jnp.asarray(rng.standard_normal((4, 4)))
    u = jnp.asarray(rng.standard_normal((4, 4, model.n_bands)))
    Hdx, vjp = jax.vjp(H, dx)
    (Htu,) = vjp(u)
    lhs = float(jnp.sum(Hdx * u))
    rhs = float(jnp.sum(dx * Htu))
    assert abs(lhs - rhs) < 1e-10 * max(abs(lhs), abs(rhs), 1.0)


def test_jit_runs(obs_model_no_optics):
    model = obs_model_no_optics
    fwd = jax.jit(model.make_forward(linear=False))
    y = fwd(jnp.zeros((5, 5)))
    assert y.shape == (5, 5, model.n_bands)
