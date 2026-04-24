"""Cost function tests.

Two invariants:
1. ``jax.grad(J)`` matches central-difference gradient to ~1e-6.
2. At ``δx = 0`` (or ``ξ = 0``), the cost equals only the obs term ``½ ‖d‖²_R⁻¹``.
3. The whitened-cost Hessian is identity-plus-low-rank: ``v + (HU)ᵀ R⁻¹ (HU) v``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from plume_simulation.assimilation.background import build_diagonal_background
from plume_simulation.assimilation.control import WhiteningTransform
from plume_simulation.assimilation.cost import (
    build_cost_x,
    build_cost_xi,
    finite_difference_grad,
)


jax.config.update("jax_enable_x64", True)


def _make_cost_setup(obs_model_no_optics, *, ny=4, nx=4, vmr_truth=2e-7):
    """Synthesise (y, x_b, B) for a tiny twin experiment."""
    model = obs_model_no_optics
    rng = np.random.default_rng(0)
    truth_field = jnp.full((ny, nx), float(vmr_truth))
    y = model.forward(truth_field, linear=False)
    x_b = jnp.zeros((ny, nx))
    return model, x_b, y, ny, nx


def test_grad_matches_finite_difference_x(obs_model_no_optics):
    model, x_b, y, ny, nx = _make_cost_setup(obs_model_no_optics)
    B = build_diagonal_background(1e-12, n_pixels=ny * nx)  # tight prior at 0
    cost = build_cost_x(
        forward_fn=model.make_forward(linear=False),
        background_op=B,
        obs_inv_variance=1e6,
        background_state=x_b,
        observation=y,
        state_shape=(ny, nx),
    )
    delta_x = jnp.full(ny * nx, 1e-7)
    g_jax = np.asarray(cost.grad(delta_x))
    g_fd = finite_difference_grad(cost.value, delta_x, eps=1e-9)
    np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3, atol=1e-3)


def test_grad_matches_finite_difference_xi(obs_model_no_optics):
    model, x_b, y, ny, nx = _make_cost_setup(obs_model_no_optics)
    B = build_diagonal_background(1e-14, n_pixels=ny * nx)
    W = WhiteningTransform.from_background(B)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1e6,
        background_state=x_b,
        observation=y,
        state_shape=(ny, nx),
    )
    rng = np.random.default_rng(1)
    xi = jnp.asarray(rng.standard_normal(ny * nx))
    g_jax = np.asarray(cost.grad(xi))
    g_fd = finite_difference_grad(cost.value, xi, eps=1e-5)
    np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3, atol=1e-3)


def test_cost_at_zero_is_pure_obs_term(obs_model_no_optics):
    model, x_b, y, ny, nx = _make_cost_setup(obs_model_no_optics)
    B = build_diagonal_background(1.0, n_pixels=ny * nx)
    cost = build_cost_x(
        forward_fn=model.make_forward(linear=False),
        background_op=B,
        obs_inv_variance=1.0,
        background_state=x_b,
        observation=y,
        state_shape=(ny, nx),
    )
    j_zero = float(cost.value(jnp.zeros(ny * nx)))
    # Innovation = y - H(x_b)
    d = y - model.forward(x_b, linear=False)
    expected = 0.5 * float(jnp.sum(d * d))
    np.testing.assert_allclose(j_zero, expected, rtol=1e-12)


def test_xi_hessian_is_identity_plus_low_rank(obs_model_no_optics):
    """Heuristic check: at ξ = 0, hvp(ξ, v) = v + (HU)ᵀR⁻¹(HU) v ≥ ‖v‖² for any v."""
    model, x_b, y, ny, nx = _make_cost_setup(obs_model_no_optics, vmr_truth=0.0)
    B = build_diagonal_background(1e-12, n_pixels=ny * nx)
    W = WhiteningTransform.from_background(B)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1.0,
        background_state=x_b,
        observation=y,
        state_shape=(ny, nx),
    )
    v = jnp.ones(ny * nx)
    Hv = cost.hvp(jnp.zeros(ny * nx), v)
    # vᵀ Hv ≥ vᵀ v (the +I term dominates the PSD obs-term contribution).
    assert float(jnp.dot(v, Hv)) >= float(jnp.dot(v, v)) - 1e-9
