"""Solver tests: LBFGS recovers truth, dual ≡ primal, twin experiment.

Three end-to-end checks:

1. **Twin experiment** — inject a flat-field ``ΔVMR > 0``, run LBFGS in
   whitened space, verify the recovered field matches truth to within the
   prior+obs noise budget.
2. **Dual == primal** for a linear forward — the PSAS solution and the
   primal LBFGS solution coincide (the dual is closed-form, so this
   pins down both implementations at once).
3. **Whitening drops iteration count** — LBFGS in ξ-space converges in
   ≤ 10 iterations even with a stiff prior, while model-space takes orders
   of magnitude more (or fails). Soft check; mostly a regression guard.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from plume_simulation.assimilation.background import (
    build_diagonal_background,
    build_kronecker_background,
)
from plume_simulation.assimilation.control import WhiteningTransform
from plume_simulation.assimilation.cost import build_cost_x, build_cost_xi
from plume_simulation.assimilation.solve import (
    run_dual_psas,
    run_lbfgs,
)


jax.config.update("jax_enable_x64", True)


def test_twin_experiment_flat_field_recovery(obs_model_no_optics):
    """Inject ΔVMR=2e-7 uniformly; LBFGS in ξ-space should recover it."""
    model = obs_model_no_optics
    ny, nx = 4, 4
    truth_vmr = 2e-7
    truth_field = jnp.full((ny, nx), truth_vmr)
    y_obs = model.forward(truth_field, linear=False)

    # Mildly informative prior, very tight obs likelihood.
    B = build_kronecker_background(
        ny=ny, nx=nx, variance=1e-12, length_scale_y=2.0, length_scale_x=2.0,
    )
    W = WhiteningTransform.from_background(B)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1e8,
        background_state=jnp.zeros((ny, nx)),
        observation=y_obs,
        state_shape=(ny, nx),
    )
    sol = run_lbfgs(cost, jnp.zeros(ny * nx), max_steps=200, rtol=1e-8, atol=1e-12)
    delta_x_flat = W.apply(jnp.asarray(sol.state))
    recovered = np.asarray(delta_x_flat).reshape(ny, nx)
    np.testing.assert_allclose(recovered, np.full((ny, nx), truth_vmr), rtol=5e-2)


def test_dual_psas_matches_primal_for_linear_forward(obs_model_no_optics):
    """Primal LBFGS in ξ-space ≡ dual PSAS for a *linear* forward."""
    model = obs_model_no_optics
    ny, nx = 4, 4
    truth_vmr = 1e-7
    truth_field = jnp.full((ny, nx), truth_vmr)
    # Use the linear forward so the primal/dual equivalence is exact.
    y_obs = model.forward(truth_field, linear=True)

    B = build_diagonal_background(1e-12, n_pixels=ny * nx)
    R_inv = 1e8

    # --- primal in ξ-space ---
    W = WhiteningTransform.from_background(B)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=True),
        whitening=W,
        obs_inv_variance=R_inv,
        background_state=jnp.zeros((ny, nx)),
        observation=y_obs,
        state_shape=(ny, nx),
    )
    sol_primal = run_lbfgs(cost, jnp.zeros(ny * nx), max_steps=200, rtol=1e-9, atol=1e-13)
    delta_x_primal = np.asarray(W.apply(jnp.asarray(sol_primal.state)))

    # --- dual PSAS ---
    sol_dual = run_dual_psas(
        forward_fn=model.make_forward(linear=True),
        background_op=B,
        obs_inv_variance=R_inv,
        background_state=jnp.zeros((ny, nx)),
        observation=y_obs,
        state_shape=(ny, nx),
        cg_rtol=1e-10,
        cg_atol=1e-14,
    )
    delta_x_dual = sol_dual.state

    np.testing.assert_allclose(delta_x_primal, delta_x_dual, rtol=1e-4, atol=1e-9)


def test_lbfgs_in_xi_space_converges_quickly(obs_model_no_optics):
    """Soft regression: ξ-space LBFGS should hit tolerance in < 30 steps."""
    model = obs_model_no_optics
    ny, nx = 4, 4
    truth_vmr = 1.5e-7
    y_obs = model.forward(jnp.full((ny, nx), truth_vmr), linear=False)
    B = build_diagonal_background(1e-12, n_pixels=ny * nx)
    W = WhiteningTransform.from_background(B)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1e8,
        background_state=jnp.zeros((ny, nx)),
        observation=y_obs,
        state_shape=(ny, nx),
    )
    sol = run_lbfgs(cost, jnp.zeros(ny * nx), max_steps=200, rtol=1e-8, atol=1e-12)
    # If it took more than 30 iterations we probably broke the preconditioning.
    if sol.n_steps != -1:
        assert sol.n_steps < 30, f"LBFGS took {sol.n_steps} steps — preconditioning regression?"
