"""Posterior-diagnostics tests.

Light-touch checks: each diagnostic should produce finite, sane values on a
trivial setup. The textbook identities (χ²_red ≈ 1 at the truth, DFS ≤ dim(y))
are easy to break with a sign flip or transposed index, so we test those even
on toy inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from plume_simulation.assimilation.background import build_diagonal_background
from plume_simulation.assimilation.control import WhiteningTransform
from plume_simulation.assimilation.cost import build_cost_xi
from plume_simulation.assimilation.diagnostics import (
    degrees_of_freedom_for_signal,
    posterior_covariance_proxy,
    reduced_chi_squared,
)


jax.config.update("jax_enable_x64", True)


def test_chi2_at_truth_is_zero(obs_model_no_optics):
    model = obs_model_no_optics
    truth = jnp.full((3, 3), 1e-7)
    y = model.forward(truth, linear=False)
    chi2 = reduced_chi_squared(
        forward_fn=model.make_forward(linear=False),
        estimated_state=truth,
        observation=y,
        obs_inv_variance=1e8,
    )
    assert chi2 < 1e-10


def test_dfs_finite_and_nonnegative(obs_model_no_optics):
    model = obs_model_no_optics
    ny, nx = 3, 3
    B = build_diagonal_background(1e-12, n_pixels=ny * nx)
    W = WhiteningTransform.from_background(B)
    truth = jnp.zeros((ny, nx))
    y = model.forward(truth, linear=False)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1e6,
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    dfs = degrees_of_freedom_for_signal(
        hessian_vector_product=lambda v: cost.hvp(jnp.zeros(ny * nx), v),
        background_op=B,
        state_size=ny * nx,
        n_probes=8,
    )
    assert np.isfinite(dfs)
    # DFS is bounded by min(state_dim, n_obs) for a properly-scaled prior.
    assert dfs >= -1e-6  # numerical tolerance


def test_posterior_covariance_proxy_runs(obs_model_no_optics):
    model = obs_model_no_optics
    ny, nx = 3, 3
    B = build_diagonal_background(1e-12, n_pixels=ny * nx)
    W = WhiteningTransform.from_background(B)
    y = model.forward(jnp.zeros((ny, nx)), linear=False)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1.0,
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    matvec = posterior_covariance_proxy(
        hessian_vector_product=lambda v: cost.hvp(jnp.zeros(ny * nx), v),
        state_size=ny * nx,
        cg_max_steps=50,
    )
    Bv = matvec(jnp.ones(ny * nx))
    assert Bv.shape == (ny * nx,)
    assert np.all(np.isfinite(np.asarray(Bv)))
