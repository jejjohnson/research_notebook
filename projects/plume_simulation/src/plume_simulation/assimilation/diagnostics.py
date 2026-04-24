"""Posterior diagnostics for a converged 3D-Var solution.

After the optimiser has produced ``x̂``, three quick scalars tell you whether
the result is statistically consistent with the prior + likelihood you
specified:

- :func:`reduced_chi_squared` — ``χ² = ‖y − H(x̂)‖²_R⁻¹ / dim(y)``. ≈ 1 means
  the posterior fits the observations to the noise level. Much smaller →
  over-fitting (R too generous); much larger → systematic errors not in R.
- :func:`degrees_of_freedom_for_signal` — ``DFS = trace(I − A)`` with the
  averaging kernel ``A = Bₐ B⁻¹`` (Rodgers, 2000). For low-rank ``B`` this is
  cheap; for the dense form we use Hutchinson's stochastic trace estimator.
- :func:`posterior_covariance_proxy` — the inverse Hessian at the optimum.
  Returned as a callable that takes a probe vector ``v`` and returns
  ``Bₐ v`` (i.e. you can evaluate variances per-pixel without forming
  the full ``Bₐ`` matrix).

These are diagnostics, not exact statistics — for the linear-Gaussian case
they coincide with the textbook posterior; for the nonlinear case they're
Laplace approximations around the optimum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np


if TYPE_CHECKING:
    pass


def reduced_chi_squared(
    *,
    forward_fn: Callable[[jax.Array], jax.Array],
    estimated_state: jax.Array,
    observation: jax.Array,
    obs_inv_variance: float | jax.Array,
) -> float:
    """``χ²_red = (y - H(x̂))ᵀ R⁻¹ (y - H(x̂)) / dim(y)``.

    Values near 1 indicate the posterior is consistent with the observation
    error model; ≪ 1 → noise overestimated; ≫ 1 → noise underestimated or
    systematic forward-model error.
    """
    y = jnp.asarray(observation)
    R_inv = jnp.asarray(obs_inv_variance)
    residual = y - forward_fn(estimated_state)
    chi2 = float(jnp.sum(R_inv * residual * residual))
    return chi2 / int(y.size)


def degrees_of_freedom_for_signal(
    *,
    hessian_vector_product: Callable[[jax.Array], jax.Array],
    background_op: lx.AbstractLinearOperator,
    state_size: int,
    n_probes: int = 32,
    seed: int = 0,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-10,
    cg_max_steps: int = 2000,
) -> float:
    """Estimate ``DFS = trace(I − Bₐ B⁻¹)`` (Rodgers 2000, §2.5).

    For the linear-Gaussian case the averaging kernel is ``A = I − Bₐ B⁻¹``
    and ``DFS = trace(A)`` measures how many independent pieces of information
    the observations contributed (``DFS = 0`` → no information; ``DFS = N`` →
    observations fully pin the state). The posterior covariance is the
    inverse Hessian at the optimum, ``Bₐ ≈ Hess⁻¹``, so each Hutchinson probe
    needs:

    - one ``B⁻¹ z`` via :func:`gaussx.solve` (structured dispatch);
    - one ``Hess⁻¹ w`` via :func:`lineax.linear_solve` (CG on the HVP).

    Cost per probe: ``O(n_CG · forward)`` — dominated by the CG solve.

    Earlier versions of this function computed ``trace(B · Hess)``, which is
    not DFS: in the zero-information limit (``Hess = B⁻¹``) it returns ``N``
    instead of the correct ``0``. See PR-review thread on diagnostics.py.
    """
    import gaussx as gx

    # Build a CG-solvable operator for ``Hess⁻¹``.
    hess_op = lx.FunctionLinearOperator(
        lambda v: hessian_vector_product(jnp.asarray(v)),
        input_structure=jax.eval_shape(lambda: jnp.zeros(state_size)),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    cg_solver = lx.CG(rtol=cg_rtol, atol=cg_atol, max_steps=cg_max_steps)

    rng = np.random.default_rng(seed)
    z_batch = rng.choice([-1.0, 1.0], size=(n_probes, state_size))
    total = 0.0
    for z in z_batch:
        z_j = jnp.asarray(z, dtype=jnp.float64)
        # w = B⁻¹ z     (structured dispatch, cheap for Kronecker / low-rank)
        w = gx.solve(background_op, z_j)
        # u = Hess⁻¹ w  (matrix-free CG)
        u = lx.linear_solve(hess_op, w, solver=cg_solver, throw=False).value
        # zᵀ (I − Bₐ B⁻¹) z ≈ zᵀ z − zᵀ (Hess⁻¹ B⁻¹) z
        total += float(jnp.dot(z_j, z_j) - jnp.dot(z_j, u))
    return total / n_probes


def posterior_covariance_proxy(
    *,
    hessian_vector_product: Callable[[jax.Array], jax.Array],
    state_size: int,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-9,
    cg_max_steps: int = 200,
) -> Callable[[jax.Array], jax.Array]:
    """Return a callable ``v → Bₐ v`` via CG on the Hessian.

    ``Bₐ ≈ Hess⁻¹`` is the Laplace-approximation posterior covariance. We
    expose it as a matvec (rather than materialising) so per-pixel variance
    estimates remain ``O(state · cg_iters)`` — fine for moderate scenes.
    """

    def matvec(v: jax.Array) -> jax.Array:
        return hessian_vector_product(jnp.asarray(v))

    op = lx.FunctionLinearOperator(
        matvec,
        input_structure=jax.eval_shape(lambda: jnp.zeros(state_size)),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )

    def apply(v: jax.Array) -> jax.Array:
        # ``throw=False`` returns the best partial solution at max_steps rather
        # than raising — for diagnostics we'd rather get a noisy estimate than
        # crash the whole notebook on a stiff probe direction.
        return lx.linear_solve(
            op, jnp.asarray(v),
            solver=lx.CG(rtol=cg_rtol, atol=cg_atol, max_steps=cg_max_steps),
            throw=False,
        ).value

    return apply
