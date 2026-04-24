"""3D-Var cost functions and gradients.

The standard incremental 3D-Var minimises

    J(δx) = ½ δxᵀ B⁻¹ δx + ½ ‖y − H(x_b + δx)‖²_R⁻¹

over the increment ``δx = x − x_b``. Two versions live here:

- :func:`build_cost_x` — model-space form. The prior term is evaluated
  via :func:`gaussx.solve(B, δx)` so any structured ``B`` from
  :mod:`.background` works without materialisation.
- :func:`build_cost_xi` — whitened (CVT) form. The prior term collapses
  to ``½ ξᵀ ξ`` and the obs term sees ``δx = U ξ`` via the supplied
  :class:`~plume_simulation.assimilation.control.WhiteningTransform`. This
  is the version we use in practice — gradients are well-scaled and the
  Hessian is identity-plus-low-rank.

In both cases the obs term is differentiated through the JAX forward map
via reverse-mode AD (``jax.grad``), so we don't need to maintain an explicit
adjoint of the observation operator. The :class:`Cost` bundle bundles
``J``, ``∇J`` and a Hessian-vector product ``Hv`` so the optimiser /
posterior-covariance machinery downstream can pick whichever it needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np


if TYPE_CHECKING:
    import lineax as lx

    from plume_simulation.assimilation.control import (
        IdentityTransform,
        WhiteningTransform,
    )


@dataclass(frozen=True)
class Cost:
    """Bundle of (value, gradient, Hessian-vector product) for an optimiser.

    The three callables share the same input signature — a 1-D state vector —
    so the optimiser doesn't have to know which control-space we're working in.
    """

    value: Callable[[jax.Array], jax.Array]
    grad: Callable[[jax.Array], jax.Array]
    value_and_grad: Callable[[jax.Array], tuple[jax.Array, jax.Array]]
    hvp: Callable[[jax.Array, jax.Array], jax.Array]


def _flatten(arr):
    """Flatten the spatial (and any leading) axes — keeps last axis intact."""
    return jnp.reshape(arr, (-1,))


# ── model-space cost ────────────────────────────────────────────────────────


def build_cost_x(
    *,
    forward_fn: Callable[[jax.Array], jax.Array],
    background_op: "lx.AbstractLinearOperator",
    obs_inv_variance: float | jax.Array,
    background_state: jax.Array,
    observation: jax.Array,
    state_shape: tuple[int, ...],
) -> Cost:
    """Build ``J(δx) = ½ δxᵀ B⁻¹ δx + ½ (y − H(x_b + δx))ᵀ R⁻¹ (y − H(x_b + δx))``.

    Parameters
    ----------
    forward_fn : Callable
        ``H : 2-D VMR field → obs cube``. Must be JAX-traceable.
    background_op : lineax operator
        ``B`` (any structured form from :mod:`.background`). The cost calls
        ``gaussx.solve(B, δx_flat)`` once per gradient evaluation.
    obs_inv_variance : float or Array
        ``R⁻¹``. Pass a scalar for ``R = σ² I`` or an array broadcastable
        against the obs cube for diagonal heteroscedastic noise.
    background_state, observation, state_shape
        ``x_b``, ``y_obs``, and the shape of the VMR field (e.g. ``(ny, nx)``).
    """
    x_b = jnp.asarray(background_state)
    y = jnp.asarray(observation)
    R_inv = jnp.asarray(obs_inv_variance)

    def value(delta_x_flat: jax.Array) -> jax.Array:
        delta_x = jnp.reshape(delta_x_flat, state_shape)
        x = x_b + delta_x
        residual = y - forward_fn(x)
        # Prior term: ½ δxᵀ B⁻¹ δx. gaussx.solve dispatches on operator structure.
        Binv_dx = gx.solve(background_op, delta_x_flat)
        prior = 0.5 * jnp.dot(delta_x_flat, Binv_dx)
        obs = 0.5 * jnp.sum(R_inv * residual * residual)
        return prior + obs

    val_and_grad = jax.value_and_grad(value)
    grad = lambda x: val_and_grad(x)[1]

    def hvp(x: jax.Array, v: jax.Array) -> jax.Array:
        # ∂²J/∂x² · v via forward-over-reverse — the standard JAX recipe.
        return jax.jvp(grad, (x,), (v,))[1]

    return Cost(value=value, grad=grad, value_and_grad=val_and_grad, hvp=hvp)


# ── whitened cost ───────────────────────────────────────────────────────────


def build_cost_xi(
    *,
    forward_fn: Callable[[jax.Array], jax.Array],
    whitening: "WhiteningTransform | IdentityTransform",
    obs_inv_variance: float | jax.Array,
    background_state: jax.Array,
    observation: jax.Array,
    state_shape: tuple[int, ...],
) -> Cost:
    """Build ``J(ξ) = ½ ξᵀ ξ + ½ (y − H(x_b + U ξ))ᵀ R⁻¹ (…)``.

    The whitening transform turns the prior into a clean ``½‖ξ‖²``, so the
    Hessian is ``I + (HU)ᵀ R⁻¹ (HU)`` — a small perturbation of the identity.
    Even vanilla LBFGS converges in a handful of iterations regardless of how
    badly conditioned ``B`` was.

    Pass an :class:`~plume_simulation.assimilation.control.IdentityTransform`
    here to recover an unpreconditioned model-space cost with implicit
    ``B = I`` — useful for unit tests of the obs term in isolation.
    """
    x_b = jnp.asarray(background_state)
    y = jnp.asarray(observation)
    R_inv = jnp.asarray(obs_inv_variance)

    def value(xi_flat: jax.Array) -> jax.Array:
        delta_x_flat = whitening.apply(xi_flat)
        delta_x = jnp.reshape(delta_x_flat, state_shape)
        x = x_b + delta_x
        residual = y - forward_fn(x)
        prior = 0.5 * jnp.dot(xi_flat, xi_flat)
        obs = 0.5 * jnp.sum(R_inv * residual * residual)
        return prior + obs

    val_and_grad = jax.value_and_grad(value)
    grad = lambda xi: val_and_grad(xi)[1]

    def hvp(xi: jax.Array, v: jax.Array) -> jax.Array:
        return jax.jvp(grad, (xi,), (v,))[1]

    return Cost(value=value, grad=grad, value_and_grad=val_and_grad, hvp=hvp)


# ── finite-difference cross-check (for unit tests) ─────────────────────────


def finite_difference_grad(
    fn: Callable[[jax.Array], jax.Array], x: jax.Array, *, eps: float = 1e-6
) -> np.ndarray:
    """Central finite differences — used only by tests to validate ``jax.grad``.

    O(N) function evaluations, so keep ``x.size`` small.
    """
    x_np = np.asarray(x, dtype=float).copy()
    g = np.zeros_like(x_np)
    for i in range(x_np.size):
        orig = x_np.flat[i]
        x_np.flat[i] = orig + eps
        f_plus = float(fn(jnp.asarray(x_np)))
        x_np.flat[i] = orig - eps
        f_minus = float(fn(jnp.asarray(x_np)))
        g.flat[i] = (f_plus - f_minus) / (2.0 * eps)
        x_np.flat[i] = orig
    return g
