"""Control-variable transforms for variational data assimilation.

3D-Var minimises a cost ``J(δx)`` in some "control space". The two choices
worth keeping straight are:

1. **Model space** — control variable *is* the increment ``δx = x − x_b``.
   The prior term is ``½ δxᵀ B⁻¹ δx``; one ``B⁻¹``-solve per gradient
   evaluation. Hessian conditioning ``= cond(B⁻¹ + H'ᵀ R⁻¹ H')`` blows up
   when ``B`` is poorly scaled.

2. **Whitened (CVT) space** — control variable is ``ξ`` with
   ``δx = U ξ`` and ``B = U Uᵀ``. The prior collapses to ``½ ξᵀ ξ``
   and the Hessian becomes ``I + (HU)ᵀ R⁻¹ (HU)`` — a small, dense
   identity-plus-low-rank matrix that even simple LBFGS solves in a handful
   of iterations regardless of how ill-conditioned ``B`` was. This is the
   single biggest practical lever you have in 3D-Var.

Both transforms expose the same minimal interface:

- ``apply(xi) → δx``        (control-space → model-space)
- ``apply_inverse(δx) → xi`` (model-space → control-space; needed once at
  init if the user gives an x-space prior mean)
- ``project_gradient(g_x) → g_xi``  (chain rule for the prior gradient when
  you're working with the cost-in-x form; not used in JAX autodiff paths)

We rely on :func:`gaussx.cholesky` to produce ``U`` as a structured operator,
so the whitening cost stays ``O(B-solve)`` rather than ``O(N²)`` for any of
the structured ``B`` flavours in :mod:`.background`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gaussx as gx
import jax.numpy as jnp
import lineax as lx


if TYPE_CHECKING:
    LinearOperator = lx.AbstractLinearOperator
else:
    LinearOperator = object


# ── identity ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IdentityTransform:
    """Pass-through transform. Use when the prior is already well-conditioned
    *or* when you want the textbook ``½ δxᵀ B⁻¹ δx`` form for didactic clarity.
    """

    def apply(self, xi):
        return xi

    def apply_inverse(self, delta_x):
        return delta_x

    def project_gradient(self, g_x):
        return g_x


# ── whitening (Cholesky CVT) ────────────────────────────────────────────────


@dataclass(frozen=True)
class WhiteningTransform:
    """``δx = U ξ`` where ``B = U Uᵀ``.

    Constructed once via :meth:`from_background`; afterwards each call is a
    single structured matvec / triangular solve from gaussx.

    Notes
    -----
    Whitening also defines a natural sampler from the prior: ``δx = U ε``
    with ``ε ∼ 𝒩(0, I)`` is exactly distributed as ``𝒩(0, B)``. We don't
    use that here but it's the same operator.
    """

    cholesky_op: lx.AbstractLinearOperator

    @classmethod
    def from_background(cls, background: LinearOperator) -> "WhiteningTransform":
        """Factor ``B = U Uᵀ`` via :func:`gaussx.cholesky`.

        gaussx returns a structured triangular operator (preserving Kronecker /
        low-rank-update structure), so subsequent matvecs and inverse-solves
        stay in the cheap regime.
        """
        U = gx.cholesky(background)
        return cls(cholesky_op=U)

    def apply(self, xi):
        """``ξ → U ξ``."""
        return self.cholesky_op.mv(jnp.asarray(xi))

    def apply_inverse(self, delta_x):
        """``δx → U⁻¹ δx`` via :func:`gaussx.solve` for structured dispatch.

        ``gaussx.cholesky`` returns operators that preserve their underlying
        structure (a Cholesky of a ``Kronecker`` is itself a ``Kronecker`` of
        Cholesky factors); ``gaussx.solve`` knows how to dispatch on those,
        whereas ``lineax.AutoLinearSolver`` would try to detect generic
        triangularity and raise ``NotImplementedError``.
        """
        return gx.solve(self.cholesky_op, jnp.asarray(delta_x))

    def project_gradient(self, g_x):
        """``∂J/∂ξ = Uᵀ ∂J/∂δx`` — the chain rule for the explicit-cost form.

        Not needed when you build the cost in ξ-space and let ``jax.grad``
        differentiate end-to-end (which is what :mod:`.cost` does).
        """
        return self.cholesky_op.transpose().mv(jnp.asarray(g_x))
