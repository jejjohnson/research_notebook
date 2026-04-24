"""Background-error covariance ``B`` constructors for 3D-Var.

The prior term ``½ δxᵀ B⁻¹ δx`` in the 3D-Var cost function is dominated, at
the operator level, by *how* you write down ``B``. We support three flavours
that cover most of what a methane-retrieval workflow ever needs:

============= ====================================================== =================
Flavour       Operator structure                                       Cost of ``B⁻¹``
============= ====================================================== =================
diagonal      ``B = diag(σ²)``                                       ``O(N)``
Kronecker     ``B = ρ_y ⊗ ρ_x``  (separable, e.g. AR(1) in each axis) ``O(N · √N)``
low-rank      ``B = λI + U diag(d) Uᵀ``  (Woodbury via gaussx)        ``O(N · k)``
============= ====================================================== =================

All three return a :class:`lineax.AbstractLinearOperator` (via gaussx) so the
preconditioning transform :class:`~plume_simulation.assimilation.control.WhiteningTransform`
and the cost gradient can both call :func:`gaussx.solve` and
:func:`gaussx.cholesky` uniformly without caring about structure.

The Kronecker constructor uses an exponential (AR(1)-style) correlation
``[exp(-|i-j|/L)]`` along each axis — simple, positive-definite by
construction, and matches the prior families used in operational variational
assimilation systems for spatially-smooth tracer fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpy as np


if TYPE_CHECKING:
    LinearOperator = lx.AbstractLinearOperator
else:
    LinearOperator = object


# ── diagonal ─────────────────────────────────────────────────────────────────


def build_diagonal_background(
    variance: float | np.ndarray, n_pixels: int | None = None
) -> "LinearOperator":
    """Build ``B = diag(σ²)`` as a :class:`lineax.DiagonalLinearOperator`.

    Parameters
    ----------
    variance : float or array
        Per-pixel prior variance. Scalar broadcasts to ``n_pixels``.
    n_pixels : int, optional
        Required when ``variance`` is a scalar.

    Returns
    -------
    LinearOperator
        ``B`` as a diagonal operator. ``gaussx.solve(B, r)`` is then a
        per-element division.
    """
    var = np.asarray(variance, dtype=float).ravel()
    if var.size == 1:
        if n_pixels is None:
            raise ValueError(
                "build_diagonal_background: scalar variance requires `n_pixels`."
            )
        var = np.full(int(n_pixels), float(var.item()))
    if np.any(var <= 0.0):
        raise ValueError("build_diagonal_background: variances must be > 0.")
    return lx.DiagonalLinearOperator(jnp.asarray(var))


# ── Kronecker (separable spatial AR(1)) ─────────────────────────────────────


def _ar1_correlation_matrix(n: int, length_scale_pixels: float) -> np.ndarray:
    """``[exp(-|i-j| / L)]`` correlation matrix — symmetric, PD for L > 0."""
    if length_scale_pixels <= 0.0:
        raise ValueError("AR(1) correlation: length_scale_pixels must be > 0.")
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    return np.exp(-diff / length_scale_pixels)


def build_kronecker_background(
    *,
    ny: int,
    nx: int,
    variance: float,
    length_scale_y: float,
    length_scale_x: float,
) -> "LinearOperator":
    """Build ``B = σ² · (ρ_y ⊗ ρ_x)`` using :class:`gaussx.Kronecker`.

    The Kronecker structure makes ``B⁻¹`` an ``O(ny³ + nx³)`` operation rather
    than ``O((ny·nx)³)``, which is the difference between feasible and not for
    realistic scene sizes (e.g. 256×256 → 65 k state vector vs.
    256³ + 256³ ≈ 30 M flops vs. 280 T flops).

    The convention here is ``B = σ² · K_y ⊗ K_x`` with row-major flattening
    ``δx[i·nx + j] = δx(i, j)`` — matches ``vmr_field.reshape(-1)`` order.
    """
    if variance <= 0.0:
        raise ValueError("build_kronecker_background: `variance` must be > 0.")
    Ky = _ar1_correlation_matrix(ny, length_scale_y)
    Kx = _ar1_correlation_matrix(nx, length_scale_x)
    # Spread the σ² factor evenly between the two Kronecker factors so each is
    # individually well-scaled. Mathematically (σ a) ⊗ (σ b) = σ² (a ⊗ b).
    sigma = float(np.sqrt(variance))
    op_y = lx.MatrixLinearOperator(
        jnp.asarray(sigma * Ky),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    op_x = lx.MatrixLinearOperator(
        jnp.asarray(sigma * Kx),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    return gx.Kronecker(op_y, op_x)


# ── low-rank update (scene-estimated background) ─────────────────────────────


def build_lowrank_background(
    *,
    samples: np.ndarray,
    rank: int,
    regularization: float = 1e-6,
) -> "LinearOperator":
    """Estimate ``B`` from a stack of state samples and wrap as ``λI + U D Uᵀ``.

    Parameters
    ----------
    samples : np.ndarray
        Sample matrix, shape ``(n_samples, n_state)`` — e.g. flattened VMR
        snapshots from a NWP ensemble or a climatology.
    rank : int
        Number of leading SVD components to keep. Costs ``O(n_state · k)``
        per solve.
    regularization : float
        Diagonal floor ``λ`` so the resulting operator is strictly PD even
        when ``samples`` is rank-deficient. Must be > 0.

    Returns
    -------
    LinearOperator
        ``gaussx.LowRankUpdate`` tagged symmetric + PSD; ``gaussx.solve(B, r)``
        uses Woodbury internally.
    """
    if samples.ndim != 2:
        raise ValueError(
            f"build_lowrank_background: samples must be 2-D (n_samples, n_state); "
            f"got {samples.shape}."
        )
    n_samples, n_state = samples.shape
    if n_samples < 2:
        raise ValueError("build_lowrank_background: need ≥ 2 samples.")
    rank = max(1, min(int(rank), n_state, n_samples))
    if regularization <= 0.0:
        raise ValueError("build_lowrank_background: `regularization` must be > 0.")
    centred = samples - samples.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(centred, full_matrices=False)
    S_k = S[:rank]
    V_k = Vt[:rank]
    U = jnp.asarray(V_k.T)  # (n_state, rank)
    d = jnp.asarray(S_k**2 / max(n_samples - 1, 1))
    base = lx.DiagonalLinearOperator(
        jnp.asarray(regularization * np.ones(n_state, dtype=float))
    )
    return gx.LowRankUpdate(
        base,
        U,
        d,
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
