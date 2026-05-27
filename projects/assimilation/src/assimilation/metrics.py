"""Comparison metrics for the Lorenz-63 benchmark.

All metrics are pure (no IO, no plotting) and operate on JAX arrays so
they JIT cleanly inside `assimilation.benchmark.run_method`.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, Float


def rmse(
    pred: Float[Array, ...],
    truth: Float[Array, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Float[Array, ""]:
    """Root mean-squared error along ``axis``.

    Defaults to a scalar over every dim. Pass ``axis=0`` for a per-
    component RMSE on shape ``(T, N)``.
    """
    return jnp.sqrt(jnp.mean((pred - truth) ** 2, axis=axis))


def sigma_coverage(
    pred_mean: Float[Array, ...],
    pred_std: Float[Array, ...],
    truth: Float[Array, ...],
    *,
    n_sigma: float = 1.0,
) -> Float[Array, ""]:
    """Fraction of entries where ``|truth - mean| <= n_sigma * std``.

    A well-calibrated 1-sigma posterior should produce a coverage near
    ``erf(1/sqrt(2)) ≈ 0.683``. Over-confident posteriors give a smaller
    fraction; under-confident a larger one.
    """
    residual = jnp.abs(truth - pred_mean)
    within = (residual <= n_sigma * jnp.maximum(pred_std, 1e-12)).astype(jnp.float32)
    return jnp.mean(within)


def nll_gaussian(
    pred_mean: Float[Array, ...],
    pred_std: Float[Array, ...],
    truth: Float[Array, ...],
) -> Float[Array, ""]:
    """Mean negative log-likelihood of ``truth`` under
    ``N(pred_mean, diag(pred_std**2))``.

    Cheap scalar diagnostic that penalises both bias and miscalibration.
    Same flat reduction (`jnp.mean`) as `rmse` so the two compose
    cleanly into a results table.
    """
    log_two_pi = math.log(2.0 * math.pi)
    log_var = 2.0 * jnp.log(jnp.maximum(pred_std, 1e-12))
    return 0.5 * jnp.mean(
        ((truth - pred_mean) ** 2) * jnp.exp(-log_var) + log_var + log_two_pi
    )
