"""Matched-filter kernel — one implementation for any covariance structure.

The classical matched-filter score is

.. math::
    y(x) = \\frac{(x - \\mu)^\\top \\Sigma^{-1} t}{t^\\top \\Sigma^{-1} t}

with the special case :math:`y(\\mu + \\alpha t) = \\alpha` (so ``y`` is an
unbiased linear estimator of the plume enhancement amplitude). The per-pixel
evaluation cost is dominated by :math:`w = \\Sigma^{-1} t`, a *single solve*
that can be precomputed once per scene; after that each pixel is a dot
product of length ``n_bands``.

This module wraps that into a single kernel that works for any
:class:`lineax.AbstractLinearOperator` covariance — dense, diagonal, low-rank
+ Tikhonov (Woodbury via :class:`gaussx.LowRankUpdate`), or Kronecker. The
solve is dispatched via :func:`gaussx.solve`, so the MF caller never sees the
structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


if TYPE_CHECKING:
    LinearOperator = lx.AbstractLinearOperator
else:
    LinearOperator = object


def _precompute_filter(
    cov_op: LinearOperator, target: Float[Array, "B"]
) -> tuple[Float[Array, "B"], Float[Array, ""]]:
    """Return ``(w, tᵀw)`` with ``w = Σ⁻¹ t``.

    Caller precomputes these once; every pixel then costs a single dot
    product against ``w``. Validates the target energy ``tᵀ Σ⁻¹ t`` so
    callers get a clear :class:`ValueError` instead of silent NaNs/infs
    when the target is zero or the covariance is singular/indefinite
    along the target direction. The check is a no-op under
    :func:`jax.jit` tracing; users who want the same fail-fast behaviour
    inside a jit-compiled pipeline should call
    :func:`validate_mf_inputs` explicitly before jitting.
    """
    w = gx.solve(cov_op, target)
    target_norm_sq = jnp.dot(target, w)
    _check_target_energy(target, target_norm_sq)
    return w, target_norm_sq


def _check_target_energy(
    target: Float[Array, "B"], target_norm_sq: Float[Array, ""]
) -> None:
    """Fail-fast validation — eager only; silently returns under jit tracing."""
    try:
        t_max = float(jnp.max(jnp.abs(target)))
        tnorm_sq_val = float(target_norm_sq)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return  # traced call: no concrete values to inspect.
    if t_max == 0.0:
        raise ValueError(
            "matched_filter: target signature is all zero — the score "
            "(x−μ)ᵀΣ⁻¹t / (tᵀΣ⁻¹t) is undefined."
        )
    if not tnorm_sq_val > 0.0:
        raise ValueError(
            f"matched_filter: target energy tᵀΣ⁻¹t = {tnorm_sq_val!r} is not "
            "strictly positive. The covariance operator is singular or "
            "indefinite along the target direction. Add a Tikhonov ridge or "
            "use a PD estimator such as estimate_cov_shrunk (Ledoit–Wolf)."
        )


def validate_mf_inputs(cov_op: LinearOperator, target: Float[Array, "B"]) -> None:
    """Public fail-fast check for ``(Σ, t)`` compatibility.

    Runs the same ``tᵀ Σ⁻¹ t > 0`` test that :func:`apply_pixel` and
    :func:`apply_image` perform automatically, but without the rest of the
    MF work — useful for catching user errors *before* jitting a pipeline
    or batching over many pixels.

    Raises
    ------
    ValueError
        If ``target`` is all zero, or if the target energy ``tᵀ Σ⁻¹ t`` is
        not strictly positive (singular or indefinite covariance along the
        target direction).
    """
    w = gx.solve(cov_op, target)
    target_norm_sq = jnp.dot(target, w)
    _check_target_energy(target, target_norm_sq)


def apply_pixel(
    pixel: Float[Array, "B"],
    mean: Float[Array, "B"],
    cov_op: LinearOperator,
    target: Float[Array, "B"],
) -> Float[Array, ""]:
    """Matched-filter score for a single pixel spectrum.

    Parameters
    ----------
    pixel
        Observed radiance spectrum at one pixel, shape ``(n_bands,)``.
    mean
        Background mean spectrum, shape ``(n_bands,)``.
    cov_op
        Background covariance as a :class:`lineax.AbstractLinearOperator`. Any
        structure is fine — :func:`gaussx.solve` dispatches on operator type
        (Woodbury for :class:`gaussx.LowRankUpdate`, Cholesky for dense, etc.).
    target
        Target signature ``t(ν)``, shape ``(n_bands,)`` — typically the
        radiance Jacobian ``∂L/∂VMR`` at the background state.

    Returns
    -------
    Float[Array, ""]
        Scalar score — an unbiased linear estimator of the plume amplitude
        under the additive-Gaussian-noise model.
    """
    w, target_norm_sq = _precompute_filter(cov_op, target)
    return jnp.dot(pixel - mean, w) / target_norm_sq


def apply_image(
    cube: Float[Array, "H W B"],
    mean: Float[Array, "B"],
    cov_op: LinearOperator,
    target: Float[Array, "B"],
) -> Float[Array, "H W"]:
    """Matched-filter score map for a hyperspectral cube.

    Precomputes ``w = Σ⁻¹ t`` once, then evaluates
    ``(cube − μ) · w / (tᵀ w)`` in a single einsum — no per-pixel Python loop.

    Parameters
    ----------
    cube
        Radiance cube with shape ``(H, W, n_bands)``.
    mean, cov_op, target
        See :func:`apply_pixel`.

    Returns
    -------
    Float[Array, "H W"]
        Score image. Pixel ``(i, j)`` is the estimated plume amplitude at
        that location.
    """
    w, target_norm_sq = _precompute_filter(cov_op, target)
    residual = cube - mean
    # Contract the last (spectral) axis with the precomputed filter
    # direction. jnp.einsum keeps the other two axes intact, matching the
    # "H W B, B -> H W" einops signature we replaced.
    scores = jnp.einsum("ijk,k->ij", residual, w)
    return scores / target_norm_sq


def matched_filter_snr(
    amplitude: float,
    cov_op: LinearOperator,
    target: Float[Array, "B"],
) -> Float[Array, ""]:
    """Theoretical SNR of the matched filter at a given plume amplitude.

    For ``x = μ + α t + ε`` with ``ε ∼ N(0, Σ)``, the MF score has
    ``mean = α`` and ``variance = 1 / (tᵀ Σ⁻¹ t)``. So the detection SNR is

    .. math::
        \\text{SNR} = \\alpha \\sqrt{t^\\top \\Sigma^{-1} t}.

    Parameters
    ----------
    amplitude
        The plume amplitude ``α`` at which to evaluate SNR.
    cov_op, target
        See :func:`apply_pixel`.

    Returns
    -------
    Float[Array, ""]
        Scalar SNR (dimensionless).
    """
    w = gx.solve(cov_op, target)
    target_norm_sq = jnp.dot(target, w)
    _check_target_energy(target, target_norm_sq)
    return jnp.asarray(amplitude) * jnp.sqrt(target_norm_sq)


def detection_threshold(
    false_alarm_rate: float,
    cov_op: LinearOperator,
    target: Float[Array, "B"],
) -> Float[Array, ""]:
    """Gaussian detection threshold on the MF score for a given FAR.

    Under the null (no plume), the score is :math:`N(0, 1/(t^\\top \\Sigma^{-1} t))`.
    The threshold above which the false-alarm rate does not exceed ``FAR`` is

    .. math::
        \\tau = \\Phi^{-1}(1 - \\text{FAR}) / \\sqrt{t^\\top \\Sigma^{-1} t}.

    Parameters
    ----------
    false_alarm_rate
        Target FAR, in ``(0, 1)`` — e.g. ``1e-6`` for one false alarm per
        million pixels under Gaussian assumptions.
    cov_op, target
        See :func:`apply_pixel`.

    Returns
    -------
    Float[Array, ""]
        Threshold in score units (same units as ``amplitude`` in
        :func:`matched_filter_snr`).
    """
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError(
            f"detection_threshold: false_alarm_rate must be in (0, 1); "
            f"got {false_alarm_rate}."
        )
    # ndtri is the inverse Gaussian CDF (aka Φ⁻¹ / probit).
    z = jax.scipy.special.ndtri(1.0 - false_alarm_rate)
    w = gx.solve(cov_op, target)
    target_norm_sq = jnp.dot(target, w)
    _check_target_energy(target, target_norm_sq)
    return z / jnp.sqrt(target_norm_sq)
