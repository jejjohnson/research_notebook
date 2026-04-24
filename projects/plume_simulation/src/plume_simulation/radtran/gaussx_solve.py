"""Gaussx-backed structured solves for the matched filter.

The default ``matched_filter_*`` functions in :mod:`.matched_filter` take a
*materialised* inverse covariance ``Σ⁻¹ ∈ ℝ^{B×B}`` and compute

    ε̂ = (x - μ)ᵀ Σ⁻¹ t / (tᵀ Σ⁻¹ t)

directly. That's fine when ``B`` is small (multispectral) but wasteful for
hyperspectral retrievals with ``B ≳ 200`` because:

1. The ``O(B³)`` dense inverse is entirely avoidable — the empirical
   covariance is naturally represented as a low-rank update
   ``Σ = λ I + U diag(d) Uᵀ``.
2. Every solve can then run in ``O(B·k)`` via the Woodbury identity,
   where ``k`` is the retained rank (typically 8–32).

This module wraps :mod:`gaussx`'s :class:`gaussx.LowRankUpdate` operator and
structural :func:`gaussx.solve` so the matched filter can stay
inversion-free on the hot path:

- :func:`build_lowrank_covariance_operator` — construct a
  :class:`gaussx.LowRankUpdate` from the same SVD the dense estimator uses.
- :func:`matched_filter_pixel_op` / :func:`matched_filter_image_op` —
  operator-backed matched filter that calls ``gaussx.solve(cov, target)``
  once and then dots the result into every pixel's innovation.

The functions accept NumPy inputs and return NumPy outputs — the gaussx
internals run on JAX arrays for us, but callers don't need to know.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # lineax is re-exported via gaussx — avoid hard import.
    import gaussx as _gx
    import lineax as _lx

    LinearOperator = _lx.AbstractLinearOperator
    LowRankUpdateOp = _gx.LowRankUpdate
else:  # pragma: no cover — runtime wildcard to dodge annotation eval cost.
    LinearOperator = object
    LowRankUpdateOp = object


# ── Covariance operator construction ────────────────────────────────────────


def build_lowrank_covariance_operator(
    radiance: np.ndarray,
    *,
    rank: int | None = None,
    trim_frac: float = 0.1,
    regularization: float = 1e-6,
    band_axis: int = 0,
) -> tuple["LowRankUpdateOp", np.ndarray]:
    """Build a :class:`gaussx.LowRankUpdate` covariance operator from a scene.

    Parameters
    ----------
    radiance : np.ndarray
        Radiance cube, shape ``(n_bands, ny, nx)`` by default.
    rank, trim_frac, regularization, band_axis
        Same semantics as :func:`.background.robust_lowrank_covariance` —
        see that function for the trimming and rank-selection details.

    Returns
    -------
    cov : gaussx.LowRankUpdate
        Structured operator ``λ I + U diag(d) Uᵀ`` (symmetric + PSD tagged)
        where ``d = S_k² / N`` are the squared singular values of the
        trimmed, mean-subtracted pixel stack.
    mu : np.ndarray
        Background mean spectrum, shape ``(n_bands,)``. Returned here
        (rather than a separate call) so callers build μ and Σ from the
        *same* trimmed pixel stack — avoiding a subtle bias where μ drops
        outliers that Σ's trim step kept.
    """
    import gaussx as gx
    import jax.numpy as jnp
    import lineax as lx

    from plume_simulation.radtran.background import (
        _flatten_pixels,
        trimmed_mean_spectrum,
    )

    flat = _flatten_pixels(radiance, band_axis)  # (n_pixels, n_bands)
    n_pixels, n_bands = flat.shape
    if n_pixels < 2:
        raise ValueError(
            f"build_lowrank_covariance_operator: need ≥ 2 pixels (got {n_pixels})"
        )
    if not (0.0 <= trim_frac < 0.5):
        raise ValueError(
            f"build_lowrank_covariance_operator: `trim_frac` must be in "
            f"[0, 0.5) (got {trim_frac!r})"
        )
    if regularization <= 0.0:
        raise ValueError(
            "build_lowrank_covariance_operator: `regularization` must be > 0."
        )
    if rank is None:
        rank = min(n_bands - 1, 16)
    rank = max(1, min(int(rank), n_bands))

    mu = trimmed_mean_spectrum(radiance, trim_frac=trim_frac, band_axis=band_axis)
    centred = flat - mu[None, :]

    if trim_frac > 0.0:
        energy = np.linalg.norm(centred, axis=1)
        lo, hi = np.quantile(energy, [trim_frac, 1.0 - trim_frac])
        keep = (energy >= lo) & (energy <= hi)
        if keep.sum() < n_bands:
            raise ValueError(
                "build_lowrank_covariance_operator: trimming left fewer "
                "pixels than bands; reduce `trim_frac` or enlarge the scene."
            )
        centred = centred[keep]

    _, S, Vt = np.linalg.svd(centred, full_matrices=False)
    S_k = S[:rank]
    V_k = Vt[:rank]  # (rank, n_bands)

    # U: principal directions as columns.
    U = jnp.asarray(V_k.T)  # (n_bands, rank)
    d = jnp.asarray(S_k**2 / max(centred.shape[0], 1))  # diag of U d Uᵀ
    # Base: regularisation · I as an explicit *diagonal* operator so
    # gaussx.solve picks up the fast per-element inverse path. A naive
    # ``regularization * IdentityLinearOperator(...)`` would build a
    # ScaledOperator that gaussx does not specialise and that would
    # compute the dense inverse instead.
    base = lx.DiagonalLinearOperator(
        regularization * jnp.ones(n_bands, dtype=jnp.float64)
    )

    cov = gx.LowRankUpdate(
        base, U, d,
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    return cov, mu


# ── Operator-backed matched filter ───────────────────────────────────────────


def matched_filter_pixel_op(
    spectrum: np.ndarray,
    mean_spectrum: np.ndarray,
    cov: "LowRankUpdateOp",
    target: np.ndarray,
) -> float:
    """Scalar matched filter using a structured covariance operator.

    Computes ``Σ⁻¹ t`` via :func:`gaussx.solve` once (O(B·k) via Woodbury),
    then projects a single pixel's innovation onto it.
    """
    import gaussx as gx
    import jax.numpy as jnp

    cov_inv_target = np.asarray(gx.solve(cov, jnp.asarray(target)))
    target_norm = float(np.dot(target, cov_inv_target))
    if not (target_norm > 0.0):
        raise ValueError(
            "matched_filter_pixel_op: `tᵀ Σ⁻¹ t` must be positive; check "
            "that Σ is PD and the target is non-trivial."
        )
    innovation = np.asarray(spectrum, dtype=float) - np.asarray(
        mean_spectrum, dtype=float,
    )
    return float(np.dot(cov_inv_target, innovation) / target_norm)


def matched_filter_image_op(
    radiance: np.ndarray,
    mean_spectrum: np.ndarray,
    cov: "LowRankUpdateOp",
    target: np.ndarray,
    *,
    band_axis: int = 0,
) -> np.ndarray:
    """Vectorised matched filter using a structured covariance operator.

    Parameters
    ----------
    radiance : np.ndarray
        Radiance cube, shape ``(n_bands, ny, nx)`` by default.
    mean_spectrum : np.ndarray
        Background mean, shape ``(n_bands,)``.
    cov : gaussx.LowRankUpdate
        Structured covariance operator (from
        :func:`build_lowrank_covariance_operator`).
    target : np.ndarray
        Target signature, shape ``(n_bands,)``.
    band_axis : int
        Band axis of ``radiance``. Default 0.

    Returns
    -------
    abundance : np.ndarray
        Pixel-wise retrieval, shape ``radiance.shape`` minus the band axis.
    """
    import gaussx as gx
    import jax.numpy as jnp

    arr = np.asarray(radiance, dtype=float)
    bands_first = np.moveaxis(arr, band_axis, 0)
    n_bands = bands_first.shape[0]
    if n_bands != target.size:
        raise ValueError(
            f"matched_filter_image_op: radiance has {n_bands} bands but "
            f"target has {target.size}."
        )

    cov_inv_target = np.asarray(gx.solve(cov, jnp.asarray(target)))  # (n_bands,)
    target_norm = float(np.dot(target, cov_inv_target))
    if not (target_norm > 0.0):
        raise ValueError(
            "matched_filter_image_op: `tᵀ Σ⁻¹ t` must be positive; check "
            "that Σ is PD and the target is non-trivial."
        )

    # Subtract μ along the band axis, project onto (Σ⁻¹ t), divide by norm.
    innovation = bands_first - np.asarray(mean_spectrum, dtype=float).reshape(
        (n_bands,) + (1,) * (bands_first.ndim - 1),
    )
    projector = cov_inv_target.reshape(
        (n_bands,) + (1,) * (bands_first.ndim - 1),
    )
    return (projector * innovation).sum(axis=0) / target_norm


def matched_filter_snr_op(
    abundance: np.ndarray,
    cov: "LowRankUpdateOp",
    target: np.ndarray,
) -> np.ndarray:
    """Detection SNR when the covariance is a gaussx operator.

    Mirrors :func:`plume_simulation.radtran.matched_filter.matched_filter_snr`
    but uses :func:`gaussx.solve` to avoid materialising ``Σ⁻¹``.
    """
    import gaussx as gx
    import jax.numpy as jnp

    cov_inv_target = np.asarray(gx.solve(cov, jnp.asarray(target)))
    target_norm = float(np.dot(target, cov_inv_target))
    return np.asarray(abundance, dtype=float) * np.sqrt(target_norm)
