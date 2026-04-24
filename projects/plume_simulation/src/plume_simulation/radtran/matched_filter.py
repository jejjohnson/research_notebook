"""Matched-filter retrieval for multispectral / hyperspectral methane detection.

The matched filter is the optimal linear detector for a known target
signature embedded in additive Gaussian noise with known covariance. For a
pixel spectrum ``x ∈ ℝ^B``, background mean ``μ``, covariance ``Σ``, and
target ``t``, the maximum-likelihood abundance estimate is

    ε̂ = (x − μ)ᵀ Σ⁻¹ t  /  (tᵀ Σ⁻¹ t) .

Positive values of ``ε̂`` indicate the target is present; ``ε̂ ≈ ΔVMR /
ΔVMR_ref`` when ``t`` is the target spectrum produced by
:func:`plume_simulation.radtran.target.target_spectrum_normalized_linear`
at ``delta_vmr = ΔVMR_ref``. Multiplying ``ε̂`` back by ``ΔVMR_ref`` gives
an estimate of the actual pixel VMR enhancement.

This module exposes the raw operation at three scopes:

- :func:`matched_filter_pixel`  — one pixel (scalar output).
- :func:`matched_filter_image`  — vectorised over spatial axes.
- :func:`matched_filter_snr`    — per-pixel detection SNR.

Ported from ``jej_vc_snippets/methane_retrieval/matched_filter.py`` with the
dask/chunked path dropped (callers who need it can chunk the input themselves
and map :func:`matched_filter_image` over chunks).
"""

from __future__ import annotations

import numpy as np


def _precompute_constants(
    target: np.ndarray,
    cov_inv: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return ``(Σ⁻¹ t, tᵀ Σ⁻¹ t)`` — precompute for all pixels."""
    if target.ndim != 1:
        raise ValueError(
            f"matched filter: `target` must be 1-D (got shape {target.shape})"
        )
    if cov_inv.shape != (target.size, target.size):
        raise ValueError(
            f"matched filter: `cov_inv` shape {cov_inv.shape} must equal "
            f"({target.size}, {target.size})"
        )
    cov_inv_target = np.einsum("ij, j -> i", cov_inv, target)
    target_norm = float(np.einsum("i, i ->", target, cov_inv_target))
    if not (target_norm > 0.0):
        raise ValueError(
            "matched filter: `tᵀ Σ⁻¹ t` must be positive; check that Σ is PD "
            "and the target is non-trivial."
        )
    return cov_inv_target, target_norm


def matched_filter_pixel(
    spectrum: np.ndarray,
    mean_spectrum: np.ndarray,
    cov_inv: np.ndarray,
    target: np.ndarray,
) -> float:
    """Matched-filter abundance for a single pixel.

    Parameters
    ----------
    spectrum : np.ndarray
        Pixel spectrum, shape ``(n_bands,)``.
    mean_spectrum : np.ndarray
        Background mean, shape ``(n_bands,)``.
    cov_inv : np.ndarray
        Inverse background covariance ``Σ⁻¹``, shape ``(n_bands, n_bands)``.
    target : np.ndarray
        Target signature, shape ``(n_bands,)``.

    Returns
    -------
    abundance : float
        Scalar retrieval in the same units as ``target``.
    """
    cov_inv_target, target_norm = _precompute_constants(target, cov_inv)
    innovation = np.asarray(spectrum, dtype=float) - np.asarray(mean_spectrum, dtype=float)
    return float(np.einsum("i, i ->", cov_inv_target, innovation) / target_norm)


def matched_filter_image(
    radiance: np.ndarray,
    mean_spectrum: np.ndarray,
    cov_inv: np.ndarray,
    target: np.ndarray,
    *,
    band_axis: int = 0,
) -> np.ndarray:
    """Matched-filter abundance map, vectorised over spatial axes.

    Parameters
    ----------
    radiance : np.ndarray
        Radiance cube, shape ``(n_bands, ny, nx)`` by default.
    mean_spectrum, cov_inv, target : np.ndarray
        As in :func:`matched_filter_pixel`.
    band_axis : int
        Band axis of ``radiance``. Default 0.

    Returns
    -------
    abundance : np.ndarray
        Shape ``radiance.shape`` minus the band axis — e.g. ``(ny, nx)`` for
        the default layout.
    """
    cov_inv_target, target_norm = _precompute_constants(target, cov_inv)
    arr = np.asarray(radiance, dtype=float)
    bands_first = np.moveaxis(arr, band_axis, 0)
    n_bands = bands_first.shape[0]
    if n_bands != target.size:
        raise ValueError(
            f"matched_filter_image: radiance has {n_bands} bands but target "
            f"has {target.size}."
        )
    # Broadcasting subtraction: (n_bands, ...) - (n_bands,)
    innovation = bands_first - np.asarray(mean_spectrum, dtype=float).reshape(
        (n_bands,) + (1,) * (bands_first.ndim - 1)
    )
    # Project along the band axis. Reshape cov_inv_target for broadcasting,
    # collapse leading band axis with sum; this matches the einops
    # "b, b ... -> ..." contraction without the einops dependency.
    broadcastable = cov_inv_target.reshape(
        (n_bands,) + (1,) * (innovation.ndim - 1)
    )
    projected = (broadcastable * innovation).sum(axis=0)
    return projected / target_norm


def matched_filter_snr(
    abundance: np.ndarray,
    cov_inv: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Detection SNR for a matched-filter abundance map.

    For the matched filter,
        SNR(x) = (target energy) · ε̂ = ε̂ · √(tᵀ Σ⁻¹ t).

    Parameters
    ----------
    abundance : np.ndarray
        Output of :func:`matched_filter_image`, shape ``(..., ny, nx)``.
    cov_inv : np.ndarray
        Inverse covariance used to build the filter.
    target : np.ndarray
        Target used to build the filter.

    Returns
    -------
    snr : np.ndarray
        Same shape as ``abundance``.
    """
    _, target_norm = _precompute_constants(target, cov_inv)
    return np.asarray(abundance, dtype=float) * np.sqrt(target_norm)
