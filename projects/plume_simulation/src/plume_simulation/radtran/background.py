"""Robust background statistics for matched-filter retrievals.

The matched filter needs a background mean spectrum ``μ`` and an inverse
covariance ``Σ⁻¹`` estimated from the *clean* pixels in the scene. Real
scenes have contaminants — plumes, clouds, shadows, bright targets — that
would bias a naive mean and inflate a naive covariance, so both estimators
here are written to be robust against outlier pixels.

- :func:`trimmed_mean_spectrum` — per-channel trimmed mean. Cheap, robust
  to a few percent of outliers, matches a Gaussian mean in the clean limit.
- :func:`robust_lowrank_covariance` — low-rank plus diagonal-regularisation
  covariance via truncated SVD on the mean-subtracted, trimmed pixel stack.
  Returns ``(Σ, Σ⁻¹)`` ready for the matched filter.

Both operate on a pixel × band array (or an ``xarray.DataArray`` with
``band`` as the last dim). The ``trim_frac`` parameter is a two-sided
fraction in ``[0, 0.5)`` — ``0.1`` removes the brightest and darkest 10%
of pixels per channel before averaging.

Heavily simplified from ``jej_vc_snippets/methane_retrieval/matched_filter_{mean,covariance}.py``
(GMM-based background estimators and dynamic-mode rejection dropped — the
trimmed mean + truncated SVD covers the demo regime with far less tuning).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import trim_mean


def _flatten_pixels(
    radiance: np.ndarray,
    band_axis: int,
) -> np.ndarray:
    """Rearrange to ``(n_pixels, n_bands)``; band_axis is the band dim of ``radiance``."""
    arr = np.asarray(radiance)
    bands_first = np.moveaxis(arr, band_axis, 0)  # (n_bands, ...)
    n_bands = bands_first.shape[0]
    return bands_first.reshape(n_bands, -1).T  # (n_pixels, n_bands)


def trimmed_mean_spectrum(
    radiance: np.ndarray,
    *,
    trim_frac: float = 0.1,
    band_axis: int = 0,
) -> np.ndarray:
    """Per-channel trimmed mean ``μ_b``.

    Parameters
    ----------
    radiance : np.ndarray
        Radiance cube, shape ``(n_bands, ny, nx)`` by default (``band_axis=0``),
        or any array where one axis indexes the band.
    trim_frac : float
        Fraction trimmed from *each* tail per channel, in ``[0, 0.5)``.
        Default 0.1 removes the top and bottom 10% per band.
    band_axis : int
        Band axis of ``radiance``. Default 0.

    Returns
    -------
    mu : np.ndarray
        Trimmed mean per band, shape ``(n_bands,)``.
    """
    if not (0.0 <= trim_frac < 0.5):
        raise ValueError(
            f"trimmed_mean_spectrum: `trim_frac` must be in [0, 0.5) (got {trim_frac!r})"
        )
    flat = _flatten_pixels(radiance, band_axis)  # (n_pixels, n_bands)
    return np.asarray(trim_mean(flat, trim_frac, axis=0), dtype=float)


def robust_lowrank_covariance(
    radiance: np.ndarray,
    *,
    rank: int | None = None,
    trim_frac: float = 0.1,
    regularization: float = 1e-6,
    band_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust low-rank covariance + inverse ``(Σ, Σ⁻¹)``.

    The pixel stack is centred with a trimmed mean, the extreme 2·``trim_frac``
    pixels (by total energy) are removed, and the covariance is estimated as

        Σ = U_k · S_k² · U_kᵀ / N + λ · I

    where ``U_k`` are the top-``k`` left singular vectors of the centred
    matrix. The added diagonal term ``λ · I`` stabilises the inverse when
    the covariance is ill-conditioned. Returns both ``Σ`` and ``Σ⁻¹`` since
    the matched filter needs the inverse and diagnostics sometimes want the
    forward covariance too.

    Parameters
    ----------
    radiance : np.ndarray
        Shape ``(n_bands, ny, nx)`` by default.
    rank : int or None
        Truncation rank. ``None`` uses ``min(n_bands - 1, 16)`` — the
        practical sweet spot for multispectral data where most of the
        variance is in a handful of modes.
    trim_frac : float
        Per-pixel energy-based trim fraction in ``[0, 0.5)``. Default 0.1.
    regularization : float
        Diagonal Tikhonov term ``λ``. Default ``1e-6``.
    band_axis : int
        Band axis of ``radiance``. Default 0.

    Returns
    -------
    Sigma, Sigma_inv : np.ndarray
        Symmetric PSD matrices, shape ``(n_bands, n_bands)``.
    """
    flat = _flatten_pixels(radiance, band_axis)  # (n_pixels, n_bands)
    n_pixels, n_bands = flat.shape
    if n_pixels < 2:
        raise ValueError(
            f"robust_lowrank_covariance: need ≥ 2 pixels (got {n_pixels})"
        )
    if not (0.0 <= trim_frac < 0.5):
        raise ValueError(
            f"robust_lowrank_covariance: `trim_frac` must be in [0, 0.5) (got {trim_frac!r})"
        )
    if regularization <= 0.0:
        raise ValueError(
            "robust_lowrank_covariance: `regularization` must be > 0."
        )
    if rank is None:
        rank = min(n_bands - 1, 16)
    rank = max(1, min(int(rank), n_bands))

    mu = trimmed_mean_spectrum(radiance, trim_frac=trim_frac, band_axis=band_axis)
    centred = flat - mu[None, :]

    # Drop the top/bottom `trim_frac` pixels by total energy to keep bright
    # outliers from dominating the SVD. This mirrors the dynamic-mode
    # rejection in the original code but without the GMM complexity.
    if trim_frac > 0.0:
        energy = np.linalg.norm(centred, axis=1)
        lo, hi = np.quantile(energy, [trim_frac, 1.0 - trim_frac])
        keep = (energy >= lo) & (energy <= hi)
        if keep.sum() < n_bands:
            raise ValueError(
                "robust_lowrank_covariance: trimming left fewer pixels than bands; "
                "reduce `trim_frac` or enlarge the scene."
            )
        centred = centred[keep]

    # Truncated SVD via numpy (scene is small for multispectral; for
    # hyperspectral callers can pre-subsample). Randomised SVD would give
    # a constant-factor speed-up but the deterministic path keeps tests
    # reproducible.
    _, S, Vt = np.linalg.svd(centred, full_matrices=False)
    S_k = S[:rank]
    V_k = Vt[:rank]  # shape (rank, n_bands); V_k.T are the principal directions.

    # Σ in the top-k subspace (1 / N) · Vᵀ S² V.
    N = centred.shape[0]
    Sigma = (V_k.T * (S_k**2)) @ V_k / max(N, 1) + regularization * np.eye(n_bands)

    # Woodbury inverse for efficiency: (λI + Uᵀ D U)⁻¹ where
    #   U = V_k (rank, n_bands), D = diag(S_k² / N) (rank, rank).
    # But since n_bands is small for multispectral, direct inversion is fine
    # and keeps the code obvious.
    Sigma_inv = np.linalg.inv(Sigma)
    return Sigma, Sigma_inv
