"""Background mean and covariance estimators for the matched filter.

Estimation is not on the gradient path — the matched filter treats the
background :math:`(\\mu, \\Sigma)` as known inputs. That lets us lean on
scikit-learn's mature estimators (which have careful numerics, shrinkage,
robust variants, etc.) without any JAX/autodiff constraints. The outputs
are wrapped as :mod:`lineax` / :mod:`gaussx` linear operators so that
:func:`plume_simulation.matched_filter.core.apply_image` can use the same
``gaussx.solve`` dispatch path for all covariance flavours:

============================== ====================================================
Estimator                      Operator returned
============================== ====================================================
``estimate_cov_empirical``     :class:`lineax.MatrixLinearOperator` (dense)
``estimate_cov_shrunk``        :class:`lineax.MatrixLinearOperator` (dense, shrunk)
``estimate_cov_lowrank``       :class:`gaussx.LowRankUpdate` (``λI + UDUᵀ``)
============================== ====================================================

The low-rank path uses scikit-learn's randomised :class:`TruncatedSVD`, which
is ``O(n · k)`` on a ``(n_samples, n_bands)`` matrix and is typically 10–100×
faster than a full SVD for the small ``k`` a matched filter needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Float


if TYPE_CHECKING:
    LinearOperator = lx.AbstractLinearOperator
else:
    LinearOperator = object


_RobustMethod = Literal["mean", "median", "trimmed", "huber"]


# ── mean estimators ──────────────────────────────────────────────────────────


def estimate_mean(
    cube: Float[Array, "H W B"] | np.ndarray,
    *,
    method: _RobustMethod = "mean",
    trim_proportion: float = 0.1,
    huber_c: float = 1.345,
) -> np.ndarray:
    """Estimate the per-band background mean ``μ`` from a scene.

    Parameters
    ----------
    cube
        Hyperspectral scene of shape ``(H, W, n_bands)``.
    method
        - ``'mean'``    — arithmetic mean (sensitive to anomalies).
        - ``'median'``  — per-band median (very robust).
        - ``'trimmed'`` — two-sided trimmed mean (``scipy.stats.trim_mean``).
        - ``'huber'``   — Huber M-estimator (sklearn's ``HuberRegressor`` on
          a constant design).
    trim_proportion
        Fraction discarded from each tail for the trimmed mean. Ignored
        otherwise.
    huber_c
        Huber influence threshold in units of the per-band MAD. Ignored
        unless ``method='huber'``.

    Returns
    -------
    np.ndarray
        Mean spectrum of length ``n_bands``.
    """
    cube_np = np.asarray(cube)
    if cube_np.ndim != 3:
        raise ValueError(
            f"estimate_mean: cube must be (H, W, n_bands), got shape {cube_np.shape}."
        )
    X = cube_np.reshape(-1, cube_np.shape[-1])
    if method == "mean":
        return X.mean(axis=0)
    if method == "median":
        return np.median(X, axis=0)
    if method == "trimmed":
        from scipy.stats import trim_mean

        return trim_mean(X, proportiontocut=trim_proportion, axis=0)
    if method == "huber":
        return _huber_mean(X, c=huber_c)
    raise ValueError(f"estimate_mean: unknown method {method!r}.")


def _huber_mean(
    X: np.ndarray, *, c: float, n_iter: int = 20, tol: float = 1e-8
) -> np.ndarray:
    """Per-band Huber M-estimator via iteratively-reweighted least squares."""
    mu = np.median(X, axis=0)
    mad = np.median(np.abs(X - mu), axis=0) + 1e-12
    scale = 1.4826 * mad
    for _ in range(n_iter):
        z = (X - mu) / scale
        w = np.where(np.abs(z) <= c, 1.0, c / np.abs(z).clip(min=1e-12))
        mu_new = (w * X).sum(axis=0) / w.sum(axis=0).clip(min=1e-12)
        if np.max(np.abs(mu_new - mu)) < tol:
            mu = mu_new
            break
        mu = mu_new
    return mu


# ── dense covariance estimators ──────────────────────────────────────────────


def _dense_operator(cov: np.ndarray) -> LinearOperator:
    """Wrap a dense PSD matrix as a lineax operator with the right tags."""
    return lx.MatrixLinearOperator(
        jnp.asarray(cov),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )


def estimate_cov_empirical(
    cube: Float[Array, "H W B"] | np.ndarray,
    mean: np.ndarray | None = None,
    *,
    ridge: float = 0.0,
) -> LinearOperator:
    """Empirical covariance via :class:`sklearn.covariance.EmpiricalCovariance`.

    Returns a dense :class:`lineax.MatrixLinearOperator`. Optionally adds a
    diagonal ``ridge`` for numerical PD-ness when ``n_samples ≲ n_bands``.
    """
    from sklearn.covariance import EmpiricalCovariance

    X = _flatten_cube(cube)
    est = EmpiricalCovariance(store_precision=False).fit(X)
    cov = est.covariance_
    if mean is not None:
        # sklearn uses the empirical mean internally; re-centre if the caller
        # supplied a robust mean so (μ, Σ) are consistent.
        Xc = X - mean
        cov = (Xc.T @ Xc) / X.shape[0]
    if ridge > 0.0:
        cov = cov + ridge * np.eye(cov.shape[0])
    return _dense_operator(cov)


def estimate_cov_shrunk(
    cube: Float[Array, "H W B"] | np.ndarray,
    mean: np.ndarray | None = None,
    *,
    method: Literal["ledoit_wolf", "oas"] = "ledoit_wolf",
) -> LinearOperator:
    """Shrinkage covariance estimator.

    Uses Ledoit–Wolf (``method='ledoit_wolf'``, default) or OAS
    (``'oas'``) — both shrink the sample covariance toward a scaled identity
    and are PD by construction even when ``n_samples < n_bands``.
    """
    X = _flatten_cube(cube)
    if mean is not None:
        X = X - mean
        assume_centered = True
    else:
        assume_centered = False
    if method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        est = LedoitWolf(assume_centered=assume_centered).fit(X)
    elif method == "oas":
        from sklearn.covariance import OAS

        est = OAS(assume_centered=assume_centered).fit(X)
    else:
        raise ValueError(
            f"estimate_cov_shrunk: method must be 'ledoit_wolf' or 'oas'; got {method!r}."
        )
    return _dense_operator(est.covariance_)


# ── low-rank covariance (Woodbury-friendly) ──────────────────────────────────


def estimate_cov_lowrank(
    cube: Float[Array, "H W B"] | np.ndarray,
    mean: np.ndarray | None = None,
    *,
    rank: int,
    tikhonov: float,
    random_state: int | None = 0,
    n_oversamples: int = 10,
) -> LinearOperator:
    """Low-rank + Tikhonov covariance: ``Σ = λI + V D Vᵀ``.

    Uses scikit-learn's randomised :class:`~sklearn.decomposition.TruncatedSVD`
    (Halko–Martinsson–Tropp) to find the top ``rank`` spectral directions of
    the sample covariance. Returned as a :class:`gaussx.LowRankUpdate` so that
    :func:`gaussx.solve` routes through the Woodbury identity — the MF
    precompute cost is ``O(n_bands · rank + rank³)`` instead of
    ``O(n_bands³)``.

    Parameters
    ----------
    cube
        Hyperspectral scene ``(H, W, n_bands)``.
    mean
        Background mean. If ``None``, subtracts the sample mean.
    rank
        Number of leading components to keep. Clamped to
        ``min(rank, n_samples, n_bands) - 1`` because ``TruncatedSVD``
        requires ``n_components < n_features``.
    tikhonov
        Diagonal floor ``λ > 0`` — ensures strict PD.
    random_state
        Seed for the randomised SVD.
    n_oversamples
        Extra random directions for the Halko sampling — higher is slower but
        better-conditioned. sklearn's default is 10.
    """
    if tikhonov <= 0.0:
        raise ValueError("estimate_cov_lowrank: tikhonov must be > 0.")
    X = _flatten_cube(cube)
    mu = X.mean(axis=0) if mean is None else np.asarray(mean)
    Xc = X - mu
    n_samples, n_bands = Xc.shape
    rank = max(1, min(int(rank), n_samples - 1, n_bands - 1))
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(
        n_components=rank,
        algorithm="randomized",
        n_oversamples=n_oversamples,
        random_state=random_state,
    )
    # TruncatedSVD(X) returns the top-k singular values s_i(X). We use MLE
    # normalization s_i² / n_samples for the covariance eigenvalues so this
    # path matches the sklearn EmpiricalCovariance / LedoitWolf / OAS paths
    # in this module (all of which divide by n_samples). Using the unbiased
    # (n_samples - 1) normalization here would have made matched_filter_snr
    # and detection_threshold dependent on *which* estimator was chosen.
    svd.fit(Xc)
    V = svd.components_  # shape (rank, n_bands), rows are right singular vectors
    s = svd.singular_values_  # shape (rank,)
    d = (s**2) / max(n_samples, 1)
    U = jnp.asarray(V.T)  # (n_bands, rank) — spectral directions as columns
    base = lx.DiagonalLinearOperator(
        jnp.asarray(tikhonov * np.ones(n_bands, dtype=float))
    )
    return gx.LowRankUpdate(
        base,
        U,
        jnp.asarray(d),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def _flatten_cube(cube: Float[Array, "H W B"] | np.ndarray) -> np.ndarray:
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError(
            f"background estimator: cube must be (H, W, n_bands), got shape {arr.shape}."
        )
    return arr.reshape(-1, arr.shape[-1])
