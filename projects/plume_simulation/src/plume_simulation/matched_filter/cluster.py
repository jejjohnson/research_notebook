"""Cluster-based and spatially-adaptive background estimators.

When a scene is heterogeneous (land/water, bright/dark surfaces, different
land-cover classes) a single global ``(μ, Σ)`` smears together statistically
distinct populations and the MF loses power. Two standard fixes:

- **GMM clustering** — fit a Gaussian mixture on the per-pixel spectra,
  label each pixel, and run the MF with the local cluster's ``(μ_k, Σ_k)``.
  :func:`gmm_cluster_background` returns the labels and a list of
  ``(mean, cov_operator)`` pairs, one per cluster.
- **Adaptive spatial windows** — for each pixel, estimate
  ``(μ_{ij}, \\sigma^2_{ij})`` from a local neighbourhood. Only supports a
  diagonal local covariance (keeps memory at ``O(H·W·n_bands)``) — for full
  local covariance, prefer the GMM route.

Both use scikit-learn / scipy under the hood; neither is on a JAX gradient
path (the MF treats the background as a fixed input).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import lineax as lx
import numpy as np

from plume_simulation.matched_filter.background import (
    estimate_cov_empirical,
    estimate_cov_shrunk,
)


if TYPE_CHECKING:
    LinearOperator = lx.AbstractLinearOperator
else:
    LinearOperator = object


@dataclass(frozen=True)
class ClusterBackground:
    """Per-cluster background model.

    Attributes
    ----------
    labels
        Cluster assignment per pixel, shape ``(H, W)`` with values in
        ``[0, n_clusters)``.
    means
        List of per-cluster mean spectra.
    cov_operators
        List of per-cluster covariance operators (same structure across
        clusters — dense from :func:`estimate_cov_empirical` or
        :func:`estimate_cov_shrunk`).
    """

    labels: np.ndarray
    means: list[np.ndarray]
    cov_operators: list[LinearOperator]


def gmm_cluster_background(
    cube: np.ndarray,
    *,
    n_clusters: int,
    cov_estimator: str = "ledoit_wolf",
    random_state: int | None = 0,
    bayesian: bool = False,
) -> ClusterBackground:
    """Fit a Gaussian mixture and build per-cluster background models.

    Uses :class:`sklearn.mixture.GaussianMixture` (or
    :class:`BayesianGaussianMixture` when ``bayesian=True``) on the per-pixel
    spectra. Each cluster gets its own shrunk covariance estimate so that
    clusters with few pixels remain well-conditioned.

    Parameters
    ----------
    cube
        Hyperspectral cube, shape ``(H, W, n_bands)``.
    n_clusters
        Number of mixture components (upper bound for the Bayesian variant).
    cov_estimator
        ``'ledoit_wolf'`` (default), ``'oas'``, or ``'empirical'``.
    random_state
        Passed through to sklearn for reproducibility.
    bayesian
        If True, use Dirichlet-process-style :class:`BayesianGaussianMixture`,
        which prunes unused components automatically.

    Returns
    -------
    ClusterBackground
    """
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError(
            f"gmm_cluster_background: cube must be (H, W, n_bands); got {arr.shape}."
        )
    h, w, _ = arr.shape
    X = arr.reshape(-1, arr.shape[-1])
    if bayesian:
        from sklearn.mixture import BayesianGaussianMixture

        gmm = BayesianGaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_state,
        )
    else:
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_state,
        )
    labels = gmm.fit_predict(X)
    # Pre-compute a global (μ, Σ) fallback once so under-populated clusters
    # get a *consistent* pair — the old code returned the per-cluster mean
    # paired with a scene-wide Ledoit–Wolf Σ, which biases scores.
    global_mean = X.mean(axis=0)
    if cov_estimator == "empirical":
        global_cov_op = estimate_cov_empirical(arr, mean=global_mean)
    elif cov_estimator in {"ledoit_wolf", "oas"}:
        global_cov_op = estimate_cov_shrunk(arr, mean=global_mean, method=cov_estimator)  # type: ignore[arg-type]
    else:
        raise ValueError(
            f"gmm_cluster_background: unknown cov_estimator {cov_estimator!r}."
        )
    means: list[np.ndarray] = []
    ops: list[LinearOperator] = []
    for k in range(int(labels.max()) + 1):
        mask = labels == k
        if mask.sum() < 2:
            # Under-populated cluster: return the *same* (μ, Σ) pair for both
            # attributes so callers that use (means[k], cov_operators[k])
            # together see a self-consistent global-scene fallback.
            means.append(global_mean)
            ops.append(global_cov_op)
            continue
        # Reshape the flat cluster subset into a fake (n_k, 1, n_bands) cube
        # so the cube-shaped estimators accept it without dedicated per-cluster
        # code paths.
        fake_cube = X[mask].reshape(-1, 1, X.shape[1])
        mu_k = X[mask].mean(axis=0)
        if cov_estimator == "empirical":
            cov_op = estimate_cov_empirical(fake_cube, mean=mu_k)
        else:
            cov_op = estimate_cov_shrunk(fake_cube, mean=mu_k, method=cov_estimator)  # type: ignore[arg-type]
        means.append(mu_k)
        ops.append(cov_op)
    return ClusterBackground(
        labels=labels.reshape(h, w),
        means=means,
        cov_operators=ops,
    )


def adaptive_window_background(
    cube: np.ndarray,
    *,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel local mean and per-band variance over a square window.

    For each pixel ``(i, j)``, estimate ``μ_{ij}`` and a *diagonal* local
    covariance ``diag(σ²_{ij,b})`` using a ``window_size × window_size``
    uniform window centred on the pixel (reflecting at boundaries).

    Full local covariances would be ``(H · W · n_bands²)`` — too big for
    realistic scenes. For a dense local covariance, cluster with GMM instead.

    Parameters
    ----------
    cube
        Hyperspectral cube, shape ``(H, W, n_bands)``.
    window_size
        Edge length of the averaging window. Must be odd and ``>= 3``.

    Returns
    -------
    local_mean
        Per-pixel mean spectrum, shape ``(H, W, n_bands)``.
    local_variance
        Per-pixel per-band variance, shape ``(H, W, n_bands)``.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(
            f"adaptive_window_background: window_size must be an odd int ≥ 3; got {window_size}."
        )
    arr = np.asarray(cube, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"adaptive_window_background: cube must be (H, W, n_bands); got {arr.shape}."
        )
    from scipy.ndimage import uniform_filter

    # Apply a 2-D moving average independently to each band — scipy handles
    # the reflect-boundary by default, matching how most MF codes handle
    # scene edges.
    mean = np.empty_like(arr)
    var = np.empty_like(arr)
    for b in range(arr.shape[-1]):
        m = uniform_filter(arr[..., b], size=window_size, mode="reflect")
        m2 = uniform_filter(arr[..., b] ** 2, size=window_size, mode="reflect")
        mean[..., b] = m
        # Population variance — E[x²] - E[x]². Clipped to zero for numerics.
        var[..., b] = np.maximum(m2 - m * m, 0.0)
    return mean, var
