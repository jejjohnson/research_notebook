"""Streaming background estimation via Welford's algorithm.

For multi-scene MF workflows (e.g. aggregating background statistics across a
campaign of flights), the full ``(n_samples, n_bands)`` stack does not fit in
memory. The parallel Welford update [#welford1962]_ [#chan1979]_ maintains a
mean and (scaled) covariance in one pass, merges partial results from
separate chunks exactly, and is numerically stable:

.. math::
    n \\leftarrow n_A + n_B
    \\delta = \\mu_B - \\mu_A
    \\mu \\leftarrow \\mu_A + \\delta \\cdot n_B / n
    M_2 \\leftarrow M_{2,A} + M_{2,B} + \\delta \\delta^\\top \\cdot n_A n_B / n

Covariance is then ``M_2 / (n - 1)``.

.. [#welford1962] Welford (1962), Note on a method for calculating corrected
   sums of squares and products.
.. [#chan1979] Chan, Golub & LeVeque (1979), Updating formulae and a pairwise
   algorithm for computing sample variances.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import lineax as lx
import numpy as np


if TYPE_CHECKING:
    LinearOperator = lx.AbstractLinearOperator
else:
    LinearOperator = object


@dataclass
class WelfordAccumulator:
    """Online mean and covariance for ``(n_samples, n_bands)`` streams.

    Use :meth:`update` to feed one batch at a time; :meth:`merge` to combine
    two accumulators computed on disjoint subsets; :meth:`mean` and
    :meth:`covariance` to finalise.

    Attributes
    ----------
    n_bands
        Number of spectral bands.
    count
        Running sample count.
    """

    n_bands: int
    count: int = 0
    _mean: np.ndarray = field(init=False)
    _M2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.n_bands <= 0:
            raise ValueError("WelfordAccumulator: n_bands must be > 0.")
        self._mean = np.zeros(self.n_bands, dtype=float)
        self._M2 = np.zeros((self.n_bands, self.n_bands), dtype=float)

    def update(self, batch: np.ndarray) -> WelfordAccumulator:
        """Ingest a batch of pixel spectra (shape ``(n_samples, n_bands)``)."""
        arr = np.asarray(batch, dtype=float)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.ndim != 2 or arr.shape[1] != self.n_bands:
            raise ValueError(
                f"WelfordAccumulator.update: expected shape (*, {self.n_bands}); "
                f"got {arr.shape}."
            )
        n_b = arr.shape[0]
        if n_b == 0:
            return self
        mean_b = arr.mean(axis=0)
        diff = arr - mean_b
        M2_b = diff.T @ diff
        n_a = self.count
        n = n_a + n_b
        delta = mean_b - self._mean
        self._mean = self._mean + delta * (n_b / n)
        self._M2 = self._M2 + M2_b + np.outer(delta, delta) * (n_a * n_b / n)
        self.count = n
        return self

    def merge(self, other: WelfordAccumulator) -> WelfordAccumulator:
        """Combine with another accumulator on disjoint data."""
        if other.n_bands != self.n_bands:
            raise ValueError(
                f"WelfordAccumulator.merge: n_bands mismatch "
                f"({self.n_bands} vs {other.n_bands})."
            )
        if other.count == 0:
            return self
        if self.count == 0:
            self._mean = other._mean.copy()
            self._M2 = other._M2.copy()
            self.count = other.count
            return self
        n_a, n_b = self.count, other.count
        n = n_a + n_b
        delta = other._mean - self._mean
        self._mean = self._mean + delta * (n_b / n)
        self._M2 = self._M2 + other._M2 + np.outer(delta, delta) * (n_a * n_b / n)
        self.count = n
        return self

    def mean(self) -> np.ndarray:
        if self.count == 0:
            raise ValueError("WelfordAccumulator.mean: no data yet.")
        return self._mean.copy()

    def covariance(self, ddof: int = 1) -> np.ndarray:
        if self.count <= ddof:
            raise ValueError(
                f"WelfordAccumulator.covariance: need count > ddof; got count={self.count}."
            )
        return self._M2 / (self.count - ddof)


def streaming_background(
    batches: Iterable[np.ndarray],
    *,
    n_bands: int,
    ridge: float = 0.0,
    ddof: int = 0,
) -> tuple[np.ndarray, LinearOperator]:
    """Streamed ``(μ, Σ)`` from an iterable of batches.

    Convenience wrapper for the common pattern of looping over scenes/tiles.
    Returns the final mean and a dense lineax operator for the covariance —
    pass the covariance through :func:`~plume_simulation.matched_filter.core.apply_image`
    exactly like any other ``cov_op``.

    Parameters
    ----------
    batches
        Iterable yielding arrays of shape ``(n_samples, n_bands)`` or
        ``(H, W, n_bands)``.
    n_bands
        Number of spectral bands (declared up front since the iterable can be
        lazy).
    ridge
        Optional diagonal ridge added to the final covariance.
    ddof
        Delta degrees of freedom for the covariance denominator:
        ``M_2 / (n - ddof)``. Defaults to ``0`` (MLE, matches sklearn's
        ``EmpiricalCovariance``, ``LedoitWolf``, ``OAS`` and our
        :func:`estimate_cov_lowrank`), so
        :func:`~plume_simulation.matched_filter.core.matched_filter_snr` and
        :func:`~plume_simulation.matched_filter.core.detection_threshold`
        give the same scaling whether the covariance came from a streaming
        pass or a batch estimator. Set ``ddof=1`` to match ``np.cov`` /
        sample-covariance conventions instead.
    """
    acc = WelfordAccumulator(n_bands=n_bands)
    for batch in batches:
        acc.update(batch)
    mu = acc.mean()
    cov = acc.covariance(ddof=ddof)
    if ridge > 0.0:
        cov = cov + ridge * np.eye(n_bands)
    import jax.numpy as jnp

    return mu, lx.MatrixLinearOperator(
        jnp.asarray(cov),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
