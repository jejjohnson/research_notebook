"""Ornstein-Uhlenbeck turbulence for Gaussian-puff position disturbances.

The base :mod:`plume_simulation.gauss_puff.puff` model advects every puff by
the cumulative wind integral

    x_i(t) = x_src + ∫_{t_r^{(i)}}^{t} u(τ) dτ ,          y_i analogous,

so two puffs released at the same instant sit on top of each other and puffs
released close together are near-collinear along the mean wind. Real plumes
show sub-grid meander and turbulence that this model cannot resolve. Following
the Gorroño-calibrated approach used by Orbio's Project Eucalyptus we add a
per-puff offset drawn from a 2-D Ornstein-Uhlenbeck (OU) process indexed by
release time:

    dp(τ) = -p(τ)/τ_c · dτ + σ_f · dW_τ                    (SDE)

where τ_c is the OU correlation time [s] and σ_f is the fluctuation
amplitude [m/√s]. Under the exact transition

    p(τ + Δτ) = p(τ) · exp(-Δτ / τ_c)
              + σ_f · √(τ_c/2 · (1 - exp(-2 Δτ / τ_c))) · ξ,   ξ ~ 𝒩(0, I)

the steady-state variance is Var[p_∞] = σ_f² · τ_c / 2 (per component). We
evaluate the process at the puff release times and add the result to each
puff's advected centre as a *time-of-release* offset — one sample per puff,
held constant over the puff's lifetime. This reproduces the observed meander
structure (neighbouring puffs share similar offsets, distant-in-time puffs are
independent) without introducing a second SDE into :func:`evolve_puffs`.

Public surface
--------------
- :class:`OUTurbulence`  — dataclass bundling (σ_f, τ_c).
- :func:`sample_ou_offsets` — draw per-release-time (Δx, Δy) from the SDE.

Design choices
~~~~~~~~~~~~~~
* The sampler returns NumPy arrays (not JAX): OU samples are consumed by the
  JIT-compiled :func:`evolve_puffs` as static per-puff offsets, so staying on
  the CPU side avoids a spurious recompilation per key and keeps the API
  friendly to callers who prefer :mod:`numpy.random` for reproducibility.
* ``seed`` (an ``int`` or :class:`numpy.random.Generator`) is an explicit
  argument rather than a thread-local — tests and notebooks pin it for
  reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float


@dataclass(frozen=True)
class OUTurbulence:
    """Ornstein-Uhlenbeck turbulence for puff-release position disturbances.

    Attributes
    ----------
    sigma_fluctuations : float
        OU fluctuation amplitude [m/√s]. Together with ``correlation_time``
        it sets the steady-state RMS offset ``σ_∞ = σ_f · √(τ_c / 2)`` [m].
    correlation_time : float
        OU correlation time [s]. Offsets for puffs released within ``τ_c``
        of each other are strongly correlated; puffs released more than a
        few ``τ_c`` apart are effectively independent.

    Notes
    -----
    Gorroño et al. (2023) calibrated comparable parameters against
    large-eddy simulations of methane plumes; defaults in the Eucalyptus
    release are (σ_f, τ_c) = (0.5 m/s, 60 s) giving a steady-state RMS
    offset of about 2.7 m — appropriate for 20 m-GSD Sentinel-2 pixels.
    """

    sigma_fluctuations: float = 0.5
    correlation_time: float = 60.0

    def __post_init__(self) -> None:
        if not (self.sigma_fluctuations >= 0.0):
            raise ValueError(
                "OUTurbulence: `sigma_fluctuations` must be ≥ 0 "
                f"(got {self.sigma_fluctuations!r})"
            )
        if not (self.correlation_time > 0.0):
            raise ValueError(
                "OUTurbulence: `correlation_time` must be > 0 "
                f"(got {self.correlation_time!r})"
            )

    @property
    def stationary_std(self) -> float:
        """Steady-state RMS per-component offset ``σ_f · √(τ_c / 2)`` [m]."""
        return float(self.sigma_fluctuations * np.sqrt(self.correlation_time / 2.0))


def sample_ou_offsets(
    turbulence: OUTurbulence,
    release_times: Float[np.ndarray, "N"] | np.ndarray,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw per-puff (Δx, Δy) offsets from a 2-D OU process at release times.

    Integrates the OU SDE via its *exact* transition density — the two-point
    Gaussian update that is exact for constant coefficients — so the result
    is statistically correct even for coarse ``release_times`` spacing and
    does not accumulate Euler-Maruyama bias.

    Parameters
    ----------
    turbulence : OUTurbulence
        Turbulence parameters.
    release_times : ndarray, shape (N,)
        Puff release times [s]. Must be monotone non-decreasing. The first
        puff starts at the OU stationary distribution so the ensemble is
        already mixed (no warm-up transient visible in the field).
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility. If ``None`` a fresh default
        generator is used and results will vary across calls.

    Returns
    -------
    dx, dy : ndarray, shape (N,)
        Per-puff position offsets [m], to be added to the advected puff
        centres in :func:`~plume_simulation.gauss_puff.puff.evolve_puffs`.
    """
    rel = np.asarray(release_times, dtype=np.float64).reshape(-1)
    if rel.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    if np.any(np.diff(rel) < 0.0):
        raise ValueError(
            "sample_ou_offsets: `release_times` must be monotone non-decreasing."
        )

    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )

    sigma_f = float(turbulence.sigma_fluctuations)
    tau_c = float(turbulence.correlation_time)
    sigma_inf = turbulence.stationary_std  # per-component stationary std

    n = rel.size
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)

    # Initial puff drawn from the stationary distribution so the trajectory
    # is already mixed — no need to run burn-in before the first release.
    z0 = rng.standard_normal(2) * sigma_inf
    dx[0] = z0[0]
    dy[0] = z0[1]

    # σ_f = 0 → deterministic OU, stays at the initial draw (which will
    # itself be exactly zero since σ_inf = 0). Shortcut to avoid spurious
    # zero-variance noise draws.
    if sigma_f == 0.0:
        dx[:] = 0.0
        dy[:] = 0.0
        return dx, dy

    for i in range(1, n):
        dt = rel[i] - rel[i - 1]
        decay = np.exp(-dt / tau_c)
        # Conditional variance of p(t+Δt) | p(t) under the exact OU transition.
        step_var = sigma_inf**2 * (1.0 - decay**2)
        step_std = np.sqrt(max(step_var, 0.0))
        eps = rng.standard_normal(2)
        dx[i] = decay * dx[i - 1] + step_std * eps[0]
        dy[i] = decay * dy[i - 1] + step_std * eps[1]

    return dx, dy
