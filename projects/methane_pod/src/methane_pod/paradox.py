"""Missing Mass Paradox — Monte Carlo library (no plotting, no I/O).

A pedagogical numerical proof of the satellite "Missing Mass Paradox":

    A satellite alerting system simultaneously OVERESTIMATES the average
    emission rate of a facility while strictly UNDERESTIMATING the total
    emitted mass.

The paradox emerges from the Thinned Marked Temporal Point Process framework:
a size-dependent Probability of Detection preferentially destroys small events,
biasing both sample means (upward) and sample sums (downward).

This module is a pure-computation library — drivers, plotting, and reporting
live in the accompanying notebooks.

Architecture (three strictly separated layers)
----------------------------------------------
  1. Event Generator  : Temporal Point Process  λ(t)   [events day⁻¹]
  2. Mark Generator   : Lognormal Mark Distribution f(Q) over Q [kg hr⁻¹]
  3. Atmospheric Filter: Parametric PoD P_d(Q) logistic sigmoid

Units convention
----------------
  t      : days
  λ      : events day⁻¹
  Q      : kg hr⁻¹
  D      : hr event⁻¹
  M      : kg

Dependencies: numpy only (pure CPU; the JAX/equinox intensity and POD modules
live in `intensity.py` / `pod_functions.py`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit


# ═════════════════════════════════════════════════════════════════════════════
# §1  PHYSICAL CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FacilityConfig:
    """Immutable physical configuration of a single leaking facility.

    Parameters
    ----------
    name : str
        Human-readable scenario label.
    lambda_true : float
        True homogeneous Poisson intensity [events day⁻¹].
    mu : float
        Lognormal location parameter [ln(kg hr⁻¹)].
    sigma : float
        Lognormal scale parameter [dimensionless].
    Q_50 : float
        Detection threshold at P_d = 0.5 [kg hr⁻¹].
    k : float
        Logistic-sigmoid steepness [hr kg⁻¹].
    duration : float
        Mean event duration [hr event⁻¹].
    observation_window : float
        Total observation period [days].
    seed : int
        PRNG seed for reproducibility.
    """

    name: str = "Default Facility"
    lambda_true: float = 2.0
    mu: float = 2.0
    sigma: float = 1.5
    Q_50: float = 100.0
    k: float = 0.02
    duration: float = 2.0
    observation_window: float = 365.0
    seed: int = 42

    @property
    def E_Q_true(self) -> float:
        """Analytic true mean emission rate E[Q_true] = exp(μ + σ²/2) [kg hr⁻¹]."""
        return float(np.exp(self.mu + 0.5 * self.sigma**2))

    @property
    def Lambda_true(self) -> float:
        """Expected total number of true events Λ_true = λ·T [events]."""
        return self.lambda_true * self.observation_window

    @property
    def M_true_analytic(self) -> float:
        """Analytic true total mass M_true = Λ·E[Q]·D [kg]."""
        return self.Lambda_true * self.E_Q_true * self.duration


# ═════════════════════════════════════════════════════════════════════════════
# §2  CORE PHYSICS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════


def lognormal_pdf(Q: NDArray, mu: float, sigma: float) -> NDArray:
    """Evaluate the Lognormal probability density function.

    f(Q) = (1 / (Q · σ · √(2π))) · exp(-(ln(Q) - μ)² / (2σ²))

    Parameters
    ----------
    Q : ndarray, shape (*batch,)
        Emission flux rates [kg hr⁻¹]. Must be > 0.
    mu, sigma : float
        Lognormal parameters.

    Returns
    -------
    ndarray, shape (*batch,)
        PDF values [hr kg⁻¹].
    """
    Q_safe = np.maximum(Q, 1e-30)
    log_Q = np.log(Q_safe)
    exponent = -((log_Q - mu) ** 2) / (2.0 * sigma**2)
    normalization = 1.0 / (Q_safe * sigma * np.sqrt(2.0 * np.pi))
    return normalization * np.exp(exponent)


def logistic_pod(Q: NDArray, Q_50: float, k: float) -> NDArray:
    """Parametric Probability of Detection (logistic sigmoid).

    P_d(Q) = 1 / (1 + exp(-k · (Q - Q₅₀)))

    Uses ``scipy.special.expit`` for a numerically stable sigmoid.
    """
    return expit(k * (Q - Q_50))


def compute_E_Pd(
    mu: float,
    sigma: float,
    Q_50: float,
    k: float,
    n_quad: int = 10_000,
) -> float:
    """Compute E[P_d] = ∫₀^∞ P_d(Q) · f(Q) dQ by log-space quadrature.

    We integrate in log-space u = ln(Q) so the lognormal kernel becomes a
    Gaussian in u, truncated to [μ - 6σ, μ + 6σ].

    Parameters
    ----------
    mu, sigma : float
        Lognormal parameters.
    Q_50, k : float
        Logistic PoD parameters.
    n_quad : int
        Number of trapezoid points.

    Returns
    -------
    float
        E[P_d] ∈ [0, 1].
    """
    u = np.linspace(mu - 6.0 * sigma, mu + 6.0 * sigma, n_quad)
    Q = np.exp(u)
    gauss_kernel = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -((u - mu) ** 2) / (2.0 * sigma**2)
    )
    pod_values = logistic_pod(Q, Q_50, k)
    return float(np.trapezoid(pod_values * gauss_kernel, u))


# ═════════════════════════════════════════════════════════════════════════════
# §3  SIMULATION RESULT CONTAINER
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ParadoxResult:
    """Container for a single Monte Carlo paradox simulation."""

    config: FacilityConfig
    N_true: int
    N_obs: int
    marks_true: NDArray
    marks_obs: NDArray
    E_Q_true_mc: float
    E_Q_obs_mc: float
    M_true_mc: float
    M_obs_mc: float
    E_Pd: float
    M_true_analytic: float
    average_overestimation_ratio: float
    mass_underestimation_ratio: float
    missing_mass_fraction: float
    MMSF: float


# ═════════════════════════════════════════════════════════════════════════════
# §4  SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════


def simulate_paradox(
    config: FacilityConfig,
    *,
    E_Pd: float | None = None,
) -> ParadoxResult:
    """Run a single Monte Carlo realisation of the Missing Mass Paradox.

    Implements the three-layer stochastic architecture:

      Layer 1: draw N_true ~ Poisson(Λ_true).
      Layer 2: draw Q_i ~ LogNormal(μ, σ) for i = 1..N_true.
      Layer 3: thin each event independently with Bernoulli(P_d(Q_i)).

    Then computes the paradox metrics:

      PROOF 1: E[Q_obs] > E[Q_true]  (average overestimation)
      PROOF 2: M_obs    < M_true     (total mass underestimation)

    Parameters
    ----------
    config : FacilityConfig
        Physical scenario parameters.
    E_Pd : float, optional
        Pre-computed ``compute_E_Pd(mu, sigma, Q_50, k)`` for this scenario.
        If ``None`` (default), it is computed inside the call. Pass the cached
        value when running many MC seeds on the same scenario — it saves a
        10k-point quadrature per trial.

    Returns
    -------
    ParadoxResult
        Full simulation output with diagnostic metrics.
    """
    rng = np.random.default_rng(config.seed)

    Lambda_true = config.Lambda_true
    N_true = int(rng.poisson(lam=Lambda_true))

    if E_Pd is None:
        E_Pd = compute_E_Pd(config.mu, config.sigma, config.Q_50, config.k)

    if N_true == 0:
        return ParadoxResult(
            config=config,
            N_true=0,
            N_obs=0,
            marks_true=np.array([]),
            marks_obs=np.array([]),
            E_Q_true_mc=0.0,
            E_Q_obs_mc=0.0,
            M_true_mc=0.0,
            M_obs_mc=0.0,
            E_Pd=E_Pd,
            M_true_analytic=config.M_true_analytic,
            average_overestimation_ratio=np.nan,
            mass_underestimation_ratio=np.nan,
            missing_mass_fraction=np.nan,
            MMSF=np.nan,
        )

    marks_true = rng.lognormal(mean=config.mu, sigma=config.sigma, size=N_true)

    pod_per_event = logistic_pod(marks_true, config.Q_50, config.k)
    survived = rng.uniform(size=N_true) < pod_per_event
    marks_obs = marks_true[survived]
    N_obs = int(survived.sum())

    D = config.duration

    E_Q_true_mc = float(np.mean(marks_true))
    M_true_mc = float(np.sum(marks_true) * D)

    if N_obs > 0:
        E_Q_obs_mc = float(np.mean(marks_obs))
        M_obs_mc = float(np.sum(marks_obs) * D)
    else:
        E_Q_obs_mc = 0.0
        M_obs_mc = 0.0

    if E_Q_true_mc > 0 and E_Q_obs_mc > 0:
        avg_overest = E_Q_obs_mc / E_Q_true_mc
    else:
        avg_overest = np.nan

    if M_true_mc > 0:
        mass_underest = M_obs_mc / M_true_mc
        missing_frac = 1.0 - mass_underest
        mmsf = M_true_mc / M_obs_mc if M_obs_mc > 0 else np.inf
    else:
        mass_underest = np.nan
        missing_frac = np.nan
        mmsf = np.nan

    return ParadoxResult(
        config=config,
        N_true=N_true,
        N_obs=N_obs,
        marks_true=marks_true,
        marks_obs=marks_obs,
        E_Q_true_mc=E_Q_true_mc,
        E_Q_obs_mc=E_Q_obs_mc,
        M_true_mc=M_true_mc,
        M_obs_mc=M_obs_mc,
        E_Pd=E_Pd,
        M_true_analytic=config.M_true_analytic,
        average_overestimation_ratio=avg_overest,
        mass_underestimation_ratio=mass_underest,
        missing_mass_fraction=missing_frac,
        MMSF=mmsf,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §5  MULTI-SCENARIO RUNNER
# ═════════════════════════════════════════════════════════════════════════════


def run_scenario_grid(
    configs: list[FacilityConfig],
    n_mc_per_scenario: int = 50,
) -> list[ParadoxResult]:
    """Run the paradox simulation across multiple scenarios, returning the
    median trial (by MMSF) per scenario to reduce sampling noise.

    Parameters
    ----------
    configs : list of FacilityConfig
    n_mc_per_scenario : int
        Number of independent Monte Carlo trials per scenario.

    Returns
    -------
    list of ParadoxResult
        One representative result per scenario.
    """
    results: list[ParadoxResult] = []
    for cfg in configs:
        # Quadrature for E[P_d] depends only on (mu, sigma, Q_50, k), not on
        # the seed — compute it once per scenario and reuse across trials.
        e_pd = compute_E_Pd(cfg.mu, cfg.sigma, cfg.Q_50, cfg.k)
        trials = [
            simulate_paradox(
                dataclasses.replace(cfg, seed=cfg.seed + i), E_Pd=e_pd
            )
            for i in range(n_mc_per_scenario)
        ]
        mmsfs = [t.MMSF for t in trials if np.isfinite(t.MMSF)]
        if mmsfs:
            median_mmsf = float(np.median(mmsfs))
            best = min(trials, key=lambda t: abs(t.MMSF - median_mmsf))
        else:
            best = trials[0]
        results.append(best)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# §6  CANONICAL SCENARIOS
# ═════════════════════════════════════════════════════════════════════════════


def build_canonical_scenarios() -> list[FacilityConfig]:
    """Six canonical test scenarios spanning the paradox regime.

    Detection midpoints (Q₅₀) are set to be representative of the sparse-revisit
    satellite constellation actually in orbit today — Sentinel-2, Landsat-8/9,
    EMIT, EnMAP, PRISMA, TROPOMI, Sentinel-3. For a given overpass the
    least-sensitive platform dominates; stacked multi-sensor POD is on the
    order of 500–3000 kg/hr. GHGSat/Tanager (Q₅₀ ~ 100 kg/hr) is the
    hyperspectral floor but only provides sparse tasked coverage.

    1. Abandoned Well             — Q₅₀=2000 (TROPOMI-heavy). Severe paradox.
    2. Active Compressor          — Q₅₀=1000 (mixed constellation). Moderate.
    3. Super-Emitter (tasked)     — Q₅₀=300 (hyperspectral tasking). Mild.
    4. Extreme Skew (σ=2.5)       — Q₅₀=1000. A few blowouts dominate mass.
    5. Sharp Sensor (ideal limit) — Q₅₀=1000, k large. Near-binary threshold.
    6. Soft Sensor                — Q₅₀=1000, k small. Gradual stacked POD.
    """
    return [
        FacilityConfig(
            name="1. Abandoned Well (TROPOMI-dominant, severe)",
            lambda_true=1.0, mu=1.5, sigma=1.8,
            Q_50=2000.0, k=0.002, duration=3.0,
            observation_window=365.0, seed=42,
        ),
        FacilityConfig(
            name="2. Active Compressor (mixed constellation, moderate)",
            lambda_true=3.0, mu=3.0, sigma=1.2,
            Q_50=1000.0, k=0.003, duration=2.0,
            observation_window=365.0, seed=123,
        ),
        FacilityConfig(
            name="3. Super-Emitter (tasked hyperspectral, mild)",
            lambda_true=5.0, mu=5.0, sigma=0.8,
            Q_50=300.0, k=0.01, duration=1.5,
            observation_window=365.0, seed=456,
        ),
        FacilityConfig(
            name="4. Extreme Skew (sigma=2.5, mixed)",
            lambda_true=2.0, mu=2.0, sigma=2.5,
            Q_50=1000.0, k=0.003, duration=2.0,
            observation_window=365.0, seed=789,
        ),
        FacilityConfig(
            name="5. Sharp Sensor (ideal hyperspectral limit)",
            lambda_true=2.0, mu=2.5, sigma=1.5,
            Q_50=1000.0, k=0.02, duration=2.0,
            observation_window=365.0, seed=101,
        ),
        FacilityConfig(
            name="6. Soft Sensor (stacked multi-platform POD)",
            lambda_true=2.0, mu=2.5, sigma=1.5,
            Q_50=1000.0, k=0.0008, duration=2.0,
            observation_window=365.0, seed=202,
        ),
    ]
