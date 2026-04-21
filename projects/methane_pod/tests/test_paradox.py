"""Monte Carlo paradox: determinism, shape, and qualitative-direction tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from methane_pod.paradox import (
    FacilityConfig,
    ParadoxResult,
    build_canonical_scenarios,
    compute_E_Pd,
    lognormal_pdf,
    logistic_pod,
    simulate_paradox,
)


def test_logistic_pod_limits_and_midpoint():
    Q = np.array([0.0, 100.0, 1e6])
    pod = logistic_pod(Q, Q_50=100.0, k=0.02)
    assert pytest.approx(pod[1], abs=1e-10) == 0.5
    assert pod[0] < 0.5 < pod[2]
    assert pod[2] > 0.99


def test_lognormal_pdf_integrates_to_one():
    # Integrate in log-space u = ln(Q) so the kernel is Gaussian(μ, σ)
    mu, sigma = 2.0, 1.5
    u = np.linspace(mu - 8 * sigma, mu + 8 * sigma, 50_000)
    Q = np.exp(u)
    f = lognormal_pdf(Q, mu, sigma)
    mass = np.trapezoid(f * Q, u)  # dQ = Q du
    assert pytest.approx(mass, abs=1e-3) == 1.0


def test_E_Pd_bounded():
    val = compute_E_Pd(mu=3.0, sigma=1.5, Q_50=100.0, k=0.02)
    assert 0.0 <= val <= 1.0
    assert np.isfinite(val)


def test_E_Pd_monotone_in_threshold():
    """Raising Q₅₀ can only reduce (or leave unchanged) the average detection rate."""
    low = compute_E_Pd(mu=3.0, sigma=1.5, Q_50=10.0, k=0.02)
    high = compute_E_Pd(mu=3.0, sigma=1.5, Q_50=5_000.0, k=0.02)
    assert low > high


def test_simulate_paradox_shapes_and_determinism():
    cfg = FacilityConfig(seed=7)
    a = simulate_paradox(cfg)
    b = simulate_paradox(cfg)
    assert isinstance(a, ParadoxResult)
    assert a.N_true == b.N_true
    assert np.allclose(a.marks_true, b.marks_true)
    assert a.marks_obs.shape == (a.N_obs,)
    assert a.marks_true.shape == (a.N_true,)


def test_paradox_direction_on_severe_case():
    """For a PoD-limited scenario, E[Q_obs] > E[Q_true] AND M_obs < M_true."""
    cfg = FacilityConfig(
        name="severe",
        lambda_true=5.0,
        mu=1.5,
        sigma=1.8,
        Q_50=500.0,
        k=0.015,
        duration=2.0,
        observation_window=365.0,
        seed=11,
    )
    r = simulate_paradox(cfg)
    assert r.average_overestimation_ratio > 1.0
    assert r.mass_underestimation_ratio < 1.0
    assert r.MMSF > 1.0


def test_edge_case_zero_events(monkeypatch):
    """Tiny λ·T can yield N_true = 0; result must still be a finite container."""
    cfg = FacilityConfig(
        lambda_true=1e-6, observation_window=1e-6, seed=0
    )
    r = simulate_paradox(cfg)
    # expected to roll a zero; if not, just check the result is well-formed
    assert r.N_obs <= r.N_true
    assert np.isfinite(r.E_Pd)


def test_canonical_scenarios_all_exhibit_paradox():
    for cfg in build_canonical_scenarios():
        # Run a couple of trials; the paradox direction should hold on average.
        trials = [
            simulate_paradox(dataclasses.replace(cfg, seed=cfg.seed + i))
            for i in range(4)
        ]
        overest = np.nanmean([t.average_overestimation_ratio for t in trials])
        underest = np.nanmean([t.mass_underestimation_ratio for t in trials])
        assert overest >= 1.0 - 0.05, f"{cfg.name}: no avg overestimation"
        assert underest <= 1.0 + 0.05, f"{cfg.name}: no mass underestimation"
