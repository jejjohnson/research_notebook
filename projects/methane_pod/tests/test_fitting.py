"""Smoke test: the POD-powerlaw NUTS fit runs end-to-end on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from methane_pod.fitting import (
    X_MAX_DEFAULT,
    X_MIN_DEFAULT,
    lognorm_cdf,
    power_law,
    run_mcmc,
)


def test_lognorm_cdf_monotone_and_bounded():
    x = np.logspace(1, 5, 200)
    pod = lognorm_cdf(x, x50=500.0, s=0.5)
    assert np.all(pod >= 0.0) and np.all(pod <= 1.0)
    assert np.all(np.diff(pod) >= -1e-9), "POD should be non-decreasing"
    # midpoint ~ x50
    idx = np.argmin(np.abs(x - 500.0))
    assert abs(pod[idx] - 0.5) < 0.02


def test_lognorm_cdf_rejects_non_positive_inputs():
    with pytest.raises(ValueError, match=r"`x` must be > 0"):
        lognorm_cdf(np.array([10.0, 0.0, 20.0]), x50=100.0, s=0.5)
    with pytest.raises(ValueError, match=r"`x` must be > 0"):
        lognorm_cdf(np.array([10.0, -1.0]), x50=100.0, s=0.5)
    with pytest.raises(ValueError, match=r"`x50` must be > 0"):
        lognorm_cdf(np.array([10.0]), x50=0.0, s=0.5)
    with pytest.raises(ValueError, match=r"`s` must be > 0"):
        lognorm_cdf(np.array([10.0]), x50=100.0, s=-0.1)


def test_power_law_evaluates():
    x = np.array([10.0, 100.0, 1000.0])
    y = power_law(x, alpha=1.5)
    assert y.shape == x.shape
    assert np.all(y > 0.0)
    assert y[0] > y[1] > y[2]


def _sample_from_powerlaw(rng, alpha, x_min, x_max, n):
    """Inverse-CDF draw from the truncated power-law x^{-α}."""
    u = rng.uniform(size=n)
    if np.isclose(alpha, 1.0):
        return x_min * (x_max / x_min) ** u
    one_minus_alpha = 1.0 - alpha
    return (
        u * (x_max**one_minus_alpha - x_min**one_minus_alpha)
        + x_min**one_minus_alpha
    ) ** (1.0 / one_minus_alpha)


def _thin_by_pod(rng, x, x50, s):
    """Bernoulli thinning with lognormal-CDF PoD."""
    p = lognorm_cdf(x, x50, s)
    return x[rng.uniform(size=x.size) < p]


@pytest.mark.slow
def test_run_mcmc_recovers_alpha_on_synthetic_data():
    """Smoke test: NUTS runs without error and alpha is in a reasonable range.

    This is a smoke test — full-quality recovery needs more samples than the
    tiny config here. We only check that the fit completes and returns
    plausible values for the synthetic α = 1.8.
    """
    rng = np.random.default_rng(0)
    alpha_true = 1.8
    x_true = _sample_from_powerlaw(
        rng, alpha_true, X_MIN_DEFAULT, X_MAX_DEFAULT, n=5_000
    )
    x_obs = _thin_by_pod(rng, x_true, x50=500.0, s=0.4)
    x_obs = x_obs[(x_obs > X_MIN_DEFAULT) & (x_obs < X_MAX_DEFAULT)]
    assert x_obs.size > 100, "synthetic sample too small — bump n"

    df = run_mcmc(
        x_obs,
        num_warmup=100,
        num_samples=200,
        num_chains=1,
        seed=0,
    )
    assert set(df.columns) == {"x0", "sk", "alpha"}
    assert len(df) == 200
    # wide sanity window — short chain, not a recovery test
    assert 1.1 < df["alpha"].mean() < 4.5
    assert 1.0 < df["x0"].mean() < 20_000.0
    assert 0.1 < df["sk"].mean() < 1.5
