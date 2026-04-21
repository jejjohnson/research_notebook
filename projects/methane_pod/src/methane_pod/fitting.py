"""POD-modified power-law fitting via NumPyro NUTS (library only).

A Bayesian fit of the observed methane plume flux distribution to the model

    p_power(x) = C₁ · x⁻ᵅ,                  x_min ≤ x ≤ x_max
    q(x)       = Φ_LN(x; x₀, σ)             (lognormal CDF = PoD)
    p_obs(x)   = C₂ · x⁻ᵅ · q(x)            (detected distribution)

with parameters (α, x₀, σ) inferred by NUTS. The data loading, plotting, and
per-satellite CSV plumbing from the original `james_jax.py` script live in the
accompanying notebook — this module exposes only the pure-computation pieces:

  - `lognorm_cdf`         : POD curve evaluation on a grid
  - `power_law`           : un-normalised power-law PDF
  - `pod_powerlaw_model`  : NumPyro generative model
  - `run_mcmc`            : NUTS runner returning a posterior DataFrame

Dependencies: numpy, scipy, jax, numpyro, pandas (for the MCMC runner output).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import scipy
from jax.scipy.special import erfc
from jax.scipy.stats import norm as jax_norm
from numpy.typing import NDArray
from numpyro.infer import MCMC, NUTS

if TYPE_CHECKING:  # pandas is imported lazily inside run_mcmc
    import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

X_MIN_DEFAULT = 10.0   # kg/hr — lower bound of power-law domain
X_MAX_DEFAULT = 1e6    # kg/hr — upper bound of power-law domain


# ── Utility functions (NumPy) ────────────────────────────────────────────────


def lognorm_cdf(x: NDArray, x50: float, s: float) -> NDArray:
    """Lognormal CDF used as the probability-of-detection curve.

    PoD(x) = 1 − ½ · erfc((ln x − ln x₅₀) / (√2 · σ))

    Parameters
    ----------
    x : array_like, shape (N,)
        Emission rate [kg/hr]. Must be strictly positive.
    x50 : float
        Median detection rate [kg/hr]. Must be strictly positive.
    s : float
        Lognormal width [dimensionless]. Must be strictly positive.

    Returns
    -------
    pod : ndarray, shape (N,)
        Probability of detection ∈ [0, 1].

    Raises
    ------
    ValueError
        If any of ``x``, ``x50``, ``s`` are non-positive.
    """
    x_arr = np.asarray(x)
    if np.any(x_arr <= 0.0):
        raise ValueError("lognorm_cdf: all entries of `x` must be > 0")
    if not (x50 > 0.0):
        raise ValueError(f"lognorm_cdf: `x50` must be > 0 (got {x50!r})")
    if not (s > 0.0):
        raise ValueError(f"lognorm_cdf: `s` must be > 0 (got {s!r})")
    return 1.0 - 0.5 * scipy.special.erfc(
        (np.log(x_arr) - np.log(x50)) / (np.sqrt(2.0) * s)
    )


def power_law(x: NDArray, alpha: float) -> NDArray:
    """Un-normalised power-law PDF: x⁻ᵅ."""
    return np.power(x, -alpha)


# ── NumPyro model ────────────────────────────────────────────────────────────


def pod_powerlaw_model(
    x_obs: jnp.ndarray,
    x_min: float,
    x_max: float,
    num_integration_points: int = 5000,
) -> None:
    """NumPyro model for a POD-modified power law on observed fluxes.

    Sampled parameters
    ------------------
    alpha : power-law exponent, ``Uniform(1.1, 4.5)``         [dimensionless]
    x0    : lognormal CDF median (POD 50 %), ``Uniform(1, 20000)`` [kg/hr]
    sk    : lognormal CDF width, ``Uniform(0.1, 1.5)``        [dimensionless]

    Log-likelihood (numerical integration in log-space for I(θ)):

        ln L = −α · Σᵢ ln xᵢ + Σᵢ ln q(xᵢ) − N · ln I(θ)
        I(θ) = ∫_{x_min}^{x_max} q(x) · x⁻ᵅ dx

    Parameters
    ----------
    x_obs : jnp.ndarray, shape (N,)
        Observed emission rates [kg/hr].
    x_min, x_max : float
        Integration bounds [kg/hr].
    num_integration_points : int
        Trapezoidal grid size for I(θ). Default 5000.
    """
    N = len(x_obs)

    alpha = numpyro.sample("alpha", dist.Uniform(1.1, 4.5))
    x0 = numpyro.sample("x0", dist.Uniform(1.0, 20000.0))
    sk = numpyro.sample("sk", dist.Uniform(0.1, 1.5))

    sum_log_x = jnp.sum(jnp.log(x_obs))
    # q(x) = Φ((ln x − ln x0)/sk) as a Normal CDF; log-evaluate via
    # `norm.logcdf` so the far-left tail stays finite instead of underflowing
    # to −inf for parameter draws with large x0 / small sk (which would
    # otherwise break NUTS with NaN gradients).
    u = (jnp.log(x_obs) - jnp.log(x0)) / sk
    sum_log_q = jnp.sum(jax_norm.logcdf(u))

    xi_values = jnp.linspace(jnp.log(x_min), jnp.log(x_max), num_integration_points)
    pod_on_grid = 1.0 - 0.5 * erfc(
        (xi_values - jnp.log(x0)) / (sk * jnp.sqrt(2.0))
    )
    integrand = pod_on_grid * jnp.exp(-alpha * xi_values + xi_values)
    I_theta = jnp.trapezoid(integrand, xi_values)
    I_theta = jnp.clip(I_theta, min=1e-300)

    logL = -alpha * sum_log_x + sum_log_q - N * jnp.log(I_theta)
    numpyro.factor("log_likelihood", logL)


# ── MCMC runner ──────────────────────────────────────────────────────────────


def run_mcmc(
    data_values: NDArray,
    *,
    x_min: float = X_MIN_DEFAULT,
    x_max: float = X_MAX_DEFAULT,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 2,
    seed: int = 20,
    print_summary: bool = False,
) -> pd.DataFrame:
    """Run NumPyro NUTS on the POD-modified power-law model.

    Parameters
    ----------
    data_values : ndarray, shape (N,)
        Observed Q values [kg/hr]. Must be strictly positive and within the
        [x_min, x_max] domain of the power law.
    x_min, x_max : float
        Power-law support bounds [kg/hr].
    num_warmup : int
        NUTS warmup iterations per chain. Defaults are tuned for a
        notebook-grade fit (~seconds on CPU); bump to 2000-4000 for a
        publication-grade run.
    num_samples : int
        Posterior samples per chain after warmup.
    num_chains : int
        Number of independent MCMC chains.
    seed : int
        PRNG seed.
    print_summary : bool
        If True, call ``mcmc.print_summary()`` after the run. Default False
        to keep notebook output tidy.

    Returns
    -------
    df_mcmc : pandas.DataFrame, shape (num_samples * num_chains, 3)
        Columns: ``x0`` [kg/hr], ``sk`` [dimensionless], ``alpha`` [dimensionless].
    """
    import pandas as pd  # lazy — only `run_mcmc` requires pandas

    kernel = NUTS(pod_powerlaw_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, x_obs=data_values, x_min=x_min, x_max=x_max)
    if print_summary:
        mcmc.print_summary()

    samples = mcmc.get_samples()
    return pd.DataFrame(
        {
            "x0": np.asarray(samples["x0"]),
            "sk": np.asarray(samples["sk"]),
            "alpha": np.asarray(samples["alpha"]),
        }
    )
