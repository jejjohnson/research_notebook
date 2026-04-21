"""Bayesian inference for the Gaussian plume forward model (NumPyro).

Infers the emission rate Q from downwind concentration observations, and
optionally the atmospheric stability class as a discrete latent. The forward
model is :func:`plume_simulation.gauss_plume.plume.plume_concentration`;
observations are modelled as Gaussian around the forward prediction plus a
half-normal background.

Public surface
--------------
- ``gaussian_plume_model`` : NumPyro probabilistic model
- ``infer_emission_rate``   : NUTS runner returning a dict of posterior arrays
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs

from plume_simulation.gauss_plume.dispersion import (
    BRIGGS_DISPERSION_PARAMS,
    STABILITY_CLASSES,
    get_dispersion_params,
)
from plume_simulation.gauss_plume.plume import plume_concentration


def gaussian_plume_model(
    observations: jnp.ndarray | None = None,
    receptor_coords: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    source_location: tuple[float, float, float] | None = None,
    wind_u: float | None = None,
    wind_v: float | None = None,
    stability_class: str = "C",
    prior_emission_rate_mean: float = 0.1,
    prior_emission_rate_std: float = 0.05,
    infer_stability: bool = False,
    background_prior_std: float = 5e-7,
    obs_noise_std: float = 5e-7,
) -> jnp.ndarray:
    """NumPyro model for the steady-state Gaussian plume.

    Priors:
        emission_rate ~ LogNormal(ln(μ_Q), σ_Q / μ_Q)              [kg/s]
        stability_idx ~ Categorical(1/6, ..., 1/6)  (optional)     [A..F]
        background    ~ HalfNormal(σ_bg)                            [kg/m³]

    Likelihood:
        obs ~ Normal(f_plume(emission_rate, ...) + background, σ_obs)

    Parameters
    ----------
    observations : jnp.ndarray, shape (N,), optional
        Observed concentrations [kg/m³]. If None, runs in prior-predictive mode.
    receptor_coords : tuple of jnp.ndarray, optional
        ``(x, y, z)``, each shape (N,), in the fixed frame [m].
    source_location : tuple of float, optional
        ``(x, y, z)`` source coordinates [m].
    wind_u, wind_v : float, optional
        Wind velocity components [m/s].
    stability_class : str
        Stability class used if ``infer_stability=False``. Default ``'C'``.
    prior_emission_rate_mean, prior_emission_rate_std : float
        Prior mean / std for the emission rate [kg/s].
    infer_stability : bool
        If True, sample a categorical ``stability_idx`` latent over A-F.
    background_prior_std : float
        HalfNormal scale for the background concentration [kg/m³].
    obs_noise_std : float
        Observation noise σ [kg/m³].
    """
    emission_rate = numpyro.sample(
        "emission_rate",
        dist.LogNormal(
            jnp.log(prior_emission_rate_mean),
            prior_emission_rate_std / prior_emission_rate_mean,
        ),
    )

    if infer_stability:
        stability_probs = jnp.ones(len(STABILITY_CLASSES)) / len(STABILITY_CLASSES)
        stability_idx = numpyro.sample(
            "stability_idx", dist.Categorical(probs=stability_probs)
        )
        all_params = jnp.stack(
            [BRIGGS_DISPERSION_PARAMS[c] for c in STABILITY_CLASSES]
        )
        dispersion_params = all_params[stability_idx]
    else:
        dispersion_params = get_dispersion_params(stability_class)

    background = numpyro.sample("background", dist.HalfNormal(background_prior_std))

    if (
        receptor_coords is not None
        and source_location is not None
        and wind_u is not None
        and wind_v is not None
    ):
        x_obs, y_obs, z_obs = receptor_coords
        src_x, src_y, src_z = source_location
        predicted = plume_concentration(
            x_obs,
            y_obs,
            z_obs,
            src_x,
            src_y,
            src_z,
            wind_u,
            wind_v,
            emission_rate,
            dispersion_params,
        )
        predicted = predicted + background
    else:
        predicted = 0.0

    return numpyro.sample(
        "obs",
        dist.Normal(predicted, obs_noise_std),
        obs=observations,
    )


def infer_emission_rate(
    observations: NDArray,
    observation_coords: tuple[NDArray, NDArray, NDArray],
    source_location: tuple[float, float, float],
    wind_speed: float,
    wind_direction: float,
    stability_class: str = "C",
    infer_stability: bool = False,
    prior_mean: float = 0.1,
    prior_std: float = 0.05,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
    progress_bar: bool = False,
    print_summary: bool = False,
) -> dict[str, NDArray]:
    """Infer the plume emission rate via NUTS.

    Wraps :func:`gaussian_plume_model` with input validation, wind-from-angle
    to (u, v) conversion, and posterior unpacking into NumPy arrays.

    Parameters
    ----------
    observations : ndarray, shape (N,)
        Observed concentrations [kg/m³].
    observation_coords : tuple of ndarray
        ``(x, y, z)``, each shape (N,), in the fixed frame [m].
    source_location : tuple of float, (3,)
        ``(x, y, z)`` source coordinates [m].
    wind_speed : float
        Wind speed magnitude [m/s]. Must be > 0.
    wind_direction : float
        Wind direction [degrees from North, meteorological "from" convention].
    stability_class : str
        Stability class used if ``infer_stability=False``. Default ``'C'``.
    infer_stability : bool
        If True, jointly sample a categorical stability-class latent.
    prior_mean, prior_std : float
        Prior parameters for the emission rate [kg/s].
    num_warmup, num_samples, num_chains : int
        NUTS sampler configuration.
    seed : int
        PRNG seed.
    progress_bar : bool
        If True, show the NumPyro progress bar. Default False.
    print_summary : bool
        If True, call ``mcmc.print_summary()`` after the run.

    Returns
    -------
    samples : dict[str, ndarray]
        Posterior draws keyed by site name: ``'emission_rate'``,
        ``'background'``, and (if ``infer_stability``) ``'stability_idx'``.

    Raises
    ------
    ValueError
        On invalid stability class, non-positive wind speed, shape mismatches
        between ``observations`` and ``observation_coords``, or empty data.
    """
    if not (wind_speed > 0.0):
        raise ValueError(
            f"infer_emission_rate: `wind_speed` must be > 0 (got {wind_speed!r})"
        )
    if not infer_stability and stability_class not in BRIGGS_DISPERSION_PARAMS:
        raise ValueError(
            f"infer_emission_rate: `stability_class` must be one of "
            f"{STABILITY_CLASSES}, got {stability_class!r}"
        )
    obs = np.asarray(observations)
    if obs.size == 0:
        raise ValueError(
            "infer_emission_rate: `observations` must contain ≥ 1 point"
        )
    if len(observation_coords) != 3:
        raise ValueError(
            "infer_emission_rate: `observation_coords` must be (x, y, z); "
            f"got length {len(observation_coords)}"
        )
    coords = tuple(np.asarray(c) for c in observation_coords)
    for axis_name, arr in zip("xyz", coords, strict=False):
        if arr.shape != obs.shape:
            raise ValueError(
                f"infer_emission_rate: `observation_coords` axis '{axis_name}' "
                f"has shape {arr.shape} ≠ observations shape {obs.shape}"
            )
    if not (prior_mean > 0.0):
        raise ValueError(
            f"infer_emission_rate: `prior_mean` must be > 0 (got {prior_mean!r})"
        )
    if not (prior_std > 0.0):
        raise ValueError(
            f"infer_emission_rate: `prior_std` must be > 0 (got {prior_std!r})"
        )

    theta_rad = np.deg2rad(wind_direction)
    wind_u = float(-wind_speed * np.sin(theta_rad))
    wind_v = float(-wind_speed * np.cos(theta_rad))

    obs_jax = jnp.asarray(obs)
    coords_jax = tuple(jnp.asarray(c) for c in coords)

    inner = NUTS(gaussian_plume_model)
    # Discrete latents can't be handled by NUTS on their own without
    # `funsor` enumeration. `DiscreteHMCGibbs` wraps NUTS with a Gibbs
    # step over the categorical stability-class site, avoiding the extra
    # dependency while giving a valid joint sampler.
    kernel = DiscreteHMCGibbs(inner) if infer_stability else inner
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        observations=obs_jax,
        receptor_coords=coords_jax,
        source_location=source_location,
        wind_u=wind_u,
        wind_v=wind_v,
        stability_class=stability_class,
        prior_emission_rate_mean=prior_mean,
        prior_emission_rate_std=prior_std,
        infer_stability=infer_stability,
    )
    if print_summary:
        mcmc.print_summary()

    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
