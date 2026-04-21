"""Bayesian inference for the Gaussian puff forward model (NumPyro).

The forward model evaluates puff concentrations at observation coordinates
for a given emission rate ``Q`` (constant or time-varying) under a specified
wind schedule. This module provides:

* :func:`gaussian_puff_model` — a NumPyro model over ``Q``, a half-normal
  background, and Gaussian observation noise.
* :func:`infer_emission_rate` — NUTS runner for a **constant** Q, wrapping
  the model with input validation and posterior unpacking.
* :func:`infer_emission_timeseries` — NUTS runner for a **time-varying**
  ``Q_i`` under a random-walk prior, for state-estimation workflows.

The log-space prior parameters are derived from real-scale ``(mean, std)``
via :func:`plume_simulation.gauss_plume.inference._lognormal_from_moments`,
reused here to keep the plume and puff inference helpers aligned.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from plume_simulation.gauss_plume.inference import _lognormal_from_moments
from plume_simulation.gauss_puff.dispersion import (
    STABILITY_CLASSES,
    get_dispersion_scheme,
)
from plume_simulation.gauss_puff.puff import (
    evolve_puffs,
    frequency_to_release_interval,
    make_release_times,
    simulate_puff_field,
)
from plume_simulation.gauss_puff.wind import WindSchedule


def _forward_inputs_present(
    receptor_coords,
    observation_times,
    source_location,
    schedule,
    release_times,
    release_interval,
) -> bool:
    """Return True iff every input needed to evaluate the forward puff model is set."""
    return (
        receptor_coords is not None
        and observation_times is not None
        and source_location is not None
        and schedule is not None
        and release_times is not None
        and release_interval is not None
    )


def _require_forward_inputs(
    observations: jnp.ndarray | None,
    receptor_coords: tuple | None,
    observation_times: jnp.ndarray | None,
    source_location: tuple | None,
    schedule: WindSchedule | None,
    release_times: jnp.ndarray | None,
    release_interval: float | None,
) -> None:
    """Validate that when ``observations`` are given, every forward-model input is too.

    Prior-predictive calls (``observations=None``) can proceed with any subset of
    forward-model inputs; only full-forward evaluation is skipped in that case.
    """
    if observations is None:
        return
    missing = [
        name
        for name, value in (
            ("receptor_coords", receptor_coords),
            ("observation_times", observation_times),
            ("source_location", source_location),
            ("schedule", schedule),
            ("release_times", release_times),
            ("release_interval", release_interval),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "gaussian_puff_model: `observations` were provided but the "
            "forward model cannot be evaluated — missing inputs: "
            f"{', '.join(missing)}. Either pass all of "
            "(receptor_coords, observation_times, source_location, schedule, "
            "release_times, release_interval) or set observations=None for "
            "prior-predictive mode."
        )


def _predict_observations(
    emission_per_puff: jnp.ndarray,   # (N_puffs,) per-puff mass [kg]
    release_times: jnp.ndarray,        # (N_puffs,)
    receptor_coords: tuple,            # (x, y, z) each (N_obs,)
    observation_times: jnp.ndarray,    # (N_obs,) per-observation evaluation time
    source_location: tuple,
    schedule: WindSchedule,
    dispersion_params: jnp.ndarray,
    dispersion_fn,
) -> jnp.ndarray:
    """Vectorised forward model: concentration at each observation (coord, time).

    Returns an array of shape ``(N_obs,)``: the k-th entry is the total
    puff field evaluated at ``(x[k], y[k], z[k])`` at time ``observation_times[k]``.
    """
    x_obs, y_obs, z_obs = receptor_coords

    def predict_one(x_k, y_k, z_k, t_k):
        puff_state = evolve_puffs(
            schedule,
            release_times,
            t_k,
            source_location,
            emission_per_puff,
        )
        single = simulate_puff_field(
            (jnp.atleast_1d(x_k), jnp.atleast_1d(y_k), jnp.atleast_1d(z_k)),
            puff_state,
            dispersion_params,
            dispersion_fn,
        )
        return single[0]

    return jax.vmap(predict_one)(x_obs, y_obs, z_obs, observation_times)


def gaussian_puff_model(
    observations: jnp.ndarray | None = None,
    receptor_coords: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    observation_times: jnp.ndarray | None = None,
    source_location: tuple[float, float, float] | None = None,
    schedule: WindSchedule | None = None,
    release_times: jnp.ndarray | None = None,
    release_interval: float | None = None,
    stability_class: str = "C",
    scheme: str = "pg",
    prior_emission_rate_mean: float = 0.1,
    prior_emission_rate_std: float = 0.05,
    background_prior_std: float = 5e-7,
    obs_noise_std: float = 5e-7,
) -> jnp.ndarray:
    """NumPyro model for the Gaussian puff forward with a constant Q prior.

    Priors:
        emission_rate ~ LogNormal(μ_log, σ_log)                     [kg/s]
        background    ~ HalfNormal(σ_bg)                            [kg/m³]

    Likelihood:
        obs ~ Normal(f_puff(emission_rate, ...) + background, σ_obs)

    where ``(μ_log, σ_log)`` are the log-space parameters derived from the
    requested real-scale ``(prior_emission_rate_mean, prior_emission_rate_std)``.

    Parameters
    ----------
    observations : jnp.ndarray, shape (N,), optional
        Observed concentrations [kg/m³]. If None, runs in prior-predictive mode.
    receptor_coords : tuple of jnp.ndarray, optional
        ``(x, y, z)`` each shape (N,) [m].
    observation_times : jnp.ndarray, shape (N,), optional
        Time at which each observation was taken [s]. Matches ``observations``
        in length.
    source_location : tuple of float, (3,), optional
    schedule : WindSchedule, optional
    release_times : jnp.ndarray, shape (N_puffs,), optional
    release_interval : float, optional
        ``Δt_release`` [s]. Used to convert ``Q`` → per-puff mass
        (``m = Q · Δt``). Required when a forward evaluation is needed.
    stability_class : str
    scheme : str
        Dispersion scheme: ``'pg'`` or ``'briggs'``.
    prior_emission_rate_mean, prior_emission_rate_std : float
    background_prior_std, obs_noise_std : float
    """
    _require_forward_inputs(
        observations,
        receptor_coords,
        observation_times,
        source_location,
        schedule,
        release_times,
        release_interval,
    )
    scheme_params_dict, dispersion_fn = get_dispersion_scheme(scheme)
    if stability_class not in scheme_params_dict:
        raise ValueError(
            f"gaussian_puff_model: `stability_class` must be one of "
            f"{STABILITY_CLASSES}, got {stability_class!r}"
        )
    dispersion_params = scheme_params_dict[stability_class]

    mu_log, sigma_log = _lognormal_from_moments(
        prior_emission_rate_mean, prior_emission_rate_std
    )
    emission_rate = numpyro.sample(
        "emission_rate", dist.LogNormal(mu_log, sigma_log)
    )
    background = numpyro.sample(
        "background", dist.HalfNormal(background_prior_std)
    )

    # The forward model is evaluated whenever its inputs are present —
    # independently of whether `observations` were provided. This keeps
    # NumPyro's `Predictive` path (observations=None + full inputs) producing
    # model-driven draws, while prior-predictive calls without forward inputs
    # fall through to a background-only draw.
    forward_ready = _forward_inputs_present(
        receptor_coords,
        observation_times,
        source_location,
        schedule,
        release_times,
        release_interval,
    )

    if forward_ready:
        puff_mass = emission_rate * release_interval * jnp.ones_like(release_times)
        predicted = _predict_observations(
            puff_mass,
            release_times,
            receptor_coords,
            observation_times,
            source_location,
            schedule,
            dispersion_params,
            dispersion_fn,
        )
        predicted = predicted + background
    else:
        predicted = background

    return numpyro.sample(
        "obs",
        dist.Normal(predicted, obs_noise_std),
        obs=observations,
    )


def gaussian_puff_rw_model(
    observations: jnp.ndarray,
    receptor_coords: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    observation_times: jnp.ndarray,
    source_location: tuple[float, float, float],
    schedule: WindSchedule,
    release_times: jnp.ndarray,
    release_interval: float,
    stability_class: str = "C",
    scheme: str = "pg",
    prior_emission_rate_mean: float = 0.1,
    prior_emission_rate_std: float = 0.05,
    rw_step_std: float = 0.02,
    background_prior_std: float = 5e-7,
    obs_noise_std: float = 5e-7,
) -> jnp.ndarray:
    """Random-walk state-space model for a time-varying emission rate Q_i.

    Each puff ``i`` has its own emission rate ``Q_i``. A Gaussian random walk
    on ``ln Q_i`` gives a smoothness prior, with the first step anchored by
    the LogNormal(``prior_mean``, ``prior_std``) prior:

        ln Q_0            ~ Normal(μ_log, σ_log)
        ln Q_i | ln Q_{i−1} ~ Normal(ln Q_{i−1}, rw_step_std)        for i ≥ 1
        Q_i              = exp(ln Q_i)                              [kg/s]

    Likelihood is the same as :func:`gaussian_puff_model`.
    """
    scheme_params_dict, dispersion_fn = get_dispersion_scheme(scheme)
    if stability_class not in scheme_params_dict:
        raise ValueError(
            f"gaussian_puff_rw_model: `stability_class` must be one of "
            f"{STABILITY_CLASSES}, got {stability_class!r}"
        )
    dispersion_params = scheme_params_dict[stability_class]

    n_puffs = release_times.shape[0]
    if n_puffs == 0:
        raise ValueError(
            "gaussian_puff_rw_model: `release_times` must contain ≥ 1 puff; "
            "got n_puffs=0. Check that t_end − t_start ≥ 1/release_frequency."
        )

    mu_log, sigma_log = _lognormal_from_moments(
        prior_emission_rate_mean, prior_emission_rate_std
    )

    innovations = numpyro.sample(
        "innovations",
        dist.Normal(0.0, 1.0).expand([n_puffs]).to_event(1),
    )
    # Build ln Q_i by cumulative sum of scaled innovations, first anchored
    # to the LogNormal prior location.
    first = mu_log + sigma_log * innovations[0]
    increments = rw_step_std * innovations[1:]
    log_q = jnp.concatenate(
        [first[None], first + jnp.cumsum(increments)]
    )
    emission_series = numpyro.deterministic("emission_rate", jnp.exp(log_q))

    background = numpyro.sample(
        "background", dist.HalfNormal(background_prior_std)
    )

    puff_mass = emission_series * release_interval
    predicted = _predict_observations(
        puff_mass,
        release_times,
        receptor_coords,
        observation_times,
        source_location,
        schedule,
        dispersion_params,
        dispersion_fn,
    )
    predicted = predicted + background

    return numpyro.sample(
        "obs",
        dist.Normal(predicted, obs_noise_std),
        obs=observations,
    )


# ── High-level runners ───────────────────────────────────────────────────────


def _validate_inference_inputs(
    func_name: str,
    observations,
    observation_coords,
    observation_times,
    wind_times,
    wind_speed,
    wind_direction,
    source_location,
    release_frequency,
    t_start,
    t_end,
    prior_mean,
    prior_std,
):
    """Shared input-validation guard for the NUTS inference runners.

    Returns the coerced ``(obs, coords, times)`` NumPy views on success.
    """
    obs = np.asarray(observations)
    if obs.size == 0:
        raise ValueError(
            f"{func_name}: `observations` must contain ≥ 1 point"
        )
    if len(observation_coords) != 3:
        raise ValueError(
            f"{func_name}: `observation_coords` must be (x, y, z); "
            f"got length {len(observation_coords)}"
        )
    coords = tuple(np.asarray(c) for c in observation_coords)
    for axis_name, arr in zip("xyz", coords, strict=False):
        if arr.shape != obs.shape:
            raise ValueError(
                f"{func_name}: `observation_coords` axis '{axis_name}' "
                f"has shape {arr.shape} ≠ observations shape {obs.shape}"
            )
    times = np.asarray(observation_times)
    if times.shape != obs.shape:
        raise ValueError(
            f"{func_name}: `observation_times` shape {times.shape} "
            f"≠ observations shape {obs.shape}"
        )
    wt = np.asarray(wind_times)
    ws = np.asarray(wind_speed)
    wd = np.asarray(wind_direction)
    if wt.ndim != 1 or wt.size == 0:
        raise ValueError(
            f"{func_name}: `wind_times` must be 1-D with ≥ 1 entry"
        )
    if ws.shape != wt.shape:
        raise ValueError(
            f"{func_name}: `wind_speed` shape {ws.shape} must match "
            f"`wind_times` shape {wt.shape}"
        )
    if wd.shape != wt.shape:
        raise ValueError(
            f"{func_name}: `wind_direction` shape {wd.shape} must match "
            f"`wind_times` shape {wt.shape}"
        )
    if len(source_location) != 3:
        raise ValueError(
            f"{func_name}: `source_location` must contain exactly three values"
        )
    if not (release_frequency > 0.0):
        raise ValueError(
            f"{func_name}: `release_frequency` must be > 0 "
            f"(got {release_frequency!r})"
        )
    if not (t_end > t_start):
        raise ValueError(
            f"{func_name}: `t_end` must be > `t_start` "
            f"(got t_start={t_start!r}, t_end={t_end!r})"
        )
    if not (prior_mean > 0.0):
        raise ValueError(
            f"{func_name}: `prior_mean` must be > 0 (got {prior_mean!r})"
        )
    if not (prior_std > 0.0):
        raise ValueError(
            f"{func_name}: `prior_std` must be > 0 (got {prior_std!r})"
        )
    return obs, coords, times


def infer_emission_rate(
    observations: NDArray,
    observation_coords: tuple[NDArray, NDArray, NDArray],
    observation_times: NDArray,
    source_location: tuple[float, float, float],
    wind_times: NDArray,
    wind_speed: NDArray,
    wind_direction: NDArray,
    release_frequency: float,
    t_start: float,
    t_end: float,
    stability_class: str = "C",
    scheme: str = "pg",
    prior_mean: float = 0.1,
    prior_std: float = 0.05,
    obs_noise_std: float = 5e-7,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
    progress_bar: bool = False,
    print_summary: bool = False,
) -> dict[str, NDArray]:
    """NUTS inference of a **constant** emission rate from puff observations.

    Parameters
    ----------
    observations : ndarray, shape (N,)
        Observed concentrations [kg/m³].
    observation_coords : tuple of ndarray, each shape (N,)
        ``(x, y, z)`` receptor coordinates [m].
    observation_times : ndarray, shape (N,)
        Evaluation time for each observation [s].
    source_location : tuple of float, (3,)
    wind_times, wind_speed, wind_direction : ndarray, shape (T_w,)
        Wind schedule. Direction follows the meteorological "from" convention.
    release_frequency : float
        Puff release rate [Hz].
    t_start, t_end : float
        Simulation window [s]. Release times are laid down on ``[t_start, t_end)``.
    stability_class : str
    scheme : str
        Dispersion scheme: ``'pg'`` or ``'briggs'``.
    prior_mean, prior_std : float
        Real-scale prior moments for Q.
    num_warmup, num_samples, num_chains, seed : int
    progress_bar, print_summary : bool

    Returns
    -------
    samples : dict[str, ndarray]
        Posterior draws keyed by site name: ``'emission_rate'``, ``'background'``.
    """
    obs, coords, times = _validate_inference_inputs(
        "infer_emission_rate",
        observations,
        observation_coords,
        observation_times,
        wind_times,
        wind_speed,
        wind_direction,
        source_location,
        release_frequency,
        t_start,
        t_end,
        prior_mean,
        prior_std,
    )

    schedule = WindSchedule.from_speed_direction(
        wind_times, wind_speed, wind_direction
    )
    release_times = make_release_times(t_start, t_end, release_frequency)
    release_interval = frequency_to_release_interval(release_frequency)

    obs_jax = jnp.asarray(obs)
    coords_jax = tuple(jnp.asarray(c) for c in coords)
    times_jax = jnp.asarray(times)

    mcmc = MCMC(
        NUTS(gaussian_puff_model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        observations=obs_jax,
        receptor_coords=coords_jax,
        observation_times=times_jax,
        source_location=source_location,
        schedule=schedule,
        release_times=release_times,
        release_interval=release_interval,
        stability_class=stability_class,
        scheme=scheme,
        prior_emission_rate_mean=prior_mean,
        prior_emission_rate_std=prior_std,
        obs_noise_std=obs_noise_std,
    )
    if print_summary:
        mcmc.print_summary()
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


def infer_emission_timeseries(
    observations: NDArray,
    observation_coords: tuple[NDArray, NDArray, NDArray],
    observation_times: NDArray,
    source_location: tuple[float, float, float],
    wind_times: NDArray,
    wind_speed: NDArray,
    wind_direction: NDArray,
    release_frequency: float,
    t_start: float,
    t_end: float,
    stability_class: str = "C",
    scheme: str = "pg",
    prior_mean: float = 0.1,
    prior_std: float = 0.05,
    rw_step_std: float = 0.02,
    obs_noise_std: float = 5e-7,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
    progress_bar: bool = False,
    print_summary: bool = False,
) -> dict[str, NDArray]:
    """NUTS inference of a **time-varying** emission rate series Q_i.

    Uses a Gaussian random-walk prior on ``ln Q`` along puff indices, anchored
    by the LogNormal(``prior_mean``, ``prior_std``) prior on ``ln Q_0``. See
    :func:`gaussian_puff_rw_model`.

    Returns posterior draws with key ``'emission_rate'`` of shape
    ``(num_samples, n_puffs)``.
    """
    obs, coords, times = _validate_inference_inputs(
        "infer_emission_timeseries",
        observations,
        observation_coords,
        observation_times,
        wind_times,
        wind_speed,
        wind_direction,
        source_location,
        release_frequency,
        t_start,
        t_end,
        prior_mean,
        prior_std,
    )
    if not (rw_step_std > 0.0):
        raise ValueError(
            "infer_emission_timeseries: `rw_step_std` must be > 0 "
            f"(got {rw_step_std!r})"
        )
    if not (obs_noise_std > 0.0):
        raise ValueError(
            "infer_emission_timeseries: `obs_noise_std` must be > 0 "
            f"(got {obs_noise_std!r})"
        )

    schedule = WindSchedule.from_speed_direction(
        wind_times, wind_speed, wind_direction
    )
    release_times = make_release_times(t_start, t_end, release_frequency)
    release_interval = frequency_to_release_interval(release_frequency)

    mcmc = MCMC(
        NUTS(gaussian_puff_rw_model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        observations=jnp.asarray(obs),
        receptor_coords=tuple(jnp.asarray(c) for c in coords),
        observation_times=jnp.asarray(times),
        source_location=source_location,
        schedule=schedule,
        release_times=release_times,
        release_interval=release_interval,
        stability_class=stability_class,
        scheme=scheme,
        prior_emission_rate_mean=prior_mean,
        prior_emission_rate_std=prior_std,
        rw_step_std=rw_step_std,
        obs_noise_std=obs_noise_std,
    )
    if print_summary:
        mcmc.print_summary()
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
