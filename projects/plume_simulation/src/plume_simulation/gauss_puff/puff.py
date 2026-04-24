"""Gaussian puff forward model (JAX + diffrax).

A continuous source releasing at rate ``Q(t)`` is represented by a sequence
of instantaneous puffs at release times ``t_r^{(i)}`` carrying mass
``m_i = Q(t_r^{(i)}) · Δt_release``. Each puff is advected by the (possibly
time-varying) wind and spreads as a 3-D Gaussian:

    C_i(x, y, z; t) = m_i / [(2π)^{3/2} σ_x σ_y σ_z]
                      · exp[−(x − x_i(t))² / (2 σ_x²)]
                      · exp[−(y − y_i(t))² / (2 σ_y²)]
                      · [exp(−(z − z_s)² / (2 σ_z²))
                         + exp(−(z + z_s)² / (2 σ_z²))]          (ground reflection)

The total field is the superposition ``C = Σ_i C_i`` over all active puffs.
Dispersion coefficients are evaluated as functions of puff travel distance
``s_i(t) = ∫_{t_r^{(i)}}^{t} |V(τ)| dτ``.

Public surface
--------------
- ``release_interval_to_frequency`` / ``frequency_to_release_interval`` : helpers
- ``make_release_times``                 : build evenly spaced release times
- ``PuffState``                           : eqx.Module bundling puff positions/mass
- ``puff_concentration`` / ``puff_concentration_vmap`` : single-puff kernel
- ``evolve_puffs``                        : release_times + wind → PuffState via diffrax
- ``simulate_puff_field``                : JIT orchestrator, field at one time
- ``simulate_puff``                       : xarray wrapper with ``time`` axis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import equinox as eqx
from jax import jit, vmap
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np

from plume_simulation.gauss_puff.dispersion import (
    STABILITY_CLASSES,
    get_dispersion_scheme,
)
from plume_simulation.gauss_puff.wind import WindSchedule, cumulative_wind_integrals


if TYPE_CHECKING:  # xarray is imported lazily inside simulate_puff
    import xarray as xr

    from plume_simulation.gauss_puff.turbulence import OUTurbulence


# ── Release cadence helpers ──────────────────────────────────────────────────


def release_interval_to_frequency(release_interval: float) -> float:
    """Convert a release interval Δt [s] into a release frequency f [Hz].

    ``f = 1 / Δt``. Both Δt and f are common ways to parametrise puff cadence;
    this pair of helpers makes the conversion explicit.
    """
    if not (release_interval > 0.0):
        raise ValueError(
            f"release_interval_to_frequency: `release_interval` must be > 0 "
            f"(got {release_interval!r})"
        )
    return 1.0 / release_interval


def frequency_to_release_interval(release_frequency: float) -> float:
    """Convert a release frequency f [Hz] into a release interval Δt [s].

    ``Δt = 1 / f``.
    """
    if not (release_frequency > 0.0):
        raise ValueError(
            f"frequency_to_release_interval: `release_frequency` must be > 0 "
            f"(got {release_frequency!r})"
        )
    return 1.0 / release_frequency


def make_release_times(
    t_start: float,
    t_end: float,
    release_frequency: float,
) -> jnp.ndarray:
    """Evenly spaced puff release times in ``[t_start, t_end)``.

    Parameters
    ----------
    t_start, t_end : float
        Simulation time window [s]. ``t_start < t_end`` required.
    release_frequency : float
        Puff release rate [Hz]. Must be > 0.

    Returns
    -------
    release_times : jnp.ndarray, shape (N,)
        Times ``[t_start, t_start + Δt, t_start + 2Δt, ...]`` with
        ``Δt = 1 / release_frequency``, truncated before ``t_end``.
    """
    if not (t_end > t_start):
        raise ValueError(
            f"make_release_times: `t_end` must be > `t_start` "
            f"(got t_start={t_start!r}, t_end={t_end!r})"
        )
    dt = frequency_to_release_interval(release_frequency)
    n = int(np.floor((t_end - t_start) / dt))
    return jnp.asarray(t_start + np.arange(n) * dt, dtype=jnp.float32)


# ── Puff state container ─────────────────────────────────────────────────────


class PuffState(eqx.Module):
    """Container for the Lagrangian state of a puff ensemble at a single time.

    Attributes
    ----------
    release_times : Float[Array, "N"]
        Puff release times [s].
    x, y, z : Float[Array, "N"]
        Puff-centre positions at the evaluation time [m].
    travel_distance : Float[Array, "N"]
        Cumulative travel distance since release [m].
    mass : Float[Array, "N"]
        Puff mass [kg]. Inactive puffs (not yet released) have mass 0.
    """

    release_times: Float[Array, "N"]
    x: Float[Array, "N"]
    y: Float[Array, "N"]
    z: Float[Array, "N"]
    travel_distance: Float[Array, "N"]
    mass: Float[Array, "N"]


# ── Single-puff concentration kernel ─────────────────────────────────────────


@jit
def puff_concentration(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    puff_x: float,
    puff_y: float,
    puff_z: float,
    sigma_x: float,
    sigma_y: float,
    sigma_z: float,
    puff_mass: float,
) -> jnp.ndarray:
    """Evaluate the concentration from a single Gaussian puff (JIT).

    Includes ground-reflection via an image source at ``z = −puff_z``.

    Parameters
    ----------
    x, y, z : jnp.ndarray
        Receptor coordinates [m]. Broadcast-compatible.
    puff_x, puff_y, puff_z : float
        Puff-centre position [m]. ``puff_z ≥ 0``.
    sigma_x, sigma_y, sigma_z : float
        Dispersion coefficients [m]. All > 0.
    puff_mass : float
        Mass of the puff [kg]. Must be ≥ 0.

    Returns
    -------
    concentration : jnp.ndarray
        Mass concentration [kg/m³], same shape as the broadcast of the input
        coordinates.
    """
    dx = x - puff_x
    dy = y - puff_y
    dz_direct = z - puff_z
    dz_reflected = z + puff_z

    normalization = (
        jnp.power(2.0 * jnp.pi, 1.5) * sigma_x * sigma_y * sigma_z
    )

    exp_x = jnp.exp(-0.5 * jnp.square(dx / sigma_x))
    exp_y = jnp.exp(-0.5 * jnp.square(dy / sigma_y))
    exp_z_direct = jnp.exp(-0.5 * jnp.square(dz_direct / sigma_z))
    exp_z_reflected = jnp.exp(-0.5 * jnp.square(dz_reflected / sigma_z))

    return (puff_mass / normalization) * exp_x * exp_y * (
        exp_z_direct + exp_z_reflected
    )


puff_concentration_vmap = vmap(
    puff_concentration,
    in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0),
    out_axes=0,
)
"""Vectorised over puffs (per-puff params batched along axis 0, receptor shared)."""


# ── Diffrax-driven puff evolution ────────────────────────────────────────────


@eqx.filter_jit
def evolve_puffs(
    schedule: WindSchedule,
    release_times: Float[Array, "N"],
    current_time: Float[Array, ""],
    source_location: tuple[float, float, float],
    puff_mass: Float[Array, "N"] | float,
    position_disturbance: tuple[Float[Array, "N"], Float[Array, "N"]] | None = None,
) -> PuffState:
    """Advance a puff ensemble to ``current_time`` under a time-varying wind.

    Uses :func:`cumulative_wind_integrals` to compute ``(I_u, I_v, S)`` at the
    release times and at ``current_time`` in a single diffrax solve, then
    derives per-puff positions and travel distances as differences.

    Puffs with ``release_times > current_time`` are "not yet released" — their
    mass is set to 0 and position is placed at the source, so they contribute
    nothing to the downstream field.

    Parameters
    ----------
    schedule : WindSchedule
        Time-varying wind field.
    release_times : Float[Array, "N"]
        Monotone increasing puff release times [s].
    current_time : Float[Array, ""]
        Evaluation time [s].
    source_location : tuple of float, (3,)
        ``(x, y, z)`` source coordinates [m]. ``z ≥ 0``.
    puff_mass : Float[Array, "N"] or float
        Per-puff mass [kg], or a scalar applied to all puffs.
    position_disturbance : tuple of Float[Array, "N"], optional
        Per-puff ``(Δx, Δy)`` offsets [m] applied on top of the wind advection.
        Intended for OU-process turbulence samples (see
        :func:`plume_simulation.gauss_puff.turbulence.sample_ou_offsets`),
        but any caller-provided offset array is accepted. Offsets for
        not-yet-released puffs are ignored.

    Returns
    -------
    puff_state : PuffState
        Puff positions, travel distances, and masses at ``current_time``.
    """
    src_x, src_y, src_z = source_location

    save_at = jnp.concatenate(
        [release_times, jnp.atleast_1d(jnp.asarray(current_time))]
    )
    sort_idx = jnp.argsort(save_at)
    save_at_sorted = save_at[sort_idx]

    I_u_s, I_v_s, S_s = cumulative_wind_integrals(schedule, save_at_sorted)

    unsort_idx = jnp.argsort(sort_idx)
    I_u = I_u_s[unsort_idx]
    I_v = I_v_s[unsort_idx]
    S = S_s[unsort_idx]

    I_u_rel, I_v_rel, S_rel = I_u[:-1], I_v[:-1], S[:-1]
    I_u_now, I_v_now, S_now = I_u[-1], I_v[-1], S[-1]

    active = release_times <= current_time
    x = src_x + jnp.where(active, I_u_now - I_u_rel, 0.0)
    y = src_y + jnp.where(active, I_v_now - I_v_rel, 0.0)
    if position_disturbance is not None:
        dx_puff, dy_puff = position_disturbance
        x = x + jnp.where(active, dx_puff, 0.0)
        y = y + jnp.where(active, dy_puff, 0.0)
    z = jnp.full_like(release_times, src_z)
    s = jnp.where(active, S_now - S_rel, 0.0)

    mass_arr = jnp.broadcast_to(jnp.asarray(puff_mass), release_times.shape)
    mass = jnp.where(active, mass_arr, 0.0)

    return PuffState(
        release_times=release_times,
        x=x,
        y=y,
        z=z,
        travel_distance=s,
        mass=mass,
    )


# ── Field assembly ───────────────────────────────────────────────────────────


def simulate_puff_field(
    receptor_coords: tuple[
        Float[Array, "P"], Float[Array, "P"], Float[Array, "P"]
    ],
    puff_state: PuffState,
    dispersion_params: Float[Array, "6"],
    dispersion_fn: Callable,
) -> Float[Array, "P"]:
    """Sum the contributions of all puffs at each receptor.

    Parameters
    ----------
    receptor_coords : tuple of Float[Array, "P"]
        ``(x, y, z)`` receptor coordinates [m].
    puff_state : PuffState
        Per-puff positions, travel distances, and masses.
    dispersion_params : Float[Array, "6"]
        Dispersion coefficient vector for the chosen scheme and stability class.
    dispersion_fn : callable
        A function with signature ``(distance, params) → (σ_x, σ_y, σ_z)``.
        See :func:`plume_simulation.gauss_puff.dispersion.calculate_pg_dispersion`
        and :func:`calculate_briggs_dispersion_xyz`.

    Returns
    -------
    concentration : Float[Array, "P"]
        Total concentration [kg/m³] at each receptor.
    """
    x_r, y_r, z_r = receptor_coords

    sigma_x, sigma_y, sigma_z = dispersion_fn(
        puff_state.travel_distance, dispersion_params
    )

    per_puff = puff_concentration_vmap(
        x_r, y_r, z_r,
        puff_state.x, puff_state.y, puff_state.z,
        sigma_x, sigma_y, sigma_z,
        puff_state.mass,
    )  # (N_puffs, P)
    return jnp.sum(per_puff, axis=0)


# ── Xarray wrapper ───────────────────────────────────────────────────────────


def simulate_puff(
    emission_rate: float | np.ndarray,
    source_location: tuple[float, float, float],
    wind_speed: np.ndarray,
    wind_direction: np.ndarray,
    stability_class: str,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    domain_z: tuple[float, float, int],
    time_array: np.ndarray,
    release_frequency: float = 1.0,
    scheme: str = "pg",
    background_conc: float = 0.0,
    turbulence: "OUTurbulence | None" = None,
    turbulence_seed: int | np.random.Generator | None = None,
) -> xr.Dataset:
    """Simulate a time-resolved Gaussian-puff dispersion field on a 3-D grid.

    Wraps :func:`evolve_puffs` + :func:`simulate_puff_field` with grid
    construction, a diffrax-driven time-varying wind solve, and optional
    column integration. Returns an :class:`xarray.Dataset` with variables
    on a ``(time, x, y, z)`` grid.

    Parameters
    ----------
    emission_rate : float or ndarray
        Continuous emission rate Q [kg/s]. Scalar constant Q, or an array of
        length ``n_puffs`` specifying Q at each puff release time. The puff
        mass is ``Q · Δt_release`` with ``Δt_release = 1/release_frequency``.
        Entries must be ≥ 0 (scalar or array); ``Q = 0`` yields a zero field.
    source_location : tuple of float, (3,)
        ``(x, y, z)`` source coordinates [m]. ``z ≥ 0``.
    wind_speed : ndarray, shape (T_w,)
        Wind speed [m/s] at each entry in ``time_array``. Must be same length.
    wind_direction : ndarray, shape (T_w,)
        Wind direction [deg from North]. Same length as ``time_array``.
    stability_class : str
        One of ``'A'``, …, ``'F'``.
    domain_x, domain_y, domain_z : tuple of (float, float, int)
        ``(start, stop, n_points)`` for each axis.
    time_array : ndarray, shape (T,)
        Monotone increasing evaluation times [s].
    release_frequency : float
        Puff release rate [Hz]. Δt_release = 1/release_frequency. Default 1 Hz.
    scheme : str
        Dispersion scheme: ``'pg'`` (Pasquill-Gifford, default) or
        ``'briggs'`` (sharing coefficients with the steady plume model).
    background_conc : float
        Additive background concentration [kg/m³]. Default 0.
    turbulence : OUTurbulence, optional
        Ornstein-Uhlenbeck sub-grid turbulence applied as per-puff ``(Δx, Δy)``
        offsets at release time. Adds realistic meander on top of the
        wind-driven advection. Defaults to ``None`` (deterministic puffs).
    turbulence_seed : int or numpy.random.Generator, optional
        Seed for the OU sampler; required for bit-exact reproducibility
        when ``turbulence`` is set.

    Returns
    -------
    ds : xarray.Dataset
        Variables: ``concentration`` ``(time, x, y, z)`` [kg/m³],
        ``column_concentration`` ``(time, x, y)`` [kg/m²],
        ``wind_speed``/``wind_direction`` ``(time,)``.

    Raises
    ------
    ValueError
        On invalid stability class, non-positive emission rate, shape
        mismatches, unknown scheme, or malformed domain tuples.
    """
    import xarray as xr  # lazy — only this entry point requires xarray

    scheme_params_dict, dispersion_fn = get_dispersion_scheme(scheme)

    if stability_class not in scheme_params_dict:
        raise ValueError(
            f"simulate_puff: `stability_class` must be one of "
            f"{STABILITY_CLASSES} (got {stability_class!r})"
        )
    if len(source_location) != 3:
        raise ValueError(
            "simulate_puff: `source_location` must be (x, y, z); "
            f"got length {len(source_location)}"
        )
    if source_location[2] < 0.0:
        raise ValueError(
            f"simulate_puff: source height must be ≥ 0 "
            f"(got z={source_location[2]!r})"
        )
    time_array = np.asarray(time_array, dtype=np.float32)
    wind_speed = np.asarray(wind_speed, dtype=np.float32)
    wind_direction = np.asarray(wind_direction, dtype=np.float32)
    if time_array.ndim != 1 or time_array.size < 2:
        raise ValueError(
            "simulate_puff: `time_array` must be 1-D with ≥ 2 entries"
        )
    if wind_speed.shape != time_array.shape:
        raise ValueError(
            f"simulate_puff: `wind_speed` shape {wind_speed.shape} must match "
            f"`time_array` shape {time_array.shape}"
        )
    if wind_direction.shape != time_array.shape:
        raise ValueError(
            f"simulate_puff: `wind_direction` shape {wind_direction.shape} "
            f"must match `time_array` shape {time_array.shape}"
        )
    if np.any(wind_speed < 0.0):
        raise ValueError("simulate_puff: `wind_speed` entries must be ≥ 0")

    for name, domain in (
        ("domain_x", domain_x),
        ("domain_y", domain_y),
        ("domain_z", domain_z),
    ):
        if len(domain) != 3:
            raise ValueError(
                f"simulate_puff: `{name}` must be (start, stop, n_points); "
                f"got length {len(domain)}"
            )
        start, stop, n = domain
        if not (start < stop):
            raise ValueError(
                f"simulate_puff: `{name}` requires start < stop "
                f"(got start={start!r}, stop={stop!r})"
            )
        if int(n) < 2:
            raise ValueError(
                f"simulate_puff: `{name}` requires n_points ≥ 2 (got n={n!r})"
            )
    if not (release_frequency > 0.0):
        raise ValueError(
            f"simulate_puff: `release_frequency` must be > 0 "
            f"(got {release_frequency!r})"
        )

    dispersion_params = scheme_params_dict[stability_class]

    x_grid = np.linspace(domain_x[0], domain_x[1], int(domain_x[2]))
    y_grid = np.linspace(domain_y[0], domain_y[1], int(domain_y[2]))
    z_grid = np.linspace(domain_z[0], domain_z[1], int(domain_z[2]))
    n_x, n_y, n_z = len(x_grid), len(y_grid), len(z_grid)
    n_t = len(time_array)

    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    x_flat = jnp.asarray(X.ravel(), dtype=jnp.float32)
    y_flat = jnp.asarray(Y.ravel(), dtype=jnp.float32)
    z_flat = jnp.asarray(Z.ravel(), dtype=jnp.float32)

    dt_release = frequency_to_release_interval(release_frequency)
    release_times = make_release_times(
        float(time_array[0]), float(time_array[-1]), release_frequency
    )
    n_puffs = int(release_times.shape[0])

    if np.ndim(emission_rate) == 0:
        q_scalar = float(emission_rate)
        if q_scalar < 0.0:
            raise ValueError(
                "simulate_puff: scalar `emission_rate` must be ≥ 0 "
                f"(got {emission_rate!r})"
            )
        puff_mass = jnp.full((n_puffs,), q_scalar * dt_release,
                             dtype=jnp.float32)
    else:
        Q = np.asarray(emission_rate, dtype=np.float32)
        if Q.shape != (n_puffs,):
            raise ValueError(
                f"simulate_puff: array `emission_rate` must have shape "
                f"({n_puffs},) matching the number of releases in "
                f"[t_start, t_end) at release_frequency={release_frequency}, "
                f"got {Q.shape}"
            )
        if np.any(Q < 0.0):
            raise ValueError(
                "simulate_puff: `emission_rate` entries must be ≥ 0"
            )
        puff_mass = jnp.asarray(Q * dt_release, dtype=jnp.float32)

    schedule = WindSchedule.from_speed_direction(
        time_array, wind_speed, wind_direction
    )

    if turbulence is not None:
        from plume_simulation.gauss_puff.turbulence import sample_ou_offsets

        dx_np, dy_np = sample_ou_offsets(
            turbulence, np.asarray(release_times), seed=turbulence_seed,
        )
        position_disturbance = (
            jnp.asarray(dx_np, dtype=jnp.float32),
            jnp.asarray(dy_np, dtype=jnp.float32),
        )
    else:
        position_disturbance = None

    concentration = np.zeros((n_t, n_x, n_y, n_z), dtype=np.float32)
    for i_t, t_current in enumerate(time_array):
        puff_state = evolve_puffs(
            schedule,
            release_times,
            jnp.asarray(float(t_current), dtype=jnp.float32),
            source_location,
            puff_mass,
            position_disturbance=position_disturbance,
        )
        field_flat = simulate_puff_field(
            (x_flat, y_flat, z_flat),
            puff_state,
            dispersion_params,
            dispersion_fn,
        )
        concentration[i_t] = np.asarray(field_flat).reshape((n_x, n_y, n_z))

    concentration = concentration + background_conc

    dz = float(z_grid[1] - z_grid[0])
    column = concentration.sum(axis=3) * dz

    ds = xr.Dataset(
        data_vars={
            "concentration": (["time", "x", "y", "z"], concentration),
            "column_concentration": (["time", "x", "y"], column),
            "wind_speed": (["time"], np.asarray(wind_speed)),
            "wind_direction": (["time"], np.asarray(wind_direction)),
        },
        coords={
            "time": np.asarray(time_array),
            "x": x_grid,
            "y": y_grid,
            "z": z_grid,
        },
        attrs={
            "title": "Gaussian puff dispersion (JAX + diffrax)",
            "source_x": float(source_location[0]),
            "source_y": float(source_location[1]),
            "source_z": float(source_location[2]),
            "stability_class": stability_class,
            "dispersion_scheme": scheme,
            "release_frequency": release_frequency,
            "release_interval": dt_release,
            "n_puffs": n_puffs,
            "background_concentration": background_conc,
            "ou_sigma_fluctuations": (
                float(turbulence.sigma_fluctuations) if turbulence else 0.0
            ),
            "ou_correlation_time": (
                float(turbulence.correlation_time) if turbulence else 0.0
            ),
        },
    )
    ds["concentration"].attrs = {"long_name": "Mass concentration", "units": "kg/m^3"}
    ds["column_concentration"].attrs = {
        "long_name": "Column-integrated concentration",
        "units": "kg/m^2",
    }
    ds["wind_speed"].attrs = {"long_name": "Wind speed", "units": "m/s"}
    ds["wind_direction"].attrs = {
        "long_name": "Wind direction",
        "units": "degrees from North (meteorological)",
    }
    return ds
