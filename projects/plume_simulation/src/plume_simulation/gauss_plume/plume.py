"""Steady-state Gaussian plume forward model (JAX).

Implements the analytical solution to the steady-state advection-diffusion
equation with a ground-reflection boundary condition, in wind-aligned
coordinates (x', y', z):

    C(x', y', z) = (Q / (2π u σ_y σ_z))
                   · exp(-y'² / (2 σ_y²))
                   · [exp(-(z - H)² / (2 σ_z²)) + exp(-(z + H)² / (2 σ_z²))]

where Q is the emission rate [kg/s], u = √(u² + v²) the wind speed [m/s],
σ_y(x'), σ_z(x') the Briggs dispersion coefficients, and H the source height.

Public surface
--------------
- ``rotate_to_wind_frame``       : JIT-compiled (x, y) → (x', y') rotation
- ``plume_concentration``        : JIT-compiled forward model
- ``plume_concentration_vmap``   : vmapped over receptor locations
- ``simulate_plume``              : xarray wrapper building a 3-D field
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax import jit, vmap
import jax.numpy as jnp
import numpy as np

from plume_simulation.gauss_plume.dispersion import (
    calculate_briggs_dispersion,
    get_dispersion_params,
)


if TYPE_CHECKING:  # xarray is imported lazily inside simulate_plume
    import xarray as xr


# ── Coordinate transform ─────────────────────────────────────────────────────


@jit
def rotate_to_wind_frame(
    x: jnp.ndarray,
    y: jnp.ndarray,
    source_x: float,
    source_y: float,
    wind_u: float,
    wind_v: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rotate fixed-frame coordinates into the wind-aligned frame.

    The wind-aligned frame has its origin at the source, its x'-axis pointing
    downwind, and its y'-axis pointing crosswind (90° counter-clockwise from
    x', following the right-hand rule). The rotation angle is
    ``θ = atan2(wind_v, wind_u)`` [rad].

    Parameters
    ----------
    x, y : jnp.ndarray
        Receptor coordinates in the fixed frame [m]. Must have matching
        shapes (or be broadcast-compatible).
    source_x, source_y : float
        Source location in the fixed frame [m].
    wind_u, wind_v : float
        Wind velocity components [m/s].

    Returns
    -------
    x_wind : jnp.ndarray
        Downwind coordinate x' [m] (> 0 downwind, < 0 upwind of source).
    y_wind : jnp.ndarray
        Crosswind coordinate y' [m].
    """
    dx = x - source_x
    dy = y - source_y

    theta = jnp.arctan2(wind_v, wind_u)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    x_wind = dx * cos_theta + dy * sin_theta
    y_wind = -dx * sin_theta + dy * cos_theta
    return x_wind, y_wind


# ── Forward model ────────────────────────────────────────────────────────────


# Minimum wind speed (below which the plume model is not physically valid — a
# puff model should be used instead). The model clamps to this value rather
# than raising so that upstream Bayesian inference remains differentiable
# when wind draws dip into a calm regime.
MIN_WIND_SPEED: float = 0.5  # m/s


@jit
def plume_concentration(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    source_x: float,
    source_y: float,
    source_z: float,
    wind_u: float,
    wind_v: float,
    emission_rate: float,
    dispersion_params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the steady-state Gaussian plume concentration.

    Computes the mass concentration [kg/m³] at receptor points ``(x, y, z)``
    in the fixed reference frame. Coordinates are rotated into the
    wind-aligned frame internally, upwind points are masked to zero, and
    ground reflection is applied.

    Parameters
    ----------
    x, y, z : jnp.ndarray
        Receptor coordinates in the fixed frame [m]. Broadcast-compatible.
    source_x, source_y, source_z : float
        Source location [m]. ``source_z`` is the emission height above ground.
    wind_u, wind_v : float
        Wind velocity components [m/s]. The total wind speed is clamped
        below to :data:`MIN_WIND_SPEED` for numerical stability.
    emission_rate : float
        Continuous emission rate Q [kg/s].
    dispersion_params : jnp.ndarray, shape (6,)
        ``[a_y, b_y, c_y, a_z, b_z, c_z]`` (Briggs parameters).

    Returns
    -------
    concentration : jnp.ndarray
        Mass concentration [kg/m³], same shape as the broadcast of the
        input coordinates. Zero at upwind locations (x' ≤ 0).
    """
    wind_speed = jnp.sqrt(wind_u**2 + wind_v**2)
    wind_speed = jnp.maximum(wind_speed, MIN_WIND_SPEED)

    x_downwind, y_crosswind = rotate_to_wind_frame(
        x, y, source_x, source_y, wind_u, wind_v
    )

    x_positive = jnp.maximum(x_downwind, 1.0)
    sigma_y, sigma_z = calculate_briggs_dispersion(x_positive, dispersion_params)

    normalization = 2.0 * jnp.pi * wind_speed * sigma_y * sigma_z

    exp_y = jnp.exp(-0.5 * jnp.square(y_crosswind / sigma_y))

    z_diff = z - source_z
    z_sum = z + source_z
    exp_z_direct = jnp.exp(-0.5 * jnp.square(z_diff / sigma_z))
    exp_z_reflected = jnp.exp(-0.5 * jnp.square(z_sum / sigma_z))

    concentration = (
        (emission_rate / normalization) * exp_y * (exp_z_direct + exp_z_reflected)
    )
    return jnp.where(x_downwind > 0.0, concentration, 0.0)


plume_concentration_vmap = vmap(
    plume_concentration,
    in_axes=(0, 0, 0, None, None, None, None, None, None, None),
    out_axes=0,
)
"""Vectorised over receptor locations — x, y, z batched along axis 0."""


# ── Xarray wrapper ───────────────────────────────────────────────────────────


def simulate_plume(
    emission_rate: float,
    source_location: tuple[float, float, float],
    wind_speed: float,
    wind_direction: float,
    stability_class: str,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    domain_z: tuple[float, float, int],
    background_conc: float = 0.0,
) -> xr.Dataset:
    """Simulate a steady-state Gaussian plume on a 3-D grid.

    Wraps :func:`plume_concentration` with grid construction, wind-from-angle
    to (u, v) conversion, and column integration, returning an
    :class:`xarray.Dataset` suitable for plotting / I/O.

    Wind-direction convention follows the meteorological "from" convention:
    ``wind_direction = 270°`` is a wind *from* the west (flowing east);
    ``wind_direction = 0°`` is a wind *from* the north (flowing south).

        u = -wind_speed · sin(θ),     v = -wind_speed · cos(θ),
            with θ = wind_direction in radians.

    Parameters
    ----------
    emission_rate : float
        Continuous emission rate Q [kg/s]. Must be strictly positive.
    source_location : tuple of float, (3,)
        (x, y, z) source coordinates [m]. ``z ≥ 0``.
    wind_speed : float
        Wind speed magnitude [m/s]. Must be strictly positive.
    wind_direction : float
        Wind direction [degrees from North, meteorological "from" convention].
    stability_class : str
        One of ``'A'``, ..., ``'F'``.
    domain_x, domain_y, domain_z : tuple of (float, float, int)
        ``(start, stop, n_points)`` for each axis. ``start < stop`` and
        ``n_points ≥ 2``.
    background_conc : float
        Additive background concentration [kg/m³]. Default ``0.0``.

    Returns
    -------
    ds : xarray.Dataset
        Variables: ``concentration`` (x, y, z) [kg/m³],
        ``column_concentration`` (x, y) [kg/m²].

    Raises
    ------
    ValueError
        On invalid stability class, non-positive emission rate, non-positive
        wind speed, malformed domain tuples, or negative source height.
    """
    import xarray as xr  # lazy — only this entry point requires xarray

    if not (emission_rate > 0.0):
        raise ValueError(
            f"simulate_plume: `emission_rate` must be > 0 (got {emission_rate!r})"
        )
    if not (wind_speed > 0.0):
        raise ValueError(
            f"simulate_plume: `wind_speed` must be > 0 (got {wind_speed!r})"
        )
    if len(source_location) != 3:
        raise ValueError(
            "simulate_plume: `source_location` must be (x, y, z); "
            f"got length {len(source_location)}"
        )
    if source_location[2] < 0.0:
        raise ValueError(
            f"simulate_plume: source height must be ≥ 0 "
            f"(got z={source_location[2]!r})"
        )
    for name, domain in (
        ("domain_x", domain_x),
        ("domain_y", domain_y),
        ("domain_z", domain_z),
    ):
        if len(domain) != 3:
            raise ValueError(
                f"simulate_plume: `{name}` must be (start, stop, n_points); "
                f"got length {len(domain)}"
            )
        start, stop, n = domain
        if not (start < stop):
            raise ValueError(
                f"simulate_plume: `{name}` requires start < stop "
                f"(got start={start!r}, stop={stop!r})"
            )
        if int(n) < 2:
            raise ValueError(
                f"simulate_plume: `{name}` requires n_points ≥ 2 (got n={n!r})"
            )

    dispersion_params = get_dispersion_params(stability_class)

    x = np.linspace(domain_x[0], domain_x[1], int(domain_x[2]))
    y = np.linspace(domain_y[0], domain_y[1], int(domain_y[2]))
    z = np.linspace(domain_z[0], domain_z[1], int(domain_z[2]))
    n_x, n_y, n_z = len(x), len(y), len(z)

    theta_rad = np.deg2rad(wind_direction)
    wind_u = -wind_speed * np.sin(theta_rad)
    wind_v = -wind_speed * np.cos(theta_rad)

    src_x, src_y, src_z = source_location

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    conc_flat = plume_concentration(
        jnp.array(X.ravel()),
        jnp.array(Y.ravel()),
        jnp.array(Z.ravel()),
        src_x,
        src_y,
        src_z,
        float(wind_u),
        float(wind_v),
        emission_rate,
        dispersion_params,
    )
    concentration = np.asarray(conc_flat).reshape((n_x, n_y, n_z))
    concentration = concentration + background_conc

    dz = float(z[1] - z[0])
    column = concentration.sum(axis=2) * dz

    ds = xr.Dataset(
        data_vars={
            "concentration": (["x", "y", "z"], concentration),
            "column_concentration": (["x", "y"], column),
        },
        coords={"x": x, "y": y, "z": z},
        attrs={
            "title": "Steady-state Gaussian plume (JAX)",
            "emission_rate": emission_rate,
            "emission_rate_units": "kg/s",
            "source_x": src_x,
            "source_y": src_y,
            "source_z": src_z,
            "wind_speed": wind_speed,
            "wind_speed_units": "m/s",
            "wind_direction": wind_direction,
            "wind_direction_units": "degrees from North (meteorological)",
            "wind_u": float(wind_u),
            "wind_v": float(wind_v),
            "stability_class": stability_class,
            "background_concentration": background_conc,
        },
    )
    ds["concentration"].attrs = {"long_name": "Mass concentration", "units": "kg/m^3"}
    ds["column_concentration"].attrs = {
        "long_name": "Column-integrated concentration",
        "units": "kg/m^2",
    }
    return ds
