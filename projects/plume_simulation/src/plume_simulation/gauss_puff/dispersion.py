"""Pasquill-Gifford dispersion coefficients for the Gaussian puff model (JAX).

The Pasquill-Gifford (PG) parameterisation expresses the lateral and vertical
dispersion coefficients of a Gaussian puff as log-quadratic functions of the
puff travel distance:

    σ_y(s) = exp(a_y + b_y · ln(s) + c_y · (ln s)²)       [m]
    σ_z(s) = exp(a_z + b_z · ln(s) + c_z · (ln s)²)       [m]

The along-wind dispersion coefficient is taken equal to the crosswind one
(``σ_x = σ_y``), reflecting isotropic horizontal spread of an instantaneous
release.

Coefficient values come from the classic Pasquill-Gifford (1961/1976) curves
re-expressed in log-quadratic form by several later authors (Turner, 1994;
Beychok, 2005). Parameters are stored per stability class A-F.

Public surface
--------------
- ``PG_DISPERSION_PARAMS`` : mapping {stability class → jnp.ndarray(6,)}
- ``STABILITY_CLASSES``     : canonical ordering ('A', ..., 'F')
- ``calculate_pg_dispersion`` : JIT-compiled σ_x, σ_y, σ_z evaluator
- ``get_pg_params``         : validated dict lookup
- ``calculate_briggs_dispersion_xyz`` : Briggs wrapper exposing σ_x = σ_y
"""

from __future__ import annotations

from jax import jit
import jax.numpy as jnp

from plume_simulation.gauss_plume.dispersion import (
    BRIGGS_DISPERSION_PARAMS,
    calculate_briggs_dispersion,
)


STABILITY_CLASSES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")


# Format: [a_y, b_y, c_y, a_z, b_z, c_z]
# Log-quadratic PG coefficients (Beychok 2005, Table 3.2).
PG_DISPERSION_PARAMS: dict[str, jnp.ndarray] = {
    "A": jnp.array([-1.104, 0.9878, -0.0076, 4.679, -1.172, 0.2770]),
    "B": jnp.array([-1.634, 1.0350, -0.0096, -1.999, 0.8752, 0.0136]),
    "C": jnp.array([-2.054, 1.0231, -0.0076, -2.341, 0.9477, -0.0020]),
    "D": jnp.array([-2.555, 1.0423, -0.0087, -3.186, 1.1737, -0.0316]),
    "E": jnp.array([-2.754, 1.0106, -0.0064, -3.783, 1.3010, -0.0450]),
    "F": jnp.array([-3.143, 1.0148, -0.0070, -4.490, 1.4024, -0.0540]),
}


def get_pg_params(stability_class: str) -> jnp.ndarray:
    """Return the Pasquill-Gifford parameter vector for a stability class.

    Parameters
    ----------
    stability_class : str
        One of ``'A'``, ``'B'``, ``'C'``, ``'D'``, ``'E'``, ``'F'``.

    Returns
    -------
    params : jnp.ndarray, shape (6,)
        ``[a_y, b_y, c_y, a_z, b_z, c_z]``.

    Raises
    ------
    ValueError
        If ``stability_class`` is not one of A-F.
    """
    if stability_class not in PG_DISPERSION_PARAMS:
        raise ValueError(
            f"get_pg_params: stability_class must be one of "
            f"{STABILITY_CLASSES}, got {stability_class!r}"
        )
    return PG_DISPERSION_PARAMS[stability_class]


@jit
def calculate_pg_dispersion(
    distance: jnp.ndarray,
    params: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate σ_x(s), σ_y(s), σ_z(s) from Pasquill-Gifford parameters.

    σ_i(s) = exp(a_i + b_i · ln(s) + c_i · (ln s)²) for i ∈ {y, z};
    σ_x := σ_y.

    Parameters
    ----------
    distance : jnp.ndarray
        Puff travel distance [m]. Non-positive values are clamped to 1 m
        so that ``ln(s)`` stays finite.
    params : jnp.ndarray, shape (6,)
        ``[a_y, b_y, c_y, a_z, b_z, c_z]``.

    Returns
    -------
    sigma_x, sigma_y : jnp.ndarray
        Horizontal dispersion coefficients [m]. ``σ_x = σ_y``.
    sigma_z : jnp.ndarray
        Vertical dispersion coefficient [m].
    """
    ay, by, cy, az, bz, cz = params
    s = jnp.maximum(distance, 1.0)
    ln_s = jnp.log(s)
    sigma_y = jnp.exp(ay + by * ln_s + cy * ln_s**2)
    sigma_z = jnp.exp(az + bz * ln_s + cz * ln_s**2)
    return sigma_y, sigma_y, sigma_z


@jit
def calculate_briggs_dispersion_xyz(
    distance: jnp.ndarray,
    params: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate σ_x(s), σ_y(s), σ_z(s) from Briggs parameters.

    Thin wrapper around the plume model's Briggs evaluator that additionally
    exposes ``σ_x = σ_y`` so this routine is signature-compatible with
    :func:`calculate_pg_dispersion`. Lets callers swap dispersion schemes
    without changing downstream code.

    Parameters
    ----------
    distance : jnp.ndarray
        Puff travel distance [m].
    params : jnp.ndarray, shape (6,)
        Briggs parameters ``[a_y, b_y, c_y, a_z, b_z, c_z]``.

    Returns
    -------
    sigma_x, sigma_y, sigma_z : jnp.ndarray
        Dispersion coefficients [m]. ``σ_x = σ_y``.
    """
    sigma_y, sigma_z = calculate_briggs_dispersion(distance, params)
    return sigma_y, sigma_y, sigma_z


# Dispatcher registry mapping scheme names to (params_dict, calculator).
# `simulate_puff` and friends use this to pick PG vs Briggs by name.
DISPERSION_SCHEMES: dict[
    str,
    tuple[dict[str, jnp.ndarray], "callable"],  # (params registry, evaluator)
] = {
    "pg": (PG_DISPERSION_PARAMS, calculate_pg_dispersion),
    "briggs": (BRIGGS_DISPERSION_PARAMS, calculate_briggs_dispersion_xyz),
}


def get_dispersion_scheme(scheme: str):
    """Return ``(params_dict, calculator_fn)`` for a named dispersion scheme.

    Parameters
    ----------
    scheme : str
        One of ``'pg'`` (Pasquill-Gifford, default) or ``'briggs'``.

    Raises
    ------
    ValueError
        If ``scheme`` is not a registered key.
    """
    if scheme not in DISPERSION_SCHEMES:
        raise ValueError(
            f"get_dispersion_scheme: `scheme` must be one of "
            f"{tuple(DISPERSION_SCHEMES)}, got {scheme!r}"
        )
    return DISPERSION_SCHEMES[scheme]
