"""Briggs-McElroy-Pooler dispersion coefficients (JAX).

The Briggs parameterisation expresses the lateral and vertical dispersion
coefficients of a steady-state Gaussian plume as

    σ_y(x) = a_y · x · (1 + b_y · x)^{c_y}       [m]
    σ_z(x) = a_z · x · (1 + b_z · x)^{c_z}       [m]

with one parameter set per Pasquill-Gifford stability class A-F.

Public surface
--------------
- ``BRIGGS_DISPERSION_PARAMS`` : mapping {stability class → jnp.ndarray(6,)}
- ``STABILITY_CLASSES``        : canonical ordering ('A', ..., 'F')
- ``calculate_briggs_dispersion`` : JIT-compiled σ_y, σ_z evaluator
- ``get_dispersion_params``    : validated dict lookup
"""

from __future__ import annotations

from jax import jit
import jax.numpy as jnp


STABILITY_CLASSES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")


# Format: [a_y, b_y, c_y, a_z, b_z, c_z]
BRIGGS_DISPERSION_PARAMS: dict[str, jnp.ndarray] = {
    "A": jnp.array([0.22, 0.0001, -0.5, 0.20, 0.0, 0.0]),
    "B": jnp.array([0.16, 0.0001, -0.5, 0.12, 0.0, 0.0]),
    "C": jnp.array([0.11, 0.0001, -0.5, 0.08, 0.0002, -0.5]),
    "D": jnp.array([0.08, 0.0001, -0.5, 0.06, 0.0015, -0.5]),
    "E": jnp.array([0.06, 0.0001, -0.5, 0.03, 0.0003, -1.0]),
    "F": jnp.array([0.04, 0.0001, -0.5, 0.016, 0.0003, -1.0]),
}


def get_dispersion_params(stability_class: str) -> jnp.ndarray:
    """Return the Briggs parameter vector for a stability class.

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
    if stability_class not in BRIGGS_DISPERSION_PARAMS:
        raise ValueError(
            f"get_dispersion_params: stability_class must be one of "
            f"{STABILITY_CLASSES}, got {stability_class!r}"
        )
    return BRIGGS_DISPERSION_PARAMS[stability_class]


@jit
def calculate_briggs_dispersion(
    distance: jnp.ndarray,
    params: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate σ_y(x), σ_z(x) from Briggs parameters.

    σ_i(x) = a_i · x · (1 + b_i · x)^{c_i} for i ∈ {y, z}.

    Parameters
    ----------
    distance : jnp.ndarray
        Downwind distance from source [m]. Negative / zero values are clamped
        to 1 m so that the coefficient evaluation stays finite; upwind points
        are masked elsewhere in the plume model.
    params : jnp.ndarray, shape (6,)
        ``[a_y, b_y, c_y, a_z, b_z, c_z]``.

    Returns
    -------
    sigma_y : jnp.ndarray
        Lateral dispersion coefficient [m], same shape as ``distance``.
    sigma_z : jnp.ndarray
        Vertical dispersion coefficient [m], same shape as ``distance``.
    """
    ay, by, cy, az, bz, cz = params
    x = jnp.maximum(distance, 1.0)
    sigma_y = ay * x * jnp.power(1.0 + by * x, cy)
    sigma_z = az * x * jnp.power(1.0 + bz * x, cz)
    return sigma_y, sigma_z
