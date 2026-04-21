"""K-theory eddy diffusion for the Eulerian dispersion solver.

Under the K-theory closure the turbulent flux is modelled as a gradient
term

    u' C'  ≈  -K · ∇C,

giving a diffusion contribution ``∂_t C ⊇ ∇·(K ∇C)`` that closes the
mean-flow tracer equation.  Here ``K = diag(K_h, K_h, K_z)`` is an
anisotropic diagonal diffusivity — horizontal eddies are taken isotropic
in (x, y) while the vertical component is smaller in a stable boundary
layer.

Horizontal diffusion is assembled via :class:`finitevolx.Diffusion3D`
(one 2-D Laplacian per z-level) and vertical diffusion by the local
:func:`_vertical_ops.vertical_diffusion_tendency` helper.

The :func:`pg_eddy_diffusivity` helper converts Pasquill-Gifford σ rates
into an effective ``(K_h, K_z)`` using the standard Taylor relation

    K = (1/2) d(σ²)/dt  ≈  σ² / (2 t),

evaluated at a user-supplied reference travel distance.  This is a
rough calibration and is exposed primarily as a convenience for
cross-checks against ``gauss_puff`` at the reference distance.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import Diffusion3D
from jaxtyping import Array, Float

from plume_simulation.gauss_puff.dispersion import (
    calculate_pg_dispersion,
    get_pg_params,
)
from plume_simulation.les_fvm._vertical_ops import vertical_diffusion_tendency
from plume_simulation.les_fvm.grid import PlumeGrid3D


class EddyDiffusivity(eqx.Module):
    """Scalar or field-valued ``(K_h, K_z)`` eddy diffusivity.

    Attributes
    ----------
    horizontal : float or Float[Array, "nz ny nx"]
        Horizontal eddy diffusivity [m²/s].  A scalar is broadcast over
        the grid.  A field is expected on interior T-points.
    vertical : float or Float[Array, "nz ny nx"]
        Vertical eddy diffusivity [m²/s].  Same broadcasting rules.
    """

    horizontal: float | Float[Array, "nz ny nx"]
    vertical: float | Float[Array, "nz ny nx"]

    def as_arrays(
        self,
    ) -> tuple[
        float | Float[Array, "Nz Ny Nx"],
        float | Float[Array, "Nz Ny Nx"],
    ]:
        """Return ``(K_h, K_z)`` in forms consumable by the tendency calls."""
        return _broadcast_or_scalar(self.horizontal), _broadcast_or_scalar(
            self.vertical
        )


def _broadcast_or_scalar(
    k: float | Float[Array, "nz ny nx"],
) -> float | Float[Array, "Nz Ny Nx"]:
    """Pass scalars through, pad interior fields to the full ghosted layout."""
    k_arr = jnp.asarray(k)
    if k_arr.ndim == 0:
        return k_arr
    return jnp.pad(k_arr, ((1, 1), (1, 1), (1, 1)), mode="edge")


@eqx.filter_jit
def diffusion_tendency(
    concentration: Float[Array, "Nz Ny Nx"],
    eddy_diffusivity: EddyDiffusivity,
    plume_grid: PlumeGrid3D,
) -> Float[Array, "Nz Ny Nx"]:
    """Full 3-D eddy diffusion tendency ``∇·(K ∇C)`` at interior T-points.

    Parameters
    ----------
    concentration : Float[Array, "Nz Ny Nx"]
        Tracer at T-points.
    eddy_diffusivity : EddyDiffusivity
        ``(K_h, K_z)`` diagonal diffusivity.
    plume_grid : PlumeGrid3D

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Diffusive tendency at T-points, zero on every ghost face.
    """
    k_h, k_z = eddy_diffusivity.as_arrays()
    horizontal_op = Diffusion3D(grid=plume_grid.grid)
    horizontal = horizontal_op(concentration, k_h)
    vertical = vertical_diffusion_tendency(concentration, k_z, plume_grid.dz)
    return horizontal + vertical


def pg_eddy_diffusivity(
    stability_class: str,
    wind_speed: float,
    reference_distance: float = 200.0,
) -> EddyDiffusivity:
    """Rough K-theory calibration from Pasquill-Gifford σ curves.

    For a neutral puff travelling at ``u`` under stability class ``C``,
    the cross-wind and vertical spreads at a reference distance
    ``x_ref`` are ``σ_y(x_ref)`` and ``σ_z(x_ref)``.  Taylor's identity
    ``K = σ² / (2 t)`` with ``t = x_ref / u`` gives a simple constant
    effective diffusivity.  The result is the *average* K over
    ``[0, x_ref]``; users wanting a sharper match across a wide range
    of travel distances should pass a spatially-varying field directly.

    Parameters
    ----------
    stability_class : str
        One of ``"A"``..``"F"``.
    wind_speed : float
        Representative wind speed ``u`` [m/s].  Must be > 0.
    reference_distance : float, default 200.0
        Reference downwind distance ``x_ref`` [m] at which the σ curves
        are sampled.

    Returns
    -------
    EddyDiffusivity
        Scalar ``(K_h, K_z)`` calibrated at ``x_ref``.

    Raises
    ------
    ValueError
        If ``wind_speed`` or ``reference_distance`` is non-positive.
    """
    if wind_speed <= 0.0:
        raise ValueError(
            "pg_eddy_diffusivity: wind_speed must be > 0 "
            f"(got {wind_speed!r})"
        )
    if reference_distance <= 0.0:
        raise ValueError(
            "pg_eddy_diffusivity: reference_distance must be > 0 "
            f"(got {reference_distance!r})"
        )
    params = get_pg_params(stability_class)
    distance = jnp.asarray(reference_distance, dtype=jnp.float32)
    _, sigma_y, sigma_z = calculate_pg_dispersion(distance, params)
    t_ref = reference_distance / wind_speed
    k_h = float(sigma_y**2 / (2.0 * t_ref))
    k_z = float(sigma_z**2 / (2.0 * t_ref))
    return EddyDiffusivity(horizontal=k_h, vertical=k_z)


def make_eddy_diffusivity(
    spec: float | tuple[float, float] | str | EddyDiffusivity,
    stability_class: str | None = None,
    wind_speed: float | None = None,
    reference_distance: float = 200.0,
) -> EddyDiffusivity:
    """Coerce a user-facing diffusivity spec into an :class:`EddyDiffusivity`.

    Accepts:

    - :class:`EddyDiffusivity`                       (returned unchanged)
    - ``float``                                      (isotropic scalar K)
    - ``(K_h, K_z)`` tuple                           (anisotropic scalars)
    - ``"pg"``                                       (call :func:`pg_eddy_diffusivity`;
      requires ``stability_class`` + ``wind_speed``)
    """
    if isinstance(spec, EddyDiffusivity):
        return spec
    if isinstance(spec, str):
        if spec.lower() != "pg":
            raise ValueError(
                f"make_eddy_diffusivity: unknown string spec {spec!r}; "
                "only 'pg' is supported"
            )
        if stability_class is None or wind_speed is None:
            raise ValueError(
                "make_eddy_diffusivity: 'pg' spec requires both "
                "`stability_class` and `wind_speed`"
            )
        return pg_eddy_diffusivity(
            stability_class=stability_class,
            wind_speed=wind_speed,
            reference_distance=reference_distance,
        )
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ValueError(
                "make_eddy_diffusivity: tuple spec must be (K_h, K_z), "
                f"got length {len(spec)}"
            )
        return EddyDiffusivity(horizontal=float(spec[0]), vertical=float(spec[1]))
    k = float(spec)
    return EddyDiffusivity(horizontal=k, vertical=k)


__all__ = [
    "EddyDiffusivity",
    "diffusion_tendency",
    "make_eddy_diffusivity",
    "pg_eddy_diffusivity",
]
