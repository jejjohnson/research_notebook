"""Vertical ∂z / flux / Laplacian helpers.

finitevolX's 3-D operators are staggered only in the horizontal plane —
``Advection3D``/``Diffusion3D``/``Difference3D`` treat the vertical axis as a
batch dimension.  To close the passive-tracer transport equation

    ∂C/∂t + ∇·(u C) = ∇·(K ∇C) + S,

we need the vertical flux ``∂_z(w C)`` and the vertical diffusion
``∂_z(K_z ∂_z C)``.  These tiny wrappers live here rather than inside
``advection.py`` / ``diffusion.py`` so the boundaries between "horizontal
operators provided by finitevolX" and "vertical operators provided by us"
stay visible.

Conventions
-----------
- Fields have shape ``[Nz, Ny, Nx]`` with a single-cell ghost ring at
  ``k = 0`` and ``k = Nz - 1``.  All tendencies are written only at
  interior cells ``[1:-1, :, :]``.
- Vertical velocity ``w`` is collocated at T-points (same ``[Nz, Ny, Nx]``
  layout).  Fluxes are assembled at k-faces by averaging neighbouring
  T-point values and applying a first-order upwind reconstruction on the
  tracer.
- Boundary handling is the caller's responsibility — ``w[0, :, :]`` and
  ``w[-1, :, :]`` on the ghost rings control the flux through the ground
  and domain top; this module just computes the tendency.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


@eqx.filter_jit
def vertical_advection_tendency(
    concentration: Float[Array, "Nz Ny Nx"],
    w: Float[Array, "Nz Ny Nx"],
    dz: float,
) -> Float[Array, "Nz Ny Nx"]:
    """First-order upwind ``-∂_z(w C)`` at interior T-points.

    Parameters
    ----------
    concentration : Float[Array, "Nz Ny Nx"]
        Tracer field at T-points.
    w : Float[Array, "Nz Ny Nx"]
        Vertical velocity at T-points [m/s].
    dz : float
        Vertical grid spacing [m].

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Advective tendency at T-points.  Ghost cells (``k ∈ {0, Nz-1}``)
        are zeroed.
    """
    # k-face velocity: linear average of the two adjacent T-point values.
    # w_face[k] lives at k-1/2 and has shape [Nz-1, Ny, Nx]; w_face[k]
    # corresponds to the face between T[k-1] and T[k].
    w_face = 0.5 * (w[:-1] + w[1:])
    # Upwind reconstruction: if w_face > 0, flux is C from below (T[k-1]);
    # else C from above (T[k]).
    c_upwind = jnp.where(w_face > 0.0, concentration[:-1], concentration[1:])
    flux_face = w_face * c_upwind  # shape [Nz - 1, Ny, Nx]

    # Tendency = -(F_{k+1/2} - F_{k-1/2}) / dz at interior T[k].
    tendency_interior = -(flux_face[1:] - flux_face[:-1]) / dz
    out = jnp.zeros_like(concentration)
    return out.at[1:-1, :, :].set(tendency_interior)


@eqx.filter_jit
def vertical_diffusion_tendency(
    concentration: Float[Array, "Nz Ny Nx"],
    kappa_z: float | Float[Array, "Nz Ny Nx"],
    dz: float,
) -> Float[Array, "Nz Ny Nx"]:
    """Central-difference ``∂_z(K_z ∂_z C)`` at interior T-points.

    Parameters
    ----------
    concentration : Float[Array, "Nz Ny Nx"]
        Tracer field at T-points.
    kappa_z : float or Float[Array, "Nz Ny Nx"]
        Vertical eddy diffusivity at T-points [m²/s].  When a field is
        supplied, face values are taken as the arithmetic mean of the
        two neighbouring T-point values.
    dz : float
        Vertical grid spacing [m].

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Diffusive tendency at T-points.  Ghost cells are zeroed.
    """
    # Face-centred diffusivity at k-1/2.
    if jnp.ndim(jnp.asarray(kappa_z)) == 0:
        kappa_face = jnp.asarray(kappa_z)  # broadcast scalar
    else:
        kappa_face = 0.5 * (kappa_z[:-1] + kappa_z[1:])

    # Flux at k-1/2: -K_z * (C[k] - C[k-1]) / dz
    grad_face = (concentration[1:] - concentration[:-1]) / dz
    flux_face = kappa_face * grad_face  # shape [Nz - 1, Ny, Nx]

    # Tendency = (F_{k+1/2} - F_{k-1/2}) / dz at interior T[k].
    tendency_interior = (flux_face[1:] - flux_face[:-1]) / dz
    out = jnp.zeros_like(concentration)
    return out.at[1:-1, :, :].set(tendency_interior)


def zero_horizontal_ghosts(
    field: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz Ny Nx"]:
    """Zero the four horizontal ghost faces of a 3-D field.

    Useful after in-place interior updates when the caller wants the
    ghost ring known to be zero before a BC application overwrites it.
    """
    out = field.at[:, 0, :].set(0.0)
    out = out.at[:, -1, :].set(0.0)
    out = out.at[:, :, 0].set(0.0)
    out = out.at[:, :, -1].set(0.0)
    return out


def zero_vertical_ghosts(
    field: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz Ny Nx"]:
    """Zero the top and bottom ghost slices of a 3-D field."""
    out = field.at[0, :, :].set(0.0)
    out = out.at[-1, :, :].set(0.0)
    return out
