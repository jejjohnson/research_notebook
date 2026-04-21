"""3-D Cartesian Arakawa C-grid wrapper for the Eulerian dispersion model.

The underlying :class:`finitevolx.CartesianGrid3D` stores fields with shape
``[Nz, Ny, Nx]`` where every axis carries a one-cell ghost ring, so the
interior indices span ``[1:-1, 1:-1, 1:-1]``.  This module wraps the
construction + coordinate generation so the rest of ``les_fvm`` can treat
the grid as a single pytree leaf with convenient ``x``, ``y``, ``z``
coordinate arrays (interior cell centres only).

Stagger convention (see :class:`finitevolx.CartesianGrid3D`)
-----------------------------------------------------------
    T[k, j, i]  cell centre  at  ( i        * dx,  j        * dy,  k * dz )
    U[k, j, i]  east face    at  ((i + 1/2) * dx,  j        * dy,  k * dz )
    V[k, j, i]  north face   at  ( i        * dx, (j + 1/2) * dy,  k * dz )

Within a C-grid stagger, velocities ``u`` and ``v`` live at U- and V-points
respectively; the scalar tracer and vertical velocity ``w`` are treated as
T-point quantities in this package.  Vertical staggering is not used; we
collocate ``w`` at T-points for shape uniformity and pay a small accuracy
cost that is dwarfed by the K-theory parameterisation anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from finitevolx import CartesianGrid3D
from jaxtyping import Array, Float


if TYPE_CHECKING:
    import numpy as np


class PlumeGrid3D(eqx.Module):
    """Arakawa C-grid + cached interior coordinate arrays.

    Attributes
    ----------
    grid : CartesianGrid3D
        Underlying finitevolX grid.  All field operators (advection,
        diffusion, interpolation, difference) take this as input.
    x : Float[Array, "nx_interior"]
        Interior T-point x-coordinates [m].
    y : Float[Array, "ny_interior"]
        Interior T-point y-coordinates [m].
    z : Float[Array, "nz_interior"]
        Interior T-point z-coordinates [m].

    Notes
    -----
    ``x``, ``y``, ``z`` are the *interior* cell-centre coordinates — they
    drop the one-cell ghost rings on either side of each axis.  The sizes
    therefore match the interior slice shapes used by the advection and
    diffusion operators, not the raw field-array shapes.
    """

    grid: CartesianGrid3D
    x: Float[Array, nx_interior]
    y: Float[Array, ny_interior]
    z: Float[Array, nz_interior]

    @property
    def dx(self) -> float:
        return float(self.grid.dx)

    @property
    def dy(self) -> float:
        return float(self.grid.dy)

    @property
    def dz(self) -> float:
        return float(self.grid.dz)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Full field shape ``(Nz, Ny, Nx)`` including ghost rings."""
        return (int(self.grid.Nz), int(self.grid.Ny), int(self.grid.Nx))

    @property
    def interior_shape(self) -> tuple[int, int, int]:
        """Interior shape ``(nz, ny, nx)`` excluding ghost rings."""
        return (int(self.z.size), int(self.y.size), int(self.x.size))


def make_grid(
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    domain_z: tuple[float, float, int],
    dtype: jnp.dtype = jnp.float32,
) -> PlumeGrid3D:
    """Build a :class:`PlumeGrid3D` from ``(x_min, x_max, nx_interior)`` triples.

    Parameters
    ----------
    domain_x : tuple (x_min, x_max, nx_interior)
        Physical x-range + number of interior cells in x.
    domain_y : tuple (y_min, y_max, ny_interior)
        Physical y-range + number of interior cells in y.
    domain_z : tuple (z_min, z_max, nz_interior)
        Physical z-range + number of interior cells in z.  ``z_min`` is
        treated as the ground level (Neumann-wall BC target below).
    dtype : jnp.dtype, default float32
        Coordinate-array dtype.

    Returns
    -------
    PlumeGrid3D

    Raises
    ------
    ValueError
        If any interior cell count is < 4 (WENO5 stencils need at least
        four interior cells in every horizontal direction) or if any
        physical extent is non-positive.
    """
    x_min, x_max, nx_int = domain_x
    y_min, y_max, ny_int = domain_y
    z_min, z_max, nz_int = domain_z

    for label, count in (("x", nx_int), ("y", ny_int), ("z", nz_int)):
        if count < 4:
            raise ValueError(
                f"make_grid: need at least 4 interior cells in {label} "
                f"for WENO5 stencils (got {count})"
            )

    Lx = float(x_max - x_min)
    Ly = float(y_max - y_min)
    Lz = float(z_max - z_min)
    for label, L in (("x", Lx), ("y", Ly), ("z", Lz)):
        if L <= 0.0:
            raise ValueError(
                f"make_grid: {label}-extent must be positive "
                f"(got x_max - x_min = {L})"
            )

    grid = CartesianGrid3D.from_interior(
        nx_interior=int(nx_int),
        ny_interior=int(ny_int),
        nz_interior=int(nz_int),
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
    )
    # Interior cell centres — offset by half a cell from the lower edge,
    # matching the finitevolX convention of co-locating T-points at
    # cell centres and drop ghost cells at i = 0 and i = Nx - 1.
    x = jnp.asarray(
        x_min + (jnp.arange(nx_int, dtype=dtype) + 0.5) * (Lx / nx_int)
    )
    y = jnp.asarray(
        y_min + (jnp.arange(ny_int, dtype=dtype) + 0.5) * (Ly / ny_int)
    )
    z = jnp.asarray(
        z_min + (jnp.arange(nz_int, dtype=dtype) + 0.5) * (Lz / nz_int)
    )
    return PlumeGrid3D(grid=grid, x=x, y=y, z=z)


def coord_arrays_from_grid(
    plume_grid: PlumeGrid3D,
) -> tuple[
    Float[Array, "nz_interior ny_interior nx_interior"],
    Float[Array, "nz_interior ny_interior nx_interior"],
    Float[Array, "nz_interior ny_interior nx_interior"],
]:
    """Return broadcast 3-D ``(X, Y, Z)`` interior-coordinate arrays."""
    X, Y, Z = jnp.meshgrid(
        plume_grid.x, plume_grid.y, plume_grid.z, indexing="xy"
    )
    # meshgrid(indexing="xy") returns (Ny, Nx, Nz) — reorder to (Nz, Ny, Nx)
    # to match field layout.
    X = jnp.transpose(X, (2, 0, 1))
    Y = jnp.transpose(Y, (2, 0, 1))
    Z = jnp.transpose(Z, (2, 0, 1))
    return X, Y, Z


def coord_to_numpy(
    plume_grid: PlumeGrid3D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert interior coordinate arrays to NumPy for xarray packaging."""
    import numpy as _np

    return (
        _np.asarray(plume_grid.x),
        _np.asarray(plume_grid.y),
        _np.asarray(plume_grid.z),
    )
