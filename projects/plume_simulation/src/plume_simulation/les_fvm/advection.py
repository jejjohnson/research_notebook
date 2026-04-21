"""3-D flux-form advection of a passive tracer.

The horizontal contribution ``-∇_h · (u_h C)`` is delegated to
:class:`finitevolx.Advection3D`, which applies a WENO/TVD reconstruction
at U- and V-faces for every z-level.  The vertical contribution
``-∂_z(w C)`` is assembled by :func:`_vertical_ops.vertical_advection_tendency`
with a first-order upwind reconstruction at k-faces.

Available reconstruction methods for the horizontal term mirror
finitevolX's advection dispatch: ``upwind1``, ``upwind2``, ``upwind3``,
``weno3``, ``weno5``, ``weno7``, ``weno9``, ``minmod``, ``van_leer``,
``superbee``, ``mc``, ``naive``.  The vertical term is always
first-order upwind — vertical resolution is typically too coarse for a
higher-order scheme to pay for itself, and first-order upwind is
monotone by construction, which matters for a non-negative tracer.
"""

from __future__ import annotations

import equinox as eqx
from finitevolx import Advection3D
from jaxtyping import Array, Float

from plume_simulation.les_fvm._vertical_ops import vertical_advection_tendency
from plume_simulation.les_fvm.grid import PlumeGrid3D


@eqx.filter_jit
def advection_tendency(
    concentration: Float[Array, "Nz Ny Nx"],
    u: Float[Array, "Nz Ny Nx"],
    v: Float[Array, "Nz Ny Nx"],
    w: Float[Array, "Nz Ny Nx"],
    plume_grid: PlumeGrid3D,
    method: str = "weno5",
) -> Float[Array, "Nz Ny Nx"]:
    """Flux-form advective tendency ``-∇·(u C)`` at interior T-points.

    Parameters
    ----------
    concentration : Float[Array, "Nz Ny Nx"]
        Tracer at T-points.
    u : Float[Array, "Nz Ny Nx"]
        x-velocity at U-points (east faces).
    v : Float[Array, "Nz Ny Nx"]
        y-velocity at V-points (north faces).
    w : Float[Array, "Nz Ny Nx"]
        Vertical velocity collocated at T-points (see ``grid.py``).
    plume_grid : PlumeGrid3D
        Grid the fields live on.
    method : str, default ``"weno5"``
        Horizontal reconstruction scheme.  Passed through to
        :class:`finitevolx.Advection3D`.

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Advective tendency at T-points, zero on every ghost face.
    """
    horizontal_op = Advection3D(grid=plume_grid.grid)
    horizontal = horizontal_op(concentration, u, v, method=method)
    vertical = vertical_advection_tendency(concentration, w, plume_grid.dz)
    return horizontal + vertical
