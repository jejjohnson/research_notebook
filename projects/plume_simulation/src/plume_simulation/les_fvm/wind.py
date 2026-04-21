"""Prescribed 3-D wind fields for the Eulerian dispersion solver.

The L2 model does **not** resolve the flow — wind is an external input.
A :class:`PrescribedWindField` is a callable pytree that maps a query
time ``t`` to the three velocity components ``(u, v, w)`` on the grid,
with ``u`` at U-points, ``v`` at V-points, and ``w`` at T-points
(the C-grid convention adopted by ``les_fvm``; see ``grid.py``).

Three constructors are provided:

- :func:`uniform_wind_field`        — constant in space *and* time.
- :func:`wind_field_from_schedule`  — spatially uniform, time-varying via
  a ``gauss_puff.wind.WindSchedule``.
- :func:`wind_field_from_callable`  — general-purpose escape hatch for
  user-supplied ``(t, X, Y, Z) -> (u, v, w)`` closures.

All fields return arrays with the same ``[Nz, Ny, Nx]`` layout as the
tracer concentration, so they can be consumed directly by the advection
operators.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from plume_simulation.gauss_puff.wind import WindSchedule
from plume_simulation.les_fvm.grid import PlumeGrid3D, coord_arrays_from_grid


class PrescribedWindField(eqx.Module):
    """Wind velocity evaluator on the full ``[Nz, Ny, Nx]`` field layout.

    The interior field values are populated by ``_interior_fn``; ghost
    rings are padded with the interior boundary values so downstream
    advection operators see a well-defined field everywhere.

    Parameters
    ----------
    plume_grid : PlumeGrid3D
        Grid the field lives on.
    interior_fn : Callable[[float], tuple[Float[Array, 'nz ny nx'], ...]]
        Pure function ``t -> (u_int, v_int, w_int)`` returning interior
        velocity components at the T/U/V stagger points (the ``les_fvm``
        convention).  See :func:`uniform_wind_field` for the simplest
        example.
    """

    plume_grid: PlumeGrid3D
    interior_fn: Callable[[Float[Array, ""]], tuple]

    def __call__(
        self, t: Float[Array, ""]
    ) -> tuple[
        Float[Array, "Nz Ny Nx"],
        Float[Array, "Nz Ny Nx"],
        Float[Array, "Nz Ny Nx"],
    ]:
        """Return ``(u, v, w)`` on the full ghost-padded grid layout."""
        u_int, v_int, w_int = self.interior_fn(t)
        u = _embed_interior(u_int, self.plume_grid.shape)
        v = _embed_interior(v_int, self.plume_grid.shape)
        w = _embed_interior(w_int, self.plume_grid.shape)
        return u, v, w


def _embed_interior(
    interior: Float[Array, "nz ny nx"], full_shape: tuple[int, int, int]
) -> Float[Array, "Nz Ny Nx"]:
    """Embed an interior-shaped field into the full ghost-padded layout.

    Ghost rings are filled by edge-padding — the interior boundary
    values are copied outward.  This is a safe default because the
    advection/diffusion operators only read from ghost cells via
    first-order stencils, and edge-padding is consistent with both
    ``Outflow1D`` and a zero-gradient ``Neumann1D`` BC on the wind.
    """
    del full_shape  # Every axis gets exactly one ghost ring in finitevolX.
    return jnp.pad(interior, ((1, 1), (1, 1), (1, 1)), mode="edge")


def uniform_wind_field(
    plume_grid: PlumeGrid3D,
    u: float = 0.0,
    v: float = 0.0,
    w: float = 0.0,
) -> PrescribedWindField:
    """Space- and time-constant wind field.

    Parameters
    ----------
    plume_grid : PlumeGrid3D
    u, v, w : float, default 0.0
        Cartesian wind velocity components [m/s].

    Returns
    -------
    PrescribedWindField
    """
    nz, ny, nx = plume_grid.interior_shape
    u_arr = jnp.full((nz, ny, nx), u, dtype=plume_grid.x.dtype)
    v_arr = jnp.full((nz, ny, nx), v, dtype=plume_grid.x.dtype)
    w_arr = jnp.full((nz, ny, nx), w, dtype=plume_grid.x.dtype)

    def interior_fn(t):
        del t
        return u_arr, v_arr, w_arr

    return PrescribedWindField(plume_grid=plume_grid, interior_fn=interior_fn)


def wind_field_from_schedule(
    plume_grid: PlumeGrid3D,
    schedule: WindSchedule,
    w: float = 0.0,
) -> PrescribedWindField:
    """Time-varying but spatially uniform wind field.

    ``u(t)`` and ``v(t)`` are interpolated from the ``WindSchedule``
    (shared with ``gauss_puff``); ``w`` remains constant in time and
    space (``0`` by default).

    Parameters
    ----------
    plume_grid : PlumeGrid3D
    schedule : WindSchedule
    w : float, default 0.0
        Constant vertical velocity [m/s].

    Returns
    -------
    PrescribedWindField
    """
    nz, ny, nx = plume_grid.interior_shape
    ones = jnp.ones((nz, ny, nx), dtype=plume_grid.x.dtype)
    w_arr = w * ones

    def interior_fn(t):
        u_t, v_t = schedule.wind_at(t)
        return u_t * ones, v_t * ones, w_arr

    return PrescribedWindField(plume_grid=plume_grid, interior_fn=interior_fn)


def wind_field_from_callable(
    plume_grid: PlumeGrid3D,
    fn: Callable[
        [
            Float[Array, ""],
            Float[Array, "nz ny nx"],
            Float[Array, "nz ny nx"],
            Float[Array, "nz ny nx"],
        ],
        tuple[
            Float[Array, "nz ny nx"],
            Float[Array, "nz ny nx"],
            Float[Array, "nz ny nx"],
        ],
    ],
) -> PrescribedWindField:
    """General-purpose wind field from a user-supplied ``(t, X, Y, Z)`` function.

    Parameters
    ----------
    plume_grid : PlumeGrid3D
    fn : callable
        Pure function ``(t, X, Y, Z) -> (u, v, w)`` where each of
        ``X, Y, Z`` is an interior-shaped coordinate array.  Each output
        must have the same interior shape.

    Returns
    -------
    PrescribedWindField
    """
    X, Y, Z = coord_arrays_from_grid(plume_grid)

    def interior_fn(t):
        return fn(t, X, Y, Z)

    return PrescribedWindField(plume_grid=plume_grid, interior_fn=interior_fn)
