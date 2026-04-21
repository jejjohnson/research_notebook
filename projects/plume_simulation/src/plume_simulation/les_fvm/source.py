"""Methane source term for the Eulerian dispersion solver.

The source enters the tracer equation as

    ∂C/∂t ⊇  S(x, t) = q(t) · ρ_s(x) / (Δx Δy Δz)

where ``q(t) [kg/s]`` is the instantaneous emission rate and ``ρ_s`` is a
unit-mass spatial profile (integrates to 1 over the interior volume).  We
use a Gaussian profile centred at the release point with a small
isotropic radius; this is the standard way to regularise a Dirac point
source on a finite grid, and the radius should be comparable to the
grid spacing.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from plume_simulation.les_fvm.grid import PlumeGrid3D, coord_arrays_from_grid


class GaussianSource(eqx.Module):
    """Gaussian point source with constant or time-varying emission rate.

    Attributes
    ----------
    emission_fn : Callable[[Float[Array, '']], Float[Array, '']]
        Pure function ``t -> q(t)`` returning the instantaneous emission
        rate [kg/s].  For a constant rate, pass a closure that discards
        ``t``.
    density : Float[Array, 'nz ny nx']
        Interior-shaped unit-mass spatial density [1/m³] centred at the
        release point.  Integrates to ``1/Δx/Δy/Δz`` in discrete grid
        units (so that ``q · density * Δx Δy Δz`` sums to ``q``).
    """

    emission_fn: Callable[[Float[Array, ""]], Float[Array, ""]]
    density: Float[Array, "nz ny nx"]

    def __call__(
        self, t: Float[Array, ""]
    ) -> Float[Array, "Nz Ny Nx"]:
        """Source tendency ``q(t) * density`` at interior T-points, zero-padded."""
        q = self.emission_fn(t)
        interior = q * self.density
        return jnp.pad(interior, ((1, 1), (1, 1), (1, 1)), mode="constant")


def make_gaussian_source(
    plume_grid: PlumeGrid3D,
    emission_rate: float | Callable[[Float[Array, ""]], Float[Array, ""]],
    source_location: tuple[float, float, float],
    source_radius: float | None = None,
) -> GaussianSource:
    """Build a :class:`GaussianSource` on ``plume_grid``.

    Parameters
    ----------
    plume_grid : PlumeGrid3D
    emission_rate : float or callable
        Either a constant rate [kg/s], or a pure function ``t -> q(t)``
        that returns an instantaneous rate as a scalar JAX array.
    source_location : tuple (x, y, z)
        Release point [m].  Must lie inside the interior domain.
    source_radius : float, optional
        Isotropic Gaussian radius [m].  Defaults to ``2 · max(dx, dy, dz)``
        — wide enough to avoid single-cell aliasing without smearing the
        plume.

    Returns
    -------
    GaussianSource

    Raises
    ------
    ValueError
        If ``emission_rate`` is a constant that is negative, or if the
        source location lies outside the interior domain.
    """
    x_src, y_src, z_src = (float(c) for c in source_location)
    x_min, x_max = float(plume_grid.x[0]), float(plume_grid.x[-1])
    y_min, y_max = float(plume_grid.y[0]), float(plume_grid.y[-1])
    z_min, z_max = float(plume_grid.z[0]), float(plume_grid.z[-1])
    if not (x_min <= x_src <= x_max
            and y_min <= y_src <= y_max
            and z_min <= z_src <= z_max):
        raise ValueError(
            "make_gaussian_source: source_location "
            f"{source_location!r} lies outside the interior domain "
            f"x∈[{x_min}, {x_max}], y∈[{y_min}, {y_max}], "
            f"z∈[{z_min}, {z_max}]"
        )

    if callable(emission_rate):
        emission_fn = emission_rate
    else:
        q_const = float(emission_rate)
        if q_const < 0.0:
            raise ValueError(
                "make_gaussian_source: constant `emission_rate` must be "
                f"≥ 0 (got {emission_rate!r})"
            )
        q_const_arr = jnp.asarray(q_const, dtype=plume_grid.x.dtype)

        def emission_fn(t):
            del t
            return q_const_arr

    if source_radius is None:
        source_radius = 2.0 * max(plume_grid.dx, plume_grid.dy, plume_grid.dz)
    else:
        source_radius = float(source_radius)
        if source_radius <= 0.0:
            raise ValueError(
                "make_gaussian_source: `source_radius` must be > 0 "
                f"(got {source_radius!r})"
            )

    X, Y, Z = coord_arrays_from_grid(plume_grid)
    r2 = (X - x_src) ** 2 + (Y - y_src) ** 2 + (Z - z_src) ** 2
    profile = jnp.exp(-r2 / (2.0 * source_radius**2))
    # Normalise so ∫ density dV = 1 over the interior (discrete sum × Δx Δy Δz).
    cell_volume = plume_grid.dx * plume_grid.dy * plume_grid.dz
    total = jnp.sum(profile) * cell_volume
    density = profile / total
    return GaussianSource(emission_fn=emission_fn, density=density)
