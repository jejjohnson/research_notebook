"""diffrax-compatible RHS for the Eulerian dispersion ODE system.

The full tendency ``∂_t C = RHS(t, C)`` is assembled by composing the
advection, diffusion, source, and boundary-application modules from this
sub-package.  Wrapping the RHS in an :class:`equinox.Module` keeps it a
single pytree leaf that diffrax can JIT as ``term = ODETerm(rhs)``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from plume_simulation.les_fvm.advection import advection_tendency
from plume_simulation.les_fvm.boundary import (
    HorizontalBC,
    VerticalBC,
    apply_boundary_conditions,
)
from plume_simulation.les_fvm.diffusion import EddyDiffusivity, diffusion_tendency
from plume_simulation.les_fvm.grid import PlumeGrid3D
from plume_simulation.les_fvm.source import GaussianSource
from plume_simulation.les_fvm.wind import PrescribedWindField


class EulerianDispersionRHS(eqx.Module):
    """Full tracer-transport tendency for diffrax.

    Parameters
    ----------
    plume_grid : PlumeGrid3D
    wind_field : PrescribedWindField
        Time-queryable prescribed wind.
    eddy_diffusivity : EddyDiffusivity
        ``(K_h, K_z)`` eddy diffusivity.
    source : GaussianSource
        Methane source term.
    horizontal_bc : HorizontalBC
    vertical_bc : VerticalBC
    advection_scheme : str, default ``"weno5"``
        Horizontal reconstruction scheme for
        :func:`advection.advection_tendency`.
    """

    plume_grid: PlumeGrid3D
    wind_field: PrescribedWindField
    eddy_diffusivity: EddyDiffusivity
    source: GaussianSource
    horizontal_bc: HorizontalBC
    vertical_bc: VerticalBC
    advection_scheme: str = eqx.field(static=True, default="weno5")

    def __call__(
        self,
        t: Float[Array, ""],
        concentration: Float[Array, "Nz Ny Nx"],
        args: object = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Return ``dC/dt`` at time ``t`` for tracer field ``concentration``."""
        del args
        # Enforce BCs before reading neighbours in the advection/diffusion
        # stencils so ghost cells reflect the current physical BC state.
        c_bc = apply_boundary_conditions(
            concentration,
            horizontal_bc=self.horizontal_bc,
            vertical_bc=self.vertical_bc,
            plume_grid=self.plume_grid,
        )
        u, v, w = self.wind_field(t)
        adv = advection_tendency(
            c_bc, u, v, w, self.plume_grid, method=self.advection_scheme
        )
        diff = diffusion_tendency(c_bc, self.eddy_diffusivity, self.plume_grid)
        src = self.source(t)
        # Keep ghost-cell entries of the tendency zero; time integration writes
        # only to the interior and the BC pass above has already updated the
        # ghost ring of the state.
        rhs = adv + diff + src
        return jnp.where(_interior_mask(rhs.shape, rhs.dtype), rhs, 0.0)


def _interior_mask(shape: tuple[int, int, int], dtype) -> Float[Array, "Nz Ny Nx"]:
    """Indicator field that is 1 on interior cells and 0 on ghost cells."""
    Nz, Ny, Nx = shape
    mask = jnp.zeros(shape, dtype=dtype)
    mask = mask.at[1:-1, 1:-1, 1:-1].set(1.0)
    return mask
