"""Eulerian 3-D advection-diffusion of a methane tracer (Level-2 fidelity).

This sub-package implements a finite-volume solver for the passive-tracer
transport equation

    ∂C/∂t + ∇·(u C) = ∇·(K ∇C) + S(x, t)

on an Arakawa C-grid (via finitevolX), with a *prescribed* wind field
``u = (u, v, w)`` and K-theory (possibly anisotropic) eddy diffusivity
``K = diag(K_h, K_h, K_z)``.  No turbulence is resolved — this model sits
between the analytic Gaussian puff (L1) and a full resolved-flow LES (L3).

Submodules
----------
- ``grid``          : CartesianGrid3D wrapper + coordinate helpers.
- ``wind``          : prescribed 3-D wind fields (uniform, sheared, time-varying).
- ``source``        : Gaussian point-source emission term.
- ``advection``     : horizontal WENO advection (finitevolX) + vertical upwind.
- ``diffusion``     : K-theory eddy diffusivity + anisotropic Laplacian.
- ``boundary``      : BC application for 3-D fields (horizontal vmap + vertical).
- ``dynamics``      : diffrax-compatible RHS for the scalar transport equation.
- ``simulate``      : high-level xarray-returning runner.
- ``_vertical_ops`` : private — vertical ∂_z / flux / Laplacian helpers.
"""

from __future__ import annotations

from plume_simulation.les_fvm import (
    advection,
    boundary,
    diffusion,
    dynamics,
    grid,
    simulate,
    source,
    wind,
)
from plume_simulation.les_fvm.advection import advection_tendency
from plume_simulation.les_fvm.boundary import (
    HorizontalBC,
    VerticalBC,
    apply_boundary_conditions,
    build_default_concentration_bc,
)
from plume_simulation.les_fvm.diffusion import (
    EddyDiffusivity,
    diffusion_tendency,
    pg_eddy_diffusivity,
)
from plume_simulation.les_fvm.dynamics import EulerianDispersionRHS
from plume_simulation.les_fvm.grid import (
    PlumeGrid3D,
    make_grid,
)
from plume_simulation.les_fvm.simulate import simulate_eulerian_dispersion
from plume_simulation.les_fvm.source import (
    GaussianSource,
    make_gaussian_source,
)
from plume_simulation.les_fvm.wind import (
    PrescribedWindField,
    uniform_wind_field,
    wind_field_from_schedule,
)


__all__ = [
    "EddyDiffusivity",
    "EulerianDispersionRHS",
    "GaussianSource",
    "HorizontalBC",
    "PlumeGrid3D",
    "PrescribedWindField",
    "VerticalBC",
    "advection",
    "advection_tendency",
    "apply_boundary_conditions",
    "boundary",
    "build_default_concentration_bc",
    "diffusion",
    "diffusion_tendency",
    "dynamics",
    "grid",
    "make_gaussian_source",
    "make_grid",
    "pg_eddy_diffusivity",
    "simulate",
    "simulate_eulerian_dispersion",
    "source",
    "uniform_wind_field",
    "wind",
    "wind_field_from_schedule",
]
