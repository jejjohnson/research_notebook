---
title: Plume simulation — atmospheric dispersion forward models
---

# Plume simulation

A self-contained sub-project of atmospheric plume dispersion forward models, built on JAX for JIT / vmap / autodiff and NumPyro for Bayesian inference. Designed to grow over time — each dispersion model lives in its own sub-package so that new ports can be added as siblings.

## Sub-packages

- [`gauss_plume`](src/plume_simulation/gauss_plume/) — **steady-state Gaussian plume** with Briggs-McElroy-Pooler dispersion coefficients (stability classes A-F), ground reflection, wind-frame coordinate rotation, and a NumPyro model for Bayesian emission-rate inference (with an optional joint categorical latent over the stability class). The [derivation note](notebooks/00_gaussian_plume_derivation.md) walks from the advection-diffusion PDE to the closed-form plume expression, following Stockie (2011).
- [`gauss_puff`](src/plume_simulation/gauss_puff/) — **time-resolved Gaussian puff** with Pasquill-Gifford dispersion (and Briggs as an opt-in alternative), ground reflection, `diffrax`-driven time-varying wind integration, and NumPyro models for both constant-Q and random-walk Q_i (state-space) inference. Sharing stability-class lookups and priors with the plume port keeps the two ports consistent. The [derivation note](notebooks/gauss_puff/00_gaussian_puff_derivation.md) derives the puff via Galilean transformation and covers its Stockie-§3.5.6 plume-superposition relation.
- [`les_fvm`](src/plume_simulation/les_fvm/) — **Eulerian 3-D advection-diffusion (L2 fidelity)** on an Arakawa C-grid via [`finitevolX`](https://github.com/jejjohnson/finitevolX), with a prescribed wind field (uniform, time-varying via `WindSchedule`, or user-supplied callable), K-theory eddy diffusivity (scalar, anisotropic `(K_h, K_z)`, or PG-calibrated), flux-form WENO5 horizontal advection + first-order upwind vertical flux, finite-volume diffusion with anisotropic K, and per-face BCs (Dirichlet inlet, outflow outlet, periodic lateral, Neumann ground/top). Time integration is `diffrax`-compatible. Useful when the Gaussian puff's spatial-uniformity in wind is too restrictive. See the [derivation note](notebooks/les_fvm/00_eulerian_dispersion_derivation.md) for the math and numerical scheme, and [03_puff_vs_eulerian.ipynb](notebooks/les_fvm/03_puff_vs_eulerian.ipynb) for a cross-check against `gauss_puff`.

Future additions: full resolved-flow LES (L3) with Smagorinsky SGS, look-up-table Beer's-law retrievals.

## Layout

```
projects/plume_simulation/
├── pyproject.toml          # standalone package "plume_simulation"
├── src/plume_simulation/
│   ├── gauss_plume/
│   │   ├── dispersion.py   # Briggs σ_y, σ_z + stability-class registry
│   │   ├── plume.py        # rotation + JIT forward model + xarray wrapper
│   │   └── inference.py    # NumPyro NUTS (lazy — imported on first access)
│   ├── gauss_puff/
│   │   ├── dispersion.py   # Pasquill-Gifford σ + shared Briggs dispatcher
│   │   ├── wind.py         # WindSchedule + diffrax cumulative-integral solve
│   │   ├── puff.py         # puff kernel + evolve_puffs + simulate_puff (xarray)
│   │   └── inference.py    # NumPyro Q (constant) and Q_i (random walk)
│   └── les_fvm/
│       ├── grid.py          # Arakawa C-grid wrapper + coordinate helpers
│       ├── wind.py          # prescribed 3-D wind fields
│       ├── source.py        # Gaussian point-source emission
│       ├── advection.py     # WENO5 horizontal + upwind vertical flux form
│       ├── diffusion.py     # K-theory anisotropic (K_h, K_z) diffusion
│       ├── boundary.py      # 3-D BC composition (horizontal vmap + vertical)
│       ├── dynamics.py      # diffrax-compatible RHS
│       └── simulate.py      # xarray-returning runner
└── tests/
    ├── gauss_plume/
    ├── gauss_puff/
    └── les_fvm/
```

## Running

The parent `research_notebook` pixi file defines a `plume-simulation` feature / environment with the right pins (jax <0.9 for numpyro compatibility, numpyro, xarray).

```bash
# install + shell into the env
pixi install -e plume-simulation
pixi shell -e plume-simulation

# fast tests (no MCMC)
pixi run -e plume-simulation test-plume-simulation

# full suite (includes one short NUTS smoke test)
pixi run -e plume-simulation test-plume-simulation-all
```

## Public API (gauss_plume)

```python
from plume_simulation.gauss_plume import (
    # dispersion
    BRIGGS_DISPERSION_PARAMS, STABILITY_CLASSES,
    calculate_briggs_dispersion, get_dispersion_params,
    # forward model
    rotate_to_wind_frame, plume_concentration, plume_concentration_vmap,
    simulate_plume,
    # Bayesian inference (lazy import on first access)
    gaussian_plume_model, infer_emission_rate,
)
```

## Public API (gauss_puff)

```python
from plume_simulation.gauss_puff import (
    # dispersion
    PG_DISPERSION_PARAMS, STABILITY_CLASSES, DISPERSION_SCHEMES,
    calculate_pg_dispersion, calculate_briggs_dispersion_xyz,
    get_pg_params, get_dispersion_scheme,
    # wind (diffrax-backed)
    WindSchedule, cumulative_wind_integrals,
    # forward model
    PuffState, puff_concentration, puff_concentration_vmap,
    evolve_puffs, simulate_puff_field, simulate_puff,
    make_release_times, release_interval_to_frequency, frequency_to_release_interval,
    # Bayesian inference (lazy)
    gaussian_puff_model, gaussian_puff_rw_model,
    infer_emission_rate, infer_emission_timeseries,
)
```

## Public API (les_fvm)

```python
from plume_simulation.les_fvm import (
    # grid + coordinates
    PlumeGrid3D, make_grid,
    # wind (prescribed, not solved)
    PrescribedWindField,
    uniform_wind_field,            # constant in space + time
    wind_field_from_schedule,      # time-varying via WindSchedule
    wind_field_from_callable,      # user-supplied (t, X, Y, Z) -> (u, v, w)
    # source
    GaussianSource, make_gaussian_source,
    # physics
    EddyDiffusivity, pg_eddy_diffusivity,
    advection_tendency, diffusion_tendency,
    # BCs
    HorizontalBC, VerticalBC,
    apply_boundary_conditions, build_default_concentration_bc,
    # dynamics + high-level runner
    EulerianDispersionRHS,
    simulate_eulerian_dispersion,
)
```
