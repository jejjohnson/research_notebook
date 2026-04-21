---
title: Plume simulation — atmospheric dispersion forward models
---

# Plume simulation

A self-contained sub-project of atmospheric plume dispersion forward models, built on JAX for JIT / vmap / autodiff and NumPyro for Bayesian inference. Designed to grow over time — each dispersion model lives in its own sub-package so that new ports can be added as siblings.

## Sub-packages

- [`gauss_plume`](src/plume_simulation/gauss_plume/) — **steady-state Gaussian plume** with Briggs-McElroy-Pooler dispersion coefficients (stability classes A-F), ground reflection, wind-frame coordinate rotation, and a NumPyro model for Bayesian emission-rate inference (with an optional joint categorical latent over the stability class). The [derivation note](notebooks/00_gaussian_plume_derivation.md) walks from the advection-diffusion PDE to the closed-form plume expression, following Stockie (2011).

Future additions: Gaussian puff (time-dependent), spectral LES post-processing, finite-volume LES, look-up-table beams-law retrievals.

## Layout

```
projects/plume_simulation/
├── pyproject.toml          # standalone package "plume_simulation"
├── src/plume_simulation/
│   └── gauss_plume/
│       ├── dispersion.py   # Briggs σ_y, σ_z + stability-class registry
│       ├── plume.py        # rotation + JIT forward model + xarray wrapper
│       └── inference.py    # NumPyro NUTS (lazy — imported on first access)
└── tests/
    └── gauss_plume/        # pytest suite
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
