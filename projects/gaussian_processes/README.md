---
title: Gaussian processes
---

# Gaussian processes

Worked examples for Gaussian-process models — sparse variational GPs, inter-domain inducing features, deep-kernel constructions, Kronecker-structured priors with non-Gaussian likelihoods. Each sub-section pairs a short motivation with runnable notebooks.

## Sub-sections

- [SVGP with pyrox](notebooks/pyroxgp/README.md) — stochastic variational GP regression using [`pyrox.gp`](https://github.com/jejjohnson/pyrox): the standard inducing-point variant, a mini-batched variant, an inter-domain variant with spherical-harmonic inducing features (VISH), and a deep-kernel variant with random Fourier features chained behind a small MLP.
- [Kronecker GPs + non-Gaussian likelihoods](notebooks/kronecker/README.md) — an additive space × time GP over Spain with a custom `GeneralizedExtremeValue` observation likelihood, using `gaussx.KroneckerSum` + `gaussx.GaussHermiteIntegrator`. Produces 25/100-year return-level maps under current and warming-projected climates.

## Running

These notebooks share the **`pyrox` pixi environment** declared in the repo-root [`pixi.toml`](../../pixi.toml). That env bundles `pyrox` (from git main), `jax`/`jaxlib` clamped to the numpyro-compatible line, `cartopy` for the Kronecker-GEV maps, and a full Jupyter stack.

```bash
# Install the env (first time only)
pixi install -e pyrox

# Open Jupyter Lab and browse to a notebook
pixi run -e pyrox jupyter lab

# Or re-execute every notebook in this project in one go
pixi run -e pyrox execute-gaussian-processes
```

Run all commands from the repo root — the `projects/gaussian_processes/` directory itself is not a standalone `uv`/`pixi` project, so `cd`-ing into it and invoking a task there will not resolve.
