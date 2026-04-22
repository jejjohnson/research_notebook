---
title: Gaussian processes
---

# Gaussian processes

Worked examples for Gaussian-process models — sparse variational GPs, inter-domain inducing features, deep-kernel constructions. Each sub-section pairs a short motivation with runnable notebooks.

## Sub-sections

- [SVGP with pyrox](notebooks/pyroxgp/) — stochastic variational GP regression using [`pyrox.gp`](https://github.com/jejjohnson/pyrox): the standard inducing-point variant, an inter-domain variant with spherical-harmonic inducing features (VISH), and a deep-kernel variant with random Fourier features chained behind a small MLP.

## Running

The notebooks rely on `pyrox` + `gaussx` being importable; see the root `research_notebook` environment. To execute locally:

```bash
cd projects/gaussian_processes
uv run jupyter lab notebooks/pyroxgp/01_svgp_standard.ipynb
```
