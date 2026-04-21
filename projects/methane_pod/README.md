---
title: Methane PoD — thinned marked temporal point processes
---

# Methane PoD

A self-contained sub-project on **methane retrieval from satellite observations**, treated as a *thinned marked temporal point process* (TMTPP). It packages:

- A library of **13 intensity kernels** λ(t) for satellite-visible methane sources (wells, compressors, mines, landfills, wetlands, feedlots, offshore platforms), each as an `equinox.Module` with a NumPyro prior factory.
- A library of **10 probability-of-detection (POD) models** P_d(·), spanning logistic → varying-coefficient → spectral-aware → full GLM, each exposing the same `__call__` + `sample_priors` interface.
- A **Monte Carlo proof of the "missing mass paradox"**: a PoD-thinned sampler simultaneously *overestimates* the average emission rate and *underestimates* the total emitted mass. Pure-NumPy library plus a notebook driver.
- A **NumPyro NUTS fitter** for a POD-modified power-law model — invert noisy observed-flux histograms for (α, x₀, σ).

## Layout

```
projects/methane_pod/
├── pyproject.toml          # standalone package "methane_pod"
├── src/methane_pod/
│   ├── intensity.py        # λ(t) equinox modules + INTENSITY_REGISTRY
│   ├── pod_functions.py    # P_d(·) equinox modules + POD_REGISTRY
│   ├── paradox.py          # Monte Carlo simulation (pure NumPy)
│   └── fitting.py          # NumPyro NUTS fitter (pure library — no I/O)
├── tests/                  # pytest suite, ~80 fast tests + 1 MCMC smoke test
└── notebooks/              # jupytext .py + executed .ipynb + prose .md
```

## Running

The parent `research_notebook` pixi file defines a `methane-pod` feature / environment with the right pins (jax <0.9 for numpyro compatibility, equinox, numpyro, corner).

```bash
# install + open a shell in the env
pixi install -e methane-pod
pixi shell -e methane-pod

# fast library tests (no MCMC)
pixi run -e methane-pod test-methane-pod

# full suite (includes a ~11s NUTS smoke test)
pixi run -e methane-pod test-methane-pod-all

# re-execute all code notebooks in-place
pixi run -e methane-pod execute-methane-pod
```

## Notebooks

See the `notebooks/` directory and the MyST toc entry **Methane PoD** in the parent `myst.yml`. Prose-only `.md` files cover the theory; `.ipynb` files (with their `.py` jupytext source alongside) are executed end-to-end with outputs committed.

## Reading order

1. [01_mttpp_theory](notebooks/01_mttpp_theory.md) — point-process foundations
2. [02_intensity_zoo](notebooks/02_intensity_zoo.md) — the λ(t) catalog
3. [03_missing_mass_paradox](notebooks/03_missing_mass_paradox.ipynb) — MC proof
4. [04_intensity_gallery](notebooks/04_intensity_gallery.ipynb) — λ(t) code gallery
5. [05_pod_gallery](notebooks/05_pod_gallery.ipynb) — P_d(·) code gallery
6. [06_stationary_numpyro_mcmc](notebooks/06_stationary_numpyro_mcmc.ipynb) — NUTS fit
7. `07_pod_fitting_mcmc.ipynb` — placeholder; real-data CSV-driven NUTS fit (pending)
8. [08_persistency](notebooks/08_persistency.md) — operational predictions
