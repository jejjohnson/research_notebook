---
title: Spatial Extremes
short_title: Spatial Extremes
authors:
  - name: Juan Emmanuel Johnson
date: 2026-06-05
---

# Spatial Extremes

A step-by-step curriculum on modelling **climate extremes in space**: how often
will a temperature this high recur, and how does that risk vary across a region?
We build up from a single station to a full spatial model, one concept per
notebook, on **real station data** from the Copernicus Climate Data Store (CDS).

Three packages do the heavy lifting, one per layer:

| Layer | Package | Role |
|-------|---------|------|
| Data | [`xrtoolz`](https://github.com/jejjohnson/xrtoolz) | pull + cache CDS in-situ land stations over Iberia |
| Extremes | [`xtremax`](https://github.com/jejjohnson/xtremax) | block-maxima extraction, GEV distribution, return levels |
| Gaussian processes | [`pyrox`](https://github.com/jejjohnson/pyrox) | kernels, latent GP fields, variational inference |

## The build-up

Each notebook is short and adds exactly one idea.

**00 — Data.** Pull daily near-surface air temperature for Iberian land
stations from CDS with `xrtoolz`, cache it, and look at it.

**01–03 — Extreme-value foundations (one station).**
01 turns a daily series into annual maxima (`xtremax.extraction`); 02 fits a
Generalized Extreme Value (GEV) distribution to one station and interprets
location/scale/shape $(\mu, \sigma, \xi)$; 03 covers the extremal-types theorem
and turns the fit into **return levels** with posterior uncertainty.

**04–06 — Pooling and Gaussian processes.** 04 fits every station independently
(`04` with NUTS, `04b` with a fast Laplace approximation) and maps the
parameters — the noisy result motivates pooling. 05 pools them with a
**hierarchical** Bayesian model. 06 is a Gaussian-process primer with `pyrox`:
interpolate a field over `(lon, lat)`, then add physical features (elevation,
distance-to-coast, slope) and use ARD to see which actually matter.

**07–09 — Spatial GEV models.** Tie the strands together — the GEV parameters
become latent GP fields, inferred with NumPyro. 07 makes the **location**
$\mu(s)$ spatial; 08 adds a spatial **scale** $\sigma(s)$ driven by an elevation
covariate; 09 frees the **shape** $\xi(s)$ too, and asks honestly whether the
tail carries any recoverable geography.

## Running it

Notebooks use a shared loader, `spatial_extremes.data`, that serves **real CDS
data when cached** and a deterministic **synthetic** series otherwise — so the
whole curriculum runs offline, no credentials required.

Set up the project environment with `uv` (from `projects/spatial_extremes/`):

```bash
uv sync --extra notebooks      # build .venv with the full stack + notebook tooling
.venv/bin/python -m ipykernel install --user --name spatial-extremes
myst build --html              # execute the notebooks and render the static site
```

To use real data, accept the CDS licence, add credentials (see `.env.example`),
and fetch once into the cache the notebooks read:

```bash
.venv/bin/python scripts/fetch_cds_insitu.py   # download the real CDS record
.venv/bin/python scripts/build_features.py     # derive nb 06 covariates (needs the cache)
```

Without credentials, just open any notebook — it will report that it is running
on the synthetic fallback and otherwise behave identically.
