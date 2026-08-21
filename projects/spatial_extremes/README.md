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
| Data | [`xrtoolz-reader`](https://github.com/jejjohnson/xr_toolz/tree/main/packages/xrtoolz-reader) | pull + cache CDS in-situ land stations over Iberia (import name: `xrreader`) |
| Extremes | [`xtremax`](https://github.com/jejjohnson/xtremax) | block-maxima extraction, GEV distribution, return levels |
| Gaussian processes | [`pyrox-gp`](https://github.com/jejjohnson/pyrox/tree/main/packages/pyrox-gp) | kernels, latent GP fields, variational inference |
| Dynamics | [`diffrax`](https://github.com/patrick-kidger/diffrax) | ODE/SDE integration for the time-varying (NB10–12) trends |

## The build-up

Each notebook is short and adds exactly one idea.

**00 — Data.** Pull daily near-surface air temperature for Iberian land
stations from CDS with `xrreader`, cache it, and look at it.

**01–03 — Extreme-value foundations (one station).**
01 turns a daily series into annual maxima (`xtremax.extraction`); 02 fits a
Generalized Extreme Value (GEV) distribution to one station and interprets
location/scale/shape $(\mu, \sigma, \xi)$; 03 covers the extremal-types theorem
and turns the fit into **return levels** with posterior uncertainty.

**04–06 — Pooling and Gaussian processes.** 04 fits every station independently
(`04` with NUTS, `04b` with a fast Laplace approximation) and maps the
parameters — the noisy result motivates pooling. 05 pools them with a
**hierarchical** Bayesian model. 06 is a Gaussian-process primer with `pyrox-gp`:
interpolate a field over `(lon, lat)`, then add physical features (elevation,
distance-to-coast, slope) and use ARD to see which actually matter.

**07–09 — Spatial GEV models.** Tie the strands together — the GEV parameters
become latent GP fields, inferred with NumPyro. 07 makes the **location**
$\mu(s)$ spatial; 08 adds a spatial **scale** $\sigma(s)$ driven by an elevation
covariate; 09 frees the **shape** $\xi(s)$ too, and asks honestly whether the
tail carries any recoverable geography.

**10–12 — Non-stationary in *time* (one long station).** Switch axes: take the
single longest record (Albacete, 1901–2025) and let the GEV location drift as the
climate warms, three escalating ways. 10 fits a **parametric** linear trend
$\mu(t)=\mu_0+\mu_1 z(t)$ (Coles' model) and turns it into time-varying return
levels. 11 replaces the line with a **mechanistic ODE** — a forced energy-balance
relaxation integrated with `diffrax` inside NUTS. 12 goes nonparametric with a
**state-space Gaussian process** (a local-linear-trend / integrated random walk,
the stochastic sibling of the ODE), shows why a free stationary GP over-fits a
short record, and puts all three trends on one set of axes.

## Running it

Notebooks use a shared loader, `spatial_extremes.data`, that serves **real CDS
data when cached** and a deterministic **synthetic** series otherwise — so the
whole curriculum runs offline, no credentials required.

Set up the project environment with `uv` (from `projects/spatial_extremes/`):

```bash
uv sync --extra notebooks      # build .venv with the full stack, notebook tooling + MyST
.venv/bin/python -m ipykernel install --user --name spatial-extremes
.venv/bin/myst build --html    # execute the notebooks and render the static site
```

To use real data, accept the CDS licence, add credentials (see `.env.example`),
and fetch once into the cache the notebooks read:

```bash
.venv/bin/python scripts/fetch_cds_insitu.py   # download the real CDS record
.venv/bin/python scripts/build_features.py     # derive nb 06 covariates (needs the cache)
```

Without credentials, just open any notebook — it will report that it is running
on the synthetic fallback and otherwise behave identically.

## Warm starts across notebooks

Several notebooks fit the same quantities more than once: the Laplace
approximation of 04b re-derives what 04 samples, 12 compares against the ODE 11
already fitted, and every spatial model estimates a location for *every*
station — Albacete included, long before 10–12 study it alone.

`spatial_extremes.results` lets a notebook publish a fit and a later one start
from it, the Bayesian analogue of a pretrained checkpoint:

| Artifact | Published by | Used by |
|---|---|---|
| `gp_primer_hypers` | 06 | 07, 08, 09 — the fitted Matérn lengthscale, replacing a hard-coded literal |
| `laplace_no_pool` | 04b | 04 — seeds NUTS, which then needs a much shorter warmup |
| `spatial_gp_mu` | 07 | 08, 09 (whitened field + scalars), 12 (Albacete's location, scale, shape) |
| `ode_gev_albacete` | 11 | 12 — loads the posterior instead of refitting the ODE |

Warm starts are **initialisation only**: they move where a sampler or optimiser
begins, never what it targets, so the posterior is unchanged. (Feeding a fitted
result back as a *prior* would be different — and wrong here, since it would
count the same maxima twice.)

The cache lives beside the data cache (`<cache_root>/results/`, gitignored) and
is entirely optional. Every consumer falls back to a cold start when an artifact
is missing, so a fresh clone runs each notebook standalone and offline; running
the curriculum in order simply populates the cache and makes the later notebooks
faster and better-anchored.
