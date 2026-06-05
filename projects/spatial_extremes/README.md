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
location/scale/shape $(\mu, \sigma, \xi)$; 03 turns the fit into **return
levels** and return periods with posterior uncertainty.

**04–05 — Towards space.** 04 fits every station independently and maps the
parameters — the noisy result motivates pooling. 05 is a gentle Gaussian-process
primer with `pyrox`: regress a smooth field over `(lon, lat)`.

**06 — First spatial extreme-value model.** Tie the two strands together: let
the GEV location $\mu(s)$ be a spatial GP latent field while $\sigma, \xi$ stay
global, inferred with NumPyro SVI.

**07–10 — Capstones** (the advanced models, now on real data).
07 additive space + time GP; 08 multiplicative warming field $\beta(s)$;
09 fully non-stationary $\sigma(s), \xi(s)$; 10 a Gaussian copula for joint
exceedances across nearby stations.

## Running it

Notebooks use a shared loader, `spatial_extremes.data`, that serves **real CDS
data when cached** and a deterministic **synthetic** series otherwise — so the
whole curriculum runs offline, no credentials required.

To use real data, add CDS credentials (see `.env.example`) and fetch once:

```bash
pixi run -e spatial-extremes python projects/spatial_extremes/scripts/fetch_cds_insitu.py
pixi run -e spatial-extremes execute-spatial-extremes   # run all notebooks
```

Without credentials, just open any notebook — it will report that it is running
on the synthetic fallback and otherwise behave identically.
