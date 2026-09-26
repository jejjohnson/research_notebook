---
title: Spatial Extremes (moved)
---

# Spatial Extremes — moved to xtremax

The spatial-extremes curriculum (Iberian station data → EVT foundations → pooling → spatial GEV → non-stationary) now lives in the [xtremax](https://github.com/jejjohnson/xtremax) docs as a tutorial series:

- [Overview](https://jejjohnson.github.io/xtremax/spatial-extremes/)
- Data: [CDS in-situ Iberia](https://jejjohnson.github.io/xtremax/cds-insitu-iberia/)
- Foundations: [block maxima](https://jejjohnson.github.io/xtremax/block-maxima/), [GEV at one station](https://jejjohnson.github.io/xtremax/gev-one-station/), [extremal types](https://jejjohnson.github.io/xtremax/extremal-types/), [return levels](https://jejjohnson.github.io/xtremax/return-levels/)
- Pooling: [independent](https://jejjohnson.github.io/xtremax/many-stations-independent/), [Laplace](https://jejjohnson.github.io/xtremax/many-stations-laplace/), [hierarchical](https://jejjohnson.github.io/xtremax/hierarchical-pooling/)
- Spatial models: [GP primer](https://jejjohnson.github.io/xtremax/gp-primer/), [GP on μ](https://jejjohnson.github.io/xtremax/spatial-gp-mu/), [GP on μ, σ](https://jejjohnson.github.io/xtremax/spatial-gp-mu-sigma/), [GP on μ, σ, ξ](https://jejjohnson.github.io/xtremax/spatial-gp-mu-sigma-xi/)
- Non-stationary: [parametric](https://jejjohnson.github.io/xtremax/nonstationary-parametric/), [neural ODE](https://jejjohnson.github.io/xtremax/nonstationary-ode/), [GP](https://jejjohnson.github.io/xtremax/nonstationary-gp/)

The helper code (`data`, `features`, `places`, `viz`) and the CDS fetch scripts sit next to the notebooks in `docs/tutorials/spatial_extremes/_helpers/`. The DVC-tracked station data was not moved; re-fetch it with the tutorial's fetch script. The separate Spain notebooks under `projects/gaussian_processes/notebooks/13_applied/spatial_extremes/` stay here.
