---
title: "POD fitting on real satellite data — placeholder"
---

# POD fitting on real satellite data

This slot is reserved for the **real-data** companion to [06_stationary_numpyro_mcmc](06_stationary_numpyro_mcmc.ipynb). The original `james_jax.py` script reads the IMEO validated-plumes CSV + the Tanager plumes CSV, computes the $Q/U$ proxy per satellite, and runs a per-satellite NUTS fit using the same `pod_powerlaw_model` that we fit on synthetic data in notebook 06.

The library code for this is already in place: see `methane_pod.fitting.run_mcmc`, `pod_powerlaw_model`, `lognorm_cdf`, `power_law`. What is missing is the CSV ingestion layer (satellite-specific filters, Q/U computation, outlier handling) and the source data itself.

When the datasets land, the notebook will:

1. Load IMEO + Tanager plume tables.
2. Filter to Oil & Gas, apply per-satellite outlier thresholds.
3. Fit the POD-modified power law independently per satellite.
4. Overlay recovered POD curves with ±1σ bands on a single axis for a head-to-head platform comparison.
5. Compare the fitted $x_{50}$ and $\sigma$ to published values (Varon 2018, Cusworth 2021, Kamdar IMEO).
