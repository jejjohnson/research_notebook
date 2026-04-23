---
title: Kronecker GPs with non-Gaussian likelihoods
---

# Kronecker GPs with non-Gaussian likelihoods

Structured Gaussian processes on product domains (space × time, space × wavelength, etc.) paired with non-conjugate observation models. All notebooks lean on [`gaussx`](https://github.com/jejjohnson/gaussx) for Kronecker operator dispatch and non-conjugate ELBO helpers, and [`pyrox.gp`](https://github.com/jejjohnson/pyrox) for kernels and likelihood wrappers.

## Notebooks

- [01_spain_extremes](01_spain_extremes.ipynb) — annual maxima on a space × time grid over Spain, modelled with an **additive** space + time GP and a **Generalized Extreme Value** observation likelihood. Inference is a custom mean-field variational GP with closed-form KL + Gauss–Hermite ELL; the extreme-value angle gives us return-level maps for current and projected climates.
- [02_spain_multiplicative](02_spain_multiplicative.ipynb) — upgrades the scalar amplification factor $\beta$ to a **spatial GP** $\beta(s)$, replacing the additive Kronecker-sum prior with a `gaussx.SumKronecker` of two rank-1-in-time Kronecker products ($K_\mu \otimes J_T + K_\beta \otimes dd^\top$). Yields a spatially heterogeneous 2024 $\to$ 2050 warming map.
- [03_spain_nonstationary](03_spain_nonstationary.ipynb) — promotes the GEV scale $\sigma$ and shape $\xi$ from globals to **per-location spatial GPs**, giving four parallel latent fields ($\mu, \tilde\beta, \tilde\sigma, \tilde\xi$) with a 4-way mean-field posterior. The ELL becomes a 3-D Gauss–Hermite tensor-product quadrature — same `gaussx.GaussHermiteIntegrator` API, just a 3-D `GaussianState`. Produces a spatially varying 100yr$-$25yr return-gap map driven by the local tail $\hat\xi(s), \hat\sigma(s)$.

Follow-up directions (planned, not yet written):

- Full-rank multiplicative $K_s \otimes K_t$ with a Kronecker-structured variational posterior (lets the time response be non-linear per location instead of tied to the GMST basis).
- Temporal non-stationarity in $\sigma, \xi$ (e.g. $\log\sigma(s, t) = \log\sigma_0 + \tilde\sigma(s) + \gamma(s) \cdot d(t)$) — same SumKronecker recipe, one more spatial GP.
