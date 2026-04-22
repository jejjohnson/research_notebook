---
title: Kronecker GPs with non-Gaussian likelihoods
---

# Kronecker GPs with non-Gaussian likelihoods

Structured Gaussian processes on product domains (space × time, space × wavelength, etc.) paired with non-conjugate observation models. All notebooks lean on [`gaussx`](https://github.com/jejjohnson/gaussx) for Kronecker operator dispatch and non-conjugate ELBO helpers, and [`pyrox.gp`](https://github.com/jejjohnson/pyrox) for kernels and likelihood wrappers.

## Notebooks

- [01_spain_extremes](01_spain_extremes.ipynb) — annual maxima on a space × time grid over Spain, modelled with an **additive** space + time GP and a **Generalized Extreme Value** observation likelihood. Inference is a custom mean-field variational GP with closed-form KL + Gauss–Hermite ELL; the extreme-value angle gives us return-level maps for current and projected climates.

Follow-up directions (planned, not yet written):

- Multiplicative space × time interaction (`K_s ⊗ K_t`) with a Kronecker-structured variational posterior.
- Non-stationary GEV parameters (`σ(t)`, `ξ(s, t)`) via a second GP layer.
