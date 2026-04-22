---
title: SVGP with pyrox
---

# Sparse variational GPs with pyrox

Four SVGP notebooks, same scaffolding — each one changes exactly one thing about the setup.

| Notebook | What changes | What you take away |
|---|---|---|
| [01_svgp_standard](01_svgp_standard.ipynb) | — (baseline) | Canonical Titsias/Hensman SVGP — ELBO, `WhitenedGuide`, inducing-point migration. |
| [02_svgp_batched](02_svgp_batched.ipynb) | Full-data ELBO → unbiased mini-batch ELBO | Why you'd reach for SVGP in the first place: `(N/B) · ELL − KL` scaling, wallclock comparison, batch-size sweep. |
| [03_svgp_spherical_harmonics](03_svgp_spherical_harmonics.ipynb) | Point-inducing `Z ∈ ℝ^(M×D)` → spherical-harmonic basis on S² (VISH, Dutordoir 2020) | Diagonal `K_uu`, closed-form Funk–Hecke coefficients, inter-domain feature viewpoint. O(M) vs O(M³) solve-cost sweep. |
| [04_svgp_rff_nn](04_svgp_rff_nn.ipynb) | RBF kernel → RFF inner-product on NN-warped input | Deep kernels, finite-dim RFF projections, when the base RBF isn't expressive enough. |

## Shared ingredients

Every notebook uses the same three pyrox primitives:

- [`pyrox.gp.SparseGPPrior`](https://github.com/jejjohnson/pyrox/blob/main/src/pyrox/gp/_sparse.py) — kernel + `Z` (or `inducing=...`) bundled with a `solver` and `jitter`.
- [`pyrox.gp.WhitenedGuide`](https://github.com/jejjohnson/pyrox/blob/main/src/pyrox/gp/_guides.py) — `q(v) = 𝒩(m, LLᵀ)` in whitened coordinates, standard Hensman-et-al-2015 parameterisation.
- [`pyrox.gp.svgp_elbo`](https://github.com/jejjohnson/pyrox/blob/main/src/pyrox/gp/_inference.py) — pure-JAX differentiable ELBO returning a scalar.

Training is pure `optax` + `equinox.filter_value_and_grad` — no NumPyro handlers in sight. Kernels are defined as plain `eqx.Module`s with `log_variance` / `log_lengthscale` as unconstrained trainable scalars.

## Reading order

Do them in sequence — each notebook assumes you've seen the previous one's ELBO + training scaffold.
