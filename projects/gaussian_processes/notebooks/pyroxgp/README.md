---
title: SVGP with pyrox
---

# Sparse variational GPs with pyrox

Three SVGP notebooks, same scaffolding — the only thing that changes between them is **how the inducing distribution is parameterised**.

| Notebook | Inducing structure | What you take away |
|---|---|---|
| [01_svgp_standard](01_svgp_standard.ipynb) | Point-inducing `Z ∈ ℝ^(M×D)` | Canonical Titsias/Hensman SVGP — ELBO, `WhitenedGuide`, inducing-point migration. |
| [02_svgp_spherical_harmonics](02_svgp_spherical_harmonics.ipynb) | Real spherical harmonics on S² (VISH, Dutordoir 2020) | Diagonal `K_uu`, closed-form Funk–Hecke coefficients, inter-domain feature viewpoint. |
| [03_svgp_rff_nn](03_svgp_rff_nn.ipynb) | Point-inducing on NN-warped space, kernel ≈ RFF inner-product | Deep kernels, finite-dim RFF projections, when the base RBF isn't expressive enough. |

## Shared ingredients

Every notebook uses the same three pyrox primitives:

- [`pyrox.gp.SparseGPPrior`](https://github.com/jejjohnson/pyrox/blob/main/src/pyrox/gp/_sparse.py) — kernel + `Z` (or `inducing=...`) bundled with a `solver` and `jitter`.
- [`pyrox.gp.WhitenedGuide`](https://github.com/jejjohnson/pyrox/blob/main/src/pyrox/gp/_guides.py) — `q(v) = 𝒩(m, LLᵀ)` in whitened coordinates, standard Hensman-et-al-2015 parameterisation.
- [`pyrox.gp.svgp_elbo`](https://github.com/jejjohnson/pyrox/blob/main/src/pyrox/gp/_inference.py) — pure-JAX differentiable ELBO returning a scalar.

Training is pure `optax` + `equinox.filter_value_and_grad` — no NumPyro handlers in sight. Kernels are defined as plain `eqx.Module`s with `log_variance` / `log_lengthscale` as unconstrained trainable scalars.

## Reading order

Do them in sequence — each notebook assumes you've seen the previous one's ELBO + training scaffold.
