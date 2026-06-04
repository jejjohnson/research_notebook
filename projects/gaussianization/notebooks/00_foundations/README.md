---
title: "Part 0 — Foundations"
short_title: "Part 0 · Foundations"
subject: gaussianization tutorial
---

# Part 0 — Foundations

The conceptual bedrock for the whole Gaussianization curriculum. Before fitting
any flow, these seven notebooks build — from scratch in JAX, then confirmed
against the mature [`rbig`](https://github.com/jejjohnson/rbig) and
[`gauss_flows`](https://github.com/jejjohnson/gauss_flows) packages — the small
set of ideas every later method reuses: the change-of-variables formula and its
log-determinant, how those compose, the forward/inverse trade-off, why
$\mathcal{N}(0, I)$ is the target, what a *density destructor* is, the numerics
that keep one stable, and how to certify it converged.

Each notebook follows the same pattern: **derive the idea by hand, then show the
exact same quantity coming out of the library**. That way the packages stop
being black boxes — you have seen what every call computes.

## Notebooks

| # | notebook | master list | what you take away |
|---|----------|:-----------:|--------------------|
| 00 | [Change of variables](00_change_of_variables.ipynb) | 0.1 | $p_X(x)=p_Z(T(x))\lvert\det J_T\rvert$; the library's `log_det` *is* this Jacobian |
| 01 | [Composition & log-det](01_composition_logdet.ipynb) | 0.2 | stacking maps ⇒ log-dets **add**; rotations are free |
| 02 | [Forward vs. inverse](02_forward_vs_inverse.ipynb) | 0.3 | density vs. sampling cost; differentiating a root-find by the **adjoint** (`optimistix`) |
| 03 | [Why $\mathcal{N}(0,I)$?](03_why_standard_gaussian.ipynb) | 0.4 | max-entropy, separability (TC$\to0$), trivial score $-z$ |
| 04 | [Density destructors](04_density_destructors.ipynb) | 0.5–0.6 | Gaussianization = iterated whitening + nonlinearity (RBIG) |
| 05 | [Numerical mechanics](05_numerical_mechanics.ipynb) | 0.7–0.9 | jitter/clamp, float64 log-dets, the round-trip CI test |
| 06 | [Gaussianity diagnostics](06_gaussianity_diagnostics.ipynb) | 0.10–0.12 | QQ/moments, negentropy, energy-distance normality test |

The numbering and scope follow the Part 0 rows of the project
[`TUTORIAL_MASTER_LIST.md`](../../TUTORIAL_MASTER_LIST.md).

## Packages

These notebooks use two of the author's libraries as the source of truth:

- [`rbig`](https://github.com/jejjohnson/rbig) — mature NumPy/SciPy Rotation-Based
  Iterative Gaussianization with a full information-theory suite
  (`AnnealedRBIG`, `negentropy`, `total_correlation`, marginal/rotation bijectors).
- [`gauss_flows`](https://github.com/jejjohnson/gauss_flows) — the modern JAX /
  `flowjax` successor: trainable Gaussianization flows, bijectors with
  `transform_and_log_det`, and `optimistix`-based inverses.

:::{note} Found (and fixed) while writing these
Notebook 05's round-trip test surfaced a tail-inverse bug in
`gauss_flows.MixtureGaussianCDF`, filed as
[gauss_flows#108](https://github.com/jejjohnson/gauss_flows/issues/108) and fixed
in `gauss_flows` 0.1.7 — the notebook now shows it passing.
:::

## Running

The foundations notebooks need `rbig` + `gauss_flows` + a Jupyter stack, which
live outside the conda-forge pixi envs. Create a dedicated
[`uv`](https://github.com/astral-sh/uv) virtual environment (paths assume the
sibling checkouts of `rbig` and `gauss_flows`):

```bash
cd projects/gaussianization
uv venv .venv-tutorials --python 3.13
uv pip install --python .venv-tutorials/bin/python \
  -e ../../../rbig -e ../../../ml4eo/gauss_flows \
  ipykernel jupyter nbconvert jupytext matplotlib seaborn

# Re-execute the whole section end-to-end:
.venv-tutorials/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/00_foundations/0*.ipynb --ExecutePreprocessor.timeout=600
```

Notebooks are paired (`jupytext`, `py:percent`): edit the `.py`, then
`jupytext --sync` to update the `.ipynb`. All notebooks set
`jax_enable_x64` so log-determinant accumulation stays exact.
