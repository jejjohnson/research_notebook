---
title: "Part 5 — Coupling-based Gaussianization"
short_title: "Part 5 · Coupling"
subject: gaussianization tutorial
---

# Part 5 — Coupling-based Gaussianization

Coupling is the expressive engine of modern Gaussianization. A coupling layer splits
the coordinates with a mask, copies the passive half through, and transforms the
active half with a bijector whose parameters are predicted by a **conditioner**
network from the passive half. From that one move follow a triangular Jacobian (free
log-det), an analytic inverse (no network inversion), and arbitrary expressiveness
(the conditioner can be any network) — which is why coupling flows scale across
modalities. This part builds the pattern piece by piece, then revisits coupling
against the diagonal flows of Part 4, grounded in
[`gauss_flows`](https://github.com/jejjohnson/gauss_flows).

Training uses a small `optax` loop (gradient clipping + one-cycle cosine LR) on the
`gauss_flows` couplings directly.

## Notebooks

| # | notebook | master list | what you take away |
|---|----------|:-----------:|--------------------|
| 00 | [The coupling pattern](00_coupling_pattern.ipynb) | 5.1–5.3 | split/condition/transform; triangular Jacobian → free log-det; analytic inverse |
| 01 | [Bijector menu](01_bijector_menu.ipynb) | 5.4–5.7 | affine / mixture-CDF / deep-sigmoid / RQ-spline as 1-D maps; expressiveness |
| 02 | [Conditioner architectures](02_conditioner_architectures.ipynb) | 5.8–5.10, 5.16 | the conditioner is the engine; capacity sweep; `log_scale_bound` stability |
| 03 | [Mask design](03_mask_design.ipynb) | 5.18–5.19 | a fixed mask leaves half untouched; alternate so every coord gets both roles |
| 04 | [Diagonal vs coupling](04_diagonal_vs_coupling.ipynb) | 4.4 | parameter-fair: coupling is more param-efficient for non-separable structure |
| 05 | [RBIG warm-start (coupling)](05_coupling_warmstart.ipynb) | 3.8 | `fit_rbig_coupling`; the **zero-kernel contract** |
| 06 | [Coupling ↔ diagonal equivalence](06_coupling_equivalence.ipynb) | 5.20 | zero-kernel coupling *is* the diagonal flow; training breaks it |
| 07 | [Depth, residual coupling & stability](07_depth_residual_stability.ipynb) | 5.21–5.22 | depth → expressiveness; gradient norm vs depth; residual coupling preview |

## The headline: the conditioner

The bijector is a triangular wrapper that makes the log-det free; the **conditioner**
is where the modelling power lives (notebook 02). Every structured-data part revisits
this slot and plugs in a modality-appropriate network — CNN for images (Part 12),
RNN/Transformer for sequences (Part 11), GNN/equivariant for graphs and symmetric
domains (Parts 12–13). Coupling generalises because only the conditioner changes.

## Threads — and the two notebooks from Part 4

Notebooks **04 (diagonal vs coupling)** and **05 (coupling warm-start)** were drafted
in Part 4 and live here, where they belong: they compare coupling against the
diagonal flows of [Part 4](../04_parametric_flows/00_nll_training.ipynb) and lead
into the formal **coupling ↔ diagonal equivalence** (06). That equivalence closes a
loop from Part 4: **RBIG = diagonal Gaussianization flow = zero-kernel coupling
flow** — one function, three parameterisations, with training the only thing that
switches the conditioner on.

## Running

Same `uv` environment as the earlier parts (`rbig` + `gauss_flows` + `optax` + a
Jupyter stack):

```bash
cd projects/gaussianization
.venv-tutorials/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/05_coupling/0*.ipynb --ExecutePreprocessor.timeout=600
```

Notebooks are paired (`jupytext`, `py:percent`) and set `jax_enable_x64`. Most run in
under a minute; the bijector / conditioner sweeps (01, 02) take a couple of minutes.
