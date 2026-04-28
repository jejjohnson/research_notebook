---
title: "SIREN vs. RFF vs. VSSGP — neural fields and Bayesian Fourier features"
---

# SIREN vs. RFF vs. VSSGP

A sub-project of [interpolation](../README.md) that takes a different angle from the pathwise-GP notes (00–04). The pathwise-GP notes build a sparse-to-dense reconstruction pipeline; this sub-project asks an orthogonal question: **for a function class trained on data and then evaluated — including outside the training footprint — which is the right design for geoscience: a SIREN, an RFF network, or a VSSGP?**

The thesis: **SIREN and RFF are special cases of VSSGP with degenerate / uninformative spectral priors**, and for geoscience data (sparse, irregular, banded spectrum) the VSSGP framing strictly dominates because the spectral structure is something we already know.

```{important}
The two notes here build on each other. 00 establishes the unification and the experiment plan; 01 maps the physical-constraint question (basis / data / loss) onto the same framework. Read 00 first.
```

## Reading order

| # | Note | Layer | What you get |
|---|---|---|---|
| 00 | [SIREN ↔ RFF ↔ VSSGP — Bayesian Fourier features for geoscience](00_siren_rff_vssgp.md) | Theory + experiment plan | Applied-math foundations (Bochner, RFF as MC, weight↔function-space duality), the four-rung hierarchy (RFF → SSGP → VSSGP → SIREN), spectral-budget argument for why geoscience is wrong for SIREN, operational-prior recipe (MIOST / DYMOST / DUACS → VSSGP), per-variable physical priors (SSH / SST / SSS / OC), additive-component construction, and a 6.X-section experiment plan with datasets, metrics, ablations, and compute budget. |
| 01 | [Enforcing physics — basis, data, loss](01_physics_constraints.md) | Constraints | Three-axis taxonomy for injecting physical knowledge: basis (function class), data (input + output preprocessing + parameterisation), loss (data fit + soft physics + learned operators). Decision matrix per constraint × variable × method. The headline reframe: **for out-of-domain prediction, the leverage ordering is Data > Basis > Loss** — soft-physics losses don't generalise outside the collocation footprint, but data-axis tricks (predict scalar potential, periodic embeddings, log-transform, climatology subtraction) enforce constraints everywhere. |

## How this differs from the pathwise notes

| Concern | Pathwise notes (00–04) | This sub-project |
|---|---|---|
| Goal | Posterior reconstruction on a dense grid | Train then predict, including OOD |
| Method family | GP with closed-form posterior + Matheron's-rule sampling | Neural fields (SIREN) and VSSGPs (Bayesian linear in RFF) |
| Scaling concern | $\mathcal{O}(N^3)$ kernel inversion → matrix-free / patch decomp | Training cost / SGD budget |
| Headline win | Posterior samples on a global grid | Informative spectral prior at fixed feature budget |
| Most interesting prediction regime | Interior of training domain (interpolation) | Outside training domain (extrapolation) |

The two angles complement each other. The pathwise notes optimise the *grid-sampling* axis given a fixed kernel. This sub-project optimises the *kernel choice* (and the function class more broadly) given a fixed training paradigm. Combining the two — VSSGP with informative `p(ω)` *plus* pathwise sampling for dense grid evaluation — is the natural full pipeline.

## Layout

```text
projects/interpolation/siren_vs_rff/
├── README.md                       # this file
├── 00_siren_rff_vssgp.md           # foundations, hierarchy, priors, experiment plan
└── 01_physics_constraints.md       # basis / data / loss axes for physical constraints
```

These notes are pure Markdown — they render directly in MyST without execution. There is no per-sub-project pixi environment; the wider repo's `pyrox` env covers the package vocabulary referenced in the pseudocode.
