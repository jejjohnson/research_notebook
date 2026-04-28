---
title: Interpolation — sparse-to-dense reconstruction of geophysical fields
---

# Interpolation

Worked examples for spatial and spatiotemporal interpolation of geophysical fields from sparse, irregular observations — anchored on **sea-surface-height (SSH) reconstruction from satellite altimetry** as the running case study. The current entry point is a **Gaussian-process pathwise sampler** built on [`gaussx`](https://github.com/jejjohnson/gaussx) and [`pyrox`](https://github.com/jejjohnson/pyrox), but the notes also place that recipe in context against the wider menu of methods used in operational SSH mapping (DUACS, MIOST, DYMOST, BFN-QG, 4DVarNet) so the trade-offs are explicit.

The notes are deliberately **derivation-first**: math, complexity, and references before any code. The goal is that someone reading them — including future-me — can pick the right tool for the job without re-deriving the screening property of local GP inversions or re-discovering that MIOST wavelets are just sparse RFF features in disguise.

```{important}
Five short notes, in increasing order of structural sophistication. Each one builds directly on the previous; jumping in at 03 without reading 00–01 will not work.
```

## Reading order

| # | Note | Layer | What you get |
|---|---|---|---|
| 00 | [GP pathwise sampling for SSH](notebooks/00_ssh_pathwise_sampling.md) | Math | Matheron's-rule conditioning by sampling, derived from scratch for a Mediterranean SSH problem. Storage + compute analysis. |
| 01 | [Efficient machinery from gaussx + pyrox](notebooks/01_efficient_machinery.md) | Compute | Matrix-free kernels, preconditioned CG, Woodbury, LOVE variance, the `PathwiseSampler` drop-in. Wall-clock budgets for MedSea / N. Atlantic / global, 1–6 months. |
| 02 | [Physics-aware SSH reconstruction](notebooks/02_physics_aware_ssh.md) | Prior | Two axes — modify the data $y$ (preprocessing, derivative obs from drifters), modify the kernel $K$ (SQG spectral prior, MIOST multi-scale Gabor wavelets). |
| 03 | [Scaling to the globe with patch decomposition](notebooks/03_global_scaling_patches.md) | Geometry | Local-GP recipe at global scale via `xrpatcher` + `BallTree` + Gaspari–Cohn overlap-add + dask. ~800× speed-up over monolithic global GP. |
| 04 | [Variational SSH reconstruction with dynamical priors](notebooks/04_variational_dynamical_priors.md) | Outside framework | 3D-Var / 4D-Var / DYMOST / BFN-QG / 4DVarNet — methods that replace the GP prior with QG dynamics or a learned regulariser. Comparison targets, not extensions. |

## Highlights at a glance

```{note}
**00 — pathwise GP**: $\mathcal{O}(m^3 + nm)$ naïvely; $4\,\text{GB}$ peak memory just to materialise $K_{\mathcal{X}^*\mathcal{X}}$.
```

```{tip}
**01 — wall-clock budget** with the matrix-free machinery on an A100, full SWOT-era data:

| Region | 1 month | 3 months | 6 months |
|---|---|---|---|
| Mediterranean | 3 s (RFF+) / 2 min (exact) | 9 s / 6 min | 18 s / 12 min |
| North Atlantic | 20 s (RFF+) / 10 h (exact) | 1 min / 30 h | 2 min / 60 h |
| Global ocean | 2.5 min (RFF+) / **infeasible** (exact) | 7.5 min | 15 min |

Patch decomposition (03) brings global *exact* GP down to 12 s/day on 8× A100.
```

```{seealso}
- [`gaussx`](https://github.com/jejjohnson/gaussx) — structured linear algebra primitives the 00/01 pipeline is built on
- [`pyrox`](https://github.com/jejjohnson/pyrox) — pathwise sampler + RFF feature primitives
- [`xrpatcher`](https://github.com/jejjohnson/xrpatcher) — patch-extraction layer used in 03
- The plume-simulation project's [3D-Var note](../plume_simulation/notebooks/assimilation/00_3dvar_derivation.md) — companion derivation in the methane-retrieval setting; equations are identical
```

## Map of methods

The five notes cover three different framings of the same SSH-mapping problem:

```text
                       ┌─────────────────────────────┐
                       │   Sparse altimeter obs y    │
                       │   on a global ocean grid    │
                       └──────────────┬──────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │  GP pathwise     │    │  Variational DA  │    │  Learned variant │
    │  (00, 01, 02, 03)│    │  3DVar / 4DVar   │    │  4DVarNet  (04)  │
    │                  │    │  DYMOST / BFN-QG │    │                  │
    │  static prior K  │    │  dynamic prior   │    │  learned prior   │
    │  closed-form     │    │  via QG model    │    │  Φ_θ autoencoder │
    │  posterior + MC  │    │       (04)       │    │  + ConvLSTM solv │
    └──────────────────┘    └──────────────────┘    └──────────────────┘
```

00–03 stay inside the static-prior GP world: the prior $K$ is a kernel, the inverse problem is linear, and the gaussx/pyrox stack does the work. 04 is a separate framework — a dynamical model replaces $K$, and the SSH community's ~50 years of variational data-assimilation work takes over. The two stacks meet at the boundary: see 04 §8 for two practical hybrid recipes (GP residual after a dynamical first guess; GP warm-start for 4DVarNet).

## Sub-project — neural fields, RFF, and VSSGPs

The five notes above all live inside the **pathwise-GP-for-dense-grids** framing. A separate sub-project, [siren_vs_rff/](siren_vs_rff/README.md), takes the orthogonal angle: *for a function class trained on data and then evaluated, including outside the training footprint, which design wins for geoscience — SIREN, RFF, or VSSGP?*

```{seealso}
- [siren_vs_rff/00_siren_rff_vssgp](siren_vs_rff/00_siren_rff_vssgp.md) — SIREN and RFF as special cases of VSSGP with degenerate priors; the four-rung hierarchy; spectral-budget argument for why geoscience is the wrong domain for SIREN; per-variable physical priors (SSH / SST / SSS / OC); experiment plan with datasets, metrics, ablations.
- [siren_vs_rff/01_physics_constraints](siren_vs_rff/01_physics_constraints.md) — the three-axis constraint taxonomy (basis / data / loss) for both methods, with the headline that for out-of-domain prediction the leverage ordering is **Data > Basis > Loss**.
```

The pathwise notes (00–04) and the SIREN/VSSGP sub-project complement each other: pathwise optimises the *grid-sampling* axis given a fixed kernel; siren_vs_rff optimises the *kernel choice* given a fixed training paradigm.

## Layout

```text
projects/interpolation/
├── README.md                                       # this file
├── notebooks/
│   ├── 00_ssh_pathwise_sampling.md                 # math derivation
│   ├── 01_efficient_machinery.md                   # gaussx + pyrox primitives + wall-clock
│   ├── 02_physics_aware_ssh.md                     # data + kernel knobs (DATA / KERNEL axes)
│   ├── 03_global_scaling_patches.md                # patch decomp via xrpatcher + dask
│   └── 04_variational_dynamical_priors.md          # 3DVar / 4DVar / DYMOST / BFN-QG / 4DVarNet
└── siren_vs_rff/
    ├── README.md                                   # sub-project overview
    ├── 00_siren_rff_vssgp.md                       # SIREN ↔ RFF ↔ VSSGP unification + experiment plan
    └── 01_physics_constraints.md                   # basis / data / loss axes for OOD prediction
```

No `src/` package yet — these are derivations and pseudocode, not a runnable port. A future iteration of the project would add a `plume_simulation`-style `src/interpolation/` package implementing the patch-decomposition pipeline from 03 against real CMEMS data.

## Running

These notes are pure Markdown — they render directly in MyST without execution. There is no per-project pixi environment; the wider repo's `pyrox` env covers the package vocabulary referenced in the pseudocode.

```bash
# View locally as part of the MyST book
make docs-serve

# Or open any single note in VS Code with the MyST extension for inline math
code projects/interpolation/notebooks/00_ssh_pathwise_sampling.md
```

Run all commands from the repo root.
