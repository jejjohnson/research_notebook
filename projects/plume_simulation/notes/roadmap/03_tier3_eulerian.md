# Tier III — Eulerian finite-volume transport

**Forward model:** full PDE solved on a grid. Uses [`finitevolX`](https://github.com/jejjohnson/finitevolX) for spatial discretisation and [`spectraldiffx`](https://github.com/jejjohnson/spectraldiffx) for spectral / periodic-domain operators.

This is the **gold-standard physics tier**: full mass conservation, arbitrary wind fields, support for chemistry and deposition. Also the most expensive — emulators (Step 3) are essential, not optional.

---

## (1) Simple model

### Conservation equation

Operational implementations carry **mixing ratio** `c` (kg/kg), not mass density, because that's what the satellite observes and what WRF couples to. The conservation form is then:

```
∂_t (ρ c) + ∇·(ρ u c) = ∇·(ρ K ∇c) + S(x,t) − λ ρ c
```

with `ρ(x,t)` the dry-air density from the [met field](00_prerequisites.md#metfield-schema). When `ρ` is approximately constant over the integration window (typical at fixed altitude in a regional basin) this reduces to the textbook `∂_t c + ∇·(u c) = ∇·(K∇c) + S − λc`. Commit to the full ρ-weighted form for any work crossing significant pressure-altitude variation.

- `u(x,t)` — wind from WRF / ERA5 / HRRR.
- `K(x,t)` — eddy diffusivity. Commit to **K-theory from MO similarity** in the PBL (`K_z ≈ κ u* z (1 − z/L)` with stability correction) plus **Smagorinsky** in the free troposphere. Source: [MO prereqs](00_prerequisites.md#monin-obukhov-similarity) + WRF TKE.
- `S(x,t)` — source field, **the unknown we invert for**. Per-cell, time-resolved.
- `λ` — first-order chemical loss. Negligible for CH₄ at <1-day scales; included for portability to CO/HCHO.

### Initial and boundary conditions

Both are **load-bearing** for inversion accuracy and currently underspecified in operational code:

- **Initial conditions.** Long inversion windows (>24 h) need realistic ICs; otherwise spin-up corrupts early observations. Default: a 48-hour spin-up from a CAMS / GEOS-Chem global background, or jointly invert `c(0)` alongside `S` (adds `c_b(0)` background term to the cost — see [Inference](#2-model-based-inference)).
- **Lateral BCs.** Inflow from a coarser global model (CAMS, GEOS-Chem) is uncertain. Lateral-BC error is a known systematic in regional inversions; standard fix is to jointly invert per-face BC scaling factors `α_face ~ N(1, 0.1²)` and absorb the bias into the source posterior covariance. Document on the `boundary.py` API.

### Spatial / temporal discretisation

Cell-centred FV with flux-limited advection via `finitevolX`; explicit RK or IMEX for stiff diffusion; `spectraldiffx` for periodic / global domains.

### Build order within Tier III

1. **1D column model.** Validates numerics — diffusion against analytical Gaussian, advection against cosine pulse / Burgers'.
2. **2D horizontal slab.** What most satellite work needs (vertical column already integrated by AK).
3. **Full 3D with vertical layers.** Needed when injection altitude matters or when assimilating multi-layer in-situ data alongside columns.

---

## (2) Model-based inference

### Forward observation operator

Same column + AK pipeline as Tiers I and II:

```
y_t = AK_t · column_z(c(S, c₀, t)) + c_bg + ε_t
```

`H_t := AK_t · column_z(·)` is the observation operator at time `t`. The 4D-Var cost below uses `H_t`; the column + AK implementation is shared with [Tier I](01_tier1_gaussian.md#from-cxyz-to-a-satellite-comparable-observation) and [the prereqs](00_prerequisites.md#averaging-kernel-operator).

### 4D-Var cost — three terms

```
J(S, c₀) = ½‖S − S_b‖²_B
         + ½‖c₀ − c_b(0)‖²_{B_c}
         + ½ Σ_t ‖H_t c(S, c₀, t) − y_t‖²_{R_t}
```

Three terms — source background, IC background, and **time-summed** observation mismatch. The IC term drops out only if you commit to a long warm-up that pins `c(0)`. Treating observations as a single instantaneous mismatch (no time index) is incompatible with multi-hour assimilation windows.

### Likelihood model

```
ε_t ~ N(0, R_t),   R_t = R_retr,t + R_repr,t
```

- `R_retr,t` — heteroscedastic per-pixel from the L2 retrieval-error map at time `t`.
- `R_repr,t` — representation error (model-vs-observation footprint mismatch). Diagonal addition; rises with terrain complexity and at coarse-instrument boundaries. **Don't omit** — naive `R = R_retr` overweights observations and produces overconfident posteriors.
- Block-diagonal across overpasses; cross-time correlation only within met decorrelation scale.

### Prior on `S` — spatially correlated, sign-constrained

Same structure as Tier II:

| Choice | Form | Notes |
|--------|------|-------|
| Mean `S_b` | from [emission inventory](00_prerequisites.md#background-emission-inventory-q_a) | EDGAR / GFEI / EPA per-cell median |
| Covariance `B` | Matern-3/2, ℓ ∈ [5, 50] km | spatial regulariser; ℓ tuneable or hierarchical |
| Positivity | `log S ~ N(log S_b, B_log)` (lognormal) | non-negative emissions; conjugate when linearised |
| BC scaling | per-face `α ~ N(1, 0.1²)` | absorbs lateral-BC bias |

Diagonal `B` is wrong — produces wildly noisy spatial posteriors. Always carry spatial correlation.

### Adjoint

The adjoint of the transport equation (backward in time, conservative form) is:

```
−∂_t λ − ∇·(u λ) − ∇·(K ∇λ) + λ_chem · λ = ∂L/∂c
```

The `∇·(u λ)` term is **conservative** (matches the forward `∇·(u c)`); the doc previously had `u·∇λ`, which is equivalent only for divergence-free `u` and is incorrect for compressible WRF winds. JAX computes the discretised adjoint exactly via reverse-mode autodiff through the FV solver — no hand-derived adjoint code, no separate adjoint-correctness derivation. The mathematical form above is for *reading*, not implementation.

### Incremental 4D-Var (the operational default)

Linearise around the current iterate `S^k`, solve the linear inner minimisation, update outer iterate:

```
S^{k+1} = S^k + δS,    δS = argmin_δ J_lin(δS; S^k)
```

`J_lin` uses the **tangent linear** of the FV solver, trivially built via `jax.linearize`. Inner solves are quadratic in `δS` → CG or Lanczos via [`gaussx`](https://github.com/jejjohnson/gaussx). Cuts cost by 1–2 orders of magnitude vs. fully-nonlinear 4D-Var. **This is the default**; full nonlinear is a sanity check.

### Control-variable transform

Direct optimisation in `S`-space with `B^{-1}` is infeasible — `B` for a 200×200 grid is 40000² ≈ 10⁹ entries. Standard fix:

```
χ = B^{-1/2}(S − S_b),   optimise J(S(χ)) in χ-space (prior = identity Gaussian)
```

`B^{-1/2}` materialised via Matern factorisation in [`gaussx`](https://github.com/jejjohnson/gaussx) (Kronecker structure for separable correlation). This is the load-bearing trick that makes 4D-Var tractable; should be explicit in `assimilation/control.py` and on the API.

### Posterior covariance — three options

4D-Var alone gives MAP; downstream Tier V needs the **posterior**:

- **Gauss-Newton Hessian inversion.** `P* = (H^T R^{-1} H + B^{-1})^{-1}` evaluated at MAP. Tractable for moderate grids via `gaussx` Krylov.
- **Laplace around MAP.** Sample from `N(S*, P*)`; cheapest path.
- **En4D-Var.** Ensemble around MAP gives sample covariance; couples to [`filterax`](https://github.com/jejjohnson/filterax). Best when the posterior is non-Gaussian.

The posterior export to Tier V.A is via the same adapter pattern as Tier I/II — see the [posterior-export module](#module-layout).

### Cost / performance

For a 200×200 grid, 24-hour assimilation window, with incremental 4D-Var: ~minutes per outer iteration on CPU, seconds on GPU. Inner CG iteration is `O(n_iter × n_steps × n_grid)`.

---

## (3) Model emulator

Full 3D FV transport is expensive: `O(N³)` state, repeated time integration. Emulator is **essential** here, not optional like at Tier I.

### State variable — commit to 2D column for v1

Most satellite inversion needs only the column-integrated XCH₄, not the 3D field. Default emulator state is **2D column** `c_col(x, y, t)`; full 3D is a v2 escalation when multi-layer in-situ assimilation enters scope. Order-of-magnitude difference in training cost.

### Architecture options

- **CNN UNet.** Honest baseline for non-stationary terrain. No translation-invariance assumption.
- **Graph-network operator.** Right structural fit for unstructured analysis grids and basin-scoped problems.
- **Fourier Neural Operator (FNO).** Resolution-agnostic and natural fit with `spectraldiffx` philosophy — **but** assumes translation invariance in the kernel. Works well for periodic / homogeneous problems; breaks for urban basins with terrain. Use only when translation invariance is plausible.
- **Neural ODE.** `c(t+Δt) = f_θ(c(t), u(t))` iterated. Simple but suffers from long-horizon drift.
- **Reduced-order model (ROM).** POD on simulation snapshots → low-rank basis; learn projected dynamics via Galerkin or DMD. Great when dynamics live on a low-dim manifold.

### Training data

- **Active-learning over uniform climatology binning.** Sample WRF / ERA5 climatology adaptively — the emulator's residual error map drives where to run the next FV simulation. Reaches operational accuracy with 100–300 runs vs. ~1000 for uniform sampling.
- Sample met conditions from the **operational distribution** (facility locations of interest, overpass times) rather than a uniform-bin climatology — same critique as Tier II's emulator.

### Emulator adjoint must be calibrated

Backprop through a trained emulator gives *some* gradient — whether it matches the true PDE adjoint is empirical. **Validation requirement:** emulator-autodiff gradients vs. FV-autodiff gradients on a held-out set should agree to `<5%` relative error in operator norm. If not, the inversion built on Step 4 is biased even when forward predictions look fine. Add as a hard test in the [validation strategy](#validation-strategy).

---

## (4) Emulator-based inference

Replace the FV integrator with the FNO / neural ODE in the 4D-Var loop:

- Gradient via autodiff through the emulator (not the PDE solver). Both are `jax.grad`-able; only the cost-per-iteration changes.
- Orders of magnitude faster per iteration (typically 100–1000×).
- **Adjoint validation:** emulator gradient ≈ FV adjoint (Step 3 calibration test).
- **Posterior validation:** posterior from emulator-based 4D-Var ≈ posterior from adjoint 4D-Var on the same observations. If they don't agree, the emulator is biased — diagnose before trusting downstream.

---

## (5) Amortized inference (predictor)

```
f_θ: (y_{t_1, ..., t_n}, u_{1:T}, instrument_id) → p(log S(x,t) | y, u)
```

### Output grid commitment

`S(x,t)` is on the inversion grid (~1–10 km, basin-scoped). Predictor outputs a fixed-size 2D field per basin tile per time slice. **Per-instrument predictor heads**, dispatched by `instrument_id` — same pattern as Tier II.

### Irregular temporal sampling

Satellite passes are intermittent (1–3 per day per instrument over a basin) with gaps. Input is **not regularly sampled** — `t_1, …, t_n` are observation times, not a fixed grid. Architecture: **set-transformer with time-encoding** over irregular passes, fused with a regular-grid encoder over `u_{1:T}` from the met field. Vanilla seq2seq / ConvLSTM assumes regular sampling and will silently underperform.

### Posterior over the spatial source field

Conditional flow over images vs. score-based diffusion — same trade-off as Tier II:

- `gauss_flows` is currently 1D-only; 2D coupling layers are a multi-month extension.
- **Score-based diffusion** is the safer v1 path for the spatial posterior.
- Context conditioning via FiLM / hypernet primitives in [`pyrox.nn`](https://github.com/jejjohnson/pyrox) — same pattern as Tier I/II.

---

## (6) Improve

- **Multi-species coupling.** Add CO and CO₂ tracers; their source ratios constrain CH₄ source attribution (e.g. fossil vs. agricultural).
- **Adaptive grid refinement.** Refine near sources, coarsen elsewhere. `finitevolX` may need primitives for this.
- **Online DA.** Assimilate observations as they arrive (rolling-window 4D-Var, or filterax's ensemble Kalman smoother).
- **Sub-grid plume reconstruction.** When a source is sub-grid, the FV solver smears it; pair Tier III with a Tier I plume in the near-field for better point-source representation.
- **Hierarchical Matern length-scale.** Promote ℓ to a hyperparameter with its own posterior — let the data choose the regularisation scale (mirrors Tier II).

---

## Module layout

| Step | Concern | Module | Status |
|------|---------|--------|--------|
| 1 | FV grid | [`les_fvm/grid.py`](../../src/plume_simulation/les_fvm/grid.py) | ✓ |
| 1 | Advection | [`les_fvm/advection.py`](../../src/plume_simulation/les_fvm/advection.py) | ✓ |
| 1 | Diffusion | [`les_fvm/diffusion.py`](../../src/plume_simulation/les_fvm/diffusion.py) | ✓ |
| 1 | Source injection | [`les_fvm/source.py`](../../src/plume_simulation/les_fvm/source.py) | ✓ |
| 1 | Boundary conditions | [`les_fvm/boundary.py`](../../src/plume_simulation/les_fvm/boundary.py) | ✓ |
| 1 | Time integration | [`les_fvm/dynamics.py`](../../src/plume_simulation/les_fvm/dynamics.py), [`simulate.py`](../../src/plume_simulation/les_fvm/simulate.py) | ✓ |
| 1 | Eddy diffusivity (MO + Smagorinsky) | `plume_simulation.les_fvm.diffusivity` | ☐ |
| 1 | Column + AK pipeline | reuse `gauss_plume.observation` from Tier I | ☐ |
| 2 | Cost function (3-term) | [`assimilation/cost.py`](../../src/plume_simulation/assimilation/cost.py) | 🚧 |
| 2 | Likelihoods + spatial priors | `plume_simulation.assimilation.likelihoods` | ☐ |
| 2 | Control vector + transform | [`assimilation/control.py`](../../src/plume_simulation/assimilation/control.py) | 🚧 |
| 2 | Incremental 4D-Var solver | [`assimilation/solve.py`](../../src/plume_simulation/assimilation/solve.py) | 🚧 |
| 2 | Background (`S_b`, `c_b`, BC scaling) | [`assimilation/background.py`](../../src/plume_simulation/assimilation/background.py) | 🚧 |
| 2 | Diagnostics | [`assimilation/diagnostics.py`](../../src/plume_simulation/assimilation/diagnostics.py) | 🚧 |
| 2 | Posterior covariance (Hessian / Laplace / En4D-Var) | `plume_simulation.assimilation.posterior` | ☐ |
| 2 | Posterior export → Tier V | `plume_simulation.assimilation.posterior_export` | ☐ |
| 3 | Emulator (UNet / GNN / FNO) | `plume_simulation.les_fvm.emulator` | ☐ |
| 3 | Emulator-adjoint calibration harness | `plume_simulation.les_fvm.emulator_adjoint_test` | ☐ |
| 5 | Sequence predictor (set-transformer) | `plume_simulation.les_fvm.predictor` | ☐ |
| 6 | Multi-species coupling | `plume_simulation.les_fvm.multispecies` | ☐ |

---

## Validation strategy

- **1D diffusion.** Initial Dirac → at time `t`, solution is Gaussian with variance `2Kt`. Check L2 error vs. analytical.
- **1D advection.** Cosine pulse advected at constant `u` → after one period, recover initial condition. Tests upwind / flux-limiter consistency.
- **CFL stability.** Artificially exceed CFL → expect blow-up; stay within → expect stability. Catches silent flux-limiter regressions.
- **Mass conservation — three regimes.**
  - No sources, periodic BCs: `∫c dV` exact to floating-point.
  - With sources, no deposition: `Δ ∫c dV = ∫∫ S dV dt` summed over the window.
  - With deposition: include `−λ ∫c dV dt` term.
- **Adjoint correctness.** JAX `vjp` of the discrete forward should satisfy `<F u, v> == <u, F^T v>` for random `u, v`. Cheap, catches differentiation bugs.
- **Tier I limit.** For a single point source in a uniform wind with constant `K`, the steady-state FV solution should match the Gaussian-plume formula at downwind distances much greater than the grid spacing.
- **Emulator-adjoint calibration.** Emulator-autodiff gradients vs. FV-autodiff gradients on a held-out met set, `<5%` relative error in operator norm. **Hard test** — failure means Step 4 inversion is biased.
- **Emulator OOD generalization.** Train on one met regime (e.g. summer Permian), evaluate on another (winter Permian, or a different basin). Resists overfit-to-training-distribution.
- **Real-data benchmark.** Compare 4D-Var output to existing CAMS / GEOS-Chem-Adjoint inversions on a published time window (e.g. Maasakkers et al. Permian). Posterior credible interval should overlap the published estimate. Without this, the inversion is a synthetic exercise.

---

## Open questions

- **Choice of advection scheme.** Upwind is robust but diffusive; WENO is sharp but stencil-heavy. Default for `les_fvm` is currently flux-limited; document the choice and the cell-Peclet floor where it stops being mass-conservative.
- **Adjoint memory / checkpointing.** Long windows mean the forward state must be re-derived (recompute) or stored (memory) for backprop. Standard fix: Griewank-style binomial checkpointing. Open: pick the checkpointing strategy and benchmark against `equinox.internal.scan_checkpointed` or hand-rolled.
- **Lateral BC scaling — fit per face or per edge cell?** Per-face is parsimonious (4 scalars); per-edge-cell is flexible but underdetermined. v1: per-face Gaussian; v2: per-face Matern along the boundary.
- **IC initialisation.** Long warm-up (48 h from CAMS) vs. joint IC inversion (more parameters but no spin-up bias). Leaning: joint inversion when budget allows, warm-up for scaling tests.
- **Emulator long-term stability.** Neural ODEs notoriously drift. Use truncated BPTT during training, or train with multi-step rollouts? Initial bias: multi-step rollout with ramped horizon.
- **Data-assimilation window length.** Longer windows = more constraint per source state but worse linearisation; shorter = faster but more drift between updates. Tunable per use case; document the trade-off.
- **GPU vs. CPU defaults.** `les_fvm` runs on both via JAX. At what grid size does GPU pay off? Worth a benchmark notebook to anchor user expectations — initial guess: GPU dominant above 200×200.
- **Posterior covariance method.** Default to Laplace (cheapest) or Gauss-Newton Hessian (more accurate)? En4D-Var only when posterior is non-Gaussian. Open: criterion for promoting to ensemble.
- **Hierarchical Matern length-scale.** Promote ℓ to a hyperparameter, or fix per-basin from a pilot inversion? Tier V.A consumes the posterior — hierarchical adds another integration but gives honest UQ.
