---
title: "Tier III — Eulerian finite-volume transport"
short_title: "Tier III — Eulerian FV"
subject: "plumax — gold-standard PDE tier"
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [Eulerian dispersion, finite volume, PDE, 4D-Var, incremental 4D-Var, control variable transform, neural operator, FNO, en4D-Var, Matern prior, lateral boundary conditions]
---

# Tier III — Eulerian finite-volume transport

**Forward model:** full PDE solved on a grid. Uses [`finitevolX`](https://github.com/jejjohnson/finitevolX) for spatial discretisation and [`spectraldiffx`](https://github.com/jejjohnson/spectraldiffx) for spectral / periodic-domain operators.

This is the **gold-standard physics tier**: full mass conservation, arbitrary wind fields, support for chemistry and deposition. Also the most expensive — emulators (Step 3) are essential, not optional.

---

(tier3-simple-model)=
## (1) Simple model

(tier3-conservation)=
### Conservation equation

Operational implementations carry **mixing ratio** $c$ (kg/kg), not mass density, because that's what the satellite observes and what WRF couples to. The conservation form is then:

```{math}
:label: eq-tier3-conservation
\partial_{t} (\rho c) \;+\; \nabla\cdot(\rho \mathbf{u}\, c) \;=\; \nabla\cdot(\rho \mathbf{K} \nabla c) \;+\; S(\mathbf{x},t) \;-\; \lambda\, \rho\, c
```

with $\rho(\mathbf{x},t)$ the dry-air density from the [met field](00_prerequisites.md#prereqs-metfield-schema). When $\rho$ is approximately constant over the integration window (typical at fixed altitude in a regional basin) this reduces to the textbook $\partial_t c + \nabla\cdot(\mathbf{u}\, c) = \nabla\cdot(\mathbf{K}\nabla c) + S - \lambda c$. Commit to the full $\rho$-weighted form for any work crossing significant pressure-altitude variation.

- $\mathbf{u}(\mathbf{x},t)$ — wind from WRF / ERA5 / HRRR.
- $\mathbf{K}(\mathbf{x},t)$ — eddy diffusivity. Commit to **K-theory from MO similarity** in the PBL ($K_z \approx \kappa\, u_*\, z (1 - z/L)$ with stability correction; {cite:p}`monin1954,stull1988`) plus **Smagorinsky** in the free troposphere. Source: [MO prereqs](00_prerequisites.md#prereqs-mo-similarity) + WRF TKE.
- $S(\mathbf{x},t)$ — source field, **the unknown we invert for**. Per-cell, time-resolved.
- $\lambda$ — first-order chemical loss. Negligible for CH₄ at <1-day scales; included for portability to CO/HCHO.

(tier3-bcs)=
### Initial and boundary conditions

Both are **load-bearing** for inversion accuracy and currently underspecified in operational code:

:::{important} Initial conditions
Long inversion windows (>24 h) need realistic ICs; otherwise spin-up corrupts early observations.

**Default:** a 48-hour spin-up from a CAMS / GEOS-Chem global background, or jointly invert $c(0)$ alongside $S$ (adds $c_b(0)$ background term to the cost — see [Inference](#tier3-inference)).
:::

:::{important} Lateral BCs
Inflow from a coarser global model (CAMS, GEOS-Chem) is uncertain. Lateral-BC error is a known systematic in regional inversions; standard fix is to jointly invert per-face BC scaling factors $\alpha_\text{face} \sim \mathcal{N}(1, 0.1^2)$ and absorb the bias into the source posterior covariance. Document on the `boundary.py` API.
:::

### Spatial / temporal discretisation

Cell-centred FV with flux-limited advection via `finitevolX`; explicit RK or IMEX for stiff diffusion; `spectraldiffx` for periodic / global domains.

### Build order within Tier III

1. **1D column model.** Validates numerics — diffusion against analytical Gaussian, advection against cosine pulse / Burgers'.
2. **2D horizontal slab.** What most satellite work needs (vertical column already integrated by AK).
3. **Full 3D with vertical layers.** Needed when injection altitude matters or when assimilating multi-layer in-situ data alongside columns.

---

(tier3-inference)=
## (2) Model-based inference

### Forward observation operator

Same column + AK pipeline as Tiers I and II:

```{math}
:label: eq-tier3-forward
\mathbf{y}_t \;=\; \mathbf{A}_t\, \mathrm{col}_z\!\bigl(c(S, c_0, t)\bigr) \;+\; \mathbf{c}_\text{bg} \;+\; \boldsymbol{\varepsilon}_t
```

$\mathbf{H}_t \coloneqq \mathbf{A}_t \cdot \mathrm{col}_z(\cdot)$ is the observation operator at time $t$. The 4D-Var cost below uses $\mathbf{H}_t$; the column + AK implementation is shared with [Tier I](01_tier1_gaussian.md#tier1-column-ak) and [the prereqs](00_prerequisites.md#prereqs-ak-operator).

(tier3-cost)=
### 4D-Var cost — three terms

```{math}
:label: eq-tier3-4dvar-cost
J(S, c_0) \;=\; \tfrac{1}{2}\lVert S - S_b \rVert^{2}_{\mathbf{B}}
\;+\; \tfrac{1}{2}\lVert c_0 - c_b(0) \rVert^{2}_{\mathbf{B}_c}
\;+\; \tfrac{1}{2} \sum_{t} \lVert \mathbf{H}_t\, c(S, c_0, t) - \mathbf{y}_t \rVert^{2}_{\mathbf{R}_t}
```

Three terms — source background, IC background, and **time-summed** observation mismatch. The IC term drops out only if you commit to a long warm-up that pins $c(0)$. Treating observations as a single instantaneous mismatch (no time index) is incompatible with multi-hour assimilation windows.

(tier3-likelihood)=
### Likelihood model

```{math}
:label: eq-tier3-likelihood
\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_t), \qquad \mathbf{R}_t \;=\; \mathbf{R}_{\text{retr},t} + \mathbf{R}_{\text{repr},t}
```

- $\mathbf{R}_{\text{retr},t}$ — heteroscedastic per-pixel from the L2 retrieval-error map at time $t$.
- $\mathbf{R}_{\text{repr},t}$ — representation error (model-vs-observation footprint mismatch). Diagonal addition; rises with terrain complexity and at coarse-instrument boundaries. **Don't omit** — naive $\mathbf{R} = \mathbf{R}_\text{retr}$ overweights observations and produces overconfident posteriors.
- Block-diagonal across overpasses; cross-time correlation only within met decorrelation scale.

(tier3-prior)=
### Prior on $S$ — spatially correlated, sign-constrained

Same structure as Tier II:

```{list-table} Prior specification on the source field $S$.
:label: tbl-tier3-prior
:header-rows: 1

* - Choice
  - Form
  - Notes
* - Mean $S_b$
  - from [emission inventory](00_prerequisites.md#prereqs-emission-inventory)
  - EDGAR / GFEI / EPA per-cell median
* - Covariance $\mathbf{B}$
  - Matérn-3/2, $\ell \in [5, 50]$ km
  - spatial regulariser; $\ell$ tuneable or hierarchical
* - Positivity
  - $\log S \sim \mathcal{N}(\log S_b, \mathbf{B}_{\log})$ (lognormal)
  - non-negative emissions; conjugate when linearised
* - BC scaling
  - per-face $\alpha \sim \mathcal{N}(1, 0.1^{2})$
  - absorbs lateral-BC bias
```

:::{important} Diagonal $\mathbf{B}$ is wrong
Diagonal $\mathbf{B}$ produces wildly noisy spatial posteriors. Always carry spatial correlation.
:::

(tier3-adjoint)=
### Adjoint

The adjoint of the transport equation (backward in time, conservative form) is:

```{math}
:label: eq-tier3-adjoint
-\partial_{t} \lambda \;-\; \nabla\cdot(\mathbf{u}\, \lambda) \;-\; \nabla\cdot(\mathbf{K} \nabla \lambda) \;+\; \lambda_\text{chem}\, \lambda \;=\; \frac{\partial \mathcal{L}}{\partial c}
```

The $\nabla\cdot(\mathbf{u}\,\lambda)$ term is **conservative** (matches the forward $\nabla\cdot(\mathbf{u}\,c)$); the doc previously had $\mathbf{u}\cdot\nabla\lambda$, which is equivalent only for divergence-free $\mathbf{u}$ and is incorrect for compressible WRF winds. JAX computes the discretised adjoint exactly via reverse-mode autodiff through the FV solver — no hand-derived adjoint code, no separate adjoint-correctness derivation. The mathematical form above is for *reading*, not implementation.

(tier3-incremental)=
### Incremental 4D-Var (the operational default)

Linearise around the current iterate $S^k$, solve the linear inner minimisation, update outer iterate:

```{math}
:label: eq-tier3-incremental
S^{k+1} \;=\; S^k + \delta S, \qquad \delta S \;=\; \arg\min_{\delta} \, J_\text{lin}(\delta; S^k)
```

$J_\text{lin}$ uses the **tangent linear** of the FV solver, trivially built via `jax.linearize`. Inner solves are quadratic in $\delta S$ → CG or Lanczos via [`gaussx`](https://github.com/jejjohnson/gaussx). Cuts cost by 1–2 orders of magnitude vs. fully-nonlinear 4D-Var. **This is the default**; full nonlinear is a sanity check.

(tier3-control-transform)=
### Control-variable transform

Direct optimisation in $S$-space with $\mathbf{B}^{-1}$ is infeasible — $\mathbf{B}$ for a $200 \times 200$ grid is $40000^{2} \approx 10^{9}$ entries. Standard fix:

```{math}
:label: eq-tier3-control-xform
\boldsymbol{\chi} \;=\; \mathbf{B}^{-1/2}(S - S_b), \qquad \text{optimise } J(S(\boldsymbol{\chi})) \text{ in } \boldsymbol{\chi}\text{-space (prior = identity Gaussian)}.
```

$\mathbf{B}^{-1/2}$ materialised via Matérn factorisation in [`gaussx`](https://github.com/jejjohnson/gaussx) (Kronecker structure for separable correlation). This is the load-bearing trick that makes 4D-Var tractable; should be explicit in `assimilation/control.py` and on the API.

### Posterior covariance — three options

4D-Var alone gives MAP; downstream Tier V needs the **posterior**:

- **Gauss–Newton Hessian inversion.** $\mathbf{P}^{*} = (\mathbf{H}^{\top} \mathbf{R}^{-1} \mathbf{H} + \mathbf{B}^{-1})^{-1}$ evaluated at MAP. Tractable for moderate grids via `gaussx` Krylov.
- **Laplace around MAP.** Sample from $\mathcal{N}(S^{*}, \mathbf{P}^{*})$; cheapest path.
- **En4D-Var.** Ensemble around MAP gives sample covariance; couples to [`filterax`](https://github.com/jejjohnson/filterax). Best when the posterior is non-Gaussian.

The posterior export to Tier V.A is via the same adapter pattern as Tier I/II — see the [posterior-export module](#tier3-modules).

### Cost / performance

For a $200 \times 200$ grid, 24-hour assimilation window, with incremental 4D-Var: ~minutes per outer iteration on CPU, seconds on GPU. Inner CG iteration is $O(n_\text{iter} \times n_\text{steps} \times n_\text{grid})$.

---

(tier3-emulator)=
## (3) Model emulator

Full 3D FV transport is expensive: $O(N^{3})$ state, repeated time integration. Emulator is **essential** here, not optional like at Tier I.

### State variable — commit to 2D column for v1

Most satellite inversion needs only the column-integrated XCH₄, not the 3D field. Default emulator state is **2D column** $c_\text{col}(x, y, t)$; full 3D is a v2 escalation when multi-layer in-situ assimilation enters scope. Order-of-magnitude difference in training cost.

### Architecture options

- **CNN UNet.** Honest baseline for non-stationary terrain. No translation-invariance assumption.
- **Graph-network operator.** Right structural fit for unstructured analysis grids and basin-scoped problems.
- **Fourier Neural Operator (FNO).** Resolution-agnostic and natural fit with `spectraldiffx` philosophy — **but** assumes translation invariance in the kernel. Works well for periodic / homogeneous problems; breaks for urban basins with terrain. Use only when translation invariance is plausible.
- **Neural ODE.** $c(t+\Delta t) = f_\theta(c(t), \mathbf{u}(t))$ iterated. Simple but suffers from long-horizon drift.
- **Reduced-order model (ROM).** POD on simulation snapshots → low-rank basis; learn projected dynamics via Galerkin or DMD. Great when dynamics live on a low-dim manifold.

### Training data

- **Active-learning over uniform climatology binning.** Sample WRF / ERA5 climatology adaptively — the emulator's residual error map drives where to run the next FV simulation. Reaches operational accuracy with 100–300 runs vs. ~1000 for uniform sampling.
- Sample met conditions from the **operational distribution** (facility locations of interest, overpass times) rather than a uniform-bin climatology — same critique as Tier II's emulator.

(tier3-emulator-adjoint)=
### Emulator adjoint must be calibrated

Backprop through a trained emulator gives *some* gradient — whether it matches the true PDE adjoint is empirical.

:::{important} Hard validation requirement
Emulator-autodiff gradients vs. FV-autodiff gradients on a held-out set should agree to $<5\%$ relative error in operator norm.

**Why:** if not, the inversion built on Step 4 is biased even when forward predictions look fine — the emulator passes a forward-only acceptance check but breaks the gradient that 4D-Var depends on.
**How to apply:** include the gradient-residual test in [validation](#tier3-validation) as a hard gate, not a "nice to have".
:::

---

(tier3-emu-inference)=
## (4) Emulator-based inference

Replace the FV integrator with the FNO / neural ODE in the 4D-Var loop:

- Gradient via autodiff through the emulator (not the PDE solver). Both are `jax.grad`-able; only the cost-per-iteration changes.
- Orders of magnitude faster per iteration (typically 100–1000×).
- **Adjoint validation:** emulator gradient ≈ FV adjoint (Step 3 calibration test).
- **Posterior validation:** posterior from emulator-based 4D-Var ≈ posterior from adjoint 4D-Var on the same observations. If they don't agree, the emulator is biased — diagnose before trusting downstream.

---

(tier3-amortized)=
## (5) Amortized inference (predictor)

```{math}
:label: eq-tier3-amortized
f_\theta : (\mathbf{y}_{t_1, \dots, t_n},\, \mathbf{u}_{1:T},\, \text{instrument\_id}) \;\longmapsto\; p(\log S(\mathbf{x},t) \mid \mathbf{y}, \mathbf{u})
```

### Output grid commitment

$S(\mathbf{x},t)$ is on the inversion grid (~1–10 km, basin-scoped). Predictor outputs a fixed-size 2D field per basin tile per time slice. **Per-instrument predictor heads**, dispatched by `instrument_id` — same pattern as Tier II.

### Irregular temporal sampling

Satellite passes are intermittent (1–3 per day per instrument over a basin) with gaps. Input is **not regularly sampled** — $t_1, \dots, t_n$ are observation times, not a fixed grid. Architecture: **set-transformer with time-encoding** over irregular passes, fused with a regular-grid encoder over $\mathbf{u}_{1:T}$ from the met field. Vanilla seq2seq / ConvLSTM assumes regular sampling and will silently underperform.

### Posterior over the spatial source field

Conditional flow over images vs. score-based diffusion — same trade-off as Tier II:

- `gauss_flows` is currently 1D-only; 2D coupling layers are a multi-month extension.
- **Score-based diffusion** is the safer v1 path for the spatial posterior.
- Context conditioning via FiLM / hypernet primitives in [`pyrox.nn`](https://github.com/jejjohnson/pyrox) — same pattern as Tier I/II.

---

(tier3-improve)=
## (6) Improve

- **Multi-species coupling.** Add CO and CO₂ tracers; their source ratios constrain CH₄ source attribution (e.g. fossil vs. agricultural).
- **Adaptive grid refinement.** Refine near sources, coarsen elsewhere. `finitevolX` may need primitives for this.
- **Online DA.** Assimilate observations as they arrive (rolling-window 4D-Var, or `filterax`'s ensemble Kalman smoother).
- **Sub-grid plume reconstruction.** When a source is sub-grid, the FV solver smears it; pair Tier III with a Tier I plume in the near-field for better point-source representation.
- **Hierarchical Matérn length-scale.** Promote $\ell$ to a hyperparameter with its own posterior — let the data choose the regularisation scale (mirrors Tier II).

---

(tier3-modules)=
## Module layout

```{list-table} Tier III module layout — step, concern, target module, status.
:label: tbl-tier3-modules
:header-rows: 1

* - Step
  - Concern
  - Module
  - Status
* - 1
  - FV grid
  - [`les_fvm/grid.py`](../../src/plume_simulation/les_fvm/grid.py)
  - ✓
* - 1
  - Advection
  - [`les_fvm/advection.py`](../../src/plume_simulation/les_fvm/advection.py)
  - ✓
* - 1
  - Diffusion
  - [`les_fvm/diffusion.py`](../../src/plume_simulation/les_fvm/diffusion.py)
  - ✓
* - 1
  - Source injection
  - [`les_fvm/source.py`](../../src/plume_simulation/les_fvm/source.py)
  - ✓
* - 1
  - Boundary conditions
  - [`les_fvm/boundary.py`](../../src/plume_simulation/les_fvm/boundary.py)
  - ✓
* - 1
  - Time integration
  - [`les_fvm/dynamics.py`](../../src/plume_simulation/les_fvm/dynamics.py), [`simulate.py`](../../src/plume_simulation/les_fvm/simulate.py)
  - ✓
* - 1
  - Eddy diffusivity (MO + Smagorinsky)
  - `plume_simulation.les_fvm.diffusivity`
  - ☐
* - 1
  - Column + AK pipeline
  - reuse `gauss_plume.observation` from Tier I
  - ☐
* - 2
  - Cost function (3-term)
  - [`assimilation/cost.py`](../../src/plume_simulation/assimilation/cost.py)
  - 🚧
* - 2
  - Likelihoods + spatial priors
  - `plume_simulation.assimilation.likelihoods`
  - ☐
* - 2
  - Control vector + transform
  - [`assimilation/control.py`](../../src/plume_simulation/assimilation/control.py)
  - 🚧
* - 2
  - Incremental 4D-Var solver
  - [`assimilation/solve.py`](../../src/plume_simulation/assimilation/solve.py)
  - 🚧
* - 2
  - Background ($S_b$, $c_b$, BC scaling)
  - [`assimilation/background.py`](../../src/plume_simulation/assimilation/background.py)
  - 🚧
* - 2
  - Diagnostics
  - [`assimilation/diagnostics.py`](../../src/plume_simulation/assimilation/diagnostics.py)
  - 🚧
* - 2
  - Posterior covariance (Hessian / Laplace / En4D-Var)
  - `plume_simulation.assimilation.posterior`
  - ☐
* - 2
  - Posterior export → Tier V
  - `plume_simulation.assimilation.posterior_export`
  - ☐
* - 3
  - Emulator (UNet / GNN / FNO)
  - `plume_simulation.les_fvm.emulator`
  - ☐
* - 3
  - Emulator-adjoint calibration harness
  - `plume_simulation.les_fvm.emulator_adjoint_test`
  - ☐
* - 5
  - Sequence predictor (set-transformer)
  - `plume_simulation.les_fvm.predictor`
  - ☐
* - 6
  - Multi-species coupling
  - `plume_simulation.les_fvm.multispecies`
  - ☐
```

---

(tier3-validation)=
## Validation strategy

- **1D diffusion.** Initial Dirac → at time $t$, solution is Gaussian with variance $2Kt$. Check $L^2$ error vs. analytical.
- **1D advection.** Cosine pulse advected at constant $u$ → after one period, recover initial condition. Tests upwind / flux-limiter consistency.
- **CFL stability.** Artificially exceed CFL → expect blow-up; stay within → expect stability. Catches silent flux-limiter regressions.
- **Mass conservation — three regimes.**
  - No sources, periodic BCs: $\int c\, \mathrm{d}V$ exact to floating-point.
  - With sources, no deposition: $\Delta \int c\, \mathrm{d}V = \iint S\, \mathrm{d}V\, \mathrm{d}t$ summed over the window.
  - With deposition: include $-\lambda \int c\, \mathrm{d}V\, \mathrm{d}t$ term.
- **Adjoint correctness.** JAX `vjp` of the discrete forward should satisfy $\langle \mathbf{F}\mathbf{u}, \mathbf{v}\rangle = \langle \mathbf{u}, \mathbf{F}^{\top} \mathbf{v}\rangle$ for random $\mathbf{u}, \mathbf{v}$. Cheap, catches differentiation bugs.
- **Tier I limit.** For a single point source in a uniform wind with constant $\mathbf{K}$, the steady-state FV solution should match the [Gaussian-plume formula](01_tier1_gaussian.md#tier1-gaussian-plume) at downwind distances much greater than the grid spacing.
- **Emulator-adjoint calibration.** Emulator-autodiff gradients vs. FV-autodiff gradients on a held-out met set, $<5\%$ relative error in operator norm. **Hard test** — failure means Step 4 inversion is biased.
- **Emulator OOD generalization.** Train on one met regime (e.g. summer Permian), evaluate on another (winter Permian, or a different basin). Resists overfit-to-training-distribution.
- **Real-data benchmark.** Compare 4D-Var output to existing CAMS / GEOS-Chem-Adjoint inversions on a published time window (e.g. {cite:p}`maasakkers2023ghgi,jacob2022quantifying` Permian inversions). Posterior credible interval should overlap the published estimate. Without this, the inversion is a synthetic exercise.

---

(tier3-open-questions)=
## Open questions

:::{attention} Choice of advection scheme
Upwind is robust but diffusive; WENO is sharp but stencil-heavy. Default for `les_fvm` is currently flux-limited; document the choice and the cell-Péclet floor where it stops being mass-conservative.
:::

:::{attention} Adjoint memory / checkpointing
Long windows mean the forward state must be re-derived (recompute) or stored (memory) for backprop. Standard fix: Griewank-style binomial checkpointing. Open: pick the checkpointing strategy and benchmark against `equinox.internal.scan_checkpointed` or hand-rolled.
:::

:::{attention} Lateral BC scaling — fit per face or per edge cell?
Per-face is parsimonious (4 scalars); per-edge-cell is flexible but underdetermined. v1: per-face Gaussian; v2: per-face Matérn along the boundary.
:::

:::{attention} IC initialisation
Long warm-up (48 h from CAMS) vs. joint IC inversion (more parameters but no spin-up bias). **Leaning:** joint inversion when budget allows, warm-up for scaling tests.
:::

:::{attention} Emulator long-term stability
Neural ODEs notoriously drift. Use truncated BPTT during training, or train with multi-step rollouts? Initial bias: multi-step rollout with ramped horizon.
:::

:::{attention} Data-assimilation window length
Longer windows = more constraint per source state but worse linearisation; shorter = faster but more drift between updates. Tunable per use case; document the trade-off.
:::

:::{attention} GPU vs. CPU defaults
`les_fvm` runs on both via JAX. At what grid size does GPU pay off? Worth a benchmark notebook to anchor user expectations — initial guess: GPU dominant above $200 \times 200$.
:::

:::{attention} Posterior covariance method
Default to Laplace (cheapest) or Gauss–Newton Hessian (more accurate)? En4D-Var only when posterior is non-Gaussian. Open: criterion for promoting to ensemble.
:::

:::{attention} Hierarchical Matérn length-scale
Promote $\ell$ to a hyperparameter, or fix per-basin from a pilot inversion? Tier V.A consumes the posterior — hierarchical adds another integration but gives honest UQ.
:::
