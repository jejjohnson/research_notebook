# vardax Tutorial Master List

A reconciled, exhaustive curriculum for **learning data assimilation through [vardax](https://github.com/jejjohnson/vardax)** — JAX-native variational DA shipping seven peer `pipekit_cycle.AnalysisStep` methods (OI / 3DVar / strong / weak / incremental 4DVar / FourDVarNet / `AmortizedPosterior`) on a single set of `eqx.Module` primitives.

> Ensemble-DA tutorials live in the sister list [`TUTORIAL_MASTER_LIST_FILTER.md`](TUTORIAL_MASTER_LIST_FILTER.md); GP / SVI tutorials live in [`../../gaussian_processes/TUTORIAL_MASTER_LIST.md`](../../gaussian_processes/TUTORIAL_MASTER_LIST.md). Cross-listed items (Bayesian update, structured covariances, sigma points, validation gates, latent-space generative priors) are flagged 🔁.

**Legend** — Source columns:
- `V` = exists in vardax (`docs/notebooks/<name>`)
- `G` = exists in gaussx (`docs/notebooks/<name>`)
- `K` = exists in pipekit (`docs/notebooks/<name>`)
- `R` = exists in research_notebook (`projects/assimilation/notebooks/<path>`)
- `—` = does not exist yet (gap)

**Scope tag**: 🧱 fundamental · 🔬 research · 🌉 bridge · 🔁 cross-listed (filterax / GP master list)

**Refs column**: `gh#N` = open GitHub issue · `dd:path` = vardax `docs/design/<path>` · `math:N` = vardax `docs/0N_<chapter>.md` · `api:foo` = vardax exported symbol.

**Math policy**: every section's first notebook (`00_*`) ports the relevant math chapter from `vardax/docs/` directly into the tutorial, so each part is self-contained for someone learning DA from scratch. Subsequent notebooks in the section are API-driven.

---

## Curriculum at a glance

- **Part 0 — Foundations** *(learning DA from scratch)*
  - 0.A — The Bayesian update
  - 0.B — Linear-Gaussian closed form (BLUE / Kalman)
  - 0.C — Variational reformulation (cost ⇄ posterior)
  - 0.D — pipekit-cycle protocols (the vardax shape)
  - 0.E — Your first assimilation
- **Part 1 — Observation operators**
  - 1.A — `MaskedIdentity` and the symmetric-masking invariant
  - 1.B — `LinearObs` with explicit H
  - 1.C — `AveragingKernel` (satellite footprints, RTM intuition)
  - 1.D — `MultiInstrumentFusion` and heterogeneous obs
  - 1.E — Writing a custom `ObservationOperator`
- **Part 2 — Forward models**
  - 2.A — The `pipekit_cycle.ForwardModel` protocol
  - 2.B — Lorenz family wrappers (L63, L96, L96-2L)
  - 2.C — Shallow water (`somax`)
  - 2.D — Quasi-geostrophic (`somax`)
  - 2.E — Plume models (`plumax`)
  - 2.F — Writing your own forward
- **Part 3 — Static priors & covariances**
  - 3.A — Diagonal B (the simplest case)
  - 3.B — Structured priors via `gaussx` (Matérn, Kronecker)
  - 3.C — Dynamical priors (forward-as-φ)
- **Part 4 — Encoders & decoders (latent-space DA)**
  - 4.A — AE priors overview (`BilinAE`, `ConvAE`, `MLP`)
  - 4.B — Pretraining an AE prior
  - 4.C — Latent-space DA (assimilate in z, decode to x)
  - 4.D — Observation encoders for amortized inference
- **Part 5 — Optimal Interpolation (BLUE)**
  - 5.A — BLUE derivation (closed-form Kalman update)
  - 5.B — `OptimalInterpolation` API walkthrough
  - 5.C — OI with structured B
- **Part 6 — 3DVar**
  - 6.A — The 3DVar cost
  - 6.B — `ThreeDVar` API + minimiser choice
  - 6.C — Nonlinear H with 3DVar
- **Part 7 — Strong-constraint 4DVar**
  - 7.A — The 4DVar cost (adjoint of M)
  - 7.B — `StrongFourDVar` API
  - 7.C — `forward_adjoint` choices (diffrax variants)
  - 7.D — Lorenz demo (deep dive, links to assimilation/)
- **Part 8 — Weak-constraint 4DVar**
  - 8.A — Model-error term
  - 8.B — `WeakFourDVar` API
  - 8.C — Imbalance failure modes
- **Part 9 — Incremental 4DVar**
  - 9.A — GN outer + CG inner
  - 9.B — `IncrementalFourDVar` API
  - 9.C — Control-variable transform (`gaussx √B` preconditioning)
- **Part 10 — FourDVarNet (learned solvers)**
  - 10.A — Unrolled iteration math
  - 10.B — `FourDVarNet1D` / `FourDVarNet2D` API
  - 10.C — Training loop (`train_step` composition)
  - 10.D — `solver_adjoint` dispatch (D15)
- **Part 11 — Amortized posterior**
  - 11.A — Simulation-based inference framing
  - 11.B — Regression head
  - 11.C — Conditional-flow head *(stub — pending `gauss_flows`)*
  - 11.D — Score-diffusion head *(stub — pending diffrax reverse-SDE)*
- **Part 12 — Adjoints deep dive**
  - 12.A — What is an adjoint?
  - 12.B — `diffrax` adjoint variants
  - 12.C — `optimistix` adjoint variants
  - 12.D — Writing your own `AbstractAdjoint`
- **Part 13 — Posterior uncertainty**
  - 13.A — Laplace at MAP
  - 13.B — Gauss-Newton Hessian
  - 13.C — Ensemble covariances
  - 13.D — `GaussianMarkLikelihood` serialisation
- **Part 14 — Validation gates (six-step cycle)**
  - 14.A — Six-step cycle motivation (D12)
  - 14.B — `assert_posterior_agreement`
  - 14.C — `assert_adjoint_calibrated`
  - 14.D — `simulation_based_calibration` (Talts et al.)
- **Part 15 — Orchestration (pipekit composition)**
  - 15.A — `VarDACycle` for sequential DA
  - 15.B — `VarSmootherCycle` for retrospective windows
  - 15.C — `obs_source` operators
  - 15.D — Composing with `pipekit.Sequential`
- **Part 16 — Training at scale (pipekit-train)**
  - 16.A — `train_step` / `amortized_train_step` primitives
  - 16.B — `pipekit_train.Loss` adapter
  - 16.C — `TrainingLoop` integration
  - 16.D — Checkpoints & sweeps
- **Part 17 — Performance**
  - 17.A — `jit` basics
  - 17.B — `eqx.filter_vmap` batching
  - 17.C — Memory profiling adjoint variants
  - 17.D — When to unroll vs `scan`
- **Part 18 — Debugging**
  - 18.A — Cost decomposition (`decomposed_loss`)
  - 18.B — Inspecting residuals
  - 18.C — Common failure modes
- **Part 19 — Extending vardax**
  - 19.A — Writing your own `AnalysisStep`
  - 19.B — Protocol-conformance tests
  - 19.C — Publishing as a plugin
- **Part 20 — Applied** *(cross-link to applied projects)*
  - 20.A — Lorenz benchmarks (→ `projects/assimilation/`)
  - 20.B — SSH satellite interpolation
  - 20.C — Methane single-overpass retrieval
  - 20.D — Multi-instrument atmospheric fusion

---

## Part 0 — Foundations

### 0.A — The Bayesian update

**Key equations / models:**
- Prior, likelihood, posterior: $p(x \mid y) \propto p(y \mid x)\,p(x)$
- Gaussian-Gaussian closure: $p(x) = \mathcal{N}(x_b, B)$, $p(y \mid x) = \mathcal{N}(Hx, R)$
- Posterior in information form: $\Lambda^a = B^{-1} + H^\top R^{-1} H$, $\eta^a = B^{-1} x_b + H^\top R^{-1} y$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.1 | The DA problem from scratch — prior + likelihood → posterior | — | 🧱 🔁 | math:01_problem_setting; graphical-model diagram; pairs with filterax 0.1 / GP 0.6 |
| 0.2 | The Gaussian-Gaussian closure — covariance form vs information form | — | 🧱 🔁 | math:04_oi_blue §1; cross-listed with filterax 0.9 |

### 0.B — Linear-Gaussian closed form (BLUE)

**Key equations / models:**
- Kalman gain: $K = B H^\top (H B H^\top + R)^{-1}$
- Update: $x^a = x_b + K(y - H x_b)$
- Posterior cov: $P^a = (I - K H) B$ (or Joseph form)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.3 | BLUE from scratch — the Kalman update derivation | — | 🧱 🔁 | math:04_oi_blue; pairs with filterax 0.3; covariance-ellipse diagram |

### 0.C — Variational reformulation (cost ⇄ posterior)

**Key equations / models:**
- 3DVar cost: $J(x) = \tfrac{1}{2}\|x - x_b\|^2_{B^{-1}} + \tfrac{1}{2}\|y - H x\|^2_{R^{-1}}$
- Argmin ↔ posterior mean equivalence (linear-Gaussian)
- Why we minimise: nonlinear $H$, structured $B$, large state dim

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.4 | The variational reformulation — argmin = posterior mean in the LG case | — | 🧱 | math:05_threedvar §1; show 3DVar reduces to BLUE on linear H |

### 0.D — pipekit-cycle protocols (the vardax shape)

**Key equations / models:**
- `ForwardModel`: `step(state, dt) → state`, `dt` property
- `ObservationOperator`: `__call__(state, mask=...) → obs`, `linearize(x)`
- `AnalysisStep`: `__call__(forecast, obs, *, obs_op, obs_err_cov) → analysis`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.5 | The pipekit-cycle protocol family — `ForwardModel` / `ObservationOperator` / `AnalysisStep` | — | 🧱 | dd:pipekit_composition.md; api:`vardax.protocols`; class-diagram with conformance examples |

### 0.E — Your first assimilation

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.6 | Five-line BLUE on a 1-D toy — `vdx.OptimalInterpolation` end-to-end | — | 🧱 | api:`OptimalInterpolation`; minimal `Batch1D`; bridge to Part 5 |

---

## Part 1 — Observation operators

Every method in vardax accepts an `ObservationOperator`. This part walks through the shipped catalogue and the protocol.

### 1.A — `MaskedIdentity` and the symmetric-masking invariant

**Key equations / models:**
- $H(x) = m \odot x$, $\mathrm{linearize}(x) = \mathrm{diag}(m)$
- Symmetric masking: $r = m \odot (y - H(x))$ — both sides masked
- Why naive `m \odot y - H(x)` biases the analysis (PR #41 review story)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.1 | `MaskedIdentity` — the dense-but-sparse-observed primitive | — | 🧱 | api:`MaskedIdentity`; math:11_observation_operators §2; symmetric-masking gotcha |

### 1.B — `LinearObs` with explicit H

**Key equations / models:**
- $H \in \mathbb{R}^{N_y \times N_x}$ as a `lineax.AbstractLinearOperator`
- Structured H via `gaussx` operators (Toeplitz, block-diagonal)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.2 | `LinearObs` — explicit H, structured H, computational dispatch | — | 🧱 🔁 | api:`LinearObs`; cross-listed with filterax 1.5 (structured R) |

### 1.C — `AveragingKernel` (satellite footprints, RTM intuition)

**Key equations / models:**
- $H(x) = A(h \odot x + (1 - h) x_a)$
- Smoothing-kernel + a-priori interpretation (Rodgers 2000)
- Why $\mathrm{linearize}$ gives a structured Jacobian

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.3 | `AveragingKernel` — satellite retrievals, $A$ + prior contribution | — | 🧱 | api:`AveragingKernel`; math:02_observation_model §3; adjoint sanity-check pattern |

### 1.D — `MultiInstrumentFusion` and heterogeneous obs

**Key equations / models:**
- Per-instrument $H_i$, $R_i$; output = `dict[str, Array]`
- `InstrumentRegistry` / `InstrumentSpec` plumbing

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.4 | `MultiInstrumentFusion` — combining heterogeneous instruments | — | 🌉 | api:`MultiInstrumentFusion`, `InstrumentRegistry`, `InstrumentSpec`; design D9 |

### 1.E — Writing a custom `ObservationOperator`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.5 | Writing a custom obs op — protocol conformance + adjoint test | — | 🧱 | dd:pipekit_composition.md §`ObservationOperator`; reuses `tests/test_pipekit_protocols.py` pattern |

---

## Part 2 — Forward models

### 2.A — The `pipekit_cycle.ForwardModel` protocol

**Key equations / models:**
- `step(state, dt) → state` — one integration step
- `state_signature` — JAX shape/dtype declaration
- One-step + `jax.lax.scan` for rollouts

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.1 | `ForwardModel` protocol — what makes a forward and how vardax uses it | — | 🧱 | dd:pipekit_composition.md §`ForwardModel`; identity / linear / nonlinear examples |

### 2.B — Lorenz family wrappers (L63, L96, L96-2L)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.2 | Lorenz wrappers — RK4 + `ForwardModel` interface | R | 🧱 | source: `projects/assimilation/src/assimilation/{lorenz63,lorenz96,lorenz96_2l}.py`; one-paragraph derivation per system |

### 2.C — Shallow water (`somax`)

**Key equations / models:**
- $\partial_t h + \nabla \cdot (hv) = 0$, $\partial_t v + v \cdot \nabla v + g \nabla h = -fk \times v + \nu \Delta v$
- Arakawa C-grid via `finitevolx`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.3 | Shallow water on a sphere — `somax.ShallowWaterModel` as a `ForwardModel` | — | 🌉 | external: `somax`; pair with strong-4DVar in Part 7 |

### 2.D — Quasi-geostrophic (`somax`)

**Key equations / models:**
- QG potential vorticity: $q = \nabla^2 \psi + \beta y - F\psi$
- Layered QG for ocean / atmosphere

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.4 | Quasi-geostrophic — classic oceanographic DA testbed | — | 🌉 | external: `somax`; pair with `IncrementalFourDVar` in Part 9 |

### 2.E — Plume models (`plumax`)

**Key equations / models:**
- Gaussian plume: $C(x, y, z) = \frac{Q}{2\pi u \sigma_y \sigma_z} \exp(\ldots)$
- Gaussian puff (time-resolved): superposed instantaneous releases

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.5 | Gaussian-puff atmospheric dispersion as a `ForwardModel` | — | 🌉 | external: `plumax`; pair with `AmortizedPosterior` in Part 11 (methane single-overpass) |

### 2.F — Writing your own forward

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.6 | Writing a custom `ForwardModel` — 1-D advection-diffusion worked example | — | 🧱 | dd:pipekit_composition.md §`ForwardModel`; `diffrax` integration recipes |

---

## Part 3 — Static priors & covariances

### 3.A — Diagonal B (the simplest case)

**Key equations / models:**
- $B = \sigma_b^2 I$
- The PSD-tag requirement for `lineax.CG` consumers
- Why `lx.DiagonalLinearOperator` must be wrapped in `TaggedLinearOperator`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.1 | Diagonal B — the simplest case, with the tagged-operator gotcha | — | 🧱 | math:01_problem_setting §3; api:`lx.DiagonalLinearOperator`; `positive_semidefinite_tag` |

### 3.B — Structured priors via `gaussx` (Matérn, Kronecker)

**Key equations / models:**
- Matérn-$\nu$ covariance with length-scale $\ell$
- Kronecker structure for space × time
- Half-operator $\sqrt{B}$ for control-variable transform

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.2 | Structured B via `gaussx` — Matérn, Kronecker, $\sqrt{B}$ | G | 🧱 🔁 | external: `gaussx`; cross-listed with GP 1.B / filterax 1.5 |

### 3.C — Dynamical priors (forward-as-φ)

**Key equations / models:**
- $\varphi(x) = M(x)$ — N applications of the forward
- `DynamicalPrior(forward, n_steps, forward_adjoint)`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.3 | Dynamical priors — forward as $\varphi$, gradient flow choice | — | 🌉 | api:`DynamicalPrior`; dd:pipekit_composition.md; bridge to Part 10 |

---

## Part 4 — Encoders & decoders (latent-space DA)

The "latent-space" angle on vardax — autoencoder priors as a regulariser, latent-space assimilation, and observation encoders for the amortized head.

### 4.A — AE priors overview (`BilinAE`, `ConvAE`, `MLP`)

**Key equations / models:**
- $\varphi_\theta(x) = D_\theta(E_\theta(x))$ — encode-then-decode
- Bottleneck dim $\ll N_x$ enforces a manifold prior
- Bilinear, conv, MLP architectures shipped in `vardax.priors`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.1 | AE prior architectures — `BilinAEPrior`, `ConvAEPrior`, `MLPAEPrior` | — | 🧱 🔁 | api:`BilinAEPrior1D`, `ConvAEPrior1D`, `MLPAEPrior1D`, `BilinAEPrior2D`, `BilinAEPrior2DMultivar`; cross-listed with GP 6.A (deep kernels) |

### 4.B — Pretraining an AE prior

**Key equations / models:**
- Reconstruction loss: $\mathcal{L} = \mathbb{E}_x\|x - \varphi_\theta(x)\|^2$
- Simulation-based pretraining via vmap'd `generate_problem`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.2 | Pretraining an AE prior on simulated trajectories | — | 🧱 | api:`train_step`; pre-training schedule for L96 |

### 4.C — Latent-space DA (assimilate in z, decode to x)

**Key equations / models:**
- Latent posterior: $q(z \mid y) \propto p(y \mid D(z))\,p(z)$
- Decode-then-observe: $H \circ D$ as the effective obs op
- Why this conditions the problem (lower-dim z = better-posed)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.3 | Latent-space DA — assimilate in $z$, decode to $x$ | — | 🔬 🔁 | math:09_4dvarnet §4; cross-listed with GP latent-variable models; ill-conditioning recipe |

### 4.D — Observation encoders for amortized inference

**Key equations / models:**
- Context encoder $c_\psi(y, m)$ for `AmortizedPosterior`
- Different role from prior encoder: maps *obs* not *state*
- `IdentityObsEncoder` vs `MLPObsEncoder`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.4 | Observation encoders for `AmortizedPosterior` | — | 🧱 | api:`IdentityObsEncoder`, `MLPObsEncoder`; bridge to Part 11 |

---

## Part 5 — Optimal Interpolation (BLUE)

### 5.A — BLUE derivation (closed-form Kalman update)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.1 | BLUE — closed-form Kalman update derivation, ported from `math:04_oi_blue` | — | 🧱 🔁 | math:04_oi_blue; cross-listed with filterax 0.3 |

### 5.B — `OptimalInterpolation` API walkthrough

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.2 | `OptimalInterpolation` API — `lineax.CG` inner solve, `mask_aware` dispatch | R | 🧱 | api:`OptimalInterpolation`; source: `projects/assimilation/notebooks/01_optimal_interpolation.ipynb` |

### 5.C — OI with structured B

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.3 | OI with a Matérn B — observed → unobserved coupling via prior cross-cov | — | 🔬 🔁 | api:`OptimalInterpolation`; uses gaussx-backed B; cross-listed with GP 1.B |

---

## Part 6 — 3DVar

### 6.A — The 3DVar cost

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.1 | The 3DVar cost — derivation + ⇄ BLUE equivalence (Decision D14) | — | 🧱 | math:05_threedvar; D14 invariant verified empirically |

### 6.B — `ThreeDVar` API + minimiser choice

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.2 | `ThreeDVar` — choosing the minimiser (BFGS / NonlinearCG / GaussNewton) | R | 🧱 | api:`ThreeDVar`; `optimistix.AbstractMinimiser` choice; source: `projects/assimilation/notebooks/02_threedvar.ipynb` |

### 6.C — Nonlinear H with 3DVar

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.3 | 3DVar with `AveragingKernel` — when iterative pays off | — | 🌉 | api:`ThreeDVar`, `AveragingKernel`; satellite-retrieval example |

---

## Part 7 — Strong-constraint 4DVar

### 7.A — The 4DVar cost (adjoint of M)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.1 | The 4DVar cost — control = $x_0$, perfect-model assumption | — | 🧱 🔁 | math:06_strong_4dvar; cross-listed with filterax 0.8 (4D-Var contrast) |

### 7.B — `StrongFourDVar` API

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.2 | `StrongFourDVar` end-to-end — Lorenz-63 long-window forecast | R | 🧱 | api:`StrongFourDVar`; source: `projects/assimilation/notebooks/03_strong_4dvar.ipynb` |

### 7.C — `forward_adjoint` choices (diffrax variants)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.3 | `forward_adjoint` — `RecursiveCheckpointAdjoint` vs `BacksolveAdjoint` vs `ForwardMode` | — | 🧱 | dd:decisions.md (D15); memory / time table; sets up Part 12 |

### 7.D — Lorenz demo (deep dive, links to assimilation/)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.4 | L96 strong-4DVar — partial obs, what dynamics buy you | R | 🧱 | source: `projects/assimilation/notebooks/10_lorenz96_benchmark.ipynb` |

---

## Part 8 — Weak-constraint 4DVar

### 8.A — Model-error term

**Key equations / models:**
- $x_t = M(x_{t-1}) + \eta_t$, $\eta_t \sim \mathcal{N}(0, Q)$
- Augmented control $(x_0, \eta_1, \ldots, \eta_T)$ in $\mathbb{R}^{N + TN}$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.1 | Weak-4DVar cost — the model-error term, $Q$ vs $B$ vs $R$ | — | 🧱 | math:07_weak_4dvar |

### 8.B — `WeakFourDVar` API

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.2 | `WeakFourDVar` — running it; convergence pitfalls on long windows | R | 🧱 | api:`WeakFourDVar`; source: `projects/assimilation/notebooks/04_weak_4dvar.ipynb` |

### 8.C — Imbalance failure modes

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.3 | Imbalance failure modes — when augmented control diverges | — | 🔬 | known L63 long-window failure (PR #73 discussion); when to lower $Q$ |

---

## Part 9 — Incremental 4DVar

### 9.A — GN outer + CG inner

**Key equations / models:**
- Linearise $M$, $H$ at $x_b^{(k)}$
- GN Hessian: $J''_{GN} = B^{-1} + \sum_t (H'_t M'_t)^\top R^{-1} (H'_t M'_t)$
- Solve $J''_{GN}\,\delta x^* = -\nabla J$ by `lineax.CG`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.1 | Incremental idea — GN outer + CG inner, why the linearised quadratic | — | 🧱 🔁 | math:08_incremental_4dvar; cross-listed with filterax 5.3 (GNKI) |

### 9.B — `IncrementalFourDVar` API

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.2 | `IncrementalFourDVar` — tuning `(n_outer, n_inner, cg_tol)` | R | 🧱 | api:`IncrementalFourDVar`, `IncrementalConfig`; source: `projects/assimilation/notebooks/05_incremental_4dvar.ipynb` |

### 9.C — Control-variable transform (`gaussx √B` preconditioning)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.3 | CVT preconditioning — $\chi = B^{-1/2}\delta x$ via `gaussx.sqrt(B)` | — | 🔬 🔁 | math:08_incremental_4dvar §4; cross-listed with GP 1.B (root decompositions) |

---

## Part 10 — FourDVarNet (learned solvers)

### 10.A — Unrolled iteration math

**Key equations / models:**
- $x^{(k+1)} = x^{(k)} - \alpha\,\Phi_\phi(\nabla J(x^{(k)}), x^{(k)}, h^{(k)})$
- Learned ConvLSTM modulator $\Phi_\phi$
- Learned prior $\varphi_\theta$ inside $J$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.1 | Unrolled-iteration math — learned solver as a $K$-step net | — | 🧱 🔁 | math:09_4dvarnet; bridges to filterax Part 8 (differentiable DA) |

### 10.B — `FourDVarNet1D` / `FourDVarNet2D` API

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.2 | `FourDVarNet1D` / `FourDVarNet2D` walkthrough | R | 🧱 | api:`FourDVarNet1D`, `FourDVarNet2D`; source: `projects/assimilation/notebooks/06_fourdvarnet.ipynb` |

### 10.C — Training loop (`train_step` composition)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.3 | Training a FourDVarNet — simulation-based + reconstruction loss | — | 🧱 | api:`train_step`, `reconstruction_loss`; bridge to Part 16 |

### 10.D — `solver_adjoint` dispatch (D15)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.4 | `solver_adjoint` dispatch — `RecursiveCheckpoint` vs `OneStep` vs `Implicit` | — | 🔬 | dd:decisions.md (D15); api:`OneStepAdjoint`; bridge to Part 12 |

---

## Part 11 — Amortized posterior

### 11.A — Simulation-based inference framing

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.1 | SBI framing — when amortization wins | — | 🧱 🔁 | math:10_amortized_inference §1; cross-listed with filterax 10.E (amortised inference) |

### 11.B — Regression head

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.2 | `RegressionHead` — diagonal-Gaussian head, training via NLL | R | 🧱 | api:`AmortizedPosterior`, `RegressionHead`, `amortized_train_step`; source: `projects/assimilation/notebooks/07_amortized_posterior.ipynb` |

### 11.C — Conditional-flow head *(stub — pending `gauss_flows`)*

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.3 | `ConditionalFlowHead` — exact density via change-of-variables *(stub)* | — | 🔬 🔁 | api:`ConditionalFlowHead`; cross-listed with GP normalising-flow tutorials; documents `NotImplementedError` until `gauss_flows` lands |

### 11.D — Score-diffusion head *(stub — pending diffrax reverse-SDE)*

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.4 | `ScoreDiffusionHead` — sampling via reverse SDE *(stub)* | — | 🔬 | api:`ScoreDiffusionHead`; documents `NotImplementedError` until diffrax reverse-SDE lands |

---

## Part 12 — Adjoints deep dive

### 12.A — What is an adjoint?

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.1 | What is an adjoint? — VJP vs hand-coded adjoint model | — | 🧱 🔁 | math:12_adjoint_methods; cross-listed with filterax Part 8 |

### 12.B — `diffrax` adjoint variants

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.2 | `diffrax.RecursiveCheckpointAdjoint` vs `BacksolveAdjoint` vs `ForwardMode` | — | 🧱 | dd:decisions.md (D15); api:`RecursiveCheckpointAdjoint`; memory / time table on Lorenz windows |

### 12.C — `optimistix` adjoint variants

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.3 | `optimistix.ImplicitAdjoint` vs `RecursiveCheckpointAdjoint` for the inner solver | — | 🧱 | api:`ImplicitAdjoint`, `OneStepAdjoint`; when implicit applies (fixed-point) |

### 12.D — Writing your own `AbstractAdjoint`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.4 | Writing your own `optx.AbstractAdjoint` — `OneStepAdjoint` as the template | — | 🔬 | api:`OneStepAdjoint` (Bolte 2023); upstream-contribution arc |

---

## Part 13 — Posterior uncertainty

### 13.A — Laplace at MAP

**Key equations / models:**
- $P^* \approx \bigl((H')^\top R^{-1} H' + B^{-1}\bigr)^{-1}$
- Returned as `lineax.AbstractLinearOperator` (lazy CG-backed inverse)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.1 | `LaplaceCovariance` — Gauss-Newton Hessian at MAP, lazy inverse | — | 🧱 🔁 | math:13_posterior_covariance §2; api:`LaplaceCovariance`; cross-listed with GP 0.C (variational Laplace) |

### 13.B — Gauss-Newton Hessian

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.2 | `GaussNewtonHessian` — reuse the incremental Hessian as posterior cov | — | 🔬 | api:`GaussNewtonHessian`; ties to Part 9 |

### 13.C — Ensemble covariances

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.3 | `EnsembleCovariance` — sample covariance from M analyses | — | 🌉 🔁 | api:`EnsembleCovariance`; cross-listed with filterax Part 1 (ensemble stats) |

### 13.D — `GaussianMarkLikelihood` serialisation

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.4 | `GaussianMarkLikelihood` — `Posterior` → `dict` for downstream population models | — | 🌉 | api:`GaussianMarkLikelihood`; dd:posterior.md |

---

## Part 14 — Validation gates (six-step cycle)

### 14.A — Six-step cycle motivation (D12)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 14.1 | The six-step cycle — why amortized heads need calibration gates | — | 🧱 🔁 | math:14_six_step_cycle; dd:decisions.md (D12); cross-listed with filterax Part 7 |

### 14.B — `assert_posterior_agreement`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 14.2 | `assert_posterior_agreement` — marginal z-score check against an oracle | — | 🧱 | api:`assert_posterior_agreement`; pair amortized with strong-4DVar oracle |

### 14.C — `assert_adjoint_calibrated`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 14.3 | `assert_adjoint_calibrated` — random-vector JVP probe of $\partial x^* / \partial y$ | — | 🔬 | api:`assert_adjoint_calibrated`; Hutchinson-style |

### 14.D — `simulation_based_calibration` (Talts et al.)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 14.4 | `simulation_based_calibration` — rank histograms (Talts 2018) | — | 🧱 🔁 | api:`simulation_based_calibration`; cross-listed with GP / SVI calibration tutorials |

---

## Part 15 — Orchestration (pipekit composition)

### 15.A — `VarDACycle` for sequential DA

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 15.1 | `VarDACycle` — chaining many forecast → analyse windows | — | 🧱 🔁 | api:`VarDACycle`; dd:pipekit_composition.md; cross-listed with filterax Part 3 |

### 15.B — `VarSmootherCycle` for retrospective windows

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 15.2 | `VarSmootherCycle` — windowed retrospective analysis | — | 🔬 🔁 | api:`VarSmootherCycle`; cross-listed with filterax Part 4 (smoothers) |

### 15.C — `obs_source` operators

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 15.3 | `obs_source` operators — loading obs per cycle from a stream / catalogue | — | 🧱 | dd:pipekit_composition.md; xarray / georeader recipes |

### 15.D — Composing with `pipekit.Sequential`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 15.4 | Composing vardax in a `pipekit.Sequential` pipeline | — | 🌉 | dd:pipekit_composition.md §5; `pk.Lambda` wrap; `JaxModelOp` (when shipped) |

---

## Part 16 — Training at scale (pipekit-train)

### 16.A — `train_step` / `amortized_train_step` primitives

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 16.1 | The `train_step` primitive — `eqx.filter_value_and_grad` + `optax` | — | 🧱 | api:`train_step`, `amortized_train_step`; dd:decisions.md (D5) |

### 16.B — `pipekit_train.Loss` adapter

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 16.2 | `pipekit_train.Loss` adapter — `VardaxReconLoss` over `train_step` | — | 🧱 | api:`vardax.adapters.pipekit_train.VardaxReconLoss` |

### 16.C — `TrainingLoop` integration

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 16.3 | `pipekit_train.TrainingLoop` end-to-end for FourDVarNet | — | 🌉 | external: `pipekit_train`; checkpointing + early stopping |

### 16.D — Checkpoints & sweeps

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 16.4 | `pipekit_train.HyperSweep` for FourDVarNet hyperparameters | — | 🔬 | external: `pipekit_train.HyperSweep`; grid over `(n_solver_steps, hidden_dim)` |

---

## Part 17 — Performance

### 17.A — `jit` basics

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 17.1 | `eqx.filter_jit` — what gets traced, what stays static | — | 🧱 | `Batch1D` shape stability; `static=True` field pitfalls |

### 17.B — `eqx.filter_vmap` batching

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 17.2 | `eqx.filter_vmap` over batches and over methods | — | 🧱 | source: `assimilation/benchmark.py` patterns |

### 17.C — Memory profiling adjoint variants

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 17.3 | Memory profiling — adjoint variant comparison on long L96 windows | — | 🔬 | `jax.profiler`; ties to Part 12 |

### 17.D — When to unroll vs `scan`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 17.4 | Unroll vs `jax.lax.scan` — when each pays off | — | 🌉 | rollout pattern; FourDVarNet `n_solver_steps` choice |

---

## Part 18 — Debugging

### 18.A — Cost decomposition (`decomposed_loss`)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 18.1 | `decomposed_loss` — splitting prior / obs cost for inspection | — | 🧱 | api:`decomposed_loss`, `obs_cost_1d`, `prior_cost`; sanity-check recipes |

### 18.B — Inspecting residuals

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 18.2 | Residual diagnostics — what a healthy assim window looks like | — | 🧱 | innovation $y - H(x^*)$, masked-vs-unmasked stats |

### 18.C — Common failure modes

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 18.3 | Common failure modes — Weak div, GN NaN, untagged operators, etc. | — | 🌉 | `lineax.CG` tag requirement; PRs #41 #42 #73 lessons distilled |

---

## Part 19 — Extending vardax

### 19.A — Writing your own `AnalysisStep`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 19.1 | Writing your own `AnalysisStep` — protocol conformance + `.as_analysis_step()` | — | 🧱 | dd:pipekit_composition.md §`AnalysisStep`; eqx.Module template |

### 19.B — Protocol-conformance tests

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 19.2 | Protocol-conformance tests — `tests/test_pipekit_protocols.py` template | V | 🧱 | source: `vardax/tests/test_pipekit_protocols.py`; isinstance-check pattern |

### 19.C — Publishing as a plugin

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 19.3 | Publishing your method as a vardax-compatible plugin | — | 🔬 | packaging recipe; entry-points pattern |

---

## Part 20 — Applied *(cross-link)*

### 20.A — Lorenz benchmarks (→ `projects/assimilation/`)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 20.1 | Lorenz-63, Lorenz-96 (1L + 2L) benchmarks | R | 🧱 | source: `projects/assimilation/notebooks/`; analysis-then-forecast view across all seven methods |

### 20.B — SSH satellite interpolation

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 20.2 | SSH satellite interpolation — `FourDVarNet2D` on OceanBench-style data | — | 🌉 | math:16_ssh_example; bridges to oceanography use case |

### 20.C — Methane single-overpass retrieval

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 20.3 | Methane single-overpass — `AveragingKernel` + `plumax` Gaussian puff | — | 🌉 | math:17_methane_example; pair with Part 11 amortized head |

### 20.D — Multi-instrument atmospheric fusion

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 20.4 | Multi-instrument atmospheric fusion — TROPOMI + EMIT + GHGSat | — | 🔬 | api:`MultiInstrumentFusion`; design D9; ties to Part 1.D |

---

## Status snapshot

This is a curriculum scaffold. Each row's `Source` column reflects what already exists:

- `R` rows are notebooks that already live in `projects/assimilation/notebooks/` (the L63 / L96 / L96-2L tutorials shipped with the assimilation project).
- `V` / `G` / `K` rows are notebooks that exist upstream in the `vardax` / `gaussx` / `pipekit` repos (none at the time of writing — every such row currently reads `—`, but the legend documents what those tags mean when upstream notebooks land).
- `—` rows are gaps to be filled by follow-up PRs.

Follow-up PRs will populate `projects/assimilation/notebooks/vardax/<section>/<name>.ipynb` (a new `vardax/` subdirectory under the existing notebooks tree, alongside `00_lorenz63_setup.md` and friends) and flip the `Source` column from `—` to `R`. Pure-prose chapters that port math may land as `.md` instead of `.ipynb`.

**v1 priority** (~6 sessions, ~35 notebooks): Parts 0–6 + 13 + 14 + 16 + the L63 entries that already exist in `projects/assimilation/`. Foundations + components + half the seven methods + posterior + validation + training.

**v2 priority**: Parts 7–12 dynamics + adjoints depth; Parts 15 cycling demo; Part 17 performance; Part 19 extension; Part 20 applied demos.

**v3 priority**: stub heads (11.C, 11.D) flipped to real when `gauss_flows` / diffrax reverse-SDE land; Part 9.C CVT depending on `gaussx.sqrt`; somax / plumax forwards (2.C / 2.D / 2.E).
