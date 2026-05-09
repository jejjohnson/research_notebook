# `plumax` — Offline equation registry

> **Not rendered with mystmd.** This file is a bookkeeping index of every labelled equation across the `plumax` roadmap notes. It exists so that authors can sanity-check label uniqueness, plan future cross-references, and audit equation numbering without spinning up the mystmd build.
>
> **Scope:** all `{math}` blocks under `projects/plume_simulation/notes/`.
> **Convention:** every block label is `eq-<page>-<short-name>`. `<page>` matches the file basename (`tier1`, `tier2`, …) or sub-page tag (`va`, `vb`, `vc`, `vd`, `rtm`, `prereqs`).
> **Total:** 77 labelled equations across 11 pages.

This document is not in `myst.yml`'s toc; do not link to it from other myst pages. It's a developer aid only.

---

## 00 — Prerequisites — [`roadmap/00_prerequisites.md`](roadmap/00_prerequisites.md)

| Label | Form | Section |
|-------|------|---------|
| `eq-ak-operator` | `ŷ = A (h^T x + (1−h^T) x_a)` | [Averaging-kernel operator](roadmap/00_prerequisites.md#prereqs-ak-operator) |

---

## I — Tier I (Gaussian family) — [`roadmap/01_tier1_gaussian.md`](roadmap/01_tier1_gaussian.md)

| Label | Form | Section |
|-------|------|---------|
| `eq-tier1-gaussian-plume` | Steady-state Gaussian plume with PBL image-source sum | [Gaussian plume](roadmap/01_tier1_gaussian.md#tier1-gaussian-plume) |
| `eq-tier1-gaussian-puff` | Sum of 3D Gaussian puffs advected by the wind | [Gaussian puff](roadmap/01_tier1_gaussian.md#tier1-gaussian-puff) |
| `eq-tier1-column-ak` | `y_model = A (∫c dz + c_bg)` — column + AK forward | [Column + AK](roadmap/01_tier1_gaussian.md#tier1-column-ak) |
| `eq-tier1-likelihood` | Heteroscedastic Gaussian observation likelihood | [Likelihood](roadmap/01_tier1_gaussian.md#tier1-likelihood) |
| `eq-tier1-amortized` | NPE / amortized predictor signature | [Amortized inference](roadmap/01_tier1_gaussian.md#tier1-amortized) |
| `eq-tier1-pbl-wellmixed` | Vertically well-mixed limit `c → Q / (ū L √(2π) σ_y) exp(−y²/2σ_y²)` | [Validation — PBL capping](roadmap/01_tier1_gaussian.md#tier1-validation) |

---

## II — Tier II (Lagrangian particle) — [`roadmap/02_tier2_lagrangian.md`](roadmap/02_tier2_lagrangian.md)

| Label | Form | Section |
|-------|------|---------|
| `eq-tier2-langevin` | Markov-1 Langevin SDE on particle velocity + position | [Langevin dynamics](roadmap/02_tier2_lagrangian.md#tier2-langevin) |
| `eq-tier2-dt-bound` | Adaptive timestep bound from $\tau_L$, CFL, diffusion | [Time-stepping](roadmap/02_tier2_lagrangian.md#tier2-langevin) |
| `eq-tier2-footprint` | Backward-mode source–receptor footprint integral | [Footprint definition](roadmap/02_tier2_lagrangian.md#tier2-footprint) |
| `eq-tier2-forward` | `y = A col_z(F q) + c_bg + ε` | [Forward observation operator](roadmap/02_tier2_lagrangian.md#tier2-inference) |
| `eq-tier2-likelihood` | `ε ~ N(0, R_retr + R_repr)` | [Likelihood](roadmap/02_tier2_lagrangian.md#tier2-likelihood) |
| `eq-tier2-posterior-mean` | Lognormal posterior mean — Gaussian–Gaussian linearised | [Closed form](roadmap/02_tier2_lagrangian.md#tier2-gaussian-closed-form) |
| `eq-tier2-posterior-cov` | Lognormal posterior covariance | same |
| `eq-tier2-amortized` | Predictor signature `(y, met) → p(log q(x))` | [Amortized inference](roadmap/02_tier2_lagrangian.md#tier2-amortized) |

---

## III — Tier III (Eulerian FV) — [`roadmap/03_tier3_eulerian.md`](roadmap/03_tier3_eulerian.md)

| Label | Form | Section |
|-------|------|---------|
| `eq-tier3-conservation` | $\rho$-weighted advection–diffusion–source–sink | [Conservation eq.](roadmap/03_tier3_eulerian.md#tier3-conservation) |
| `eq-tier3-forward` | `y_t = A_t col_z(c(S, c₀, t)) + c_bg + ε_t` | [Forward operator](roadmap/03_tier3_eulerian.md#tier3-inference) |
| `eq-tier3-4dvar-cost` | 3-term 4D-Var cost (source + IC + obs sum) | [4D-Var cost](roadmap/03_tier3_eulerian.md#tier3-cost) |
| `eq-tier3-likelihood` | Per-time heteroscedastic Gaussian | [Likelihood](roadmap/03_tier3_eulerian.md#tier3-likelihood) |
| `eq-tier3-adjoint` | Conservative adjoint transport equation | [Adjoint](roadmap/03_tier3_eulerian.md#tier3-adjoint) |
| `eq-tier3-incremental` | Incremental 4D-Var outer-iterate update | [Incremental 4D-Var](roadmap/03_tier3_eulerian.md#tier3-incremental) |
| `eq-tier3-control-xform` | $\boldsymbol{\chi} = \mathbf{B}^{-1/2}(S − S_b)$ control transform | [Control transform](roadmap/03_tier3_eulerian.md#tier3-control-transform) |
| `eq-tier3-amortized` | Sequence predictor signature | [Amortized inference](roadmap/03_tier3_eulerian.md#tier3-amortized) |

---

## RTM — RTM stack — [`roadmap/04_rtm_stack.md`](roadmap/04_rtm_stack.md)

| Label | Form | Section |
|-------|------|---------|
| `eq-rtm-radiance` | SWIR Beer–Lambert radiance | [Beer–Lambert](roadmap/04_rtm_stack.md#rtm-beer-lambert) |
| `eq-rtm-tau-total` | Two-way airmass-factor optical depth | same |
| `eq-rtm-tau` | Layer-integrated $\tau$ from HAPI cross-section | same |
| `eq-rtm-thermal-ir` | TIR variant (emission + path radiance) | [Thermal-IR addendum](roadmap/04_rtm_stack.md#rtm-thermal-ir) |
| `eq-rtm-state-vector` | Joint retrieval state vector | [Joint state](roadmap/04_rtm_stack.md#rtm-state-vector) |
| `eq-rtm-gauss-newton` | Iterative Gauss–Newton update | [Gauss–Newton](roadmap/04_rtm_stack.md#rtm-gauss-newton) |
| `eq-rtm-gain` | Gain matrix $\mathbf{G}_k$ + Jacobian definition | same |
| `eq-rtm-convergence` | Rodgers (2000) §5.7 convergence criterion | same |
| `eq-rtm-posterior-cov` | $\mathbf{S}^*$ posterior covariance | [Info content](roadmap/04_rtm_stack.md#rtm-info-content) |
| `eq-rtm-averaging-kernel` | $\mathbf{A} = \mathbf{G}\mathbf{K}$, DOFs = tr | same |
| `eq-rtm-info-content` | Shannon $H$ + posterior contraction $\Delta\mathbf{S}$ | same |
| `eq-rtm-factorised-lut` | Factorised LUT (gas × surf + scatt) decomposition | [Factorised LUT](roadmap/04_rtm_stack.md#rtm-emu-lut) |
| `eq-rtm-jax-grad` | `jax.grad ‖NeuralRTM(x) − y‖²` end-to-end gradient | [Emulator-based inference](roadmap/04_rtm_stack.md#rtm-emu-inference) |
| `eq-rtm-amortized` | Direct-retrieval predictor signature | [Amortized inference](roadmap/04_rtm_stack.md#rtm-amortized) |

---

## IV — Tier IV (Coupled E2E) — [`roadmap/05_tier4_coupled.md`](roadmap/05_tier4_coupled.md)

| Label | Form | Section |
|-------|------|---------|
| `eq-tier4-per-inst-forward` | Per-instrument coupled forward | [Per-instrument forward](roadmap/05_tier4_coupled.md#tier4-per-instrument-forward) |
| `eq-tier4-per-inst-likelihood` | Per-instrument noise covariance decomposition | same |
| `eq-tier4-state-vector` | Full coupled state vector enumeration | [State vector](roadmap/05_tier4_coupled.md#tier4-state-vector) |
| `eq-tier4-fused-operator` | Joint multi-instrument observation operator | [Multi-instrument fusion](roadmap/05_tier4_coupled.md#tier4-multi-instrument-fusion) |
| `eq-tier4-q-ou` | OU process for $Q(t)$ | [$Q(t)$](roadmap/05_tier4_coupled.md#tier4-q-of-t) |
| `eq-tier4-gradient` | End-to-end multi-instrument gradient | [Gradient](roadmap/05_tier4_coupled.md#tier4-gradient) |
| `eq-tier4-cost` | 3-term coupled 4D-Var cost (per-inst obs + prior + $Q(t)$ kernel) | [Cost function](roadmap/05_tier4_coupled.md#tier4-cost) |
| `eq-tier4-coupled-emulator` | Coupled emulator $g_\phi$ signature | [Coupled emulator](roadmap/05_tier4_coupled.md#tier4-coupled-emulator) |
| `eq-tier4-amortized` | Operational predictor signature | [Amortized inference](roadmap/05_tier4_coupled.md#tier4-amortized) |

---

## V — Tier V index — [`roadmap/06_tier5_population.md`](roadmap/06_tier5_population.md)

| Label | Form | Section |
|-------|------|---------|
| `eq-tier5-loglik` | TMTPP three-term log-likelihood | [Log-likelihood](roadmap/06_tier5_population.md#tier5-loglik) |
| `eq-tier5-mark` | Per-event mark integral $\int P_d L_i f \, \mathrm{d}Q$ | [Mark contribution](roadmap/06_tier5_population.md#tier5-mark-contribution) |
| `eq-tier5-importance-weights` | Importance-weighted MC estimator $f / \pi_\text{per-event}$ | same |

---

## V.A — Instantaneous emission — [`roadmap/06a_instantaneous.md`](roadmap/06a_instantaneous.md)

| Label | Form | Section |
|-------|------|---------|
| `eq-va-full-posterior` | Per-event posterior $\propto L_i \cdot \pi_\text{per-event}$ | [Full-physics posterior](roadmap/06a_instantaneous.md#va-regime-fullphysics) |
| `eq-va-wind-rescale` | Wind-source consistency rescaling | [Catalog Q](roadmap/06a_instantaneous.md#va-regime-catalog) |
| `eq-va-mark-integral` | Mark-likelihood integral form | [Mark likelihood](roadmap/06a_instantaneous.md#va-mark-likelihood) |
| `eq-va-importance-mc` | Importance-weighted MC estimator | same |
| `eq-va-thinned-rate` | Integrated thinned-rate term | [Detection-floor / non-detection](roadmap/06a_instantaneous.md#va-detection-floor) |
| `eq-va-multisource` | Multi-source product over within-overpass sources | [Multi-source](roadmap/06a_instantaneous.md#va-multi-source) |

---

## V.B — TMTPP — [`roadmap/06b_point_process.md`](roadmap/06b_point_process.md)

| Label | Form | Section |
|-------|------|---------|
| `eq-vb-lambda-constant` | Homogeneous Poisson | [Temporal](roadmap/06b_point_process.md#vb-temporal) |
| `eq-vb-lambda-diurnal` | Diurnal sinusoidal intensity | same |
| `eq-vb-lambda-step` | Step intensity (valve fail) | same |
| `eq-vb-lambda-decay` | Exponential decay (blowdown) | same |
| `eq-vb-lambda-hawkes` | Hawkes / self-exciting kernel | same |
| `eq-vb-lambda-lgcp` | Log-Gaussian Cox process | same |
| `eq-vb-pod-hill` | Hill function $P_d(Q) = 1/(1 + (Q_{50}/Q)^k)$ | [POD](roadmap/06b_point_process.md#vb-pod) |
| `eq-vb-pod-hier-prior` | Hierarchical prior on $(Q_{50}, k)$ | [POD calibration](roadmap/06b_point_process.md#vb-pod-calibration) |
| `eq-vb-loglik` | Canonical TMTPP log-likelihood | [Likelihood — canonical](roadmap/06b_point_process.md#vb-likelihood) |
| `eq-vb-mark-iw` | Importance-weighted MC for the mark integral | [Practical evaluation](roadmap/06b_point_process.md#vb-practical-eval) |
| `eq-vb-loglik-point` | Point-regime simplification | [Point regime](roadmap/06b_point_process.md#vb-point-regime) |

---

## V.C — Persistency — [`roadmap/06c_persistency.md`](roadmap/06c_persistency.md)

| Label | Form | Section |
|-------|------|---------|
| `eq-vc-wait-homogeneous` | $\mathbb{E}[\Delta t] = 1/\lambda_0$ | [Wait time](roadmap/06c_persistency.md#vc-wait-time) |
| `eq-vc-wait-inhomogeneous` | Inhomogeneous-Poisson wait integral | same |
| `eq-vc-occurrence-homogeneous` | $1 - e^{-\lambda_0 \Delta t}$ | [Occurrence](roadmap/06c_persistency.md#vc-occurrence) |
| `eq-vc-occurrence-inhomogeneous` | $1 - e^{-\int \lambda \, \mathrm{d}t}$ | same |
| `eq-vc-hawkes-bump` | Conditional intensity given prior detection | [Conditional intensity](roadmap/06c_persistency.md#vc-conditional-intensity) |
| `eq-vc-cumulative` | $\mathbb{E}[N(0,T)] = \int_0^T \lambda \, \mathrm{d}t$ | [Cumulative count](roadmap/06c_persistency.md#vc-cumulative-count) |

---

## V.D — Total emission — [`roadmap/06d_total_emission.md`](roadmap/06d_total_emission.md)

| Label | Form | Section |
|-------|------|---------|
| `eq-vd-mtotal` | $M_\text{total}(T) = \mathbb{E}[N_\text{true}] \cdot \mathbb{E}[Q]$ | [Corrected estimator](roadmap/06d_total_emission.md#vd-corrected-estimator) |
| `eq-vd-mnaive` | Biased naive estimator (sum over detected) | same |
| `eq-vd-ndetected` | $\mathbb{E}[N_\text{detected}]$ formula | same |
| `eq-vd-mtotal-posterior` | Posterior samples for $M_\text{total}$ | [Posterior](roadmap/06d_total_emission.md#vd-posterior) |
| `eq-vd-pod-union` | Union POD across $K$ satellites | [Multi-satellite fusion](roadmap/06d_total_emission.md#vd-multi-satellite-fusion) |

---

## Maintenance

When you add or remove a `{math}` block:

1. Add / remove the corresponding row in this file.
2. Re-run the count `grep -rE "^:label: eq-" projects/plume_simulation/notes/ | wc -l` and update the total at the top.
3. If a label changes, grep for `{eq}` references across the roadmap (`grep -rE "\{eq\}\`<old-label>\`" projects/plume_simulation/notes/`) and rename them too.
