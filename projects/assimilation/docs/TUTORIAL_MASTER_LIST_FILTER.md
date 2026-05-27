# filterax Tutorial Master List

A reconciled, exhaustive curriculum spanning what currently exists in **filterax**, **gaussx**, **pipekit**, and **research_notebook**, plus gaps surfaced from the filterax public API, open GitHub issues, and the `design_docs/` series under `filterax/docs/design_docs/`. Goal: the most complete ensemble-DA / data-assimilation tutorial sequence we could ship.

> GP / SVI tutorials live in [`../../gaussian_processes/TUTORIAL_MASTER_LIST.md`](../../gaussian_processes/TUTORIAL_MASTER_LIST.md). Cross-listed items (state-space GPs, ensemble VI, structured Gaussians, sigma points, shrinkage) are flagged 🔁.

**Legend** — Source columns:
- `F` = exists in filterax (`docs/notebooks/<name>`)
- `G` = exists in gaussx (`docs/notebooks/<name>`)
- `K` = exists in pipekit (`docs/notebooks/<name>`)
- `R` = exists in research_notebook (`projects/assimilation/notebooks/<path>`)
- `—` = does not exist yet (gap)

**Scope tag**: 🧱 fundamental · 🔬 research · 🌉 bridge · 🔁 cross-listed (GP master list)

**Refs column**: `gh#N` = open GitHub issue · `dd:path` = filterax `docs/design_docs/<path>` · `api:foo` = filterax exported symbol.

---

## Curriculum at a glance

- **Part 0 — Bayesian Filtering & DA Foundations**
  - 0.A — The filtering problem
  - 0.B — Linear-Gaussian Kalman filter
  - 0.C — The forecast → analysis → inflate cycle
  - 0.D — Variational DA (3D/4D-Var) contrast
  - 0.E — Information vs covariance form
- **Part 1 — Layer 0 Primitives**
  - 1.A — Ensemble statistics
  - 1.B — Gain & innovation
  - 1.C — Likelihood & innovation statistics
  - 1.D — Perturbations
  - 1.E — Localisation kernels
  - 1.F — Inflation primitives
  - 1.G — Patches & domain decomposition
- **Part 2 — Layer 1 Sequential Filters**
  - 2.A — Stochastic / perturbed-observation
  - 2.B — Deterministic square-root family
  - 2.C — Localised
  - 2.D — Symmetry-breaking variants
  - 2.E — Parametric (non-ensemble)
  - 2.F — Selection guide
- **Part 3 — Layer 2 Forecast-Analysis Loops**
  - 3.A — Protocols & extension points
  - 3.B — L2 model walkthroughs
  - 3.C — Inflator integration
  - 3.D — Stochastic key handling
- **Part 4 — Backward-Pass Smoothers**
  - 4.A — Sequential smoothers
  - 4.B — Square-root smoothers
  - 4.C — Iterative smoothers
  - 4.D — Selection & memory trade-offs
- **Part 5 — Ensemble Kalman Processes (Inversion & Sampling)**
  - 5.A — Inversion (EKI family)
  - 5.B — Sampling (EKS family)
  - 5.C — Parametric (UKI)
  - 5.D — Regularised / sparse
  - 5.E — Schedulers
- **Part 6 — Localisation, Inflation, Calibration**
  - 6.A — Why localisation
  - 6.B — Why inflation
  - 6.C — Adaptive variants
  - 6.D — Shrinkage estimators
- **Part 7 — Diagnostics & Verification**
  - 7.A — Basic spread / RMSE / rank
  - 7.B — Innovation diagnostics
  - 7.C — Reliability & sharpness
  - 7.D — Predictive likelihood
- **Part 8 — Differentiable DA**
  - 8.A — Theory & gradient stability
  - 8.B — `differentiable_assimilate` mechanics
  - 8.C — Training patterns
  - 8.D — Memory & remat
  - 8.E — Loss zoo
- **Part 9 — optax Integration**
  - 9.A — Process transforms
  - 9.B — Composition with optax chains
  - 9.C — Hybrid SGD + EKI
- **Part 10 — Sequential Variational Inference**
  - 10.A — Foundations
  - 10.B — Particle filters & SMC
  - 10.C — Variational SMC
  - 10.D — Ensemble VI for SSMs
  - 10.E — Amortised / streaming inference
  - 10.F — Sequential VB comparison
- **Part 11 — Ecosystem Integrations**
  - 11.A — gaussx (structured covariances)
  - 11.B — pipekit (orchestration)
  - 11.C — somax (SDE dynamics)
  - 11.D — geo_toolz / xr_assimilate (xarray)
  - 11.E — plumax (Tier IV)
- **Part 12 — Applied Case Studies *(research_notebook)***
  - 12.A — Canonical DA benchmarks
  - 12.B — Atmospheric & remote sensing
  - 12.C — Inverse problems
  - 12.D — Online / streaming
- **Part 13 — Reference Surfaces (Zoo)**
  - 13.A — Continuous-time
  - 13.B — Toy dynamical systems
  - 13.C — Hybrid Var-EnKF

---

## Part 0 — Bayesian Filtering & DA Foundations

### 0.A — The filtering problem

**Key equations / models:**
- Prior, dynamics, likelihood factorisation: $p(x_{0:T}, y_{1:T}) = p(x_0)\prod_t p(x_t \mid x_{t-1})\,p(y_t \mid x_t)$
- Filtering target: $p(x_t \mid y_{1:t})$
- Smoothing target: $p(x_t \mid y_{1:T})$, $T > t$
- Forecasting: $p(x_{t+h} \mid y_{1:t})$
- Chapman-Kolmogorov: $p(x_t \mid y_{1:t-1}) = \int p(x_t \mid x_{t-1})\, p(x_{t-1} \mid y_{1:t-1})\,dx_{t-1}$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.1 | The filtering problem from scratch — joint factorisation, filtering vs smoothing vs forecasting | — | 🧱 | pedagogical entry — graphical-model diagram, three target densities, recursion identity; sets the language for Parts 1–4 |
| 0.2 | Sequential Bayesian inference as natural-form addition | — | 🧱 🔁 | mirrors GP 0.6 / 0.4; conjugate update = $\eta_{t+1} = \eta_t + H^\top R^{-1} y_t$; batch = sequential = any order |

### 0.B — Linear-Gaussian Kalman filter

**Key equations / models:**
- Forecast: $\bar x^f_t = M_t \bar x^a_{t-1}$, $P^f_t = M_t P^a_{t-1} M_t^\top + Q_t$
- Analysis: $K_t = P^f_t H_t^\top S_t^{-1}$, $\bar x^a_t = \bar x^f_t + K_t(y_t - H_t \bar x^f_t)$, $P^a_t = (I - K_t H_t) P^f_t$
- Joseph form: $P^a_t = (I - K_t H_t)\, P^f_t\, (I - K_t H_t)^\top + K_t R_t K_t^\top$
- Log-marginal: $\log p(y_t \mid y_{1:t-1}) = -\tfrac{1}{2}\bigl[N_y \log 2\pi + \log|S_t| + v_t^\top S_t^{-1} v_t\bigr]$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.3 | Kalman filter from scratch — derivation, closed-form recursion, log-marginal likelihood | — | 🧱 | uses `SquareRootKF` for parametric ground truth; visual covariance ellipses; six-equation cheat sheet |
| 0.4 | Joseph-form covariance update — float32 stress test, PSD preservation | — | 🧱 🔁 | mirrors GP 0.5; four equivalent forms (standard / symmetric / information / Joseph) with PSD checks |

### 0.C — The forecast → analysis → inflate cycle

**Key equations / models:**
- Cycle structure: forecast (apply dynamics) → analysis (assimilate obs) → inflate (counteract sample collapse)
- Ensemble representation: $X \in \mathbb{R}^{N_e \times N_x}$, rows = members
- Sample covariance: $P_e = (N_e - 1)^{-1} (X - \bar X)^\top (X - \bar X)$, rank $\le N_e - 1$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.5 | Anatomy of one DA cycle — forecast / analysis / inflate, why each step exists | — | 🧱 | dd:architecture.md; three-panel diagram (prior cloud → posterior cloud → inflated cloud) |
| 0.6 | Why ensembles? — sample-covariance limits, rank ≤ $N_e − 1$, when ensemble beats parametric | — | 🧱 | eigenvalue-spectrum plot vs $N_e$; rank deficit and the null direction visualised |

### 0.D — Variational DA (3D/4D-Var) contrast

**Key equations / models:**
- 3D-Var cost: $J(x) = \tfrac{1}{2}(x - x^b)^\top B^{-1}(x - x^b) + \tfrac{1}{2}(y - H x)^\top R^{-1}(y - H x)$
- 4D-Var window: $J(x_0) = \tfrac{1}{2}\|x_0 - x^b_0\|^2_{B^{-1}} + \sum_t \tfrac{1}{2}\|y_t - H_t M_{1:t} x_0\|^2_{R_t^{-1}}$
- Adjoint vs autodiff: see Part 8

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.7 | 3D-Var vs Kalman — duality (Kalman = sequential 3D-Var with $B = P^f$) | — | 🧱 🌉 | pairs with R `projects/plume_simulation/notebooks/assimilation/00_3dvar_derivation.md`; minimisation = closed-form same answer |
| 0.8 | 4D-Var with adjoints — and how differentiable EnKF (Part 8) compares | — | 🧱 🌉 | dd:features/differentiable_da.md §4; cost / memory / Jacobian comparison table |

### 0.E — Information vs covariance form

**Key equations / models:**
- Information matrix $\Lambda = \Sigma^{-1}$, information vector $\eta = \Sigma^{-1} \mu$
- Conjugate update in natural form: $\eta_{t+1} = \eta_t + H^\top R^{-1} y_t$, $\Lambda_{t+1} = \Lambda_t + H^\top R^{-1} H$
- When to prefer information form: GMRF priors, banded $\Lambda$, sequential-by-obs updates

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.9 | Information vs covariance form — when each wins, conjugate-update identities | — | 🧱 🔁 | mirrors GP 0.4; sparsity-of-$\Lambda$ vs density-of-$\Sigma$ table; round-trip cost diagram |

---

## Part 1 — Layer 0 Primitives

filterax's pure-function building blocks. Every L1 / L2 algorithm composes from these.

### 1.A — Ensemble statistics

**Key equations / models:**
- $\bar x = N_e^{-1} \sum_j x^{(j)}$
- $X' = X - \mathbf{1}\bar x^\top$ (rows sum to zero)
- $P_e = (N_e - 1)^{-1} X'^\top X'$ — Bessel-corrected; returned as `gaussx.LowRankUpdate`
- Cross-cov $C^{xH} = (N_e - 1)^{-1} X'^\top (HX)'$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.1 | Ensemble mean, anomalies, sample covariance — the three primitives every filter starts from | — | 🧱 | api: `ensemble_mean`, `ensemble_anomalies`, `ensemble_covariance`; rank-deficit visualised; equivalence $P_e = X^\top X / (N_e-1)$ via QR / SVD |
| 1.2 | Cross-covariance for nonlinear $H$ — implicit derivative-free linearisation | — | 🧱 | api: `cross_covariance`; sanity check against finite-difference Jacobian; identity $C^{xH} = P_e H^\top$ for linear $H$ |
| 1.3 | Low-rank covariance as a structured operator — Woodbury identity preview | — | 🧱 🔁 | bridges to GP 1.B / 1.10; api: `gaussx.LowRankUpdate`; solve / logdet routing diagram |

### 1.B — Gain & innovation

**Key equations / models:**
- Ensemble Kalman gain: $K = C^{xH}(C^{HH} + R)^{-1}$
- Innovation covariance: $S = C^{HH} + R$ (low-rank update over $R$)
- Innovation: $v = y - \bar H X$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.4 | The ensemble Kalman gain — Bessel correction, Woodbury dispatch for structured $R$ | — | 🧱 | api: `kalman_gain`; dd:architecture.md; cost table for dense / diagonal / Toeplitz $R$ |
| 1.5 | Innovation covariance & gaussx structural dispatch — diag / low-rank / Toeplitz $R$ | — | 🧱 🔁 | api: `innovation_covariance`; pairs with GP 1.4 (Toeplitz) and 1.3 (Kronecker) |

### 1.C — Likelihood & innovation statistics

**Key equations / models:**
- $\log p(y \mid \text{forecast}) = -\tfrac{1}{2}[N_y \log 2\pi + \log|S| + v^\top S^{-1} v]$
- `InnovationStatistics` — packaged $v$, $S$, $\log p$ for diagnostics & training

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.6 | Predictive log-likelihood as the universal training signal | — | 🧱 | api: `log_likelihood`, `innovation_statistics`, `InnovationStatistics`; gradient sanity check ($-S^{-1}v$); feeds Parts 7 and 8 |

### 1.D — Perturbations

**Key equations / models:**
- $\epsilon^{(j)} \sim \mathcal{N}(0, R)$, structure-aware via `gaussx.root_decomposition`
- Diagonal-$R$ fast path: $\epsilon^{(j)}_k = z^{(j)}_k \sqrt{R_{kk}}$
- Determinism: identical key ⇒ identical draws

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.7 | Perturbed observations & R-aware sampling | — | 🧱 | api: `perturbed_observations`; fast-path vs dense-fallback paths, deterministic-key reproducibility |

### 1.E — Localisation kernels

**Key equations / models:**
- Gaspari-Cohn (compact, $C^4$ at origin, support $[0, 2r]$)
- Gaussian taper: $\exp(-d^2/2r^2)$
- SOAR: $(1 + d/r)\exp(-d/r)$
- Hard cutoff (non-differentiable!) & adaptive (Anderson 2007 / 2012)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.8 | Localisation taper zoo — visual + differentiability table | — | 🧱 🔁 | api: `gaspari_cohn`, `gaussian_taper`, `hard_cutoff`, `soar_taper`; pairs with GP 2.A (kernel zoo); plot $\rho(d/r)$ side-by-side |
| 1.9 | Generic `localize(cov, coords, taper_fn)` — assembling localised covariances | — | 🧱 | api: `localize`; ETKF-localized vs LETKF-localized comparison |
| 1.10 | Adaptive localisation (Anderson) — empirical correlation $\to$ taper | — | 🔬 | api: `adaptive_localization`; dd:features/localization_inflation.md |

### 1.F — Inflation primitives

**Key equations / models:**
- Multiplicative: $X'_a \leftarrow \lambda X'_a$, posterior cov $\lambda^2 P_a$
- Additive: $X_a \leftarrow X_a + \xi$, $\xi \sim \mathcal{N}(0, Q_\text{add})$
- RTPS (Whitaker-Hamill 2012): $X'_a \leftarrow X'_a [\alpha\, \sigma^f/\sigma^a + (1-\alpha)]$
- RTPP (Zhang 2004): $X'_a \leftarrow \alpha X'_f + (1-\alpha) X'_a$
- Ledoit-Wolf shrinkage: blend sample cov with structured target

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.11 | Multiplicative & additive inflation primitives | — | 🧱 | api: `inflate_multiplicative`, `inflate_additive`; spread-vs-step diagram |
| 1.12 | Relaxation: RTPS vs RTPP | — | 🧱 | api: `inflate_rtps`, `inflate_rtpp`; spread-recovery curves; $\alpha=0$ / $\alpha=1$ limiting cases |
| 1.13 | Adaptive inflation & Ledoit-Wolf shrinkage | — | 🔬 🔁 | api: `inflate_adaptive`, `ledoit_wolf_shrinkage`; pairs with GP 0.10 (jitter / shrinkage) |

### 1.G — Patches & domain decomposition

**Key equations / models:**
- Patch index sets $P_i \subseteq \{1, …, N_x\}$, obs-to-patch assignment, blending for overlapping patches
- Per-patch ETKF analysis assembled with smooth weighting in overlaps

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.14 | Domain decomposition — patches for high-dim spatial DA | — | 🔬 | api: `create_patches`, `assign_obs_to_patches`, `blend_patches`; dd:architecture.md; 2D grid example with overlap visualisation |

---

## Part 2 — Layer 1 Sequential Filters

Each filter as its own tutorial. Verified against the closed-form Kalman update on a linear-Gaussian problem; the existing `tests/test_filters.py` baselines are the template.

### 2.A — Stochastic / perturbed-observation

**Key equations / models:**
- $X^a_j = X^f_j + K(y + \epsilon^{(j)} - H X^f_j)$, $\epsilon^{(j)} \sim \mathcal{N}(0, R)$
- Monte Carlo gain noise scales as $1/\sqrt{N_e}$
- Per-window PRNG key: `jr.fold_in(base_key, step)`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.1 | Stochastic EnKF (Evensen 1994) — perturbed-obs analysis | — | 🧱 | api: `filters.StochasticEnKF`; dd:features/filters.md; MC-noise vs $N_e$ plot; pairs with 3.9 (key threading) |

### 2.B — Deterministic square-root family

**Key equations / models:**
- ETKF transform precision: $\tilde C = (N_e - 1) I + Y' R^{-1} Y'^\top$
- Rank-$N_y$ spectrum trick (no eigh of degenerate $(N_e, N_e)$ matrix) — fixed in [#82](https://github.com/jejjohnson/filterax/issues/82)
- $W_a = \sqrt{(N_e - 1) \tilde C^{-1}}$ applied to anomalies via $g(\tilde C) v = g(N_e-1) v + U_y \mathrm{diag}(g_\Delta) U_y^\top v$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.2 | ETKF (Bishop 2001) — ensemble transform, symmetric sqrt | — | 🧱 | api: `filters.ETKF`; dd:features/filters.md; rank-$N_y$ spectrum trick walked through; gradient-stability sanity check |
| 2.3 | EnSRF batch form (Whitaker & Hamill 2002) — separate mean & perturbation updates | — | 🧱 | api: `filters.EnSRF`; equivalence with ETKF in batch mode (Tippett 2003 §3) |
| 2.4 | Serial EnSRF — scalar obs processing, no eigh | — | 🧱 | api: `filters.EnSRF_Serial`; per-obs scalar gain; diagonal-$R$ requirement |
| 2.5 | ESTKF (Nerger 2012) — $(N_e − 1)$ error subspace, mean-preserving projection | — | 🧱 | api: `filters.ESTKF`; $L \in \mathbb{R}^{N_e \times (N_e-1)}$ Householder construction; reduced eigh cost |

### 2.C — Localised

**Key equations / models:**
- R-localisation (Hunt et al. 2007): inflate local $R^{-1}$ by per-obs taper $\rho$
- Per-grid-point local ETKF, vmapped across $N_x$ points
- Hard cutoff at radius: obs beyond $r$ excluded entirely (Gaspari-Cohn nonzero out to $2r$)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.6 | LETKF (Hunt 2007) — local ETKF with R-localisation | — | 🧱 | api: `filters.LETKF`; requires diagonal $R$; per-point compute diagram |
| 2.7 | LETKF hard-cutoff vs taper-only — why explicit cutoff matters | — | 🌉 | regression context for `test_letkf_hard_cutoff_at_radius`; far-obs invariance demo |

### 2.D — Symmetry-breaking variants

**Key equations / models:**
- $W_a^\text{rot} = W_a \cdot \Theta$, $\Theta \in O(N_e)$, $\Theta \mathbf{1} = \mathbf{1}$
- Counteracts preferred-direction drift over many cycles

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.8 | ETKF_Livings — mean-preserving random rotation | — | 🔬 | api: `filters.ETKF_Livings`; rotation construction via Householder + random $O(N_e-1)$; cov preserved, ensemble realisations differ |

### 2.E — Parametric (non-ensemble)

**Key equations / models:**
- Cholesky-form mean & covariance propagation
- Marginal likelihood from the innovation sequence
- PSD-preserving across $T$ steps by construction

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.9 | SquareRootKF — Cholesky-form parametric KF | — | 🧱 🔁 | api: `filters.SquareRootKF`; ground truth for Part 0.B; pairs with GP 8.D (parametric Kalman) |

### 2.F — Selection guide

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.10 | Filter selection — when each L1 variant wins (table + worked examples) | — | 🧱 | reads like an extended `design_docs/decisions.md` D-row; deterministic vs stochastic, batch vs serial, localised vs global |

---

## Part 3 — Layer 2 Forecast-Analysis Loops

### 3.A — Protocols & extension points

**Key equations / models:**
- `AbstractDynamics(state, t0, t1) -> state` — vmap over members
- `AbstractObsOperator(state) -> obs`
- `AbstractInflator(particles, forecast, **kwargs) -> particles`
- `AbstractLocalizer(cov, coords) -> cov`
- `AbstractScheduler.get_dt(state)` for processes

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.1 | The protocol family — `AbstractDynamics` / `AbstractObsOperator` / `AbstractInflator` / `AbstractLocalizer` / `AbstractScheduler` | — | 🧱 | api: `filterax._src._protocols`; dd:architecture.md; class diagram with extension points |
| 3.2 | Plugging in a JAX dynamics model | — | 🧱 | identity / linear / Lorenz-63 / SDE wrappers; pure-function rule |
| 3.3 | Plugging in a nonlinear obs operator — neural decoder warm-up for Part 8 | — | 🌉 | dd:features/differentiable_da.md §6.B; equinox-based `eqx.nn.MLP` example |

### 3.B — L2 model walkthroughs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.4 | L2 ETKF end-to-end — forecast → analysis → likelihood logging | — | 🧱 | api: `ETKF`; mirrors `test_l2_etkf_assimilate_smoke`; `AssimilationResult` field walkthrough |
| 3.5 | L2 EnSRF & L2 StochasticEnKF — when batch vs perturbed-obs matters | — | 🧱 | api: `EnSRF`, `StochasticEnKF`; side-by-side spread plots |
| 3.6 | L2 LETKF with `state_coords` / `obs_coords` | — | 🧱 | api: `LETKF`; 1D-grid worked example with localisation radius sweep |

### 3.C — Inflator integration

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.7 | Adding `MultiplicativeInflator` / `RTPS` / `RTPP` to the L2 loop | — | 🧱 | api: `MultiplicativeInflator`, `RTPS`, `RTPP`; spread-trajectory comparison |
| 3.8 | `AdditiveInflator` & per-cycle key threading | — | 🔬 | api: `AdditiveInflator`; `jr.fold_in(base_key, step)` pattern |

### 3.D — Stochastic key handling

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.9 | `jr.fold_in` per window — why naive `StochasticEnKF` repeats draws (regression for `test_stochastic_enkf_l2_uses_independent_keys_per_window`) | — | 🌉 | api: `StochasticEnKF.assimilate`; demo of identical-draw bug without folding |

---

## Part 4 — Backward-Pass Smoothers

### 4.A — Sequential smoothers

**Key equations / models:**
- Smoother gain: $G_t = C^{af}_{t,t+1}\,(C^{ff}_{t+1})^{-1}$
- Recursion: $X^s_t = X^a_t + G_t(X^s_{t+1} - X^f_{t+1})$
- Dual ensemble-space form: $G_t (X^s_{t+1} - X^f_{t+1}) = D F^\top (F F^\top)^+ A$ — avoids materialising $(N_x, N_x)$ cov

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.1 | EnKS (Evensen & van Leeuwen 2000) — standard backward pass | — | 🧱 | api: `smoothers.EnKS`; dd:features/smoothers.md; backward-scan diagram; final-time identity |
| 4.2 | EnsembleRTS — RTS interpretation, model-error placeholder | — | 🧱 | api: `smoothers.EnsembleRTS`; equivalence with EnKS without explicit $Q$ |
| 4.3 | FixedLagSmoother — windowed lookahead, online interpretation | — | 🔬 | api: `smoothers.FixedLagSmoother`; lag=0 / lag=T-1 limits; rolling-buffer interpretation |

### 4.B — Square-root smoothers

**Key equations / models:**
- Decompose mean + perturbation; apply symmetric sqrt to anomalies in ensemble space
- $\Lambda = I + K_e (D^\top D - F^\top F) K_e^\top$, $K_e = (F F^\top)^+ F$
- Smoothed perts $X^{s'}_t = \Lambda^{1/2} A_t$ live in the row span of $A_t$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.4 | EnsembleSqrtSmoother — deterministic sqrt backward pass | — | 🔬 | api: `smoothers.EnsembleSqrtSmoother`; dd:features/smoothers.md Gap 4; perts-in-column-span demo |

### 4.C — Iterative smoothers

**Key equations / models:**
- Chen-Oliver IES update with prior anchor:
  $\theta^j_{i+1} = (1 - \alpha)\,\theta^j_i + \alpha\bigl[\theta^j_0 + K_i(y + \epsilon^{(j)} - G(\theta^j_i))\bigr]$
- $K_i = C^{\theta G}_i (C^{GG}_i + \Gamma_y)^{-1}$ recomputed each iteration

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.5 | IES (Chen & Oliver 2013) — iterative ensemble smoother for inverse problems | — | 🔬 | api: `smoothers.IES`; dd:features/smoothers.md Gap 5; anchor-to-$\theta_0$ visualisation; $\alpha$ ablation |

### 4.D — Selection & memory trade-offs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.6 | Smoother selection guide — offline vs online, deterministic vs iterative, memory budget | — | 🧱 | extends `features/smoothers.md` §5 comparison table; flow chart for picking smoother |

---

## Part 5 — Ensemble Kalman Processes (Inversion & Sampling)

### 5.A — Inversion (EKI family)

**Key equations / models:**
- EKI update: $\theta^j_{n+1} = \theta^j_n + \Delta t_n\, C^{\theta G}_n (C^{GG}_n + \Delta t_n^{-1} \Gamma)^{-1}(y - G(\theta^j_n))$
- Tempered $\Delta t^{-1}\Gamma$ noise as low-rank update over $\Gamma$
- Algo-time $\to 1$ collapse: spread $\to 0$, MAP recovered

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.1 | EKI (Iglesias, Law & Stuart 2013) — iterative ensemble inversion | — | 🧱 | api: `EKI`, `processes.EKI`; dd:features/processes.md; one-step Kalman equivalence on the sample cov |
| 5.2 | TEKI — Tikhonov-regularised EKI with prior pull | — | 🌉 | api: `processes.TEKI`; augmented-identity block; unidentifiable-parameter shrinkage demo |
| 5.3 | GNKI — Gauss-Newton with explicit ensemble Jacobian | — | 🔬 | api: `processes.GNKI`; requires $J > N_p$; one-step linear-Gaussian convergence |
| 5.4 | ETKI — deterministic / sqrt EKI variant | — | 🔬 | api: `processes.ETKI`; deterministic transform analog |
| 5.5 | SparseInversion — L¹ proximal soft-threshold on EKI step | — | 🔬 | api: `processes.SparseInversion`; Schneider-Stuart-Wu 2022; inactive-parameter-to-zero demo |

### 5.B — Sampling (EKS family)

**Key equations / models:**
- EKS / interacting Langevin: $\theta^j_{n+1} = \theta^j_n + \Delta t\, C^{\theta G}_n (\ldots) + \sqrt{2\Delta t\, C^{\theta\theta}_n}\,\xi^j_n$
- Approaches the Bayesian posterior as $\Delta t \to 0$, $N_e \to \infty$
- Ergodic — spread doesn't collapse

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.6 | EKS (Garbuno-Iñigo 2020) — ergodic sampler, no collapse | — | 🧱 🔁 | api: `EKS`, `processes.EKS_Process`; cross-listed with GP 12.x (ensemble VI); spread-vs-time vs EKI plot |

### 5.C — Parametric (UKI)

**Key equations / models:**
- Sigma-point propagation: $\{\theta_k\} = \mu \pm \sqrt{(N_p + \kappa)\Sigma}_i$
- Closed-form mean + cov update; no random perturbations
- Exact second-moment match in linear-Gaussian limit

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.7 | UKI — unscented Kalman inversion, parametric mean / cov | — | 🧱 | api: `UKI`, `processes.UKI`; dd:features/processes.md; sigma-point cloud diagram |
| 5.8 | Sigma-point utilities — reusable for filter ops in Part 2 | — | 🧱 🔁 | api: `processes.sigma_points`; mirrors GP 6.3; reconstruction-of-mean-and-cov sanity check |

### 5.D — Regularised / sparse

Covered in 5.A (TEKI, SparseInversion); listed here for navigation.

### 5.E — Schedulers

**Key equations / models:**
- Fixed: $\Delta t_n = \text{const}$
- Data-misfit controller (Iglesias 2016): adapt $\Delta t$ from current misfit, freeze past $\text{algo\_time}=1$
- EKS-stable: $\Delta t = h / (\text{trace}(C^{GG}) + \delta)$ (avoids ensemble blow-up)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.9 | Scheduler zoo — fixed, data-misfit, EKS-stable | — | 🧱 | api: `FixedScheduler`, `DataMisfitController`, `EKSStableScheduler`; convergence-trajectory comparison |
| 5.10 | DataMisfitController past convergence — `algo_time ≥ 1` safety | — | 🌉 | regression context for `test_eki_update_is_finite_after_algo_time_one`; $\Delta t = 0$ floor; no-NaN guarantee |

---

## Part 6 — Localisation, Inflation, Calibration

### 6.A — Why localisation

**Key equations / models:**
- Spurious correlations: $|C_{ij}| \sim 1/\sqrt{N_e}$ for unrelated state pairs
- Sample-cov rank ≤ $N_e − 1$ in state space; rank-deficient inverse undefined without regularisation

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.1 | Spurious correlations & sample-cov rank — visual + spectrum demo | — | 🧱 | dd:features/localization_inflation.md; correlation heat-map at small / large $N_e$ |
| 6.2 | R-localisation vs B-localisation — when each is correct | — | 🧱 | Hunt 2007 §2; B-loc preserves PSD only with specific tapers; R-loc requires diagonal $R$ |

### 6.B — Why inflation

**Key equations / models:**
- Filter divergence: posterior cov collapses → obs rejected → estimates drift
- Recovery: $\lambda > 1$ multiplicative or $Q_\text{add}$ additive

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.3 | Filter divergence — overconfident analysis rejects obs | — | 🧱 | dd:features/localization_inflation.md; trajectory-plot demo with / without inflation |
| 6.4 | Multiplicative vs RTPS vs RTPP — when each wins | — | 🧱 | calibration table; spread-trajectory comparison; failure modes |

### 6.C — Adaptive variants

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.5 | Adaptive localisation in operation | — | 🔬 | api: `adaptive_localization`; empirical-correlation threshold demo |
| 6.6 | Adaptive inflation (Anderson 2007 / Miyoshi 2011) | — | 🔬 | api: `inflate_adaptive`; observation-space hierarchical inflation update |

### 6.D — Shrinkage estimators

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.7 | Ledoit-Wolf shrinkage for ensemble cov | — | 🔬 🔁 | api: `ledoit_wolf_shrinkage`; pairs with GP 0.11 (jitter / safe Cholesky); analytic shrinkage intensity |

---

## Part 7 — Diagnostics & Verification

### 7.A — Basic spread / RMSE / rank

**Key equations / models:**
- Ensemble spread: $\sigma_t = \sqrt{(N_e-1)^{-1}\sum_j \|x^{(j)}_t - \bar x_t\|^2 / N_x}$
- RMSE: $\sqrt{\langle (\bar x_t - x^\text{true}_t)^2\rangle_t}$
- Rank-histogram of obs vs ensemble (Talagrand)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.1 | Ensemble spread & RMSE — minimal pair for "is the filter alive" | — | 🧱 | dd:features/diagnostics.md; trajectory plots; spread-RMSE ratio reading |
| 7.2 | Rank histograms & reliability | — | 🧱 | Talagrand diagrams; under- / well- / over-dispersive signatures |

### 7.B — Innovation diagnostics

**Key equations / models:**
- Mahalanobis: $v^\top S^{-1} v \sim \chi^2_{N_y}$ if filter is consistent
- Desroziers (2005): $\mathbb{E}[(y - H\bar x^a)(y - H\bar x^f)^\top] = R$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.3 | Innovation chi-squared & Mahalanobis | — | 🧱 | api: `innovation_statistics`; pass/fail thresholds; cycle-averaged plots |
| 7.4 | Desroziers diagnostic — observation-error tuning | — | 🔬 | dd:features/diagnostics.md; recovering $R$ from posterior innovations |

### 7.C — Reliability & sharpness

**Key equations / models:**
- Spread-skill: $\sigma_t \approx \text{RMSE}_t$ when the filter is calibrated
- CRPS: $\int_{-\infty}^\infty \bigl(F(z) - \mathbf{1}\{y \le z\}\bigr)^2 dz$
- DFS: $\text{tr}(I - P^a (P^f)^{-1})$
- ESS: $(\sum w^{(j)})^2 / \sum w^{(j)2}$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.5 | Spread-skill relationship | — | 🌉 | calibration scatter $\sigma_t$ vs $|\text{err}_t|$ |
| 7.6 | CRPS — calibration without Gaussianity | — | 🌉 🔁 | mirrors GP calibration tutorials; ensemble-vs-pointwise CRPS |
| 7.7 | DFS — degrees of freedom for signal | — | 🔬 | observability metric; per-obs contribution table |
| 7.8 | Effective sample size (ESS) | — | 🔬 🔁 | bridges to particle filters in Part 10; degeneracy threshold |

### 7.D — Predictive likelihood

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.9 | Log predictive density as the training loss — connects to Part 8 | — | 🧱 | api: `InnovationStatistics.log_likelihood`; running sum over windows; sanity gradient sign |

---

## Part 8 — Differentiable DA

See dd:features/differentiable_da.md.

### 8.A — Theory & gradient stability

**Key equations / models:**
- Pure-function filter ⇒ `jax.grad` flows for free
- Stochastic vs deterministic filters: perturbed-obs draws inject non-smooth randomness
- eigh degeneracy: $(N_e, N_e)$ transform precision has $N_e − N_y$ repeated eigenvalues → NaN gradient
- Rank-$N_y$ QR-based fix: stable at any $N_e$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.1 | Why differentiable — learning dynamics, obs ops, hyperparams end-to-end | — | 🧱 | dd:features/differentiable_da.md §1; four motivating use cases |
| 8.2 | Stochastic vs deterministic filters under `grad` — eigh degeneracy & the rank-$N_y$ trick | — | 🔬 | regression context for [#82](https://github.com/jejjohnson/filterax/issues/82); api: `_etkf_inner_spectrum`; before / after gradient plot |

### 8.B — `differentiable_assimilate` mechanics

**Key equations / models:**
- $X^a_t = \text{EnKF}(X^f_t = f_\theta(X^a_{t-1}), y_t)$ unrolled as `jax.lax.scan`
- Memory under reverse-mode: $O(T \cdot N_e \cdot N_x)$ → $O(\sqrt{T}\cdot N_e \cdot N_x)$ with checkpoint

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.3 | The scan + vmap + remat idiom — single fused XLA `While` | — | 🧱 | api: `differentiable_assimilate`; dd:features/differentiable_da.md §8 |
| 8.4 | Carry-dtype unification & extension kwargs (LETKF coords) | — | 🌉 | regression context for `test_diff_assimilate_handles_mixed_time_dtypes`; mixed-dtype trace error reproduction |

### 8.C — Training patterns

**Key equations / models:**
- Pattern A: $\theta_f$ via $\nabla_{\theta_f} (-\sum_t \log p(y_t \mid \text{forecast}_t))$
- Pattern B: $\theta_H$ via the same loss, gradient flows through obs operator
- Pattern C: $\rho$, $r_\text{loc}$, $R$ diag via reparameterised `log_factor` etc.

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.5 | Pattern A — learn dynamics parameters (Neural ODE through filter) | — | 🔬 | dd:features/differentiable_da.md §6.A; loss-vs-epoch curve; gradient-sign sanity |
| 8.6 | Pattern B — learn observation operator (neural decoder) | — | 🔬 | dd:features/differentiable_da.md §6.B; neural RTM example (plumax Tier IV v2) |
| 8.7 | Pattern C — meta-learn inflation / localisation radius / $R$ diag | — | 🔬 | dd:features/differentiable_da.md §6.C; constrained-via-`exp` reparameterisation |

### 8.D — Memory & remat

**Key equations / models:**
- `jax.checkpoint(step)` placed on the scan body
- Binomial checkpointing schedule: $O(\sqrt{T})$ memory at 2-3× compute
- ROAD-EnKF (Chen et al. 2023): $O(1)$ memory, approximate gradient

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.8 | `jax.checkpoint` placement — $O(\sqrt{T})$ memory under reverse-mode | — | 🧱 | dd:features/differentiable_da.md §5.1; checkpoint-on-body vs checkpoint-on-scan |
| 8.9 | ROAD-EnKF — local-gradient approximation, $O(1)$ memory | — | 🔬 | dd:features/differentiable_da.md §6.D; not yet implemented — gap |

### 8.E — Loss zoo

**Key equations / models:**
- NLL: $\sum_t -\log p(y_t \mid \bar x^f_t)$
- MSE: $\sum_t \|v_t\|^2$
- CRPS: $\sum_t \text{CRPS}(\text{ens}_t, y_t)$
- Spread-skill: $\sum_t (\sigma_t - \text{err}_t)^2$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.10 | NLL vs MSE vs CRPS vs spread-skill — pick your gradient signal | — | 🧱 | dd:features/differentiable_da.md §3; calibration vs accuracy trade-off table |

---

## Part 9 — optax Integration

### 9.A — Process transforms

**Key equations / models:**
- Each process exposed as an `optax.GradientTransformation`: `init(params) -> state`, `update(grad, state, params) -> (updates, state)`
- The "gradient" slot is unused; updates come from the ensemble Kalman process internally
- `optax.apply_updates(params, updates)` advances the parameter mean

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.1 | `filterax.optax.eki` — EKI as a gradient transform | — | 🧱 | api: `filterax.optax.eki`; dd:features/optax_ekp.md; three-iter convergence smoke |
| 9.2 | `filterax.optax.eks` — EKS as a gradient transform | — | 🧱 | api: `filterax.optax.eks`; per-step key-advance demonstration |
| 9.3 | `filterax.optax.uki` — UKI with parametric carry | — | 🌉 | api: `filterax.optax.uki`; mean / covariance both updated |

### 9.B — Composition with optax chains

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.4 | Composing with `optax.chain` — gradient clipping, masking, scheduling on top of EKI | — | 🌉 | dd:features/optax_ekp.md; `clip_by_global_norm` example; mask-by-param-name |

### 9.C — Hybrid SGD + EKI

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.5 | Hybrid pipelines — SGD on neural-net params + EKI on physics params, single `optax.chain` | — | 🔬 | bridges to Part 8 (Pattern A/B); per-leaf transform via `optax.multi_transform` |

---

## Part 10 — Sequential Variational Inference

The bridge into broader filtering / sequential-VI work. Each tutorial sits next to a filterax / pyrox primitive and points at the GP master list where overlap exists.

### 10.A — Foundations

**Key equations / models:**
- Sequential VI: $q_t(x_t) \approx p(x_t \mid y_{1:t})$ updated as obs arrive
- Recursive ELBO: $\mathcal{L}_t = \mathbb{E}_{q_t}[\log p(y_t \mid x_t)] - \text{KL}(q_t \Vert q_{t-1}^\text{forecast})$
- Connection to Kalman: linear-Gaussian $q$ ⇒ exact Kalman recursion

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.1 | Sequential VI from scratch — recursive ELBO, Gaussian limit = Kalman | — | 🧱 🔁 | pairs with GP 6.14 (variational guides); duality diagram |
| 10.2 | Variational EnKF — interpreting ETKF as ELBO ascent | — | 🔬 | research bridge; minimisation-vs-update derivation |

### 10.B — Particle filters & SMC

**Key equations / models:**
- Importance sampling + resampling: $w^{(j)} \propto p(y_t \mid x^{(j)}_t)$
- ESS: $N_\text{eff} = (\sum w^{(j)})^2 / \sum (w^{(j)})^2$
- Resampling schemes: multinomial, stratified, systematic

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.3 | Bootstrap particle filter — proposal = prior | — | 🧱 | not in filterax core; build from primitives; reweighting & degeneracy demo |
| 10.4 | Auxiliary particle filter | — | 🌉 | optimal proposal $q(x_t \mid x_{t-1}, y_t)$; variance-reduction plots |
| 10.5 | SMC samplers — annealed posteriors | — | 🔬 🔁 | overlaps GP MCMC tutorials; tempered-sequence visualisation |

### 10.C — Variational SMC

**Key equations / models:**
- VSMC ELBO (Naesseth 2018): tighter than IWAE via SMC structure
- Filtering variational objectives (Maddison 2017): per-step lower-bound

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.6 | Variational SMC (Naesseth 2018) — amortised proposal trained against ELBO | — | 🔬 | research bridge; learnable-proposal example |
| 10.7 | Filtering variational objectives (Maddison 2017) — IWAE-style filtering | — | 🔬 | comparison with bootstrap PF on Lorenz |

### 10.D — Ensemble VI for SSMs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.8 | EKS as ensemble VI — Garbuno-Iñigo 2020 reading | — | 🔬 🔁 | cross-listed with GP 12.x; api: `EKS`; ergodicity & posterior recovery |
| 10.9 | Reich-style ensemble VI — coupling-based posterior approximation | — | 🔬 | research; OT-coupling-as-resampling |

### 10.E — Amortised / streaming inference

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.10 | Amortised filter — neural encoder $q_\phi(x_t \mid y_{1:t})$ trained through `differentiable_assimilate` | — | 🔬 | builds on Part 8; encoder ELBO recipe |
| 10.11 | Streaming amortised inference — online updates without retraining | — | 🔬 | bounded-memory amortisation; recurrent encoder |

### 10.F — Sequential VB comparison

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.12 | Sequential VB (Beal & Ghahramani 2003; Honkela 2003) — when classical VB beats ensemble | — | 🌉 🔁 | pairs with GP 6.16 (CVI); cost / accuracy table |

---

## Part 11 — Ecosystem Integrations

### 11.A — gaussx (structured covariances)

**Key equations / models:**
- $S = C^{HH} + R$ as `gaussx.LowRankUpdate` → Woodbury solve in $O(N_e^2 N_y + N_e^3)$ vs $O(N_y^3)$
- Diagonal / Toeplitz / Kronecker $R$ never densified

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.1 | Diagonal $R$ → Woodbury gain — never densify | — | 🧱 🔁 | api: `gaussx.LowRankUpdate`, `gaussx.solve_rows`; pairs with GP 1.9 |
| 11.2 | Toeplitz / Kronecker $R$ — spatial obs noise | — | 🌉 🔁 | dd:integrations/geostack.md; pairs with GP 1.4 / 1.3 |

### 11.B — pipekit (orchestration)

**Key equations / models:**
- D11 wrapper pattern: `FilterAsAnalysisStep`, `DynamicsAsForwardModel`
- Sequential / Graph / Cycle composition of analysis steps

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.3 | filterax filter as `pipekit.AnalysisStep` (wrapper pattern D11) | — | 🧱 | dd:integrations/pipekit.md; api: `FilterAsAnalysisStep` (user wrapper) |
| 11.4 | Sequential / Graph / Cycle composition | — | 🌉 | full multi-step Tier IV pipeline; pipekit-side notebook |

### 11.C — somax (SDE dynamics)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.5 | `somax` SDE as `AbstractDynamics` | — | 🌉 | dd:examples/integration.md; stochastic forward model worked example |

### 11.D — geo_toolz / xr_assimilate (xarray)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.6 | xarray-aware DA — coordinate-driven assimilation | — | 🌉 🔁 | pairs with coordax tutorials; named-axis filter API |

### 11.E — plumax (Tier IV)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.7 | Multi-instrument methane retrieval — `JointObsOperator`, `SequentialAssimilation`, `GeoLocalizer`, fixed-lag smoother | — | 🔬 | dd:integrations/plumax.md; the canonical end-to-end demo |

---

## Part 12 — Applied Case Studies *(research_notebook)*

### 12.A — Canonical DA benchmarks

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.1 | Lorenz-63 toy DA — ETKF / EnSRF / LETKF on the standard benchmark | — | 🧱 | mirrors gaussx `ensemble_kalman` notebook; RMSE-vs-time across filters |
| 12.2 | Lorenz-96 spatially-extended DA — LETKF + adaptive inflation | — | 🔬 | operational analogue; per-grid-point posterior |

### 12.B — Atmospheric & remote sensing

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.3 | 1D heat-equation DA — pedagogical PDE state-space | — | 🧱 | ground-truth-from-PDE; visualised assimilation |
| 12.4 | Plume dispersion DA — `plume_simulation/matched_filter` → EnKF | — | 🔬 | pairs with `projects/plume_simulation`; emission-rate estimation |
| 12.5 | Multi-instrument retrieval — TROPOMI/EMIT/GHGSat-style joint observation | — | 🔬 | extends 11.7; per-instrument $H$ stack |

### 12.C — Inverse problems

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.6 | Inverse heat conduction — EKI parameter estimation | — | 🔬 | thermal-conductivity recovery; ensemble-vs-truth contour plot |
| 12.7 | Subsurface flow history matching — IES end-to-end | — | 🔬 | reservoir engineering analogue; full Chen-Oliver iteration |

### 12.D — Online / streaming

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.8 | Online streaming smoother — fixed-lag with rolling buffer, memory budget | — | 🔬 | api: `FixedLagSmoother`; live-data DA recipe |
| 12.9 | Differentiable end-to-end — learn dynamics through 100-step assimilation | — | 🔬 | full Pattern A demo with checkpointing |

---

## Part 13 — Reference Surfaces (Zoo)

Explicitly *not* maintained as core API; lives under `zoo/` (gap, planned for [#61](https://github.com/jejjohnson/filterax/issues/61)) for educational / benchmarking use.

### 13.A — Continuous-time

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.1 | Continuous-time EnKF — derivation from continuous Kalman | — | 🔬 | zoo; SDE form of the ensemble update |
| 13.2 | 4D-EnKF — observation-time-aware ensemble update | — | 🔬 | zoo; multi-obs-window equivalence |

### 13.B — Toy dynamical systems

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.3 | Toy systems catalog — Lorenz-63 / Lorenz-96, sinusoid, double-well, brownian, OU | — | 🌉 | benchmarking fixtures; reusable `AbstractDynamics` implementations |

### 13.C — Hybrid Var-EnKF

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.4 | EnVar — hybrid 3D/4D-Var-EnKF (Bocquet 2010) | — | 🔬 | bridges Part 0.D and Parts 2-3; static-B-plus-ensemble-B cost |

---

## References

```{bibliography}
:filter: docname in docnames
```
