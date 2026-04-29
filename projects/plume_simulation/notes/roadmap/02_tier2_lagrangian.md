# Tier II — Lagrangian particle transport

**Forward model:** stochastic particle trajectories driven by wind + turbulence. The bridge between the analytical Tier I (no real wind variability) and the full PDE Tier III (no statistical efficiency). This is what FLEXPART and STILT do operationally.

This tier is **not yet started in `plume_simulation`** — module layout proposed below.

---

## (1) Simple model

### Stochastic dynamics — Markov-1 Langevin

Operational LPDM (FLEXPART, STILT, HYSPLIT) uses **Markov-1**: a Langevin equation on particle *velocity*, then position from velocity. We commit to Markov-1 because it's required for near-source super-emitter work and for non-stationary turbulence:

```
dv = a(x, v, t) dt + b(x, t) dW,    dW ~ N(0, dt) ⊗ I_3
dx = (u(x,t) + v) dt
```

- `u(x,t)` — mean wind from the [met field](00_prerequisites.md#1--forcing--meteorology) (WRF / ERA5 / HRRR).
- `v` — turbulent velocity perturbation, the actual stochastic state.
- `a, b` — drift and diffusion coefficients constructed from the Reynolds-stress tensor `σ_ij(x,t)` and Lagrangian timescale `τ_L(x,t)`. Reference: Wilson & Sawford 1996 (well-mixed condition).
- `σ_ij`, `τ_L` come from MO similarity (`u*`, `L_obukhov`, `z₀`) plus WRF TKE — see [MO prereqs](00_prerequisites.md#monin-obukhov-similarity). Markov-0 (random-displacement on position only) is **not** the model; it shows up as the well-mixed limit at long Δt.

### Sub-grid wind interpolation

WRF velocities are gridded at hourly snapshots; particles are continuous in space and time. Naïve bilinear interpolation of `u` is **not divergence-free**, which causes mass-conservation drift over long integrations. Use a **C-grid-aware interpolator** that reconstructs `u` consistently with the WRF staggered grid (operational STILT pattern). Document the chosen scheme on the integrator API; assume divergence preservation is load-bearing.

### Time-stepping

Default to **Euler-Maruyama** with adaptive Δt bounded by `Δt < min(τ_L, Δx / |u|, Δx² / σ²)`. Higher-order Milstein only if convergence diagnostics flag bias. Δt is an open parameter — see [open questions](#open-questions) for the operational target.

### Run modes

- **Forward mode:** release `N` particles from the source, track concentration by binning particle density on the analysis grid.
- **Backward mode (footprint):** release receptors backward in time → **source–receptor sensitivity matrix** `F`. The FLEXPART/STILT paradigm and the workhorse of regional-scale inversions.

### Footprint definition (formal)

For a receptor `r` and a candidate source cell `s`:

```
F(r, s) = ∫_{t_obs − T}^{t_obs}  (1 / V_s · ρ_air(s,t))  · 1[z_part(t) < f_PBL · L(s,t)]  dt
```

— integrated over particle residence time below a fraction `f_PBL ≈ 0.5` of the local PBL height `L(s,t)`. Units: `s · m² · kg⁻¹` (mixing-ratio per kg/m²/s of surface flux). The footprint is a **probability density** over candidate source cells, not an indicator: properly normalised by particle volume and air density. Numerical scaling matters because the values are tiny — work in log-space when storing.

For an overpass with `n_obs` columns and `n_grid` candidate source cells, `F` is shape `(n_obs, n_grid)` and typically 90%+ sparse.

---

## (2) Model-based inference

### Forward model with observation operator

The full forward used by inference is **not** `y = F q + ε`. It's:

```
y = AK · column_z(F q) + c_bg + ε
```

- `F q` gives a 3D concentration field from the source vector.
- `column_z` collapses to an XCH₄ column.
- `AK` applies the satellite [averaging kernel](00_prerequisites.md#averaging-kernel-operator).
- `c_bg` is the regional background — mandatory, not optional. Same critique as Tier I: predicted enhancement vs. absolute-column observation must be reconciled day-1.

### Likelihood model

```
ε ~ N(0, R),   R = R_retr + R_repr
```

- `R_retr` — heteroscedastic per-pixel from the L2 retrieval-error map.
- `R_repr` — **representation error**: model-vs-observation footprint mismatch. Typically 1–5 ppb diagonal addition; rises with terrain complexity and at coarse-instrument boundaries. Don't omit — naive `R = R_retr` overweights observations and produces overconfident posteriors.
- Block-diagonal across overpasses; cross-overpass correlation only if observation times are within met decorrelation scale (~hours).

### Prior on `q` — spatially correlated, sign-constrained

The prior is the regularizer; it's *the* methodological choice in regional inversions.

| Choice | Form | Notes |
|--------|------|-------|
| Mean `q_a` | from [emission inventory](00_prerequisites.md#background-emission-inventory-q_a) (EDGAR / GFEI / EPA) | prior median per cell |
| Covariance `B` | Matern-3/2 with correlation length ℓ ∈ [5, 50] km | smooth posterior; ℓ tuned by posterior diagnostics or hierarchical |
| Positivity | `log q ~ N(log q_a, B_log)` (lognormal) | non-negative emissions; conjugate variant: NNLS / projected-gradient on Gaussian `q` |

Diagonal `B` is wrong — produces wildly noisy spatial posteriors. Always carry spatial correlation.

### Gaussian–Gaussian closed form (linear-in-log)

When the lognormal is linearised around `log q_a` (fine for moderate enhancements over the prior), the posterior is closed form:

```
q* = q_a · exp(B_log F̃^T (F̃ B_log F̃^T + R)^{-1} (y − F̃ q_a))
P* = B_log − B_log F̃^T (F̃ B_log F̃^T + R)^{-1} F̃ B_log
```

with `F̃ = diag(q_a) F` (the linearised Jacobian). Identical structure to the equations [`gaussx`](/home/azureuser/localfiles/gaussx/) is built around.

### Scaling beyond moderate grids

- "≲10k cells × ≲1k obs" is the dense-solver limit. Beyond that:
  - **Krylov + structure-aware solves** via `gaussx` (Kronecker-Matern, low-rank `F`). Pushes direct solves to ~100k cells with sufficient sparsity.
  - **Ensemble Kalman Inversion (EKI):** ensemble of forward trajectories → ensemble-based `∂c/∂Q`. Plug into [`filterax`](/home/azureuser/localfiles/filterax/); couple with `vardaX` for the variational version.
  - **MCMC over `log q(x)`:** expensive but exact. Use only when EKI is suspected biased.

---

## (3) Model emulator

The Lagrangian model becomes expensive at large `N` particles or when running ensembles for met-uncertainty propagation.

- **Footprint emulator:** `(met snapshot, receptor location, source-candidate grid) → F(receptor, ·)`. Receptor location is **part of the input** (not just source location — the original draft had this backwards). CNN if the met grid is regular; FNO if you want to be resolution-agnostic.
- **Trajectory emulator:** replace the SDE integration with a neural ODE or normalising flow. Less common but potentially useful for backward integration.
- **Training distribution.** Sample met conditions from the **actual distribution at facility locations of interest**, not a uniform climatology bin. A coarse `(hour-of-day, season, stability)` grid covers <10% of the operational regime probability mass; train on the conditional distribution `p(met | facility_lat_lon, t_overpass)` instead.

---

## (4) Emulator-based inference

Replace `F` with the emulated footprint in the linear inversion. Enables:

- Real-time source estimation during satellite overpass (no SDE integration in the loop).
- Ensemble-based UQ at scale — sample met conditions, emulator gives footprint per sample, propagate to source posterior.
- **Validation:** posterior from emulator-inversion ≈ posterior from full-Lagrangian inversion on the same observations. Diverging posteriors flag emulator bias before it becomes a downstream problem.

---

## (5) Amortized inference (predictor)

```
f_θ: (y_multiscale, met_context, instrument_id) → p(log q(x) | y, met)
```

### Output grid commitment

`q(x)` is on the inversion grid (~1–10 km, basin-scoped). Predictor outputs a fixed-size 2D field per basin tile. **Per-instrument predictors**, dispatched by `instrument_id`, because TROPOMI (~5 km), EMIT (~60 m), Tanager (~30 m) need different summary networks.

### Multi-instrument observations

When fusing across instruments, each observation tensor carries its own AK and footprint at native resolution. The predictor consumes a list of `(y, AK, footprint, σ_retr, mask)` per instrument and aggregates internally — don't pre-regrid to a common resolution (loses information).

### Conditional flow architecture

- Posterior over `log q(x)` is a *spatial field* — natural fit for a conditional flow over images. **Caveat:** [`gauss_flows`](/home/azureuser/localfiles/gauss_flows/) is currently 1D-supported. Either extend `gauss_flows` to 2D coupling layers (multi-month effort) or fall back to score-based / diffusion posterior. Score-based is the safer path for v1.
- Context conditioning via FiLM / hypernet primitives in [`pyrox.nn`](/home/azureuser/localfiles/pyrox/) — same pattern as Tier I.

---

## (6) Improve

- **Multi-layer met fields.** Move from 2D footprints to 3D trajectories through stacked WRF layers — necessary when emissions span the inversion layer or when the source PBL is poorly mixed.
- **Chemical loss during transport.** `dc/dt = −k_OH c` along trajectories. For CH₄ over <1 day, loss is negligible (~0.5%/day); for CO it's significant.
- **Met uncertainty propagation.** Run the trajectory ensemble across `N_met` WRF / ERA5 ensemble realisations → uncertainty in `F` propagated through to source posterior. Hooks into the `MetField.ensemble_dim` from the [prereqs schema](00_prerequisites.md#metfield-schema).
- **Hierarchical `B` correlation length.** Promote ℓ in the Matern prior to a hyperparameter with its own posterior — let the data choose the regularisation scale.

---

## Module layout (proposed)

| Step | Concern | Module | Status |
|------|---------|--------|--------|
| 1 | Particle integrator (Markov-1) | `plume_simulation.lagrangian.particles` | ☐ |
| 1 | C-grid-aware wind interpolator | `plume_simulation.lagrangian.wind_interp` | ☐ |
| 1 | Backward footprint | `plume_simulation.lagrangian.footprint` | ☐ |
| 1 | Turbulence parameterisation (σ_ij, τ_L) | `plume_simulation.lagrangian.turbulence` | ☐ |
| 1 | Column + AK pipeline | reuse `gauss_plume.observation` from Tier I | ☐ |
| 2 | Likelihoods + spatial priors | `plume_simulation.lagrangian.likelihoods` | ☐ |
| 2 | Linear inversion (Gaussian / lognormal) | reuse [`assimilation/solve.py`](src/plume_simulation/assimilation/solve.py) with `F̃` injected | ☐ |
| 2 | Krylov / structure-aware solver | dispatch to [`gaussx`](/home/azureuser/localfiles/gaussx/) | dependency |
| 2 | EKI | [`filterax`](/home/azureuser/localfiles/filterax/) (external) | dependency |
| 2 | Posterior export → Tier V | `plume_simulation.lagrangian.posterior_export` | ☐ |
| 3 | Footprint emulator | `plume_simulation.lagrangian.emulator` | ☐ |
| 5 | Field predictor (per instrument) | `plume_simulation.lagrangian.predictor` | ☐ |
| 6 | Met-ensemble runner | `plume_simulation.lagrangian.met_ensemble` | ☐ |

The whole subpackage doesn't exist yet; this is the proposed shape.

---

## Validation strategy

- **Particle integration — zero-turbulence limit.** With `b → 0`, trajectories must follow streamlines exactly. Compare to streamline integration of the same wind field.
- **Mass conservation.** In the no-deposition, closed-domain limit, total particle-seconds in the domain must be conserved to floating-point precision over the integration window. Standard LPDM regression test.
- **Adjoint–finite-difference.** Verify backward ≡ adjoint of forward: perturb `q_i` in the forward, measure `Δc_j`, compare to `F[j, i]` from the backward run. Should agree within Monte Carlo error. Cheap test, catches indexing / sign / time-direction bugs that are otherwise nightmare to track down.
- **Particle-count convergence.** `N → 2N` should give converging posterior moments; unconverged means `N` is too low. Sets the operational floor and replaces the open-question guess (10⁵ forward / 10³ backward) with measurement.
- **Footprint vs. forward agreement.** For a single-source `q = e_i`, the column `F·e_i` (forward) and the receptor row `F[i, :]` (backward) should agree. Catches indexing/orientation bugs distinct from the adjoint test.
- **Linear inversion against Tier I.** In the limit of a single source, near-stationary winds, and known turbulence, the Lagrangian inversion should recover Tier I's MAP estimate within posterior σ.
- **Real-data Permian benchmark.** Invert TROPOMI + EMIT observations over the Permian for a published time window and compare to Lu et al. / Maasakkers et al. inverse-modelling estimates. Without this, the inversion is a synthetic exercise.
- **Emulator residual.** `‖F_emu − F_true‖_F / ‖F_true‖_F < 5%` on a held-out met-condition set drawn from the *operational* distribution (not the training-bin distribution).

---

## Open questions

- **Time discretisation Δt.** Operational floor for SDE convergence in non-stationary PBL turbulence? Initial guess: Δt = min(τ_L / 5, 60 s). Needs benchmarking against Markov-1 well-mixed-condition tests.
- **Positivity strategy.** Lognormal prior on `q` (smooth, conjugate when linearised) vs. NNLS on Gaussian `q` (cheap, but biased near zero) vs. reflected-Gaussian MCMC (exact, slow). v1: lognormal; revisit if posterior contracts hard at zero on real data.
- **Representativeness error magnitude.** `R_repr` diagonal: 1 ppb (well-resolved) to 5 ppb (coarse satellite over rough terrain). Open: hierarchical fit to data, or pre-tabulated by (instrument, terrain class)?
- **Number of particles — measured, not guessed.** Replace the `N=10⁵` / `N=10³` guess with the convergence test above. Operational target: posterior moments stable within 5% as `N → 2N`.
- **Random seed handling.** Treat the forward as a noisy oracle (averaged over seeds) or fix one seed and treat result as deterministic? **Leaning:** fix-seed for inference (deterministic gradients); average-seed only for final-report posterior summarisation.
- **Footprint storage / compression.** Sparse CSR storage handles the 90%+ zeros, but cross-receptor footprints share spatial structure — low-rank factorisation (`F ≈ U V^T` with `r ≪ n_obs`) might compress further. Open: empirical rank vs. accuracy trade-off on real data.
- **Backward vs. forward — refined.** Two distinct reasons backward dominates inversion: (a) `n_source` is unknown a priori (the original "you're inferring it"), and (b) backward gives a sparse `F^T` *per receptor* that's trivially parallelisable and storage-friendly. The cost ratio `O(n_obs N)` vs. `O(n_source N)` is secondary.
- **Coarse-instrument representation.** TROPOMI 5 km vs inversion grid 1 km — observation operator must include footprint-weighted spatial averaging. Open: handle as a deterministic averaging operator on `column_z(F q)`, or as a stochastic representation kernel inside `R_repr`?
- **Spatial correlation length ℓ in `B`.** Default 10 km, but real basins have stronger correlation along pipeline corridors. Open: anisotropic Matern with corridor-aligned anisotropy, or hierarchical ℓ?
