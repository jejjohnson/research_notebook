---
title: "Tier II — Lagrangian particle transport"
short_title: "Tier II — Lagrangian"
subject: "plumax — stochastic particle dispersion tier"
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [Lagrangian particle dispersion, FLEXPART, STILT, HYSPLIT, Markov-1 Langevin, footprint, source-receptor matrix, lognormal prior, ensemble Kalman inversion, Krylov inversion]
---

# Tier II — Lagrangian particle transport

**Forward model:** stochastic particle trajectories driven by wind + turbulence. The bridge between the analytical Tier I (no real wind variability) and the full PDE Tier III (no statistical efficiency). This is what FLEXPART ({cite:p}`stohl2005flexpart`) and STILT do operationally — see also HYSPLIT ({cite:p}`stein2015hysplit`).

This tier is **not yet started in `plume_simulation`** — module layout proposed below.

---

(tier2-simple-model)=
## (1) Simple model

(tier2-langevin)=
### Stochastic dynamics — Markov-1 Langevin

Operational LPDMs (FLEXPART, STILT, HYSPLIT) use **Markov-1**: a Langevin equation on particle *velocity*, then position from velocity. We commit to Markov-1 because it's required for near-source super-emitter work and for non-stationary turbulence:

```{math}
:label: eq-tier2-langevin
\begin{aligned}
\mathrm{d}\mathbf{v} &= \mathbf{a}(\mathbf{x}, \mathbf{v}, t)\, \mathrm{d}t \;+\; \mathbf{b}(\mathbf{x}, t)\, \mathrm{d}\mathbf{W}, \qquad \mathrm{d}\mathbf{W} \sim \mathcal{N}(\mathbf{0}, \mathrm{d}t \cdot \mathbf{I}_3) \\
\mathrm{d}\mathbf{x} &= \bigl(\mathbf{u}(\mathbf{x},t) + \mathbf{v}\bigr)\, \mathrm{d}t
\end{aligned}
```

- $\mathbf{u}(\mathbf{x},t)$ — mean wind from the [met field](00_prerequisites.md#prereqs-met) (WRF / ERA5 / HRRR).
- $\mathbf{v}$ — turbulent velocity perturbation, the actual stochastic state.
- $\mathbf{a}, \mathbf{b}$ — drift and diffusion coefficients constructed from the Reynolds-stress tensor $\sigma_{ij}(\mathbf{x},t)$ and Lagrangian timescale $\tau_L(\mathbf{x},t)$. Reference: {cite:p}`thomson1987`, {cite:p}`wilson1996` (well-mixed condition).
- $\sigma_{ij}, \tau_L$ come from MO similarity ({cite:p}`monin1954`) parameterised by $(u_*, L_\text{Obukhov}, z_0)$ plus WRF TKE — see [MO prereqs](00_prerequisites.md#prereqs-mo-similarity).

:::{important} Markov-0 is not the model
Markov-0 (random-displacement on position only) is **not** the model; it shows up as the well-mixed limit at long $\Delta t$. Don't bake Markov-0 into the API as a "fast option" — implement the well-mixed limit explicitly when needed.
:::

### Sub-grid wind interpolation

WRF velocities are gridded at hourly snapshots; particles are continuous in space and time. Naïve bilinear interpolation of $\mathbf{u}$ is **not divergence-free**, which causes mass-conservation drift over long integrations. Use a **C-grid-aware interpolator** that reconstructs $\mathbf{u}$ consistently with the WRF staggered grid (operational STILT pattern). Document the chosen scheme on the integrator API; assume divergence preservation is load-bearing.

### Time-stepping

Default to **Euler–Maruyama** with adaptive step bounded by

```{math}
:label: eq-tier2-dt-bound
\Delta t \;<\; \min\!\left(\tau_L,\; \frac{\Delta x}{|\mathbf{u}|},\; \frac{\Delta x^{2}}{\sigma^{2}}\right).
```

Higher-order Milstein only if convergence diagnostics flag bias. $\Delta t$ is an open parameter — see [open questions](#tier2-open-questions) for the operational target.

### Run modes

- **Forward mode:** release $N$ particles from the source, track concentration by binning particle density on the analysis grid.
- **Backward mode (footprint):** release receptors backward in time → **source–receptor sensitivity matrix** $\mathbf{F}$. The FLEXPART/STILT paradigm and the workhorse of regional-scale inversions ({cite:p}`stohl2005flexpart`).

(tier2-footprint)=
### Footprint definition (formal)

For a receptor $r$ and a candidate source cell $s$:

```{math}
:label: eq-tier2-footprint
F(r, s) \;=\; \int_{t_\text{obs} - T}^{t_\text{obs}} \frac{1}{V_s\, \rho_\text{air}(s,t)} \;\mathbb{1}\!\left[\, z_\text{part}(t) < f_\text{PBL}\, L(s, t)\, \right]\, \mathrm{d}t
```

— integrated over particle residence time below a fraction $f_\text{PBL} \approx 0.5$ of the local PBL height $L(s,t)$. Units: $\text{s} \cdot \text{m}^{2} \cdot \text{kg}^{-1}$ (mixing-ratio per kg/m²/s of surface flux). The footprint is a **probability density** over candidate source cells, not an indicator: properly normalised by particle volume and air density. Numerical scaling matters because the values are tiny — work in log-space when storing.

For an overpass with $n_\text{obs}$ columns and $n_\text{grid}$ candidate source cells, $\mathbf{F}$ is shape $(n_\text{obs}, n_\text{grid})$ and typically 90%+ sparse.

---

(tier2-inference)=
## (2) Model-based inference

### Forward model with observation operator

The full forward used by inference is **not** $\mathbf{y} = \mathbf{F}\mathbf{q} + \boldsymbol{\varepsilon}$. It's:

```{math}
:label: eq-tier2-forward
\mathbf{y} \;=\; \mathbf{A}\, \mathrm{col}_z(\mathbf{F}\mathbf{q}) \;+\; \mathbf{c}_\text{bg} \;+\; \boldsymbol{\varepsilon}
```

- $\mathbf{F}\mathbf{q}$ gives a 3D concentration field from the source vector.
- $\mathrm{col}_z$ collapses to an XCH₄ column.
- $\mathbf{A}$ applies the satellite [averaging kernel](00_prerequisites.md#prereqs-ak-operator); see {eq}`eq-ak-operator`.
- $\mathbf{c}_\text{bg}$ is the regional background — mandatory, not optional. Same critique as Tier I: predicted enhancement vs. absolute-column observation must be reconciled day-1.

(tier2-likelihood)=
### Likelihood model

```{math}
:label: eq-tier2-likelihood
\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}), \qquad \mathbf{R} \;=\; \mathbf{R}_\text{retr} + \mathbf{R}_\text{repr}
```

- $\mathbf{R}_\text{retr}$ — heteroscedastic per-pixel from the L2 retrieval-error map.
- $\mathbf{R}_\text{repr}$ — **representation error**: model-vs-observation footprint mismatch. Typically 1–5 ppb diagonal addition; rises with terrain complexity and at coarse-instrument boundaries. Don't omit — naive $\mathbf{R} = \mathbf{R}_\text{retr}$ overweights observations and produces overconfident posteriors.
- Block-diagonal across overpasses; cross-overpass correlation only if observation times are within met decorrelation scale (~hours).

(tier2-prior)=
### Prior on $\mathbf{q}$ — spatially correlated, sign-constrained

The prior is the regularizer; it's *the* methodological choice in regional inversions.

```{list-table} Spatial-prior choices for the source vector $\mathbf{q}$.
:label: tbl-tier2-prior
:header-rows: 1

* - Choice
  - Form
  - Notes
* - Mean $\mathbf{q}_a$
  - from [emission inventory](00_prerequisites.md#prereqs-emission-inventory) ({cite:p}`crippa2023edgar,scarpelli2022gfei,maasakkers2023ghgi`)
  - prior median per cell
* - Covariance $\mathbf{B}$
  - Matérn-3/2 with correlation length $\ell \in [5, 50]$ km
  - smooth posterior; $\ell$ tuned by posterior diagnostics or hierarchical
* - Positivity
  - $\log \mathbf{q} \sim \mathcal{N}(\log \mathbf{q}_a, \mathbf{B}_{\log})$ (lognormal)
  - non-negative emissions; conjugate variant: NNLS / projected-gradient on Gaussian $\mathbf{q}$
```

:::{important} Diagonal $\mathbf{B}$ is wrong
Diagonal $\mathbf{B}$ produces wildly noisy spatial posteriors. Always carry spatial correlation.

**Why:** an inversion with diagonal prior covariance has no smoothing scale, so unresolved structure goes into single-cell spikes.
**How to apply:** use a Matérn-3/2 (or any spatial kernel with a real correlation length); calibrate $\ell$ by posterior diagnostics.
:::

(tier2-gaussian-closed-form)=
### Gaussian–Gaussian closed form (linear-in-log)

When the lognormal is linearised around $\log \mathbf{q}_a$ (fine for moderate enhancements over the prior), the posterior is closed form:

```{math}
:label: eq-tier2-posterior-mean
\mathbf{q}^* \;=\; \mathbf{q}_a \cdot \exp\!\left(\mathbf{B}_{\log} \tilde{\mathbf{F}}^{\top} (\tilde{\mathbf{F}} \mathbf{B}_{\log} \tilde{\mathbf{F}}^{\top} + \mathbf{R})^{-1} (\mathbf{y} - \tilde{\mathbf{F}} \mathbf{q}_a)\right)
```

```{math}
:label: eq-tier2-posterior-cov
\mathbf{P}^* \;=\; \mathbf{B}_{\log} \;-\; \mathbf{B}_{\log} \tilde{\mathbf{F}}^{\top} (\tilde{\mathbf{F}} \mathbf{B}_{\log} \tilde{\mathbf{F}}^{\top} + \mathbf{R})^{-1} \tilde{\mathbf{F}} \mathbf{B}_{\log}
```

with $\tilde{\mathbf{F}} = \operatorname{diag}(\mathbf{q}_a)\, \mathbf{F}$ (the linearised Jacobian). Identical structure to the equations [`gaussx`](https://github.com/jejjohnson/gaussx) is built around.

### Scaling beyond moderate grids

- "≲10k cells × ≲1k obs" is the dense-solver limit. Beyond that:
  - **Krylov + structure-aware solves** via `gaussx` (Kronecker-Matérn, low-rank $\mathbf{F}$). Pushes direct solves to ~100k cells with sufficient sparsity.
  - **Ensemble Kalman Inversion (EKI):** ensemble of forward trajectories → ensemble-based $\partial \mathbf{c} / \partial \mathbf{Q}$. Plug into [`filterax`](https://github.com/jejjohnson/filterax); couple with `vardaX` for the variational version.
  - **MCMC over $\log \mathbf{q}(\mathbf{x})$:** expensive but exact. Use only when EKI is suspected biased.

---

(tier2-emulator)=
## (3) Model emulator

The Lagrangian model becomes expensive at large $N$ particles or when running ensembles for met-uncertainty propagation.

- **Footprint emulator:** $(\text{met snapshot}, \text{receptor location}, \text{source-candidate grid}) \to \mathbf{F}(\text{receptor}, \cdot)$. Receptor location is **part of the input** (not just source location). CNN if the met grid is regular; FNO if you want to be resolution-agnostic.
- **Trajectory emulator:** replace the SDE integration with a neural ODE or normalising flow. Less common but potentially useful for backward integration.
- **Training distribution.** Sample met conditions from the **actual distribution at facility locations of interest**, not a uniform climatology bin. A coarse $(\text{hour-of-day}, \text{season}, \text{stability})$ grid covers <10% of the operational regime probability mass; train on the conditional distribution $p(\text{met} \mid \text{facility}_\text{lat,lon}, t_\text{overpass})$ instead.

---

(tier2-emu-inference)=
## (4) Emulator-based inference

Replace $\mathbf{F}$ with the emulated footprint in the linear inversion. Enables:

- Real-time source estimation during satellite overpass (no SDE integration in the loop).
- Ensemble-based UQ at scale — sample met conditions, emulator gives footprint per sample, propagate to source posterior.
- **Validation:** posterior from emulator-inversion ≈ posterior from full-Lagrangian inversion on the same observations. Diverging posteriors flag emulator bias before it becomes a downstream problem.

---

(tier2-amortized)=
## (5) Amortized inference (predictor)

```{math}
:label: eq-tier2-amortized
f_\theta : (\mathbf{y}_\text{multiscale}, \text{met}_\text{context}, \text{instrument\_id}) \;\longmapsto\; p(\log \mathbf{q}(\mathbf{x}) \mid \mathbf{y}, \text{met})
```

### Output grid commitment

$\mathbf{q}(\mathbf{x})$ is on the inversion grid (~1–10 km, basin-scoped). Predictor outputs a fixed-size 2D field per basin tile. **Per-instrument predictors**, dispatched by `instrument_id`, because TROPOMI ({cite:p}`s5p_tropomi`, ~5 km), EMIT ({cite:p}`emit`, ~60 m), Tanager ({cite:p}`carbon_mapper`, ~30 m) need different summary networks.

### Multi-instrument observations

When fusing across instruments, each observation tensor carries its own AK and footprint at native resolution. The predictor consumes a list of $(\mathbf{y}, \mathbf{A}, \text{footprint}, \sigma_\text{retr}, \text{mask})$ per instrument and aggregates internally — don't pre-regrid to a common resolution (loses information).

### Conditional flow architecture

- Posterior over $\log \mathbf{q}(\mathbf{x})$ is a *spatial field* — natural fit for a conditional flow over images.
- :::{caution} `gauss_flows` is currently 1D-only
  Either extend `gauss_flows` to 2D coupling layers (multi-month effort) or fall back to score-based / diffusion posterior. Score-based is the safer path for v1.
  :::
- Context conditioning via FiLM / hypernet primitives in [`pyrox.nn`](https://github.com/jejjohnson/pyrox) — same pattern as Tier I.

---

(tier2-improve)=
## (6) Improve

- **Multi-layer met fields.** Move from 2D footprints to 3D trajectories through stacked WRF layers — necessary when emissions span the inversion layer or when the source PBL is poorly mixed.
- **Chemical loss during transport.** $\mathrm{d}c/\mathrm{d}t = -k_\text{OH}\, c$ along trajectories. For CH₄ over <1 day, loss is negligible (~0.5%/day); for CO it's significant.
- **Met uncertainty propagation.** Run the trajectory ensemble across $N_\text{met}$ WRF / ERA5 ensemble realisations → uncertainty in $\mathbf{F}$ propagated through to source posterior. Hooks into the `MetField.ensemble_dim` from the [prereqs schema](00_prerequisites.md#prereqs-metfield-schema).
- **Hierarchical $\mathbf{B}$ correlation length.** Promote $\ell$ in the Matérn prior to a hyperparameter with its own posterior — let the data choose the regularisation scale.

---

(tier2-modules)=
## Module layout (proposed)

```{list-table} Tier II proposed module layout — step, concern, target module, status.
:label: tbl-tier2-modules
:header-rows: 1

* - Step
  - Concern
  - Module
  - Status
* - 1
  - Particle integrator (Markov-1)
  - `plume_simulation.lagrangian.particles`
  - ☐
* - 1
  - C-grid-aware wind interpolator
  - `plume_simulation.lagrangian.wind_interp`
  - ☐
* - 1
  - Backward footprint
  - `plume_simulation.lagrangian.footprint`
  - ☐
* - 1
  - Turbulence parameterisation ($\sigma_{ij}, \tau_L$)
  - `plume_simulation.lagrangian.turbulence`
  - ☐
* - 1
  - Column + AK pipeline
  - reuse `gauss_plume.observation` from Tier I
  - ☐
* - 2
  - Likelihoods + spatial priors
  - `plume_simulation.lagrangian.likelihoods`
  - ☐
* - 2
  - Linear inversion (Gaussian / lognormal)
  - reuse [`assimilation/solve.py`](../../src/plume_simulation/assimilation/solve.py) with $\tilde{\mathbf{F}}$ injected
  - ☐
* - 2
  - Krylov / structure-aware solver
  - dispatch to [`gaussx`](https://github.com/jejjohnson/gaussx)
  - dependency
* - 2
  - EKI
  - [`filterax`](https://github.com/jejjohnson/filterax) (external)
  - dependency
* - 2
  - Posterior export → Tier V
  - `plume_simulation.lagrangian.posterior_export`
  - ☐
* - 3
  - Footprint emulator
  - `plume_simulation.lagrangian.emulator`
  - ☐
* - 5
  - Field predictor (per instrument)
  - `plume_simulation.lagrangian.predictor`
  - ☐
* - 6
  - Met-ensemble runner
  - `plume_simulation.lagrangian.met_ensemble`
  - ☐
```

The whole subpackage doesn't exist yet; this is the proposed shape.

---

(tier2-validation)=
## Validation strategy

- **Particle integration — zero-turbulence limit.** With $\mathbf{b} \to \mathbf{0}$, trajectories must follow streamlines exactly. Compare to streamline integration of the same wind field.
- **Mass conservation.** In the no-deposition, closed-domain limit, total particle-seconds in the domain must be conserved to floating-point precision over the integration window. Standard LPDM regression test.
- **Adjoint–finite-difference.** Verify backward ≡ adjoint of forward: perturb $q_i$ in the forward, measure $\Delta c_j$, compare to $F[j, i]$ from the backward run. Should agree within Monte Carlo error. Cheap test, catches indexing / sign / time-direction bugs that are otherwise nightmare to track down.
- **Particle-count convergence.** $N \to 2N$ should give converging posterior moments; unconverged means $N$ is too low. Sets the operational floor and replaces the open-question guess (10⁵ forward / 10³ backward) with measurement.
- **Footprint vs. forward agreement.** For a single-source $\mathbf{q} = \mathbf{e}_i$, the column $\mathbf{F}\mathbf{e}_i$ (forward) and the receptor row $\mathbf{F}[i, :]$ (backward) should agree. Catches indexing/orientation bugs distinct from the adjoint test.
- **Linear inversion against Tier I.** In the limit of a single source, near-stationary winds, and known turbulence, the Lagrangian inversion should recover Tier I's MAP estimate within posterior $\sigma$.
- **Real-data Permian benchmark.** Invert TROPOMI + EMIT observations over the Permian for a published time window and compare to Lu et al. / Maasakkers et al. inverse-modelling estimates (see {cite:p}`maasakkers2023ghgi,jacob2022quantifying`). Without this, the inversion is a synthetic exercise.
- **Emulator residual.** $\lVert \mathbf{F}_\text{emu} - \mathbf{F}_\text{true} \rVert_F / \lVert \mathbf{F}_\text{true} \rVert_F < 5\%$ on a held-out met-condition set drawn from the *operational* distribution (not the training-bin distribution).

---

(tier2-open-questions)=
## Open questions

:::{attention} Time discretisation $\Delta t$
Operational floor for SDE convergence in non-stationary PBL turbulence? Initial guess: $\Delta t = \min(\tau_L / 5,\, 60\,\text{s})$. Needs benchmarking against Markov-1 well-mixed-condition tests.
:::

:::{attention} Positivity strategy
Lognormal prior on $\mathbf{q}$ (smooth, conjugate when linearised) vs. NNLS on Gaussian $\mathbf{q}$ (cheap, but biased near zero) vs. reflected-Gaussian MCMC (exact, slow). v1: lognormal; revisit if posterior contracts hard at zero on real data.
:::

:::{attention} Representativeness error magnitude
$\mathbf{R}_\text{repr}$ diagonal: 1 ppb (well-resolved) to 5 ppb (coarse satellite over rough terrain). Open: hierarchical fit to data, or pre-tabulated by (instrument, terrain class)?
:::

:::{attention} Number of particles — measured, not guessed
Replace the $N=10^5$ / $N=10^3$ guess with the convergence test above. Operational target: posterior moments stable within 5% as $N \to 2N$.
:::

:::{attention} Random seed handling
Treat the forward as a noisy oracle (averaged over seeds) or fix one seed and treat result as deterministic? **Leaning:** fix-seed for inference (deterministic gradients); average-seed only for final-report posterior summarisation.
:::

:::{attention} Footprint storage / compression
Sparse CSR storage handles the 90%+ zeros, but cross-receptor footprints share spatial structure — low-rank factorisation ($\mathbf{F} \approx \mathbf{U} \mathbf{V}^{\top}$ with $r \ll n_\text{obs}$) might compress further. Open: empirical rank vs. accuracy trade-off on real data.
:::

:::{attention} Backward vs. forward — refined
Two distinct reasons backward dominates inversion: (a) $n_\text{source}$ is unknown a priori (the original "you're inferring it"), and (b) backward gives a sparse $\mathbf{F}^{\top}$ *per receptor* that's trivially parallelisable and storage-friendly. The cost ratio $O(n_\text{obs} N)$ vs. $O(n_\text{source} N)$ is secondary.
:::

:::{attention} Coarse-instrument representation
TROPOMI 5 km vs inversion grid 1 km — observation operator must include footprint-weighted spatial averaging. Open: handle as a deterministic averaging operator on $\mathrm{col}_z(\mathbf{F}\mathbf{q})$, or as a stochastic representation kernel inside $\mathbf{R}_\text{repr}$?
:::

:::{attention} Spatial correlation length $\ell$ in $\mathbf{B}$
Default 10 km, but real basins have stronger correlation along pipeline corridors. Open: anisotropic Matérn with corridor-aligned anisotropy, or hierarchical $\ell$?
:::
