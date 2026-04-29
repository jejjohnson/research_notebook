# Tier I — Gaussian family

**Forward model:** closed-form analytical solution for steady/transient point-source dispersion.

This is the **first tier to build to completion** (all six steps), because:

1. The forward model is microseconds, so MCMC is feasible end-to-end.
2. Every downstream tier validates against Tier I in the limit of weak turbulence and stationary winds — Tier I is the analytical reference.
3. The amortized predictor at Step 5 doubles as the working prototype for the inference UX (input/output shapes, posterior visualisation, uncertainty budget).

---

## (1) Simple model

### Gaussian plume (steady-state, continuous source)

```
c(x',y',z) = Q / (2π σ_y σ_z ū)
             × exp(−y'² / 2σ_y²)
             × Σ_n [exp(−(z − H_eff − 2nL)² / 2σ_z²)
                  + exp(−(z + H_eff − 2nL)² / 2σ_z²)]
```

The primed coordinates `(x', y')` are the source-aligned frame: `x'` is downwind along `θ_wind`, `y'` is crosswind. The image-source sum runs over `n ∈ {…,−1, 0, 1,…}` to enforce no-flux at the ground (`z=0`) **and** at the capping inversion (`z=L = PBL height`). For `L → ∞` only the `n=0` pair survives — that's the unbounded-domain case in most textbooks. For typical PBL `L ≈ 1 km` and σ_z growing past ~`L/3`, the upper image is non-negligible.

**Parameters:**

- `Q` — source strength (kg/s)
- `x₀ = (lat₀, lon₀)` — source location
- `H_stack` — physical stack height
- `Δh(F_b, F_m, ū, stability)` — Briggs plume rise from buoyancy/momentum fluxes; `H_eff = H_stack + Δh`
- `ū`, `θ_wind` — wind speed and direction (from met, with uncertainty — see [Inference](#2-model-based-inference))
- `σ_y(x'), σ_z(x')` — crosswind/vertical spread; PG-class lookup for v1, MO-similarity for v2 (see prereqs)
- `L` — PBL capping height (from met)

For methane super-emitter inversions where stack height is poorly known, `H_eff` is part of what you infer; for known facilities `H_stack` is fixed and only `Δh` is computed.

### Gaussian puff (time-varying, episodic source)

```
c(x,y,z,t) = Σ_k  Q_k / (2π)^(3/2) σ³
              × exp(−‖x − x_k(t)‖² / 2σ²)
```

Each puff `k` advects with the wind and diffuses independently — handles intermittent / burst-mode releases that violate the steady-state assumption.

### From `c(x,y,z)` to a satellite-comparable observation

The forward used by inference is **not** `c(x,y,z)` directly. It's:

```
y_model(x,y) = AK · (column_integrate_z(c) + c_bg(x,y))
```

- **Column integrate:** the satellite sees a column-integrated enhancement, not a single altitude.
- **Background `c_bg(x,y)`:** the regional background (typically ~1900 ppb for CH₄). **Not optional** — without it the model predicts the *enhancement above background*, which is what `c` actually is, but the satellite delivers absolute column densities. Either subtract `c_bg` from `y_obs` upstream, or jointly model it. Either way, it's day-1 infrastructure, not a Step-6 upgrade.
- **Averaging kernel `AK`:** the satellite-product AK from the [prereqs](00_prerequisites.md#averaging-kernel-operator). Required when comparing to L2 XCH₄ products. Skip it only when working with a flat-AK assumption (rare and worth flagging).

### Extended Gaussian (AERMOD-style)

Adds terrain corrections (plume rise over hills, beyond Briggs), building downwash (Huber-Snyder), and receptor-grid output. Useful as a sanity benchmark against operational tools, **not** a primary research target.

---

## (2) Model-based inference

The plume is analytical, so the full forward `H: (Q, x₀, H_eff, ū, θ_wind, …) → y_model(x,y)` is differentiable via JAX end-to-end (column integration + AK included).

### Likelihood model

```
y_obs(x,y) = y_model(x,y) + ε(x,y),   ε(x,y) ~ N(0, σ_retr(x,y)²)
```

- **Default:** heteroscedastic Gaussian on column-XCH₄ enhancement, with per-pixel σ from the L2 retrieval-error map.
- **Heavy-tail variant:** Student-t with `ν ~ 5` for retrievals near the detection floor where outliers dominate (TROPOMI single-pass, EMIT off-axis).
- **Mask:** quality-flag mask from the L2 product is passed through as an indicator weight on the likelihood — flagged pixels contribute zero log-likelihood.

The likelihood is load-bearing for MCMC convergence. Don't leave it to "default Gaussian" silently.

### Priors

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| `Q` | LogNormal(μ_Q, σ_Q) with μ_Q from the [emission inventory](00_prerequisites.md#background-emission-inventory-q_a) prior, σ_Q ≈ 1.0 | positive, heavy-tail; matches inventory uncertainty |
| `x₀` | N(facility_lat_lon, σ_x₀) with σ_x₀ ≈ sub-pixel | facility location is known, sub-pixel uncertainty for cluster vs. point |
| `H_eff` | Uniform(H_stack, H_stack + Δh_max) or N(H_stack + Δh_briggs, σ_H) | depends on whether stack height is known |
| `ū` | N(ū_met, σ_ū_met) — **tight prior from met** | ū is data, not a free parameter |
| `θ_wind` | N(θ_met, σ_θ_met) — tight prior from met | same |
| `c_bg` | GP or Gaussian per-tile around climatology | regional background varies smoothly |

### MAP / MCMC

- **MAP estimation:** `jax.grad(log_posterior)` → L-BFGS or Adam. Convergence in <1s for a single overpass.
- **MCMC:** NumPyro NUTS over `(Q, x₀, H_eff, ū, θ_wind, c_bg)` jointly. Forward pass is ~µs, so 10k samples is seconds. Keep `ū`, `θ_wind` as inference variables (with tight met priors) so their posterior contracts can be diagnosed downstream.
- **Linear-Gaussian special case:** if observations are linear in `c` (column-integrated XCH₄ with a fixed AK) and only `Q` is free with all geometry fixed, the posterior over `Q` is analytically tractable — exact Bayesian inversion via [`gaussx`](/home/azureuser/localfiles/gaussx/). Useful as a sanity check on the NUTS implementation.

### `Q / ū` identifiability — corrected

The classic statement is "`Q` and `ū` enter only as `Q/ū`, so they're degenerate in a single transect". This is true only if `ū` has a flat prior. In production, **the tight met-derived prior on `ū` resolves the degeneracy**: the posterior `p(Q | y, ū_met, σ_ū_met)` contracts as long as `σ_ū_met / ū_met ≪ σ_Q / Q_prior`. The "need two crosswind transects" claim is a special case for fully-free `ū` and should not be the default reading.

When the wind prior is *not* tight (rare — typically remote regions without good reanalysis coverage), then yes, multiple overpasses with different wind directions break the ratio. But the first-line fix is the prior, not the geometry.

This is the **first working end-to-end inverse pipeline.** Build it here, validate against synthetic releases with known truth, then move to real controlled-release data (see [Validation](#validation-strategy)).

---

## (3) Model emulator

The Gaussian plume is cheap enough that an emulator is **optional** at this tier. Skip in production, but build it as a training exercise for the emulator infrastructure that Tier III will need.

- **Input:** `(Q, x₀, H_eff, ū, θ_wind, stability_class, L)` — the *raw* inputs, not the evaluated σ profiles. The emulator should learn the σ functional too; otherwise you're benchmarking interpolation, not modelling.
- **Output:** `y_model(x,y)` on the satellite-pixel grid (post column-integration, post AK).
- **Architecture:** MLP for low-dim parameters → small transposed-conv decoder for spatial output, or lightweight DeepONet.
- **Value:** validates emulator training pipeline (data generation, training loop, residual checks) before applying to expensive PDE models.

---

## (4) Emulator-based inference

Same inference loop as Step 2, but the forward pass is the neural network. Required validation:

- Posterior mean from emulator-MCMC ≈ posterior mean from analytical-MCMC, within 1σ.
- Posterior covariance from emulator-MCMC ≈ analytical posterior covariance, in operator norm.

If those pass for Tier I, the same diagnostics apply unchanged at Tier III.

---

## (5) Amortized inference (predictor)

Train a **summary network + posterior network** mapping observation patches directly to the source posterior:

```
f_θ: (y_patch, context) → p(Q, x₀, H_eff | y_patch, context)
```

### Input shape — commit to a 2D patch

`y_patch` is a fixed-size 2D image patch of column XCH₄ enhancement (background-subtracted) centred on the facility candidate, plus per-pixel uncertainty and quality-mask channels. Concretely:

- Shape: `(C, H_patch, W_patch)` with `C = 3` channels (`enhancement`, `σ_retr`, `mask`).
- Resolution: instrument-native (e.g. 30 m for Tanager, 60 m for EMIT, ~5 km for TROPOMI). Train one predictor per instrument family — don't try to share across resolutions.
- Footprint: large enough to capture the plume's downwind extent at expected wind speeds (typical: 2–10 km square for point-source instruments).

This commitment matters because it pins the architecture (CNN-style backbones over set-transformers or graph nets).

### Conditioning on context

The predictor is **not** observation-only. At inference time we know `context = (facility_lat_lon, ū_met, θ_wind_met, stability_class, L_met)`. The clean way to wire this in is feature-wise modulation of the summary network — the FiLM / hypernet conditioning primitives being built in [`pyrox.nn`](/home/azureuser/localfiles/pyrox/) (see the [conditioning module plan](/home/azureuser/.claude/plans/wiggly-enchanting-thimble.md)).

```
summary  = CNN(y_patch)               # observation features
modulated = FiLM(summary, context)    # per-feature γ(context)·summary + β(context)
posterior = NPE_head(modulated)       # conditional flow over (Q, x₀, H_eff)
```

Without context conditioning the predictor has to relearn met-dependent behaviour from scratch, which is wasteful and well-documented to fail near regime boundaries (stable vs. unstable PBL, low vs. high wind speed).

### Architecture options

- **Conditional normalizing flow** (`gauss_flows`) for the posterior head — preferred for low-dim posteriors over `(Q, x₀, H_eff)`.
- **BNN posterior head** (`pyrox`) when posterior multimodality is unlikely.
- **NPE / SBI-style** — drop-in for either, gives a clean SBC validation interface.

Training dataset is **free**: simulate millions of plume configurations in seconds, sampling met context from the WRF/ERA5 climatology. Validation: posterior calibration via simulation-based calibration (SBC) — uniform rank statistics across 1k held-out simulations, stratified by met regime.

---

## (6) Improve

- **PG → MO σ swap.** Replace PG lookup tables for `σ_y, σ_z` with MO-similarity-derived functions parameterised by `(u*, L_obukhov, z₀)` from the [prereqs](00_prerequisites.md#monin-obukhov-similarity). Closer to physics, continuous in stability, and the right pre-step for tier-II validation.
- **Multi-source.** Real basins host 5–50 simultaneous emitters. This is **not** an array-shape change — the inference becomes a *mixture model with unknown component count*, requiring reversible-jump MCMC (RJMCMC) or a Dirichlet-process prior over source count. Plan for this from the start: the posterior interface that Tier V.A consumes must handle variable-`K` per overpass.
- **Learned σ from LES.** Train a small NN against LES output to learn stability-dependent σ functions that go beyond MO. Slot in as a swap-in for either PG or MO σ. Useful for super-emitter regimes that LES has resolved better than empirical fits.
- **Distributed-source field `Q(x)`.** Replace point source with a spatial source field — opens the door to Tier II/III for spatially extended emissions. At Tier I this is just a sum of point sources sharing met context; the multi-source mixture (above) is the same code path with finer support.

---

## Module layout

| Step | Concern | Module | Status |
|------|---------|--------|--------|
| 1 | Plume forward | [`gauss_plume/plume.py`](src/plume_simulation/gauss_plume/plume.py) | ✓ |
| 1 | Puff forward | [`gauss_puff/puff.py`](src/plume_simulation/gauss_puff/puff.py) | ✓ |
| 1 | Plume rise (Briggs) | `gauss_plume.plume_rise` | ☐ |
| 1 | Stability + dispersion | [`gauss_plume/dispersion.py`](src/plume_simulation/gauss_plume/dispersion.py) | 🚧 partial |
| 1 | Puff turbulence | [`gauss_puff/turbulence.py`](src/plume_simulation/gauss_puff/turbulence.py) | ✓ |
| 1 | Column + AK pipeline | `gauss_plume.observation` | ☐ — links to [`assimilation/obs_operator.py`](src/plume_simulation/assimilation/obs_operator.py) |
| 1 | Background `c_bg` loader | `plume_simulation.priors.background` | ☐ |
| 2 | Likelihoods + priors | `gauss_plume.likelihoods` | ☐ |
| 2 | Plume MAP/MCMC | [`gauss_plume/inference.py`](src/plume_simulation/gauss_plume/inference.py) | ✓ |
| 2 | Puff inference | [`gauss_puff/inference.py`](src/plume_simulation/gauss_puff/inference.py) | ✓ |
| 2 | Posterior export → Tier V | `gauss_plume.posterior_export` (mark-likelihood adapter for [V.A](06a_instantaneous.md)) | ☐ |
| 3 | Plume emulator | `gauss_plume.emulator` | ☐ |
| 4 | Emulator-based MCMC | wired in `inference.py` once emulator exists | ☐ |
| 5 | NPE / flow predictor | `gauss_plume.predictor` | ☐ |
| 5 | Context-conditioning layer | uses `pyrox.nn` FiLM/hypernet primitives | external dep |
| 6 | Multi-source (RJMCMC) | extend `inference.py` with reversible-jump kernel | ☐ |
| 6 | MO σ swap | `gauss_plume.dispersion_mo` | ☐ |

---

## Validation strategy

- **Forward model — mass flux conservation.** Integrate **mass flux** `∫∫ c · u_⊥ dA` through a transverse plane downwind of the source, compare to `Q`. Should be exact in the no-deposition, unbounded-domain limit. (Common bug: integrating `∫∫ c dA` instead — that's mass per unit wind speed, dimensionally wrong.)
- **Ground reflection consistency.** Set `H = 0`; the two image-source terms must collapse to a single Gaussian with doubled prefactor. Catches sign/index bugs in the image sum.
- **PBL capping.** With `L → ∞` the upper-image series must vanish; with finite `L` and σ_z growing past `L/2`, concentrations must converge to a vertically well-mixed `c → Q / (ū · L · √(2π) σ_y) · exp(−y²/2σ_y²)`. Tests both branches of the n-sum.
- **MAP recovery.** Synthetic release with known `(Q*, x₀*, H*)`. Recovery target: `|Q_MAP − Q*| < σ_post(Q)` — i.e. recovery within the posterior's own claimed uncertainty. (Not a fixed % — that conflates SNR with method quality.)
- **MCMC calibration.** SBC on 1000 simulated releases — rank histograms uniform across all parameters, stratified by met regime.
- **Q/ū identifiability — empirical.** Run MAP over a sweep of `σ_ū_met / ū_met ∈ [0.05, 0.5]`. Posterior CV(Q) should grow monotonically with met-wind uncertainty; flat CV signals a wiring bug.
- **Real-data benchmark.** Invert a published controlled-release flight (e.g. Stanford / Sherwin et al. 2024 controlled releases over Tanager / GHGSat / EMIT). Posterior median for `Q` must contain the metered release rate within 95% credible interval. **This is the only test that proves the inference works on real radiances** — synthetic validation is necessary but not sufficient.
- **Emulator agreement.** Step 4 posterior matches Step 2 posterior in mean and covariance to within Monte Carlo error.

---

## Aggregating across overpasses

The MAP / MCMC posterior `p(Q | overpass)` produced here is the **per-event evidence** consumed by the population layer. See [Tier V.A — Instantaneous emission estimation](06a_instantaneous.md) for the formal interface that turns this posterior into a mark likelihood for the TMTPP fit, and [Tier V.D — Total emission estimation](06d_total_emission.md) for why per-overpass averages systematically misrepresent regional totals.

The multi-source extension (Step 6) makes this interface variable-K per overpass — the V.A adapter must handle that.

## Open questions

- **Coordinate frame for the puff cloud.** Carry puff centroids in lat/lon (general but slower) or in a local frame anchored to the source (fast but breaks for puffs that travel far)?
- **Time-varying wind in the plume model.** Strictly the steady-state plume assumes constant `(ū, θ_wind)`. For ~30-min satellite overpasses winds rotate; do we time-slice (multiple steady-state plumes), advect with hourly met (piecewise-stationary), or jump straight to puff?
- **Non-Gaussian noise.** Detection-floor effects mean observation noise is heavy-tailed at low concentration. Default Student-t with `ν ~ 5` for predictor training (Step 5); does heteroscedastic Gaussian suffice for MCMC at moderate-to-high SNR? Open until SBC results across SNR regimes are in.
- **Identifiability — quantitative target.** Sherwin et al. (2024) report that ~3 overpasses with varied wind direction give `CV(Q) < 50%` on isolated super-emitters. Adopt this as the operational target: the inference is "good enough" when 3-overpass CV(Q) ≤ 50%, single-overpass CV(Q) ≤ 100% (with tight met prior). Document on the inference docstring.
- **PG sunset timing.** When do we deprecate PG `σ_y, σ_z`? Either (a) immediately swap to MO once the prereq lands, or (b) keep PG as a v1 default and add MO as opt-in. Leaning (b) — PG matches the historic literature and gives a stable baseline.
- **Multi-source prior on `K`.** RJMCMC needs a prior on source count. Poisson(λ_K) with λ_K ~ 5 (typical basin)? Geometric? Open — depends on basin and inventory coverage.
- **Background `c_bg` model.** GP residual on top of a regional climatology, or a simple per-tile constant? GP is "right" but slows MCMC; per-tile is fast but has bias near the source. v1: per-tile; v2: GP.
