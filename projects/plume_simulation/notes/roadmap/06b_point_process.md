# Tier V.B — Point process model (TMTPP)

**Generative model:** events arrive in time according to an intensity `λ(t)`, each carries a mark `Q ~ f(Q)`, and each is observed with probability `P_d(Q; satellite)`. The "thinned marked temporal point process" (TMTPP) is the mathematical object on which everything else at Tier V is built.

The full mathematical derivation lives in [`methane_pod/notebooks/01_tmtpp_theory`](projects/methane_pod/notebooks/01_tmtpp_theory.md). This page gives the architectural view: the components, their interfaces, and where they plug into the rest of `plumax`.

**Units convention:** everything in SI internally — `Q` in kg/s, time in seconds. Catalog ingestion (06a) normalises to SI; rendering layers convert to operational units (t/h, kg/h) on display.

---

## The three components

### Temporal — `λ(t)` (events / second)

The intensity function tells you how rapidly events arrive at time `t`. Examples from the catalogue in [`02_intensity_zoo.md`](projects/methane_pod/notebooks/02_intensity_zoo.md):

- **Constant** `λ(t) = λ₀` — homogeneous Poisson; baseline.
- **Diurnal** `λ(t) = λ₀ + A sin(2πt/T_day + φ)` — solar-heated tank cycles.
- **Step** `λ(t) = λ₀ · 1[t > t_open]` — valve fails open at known time.
- **Decaying** `λ(t) = λ₀ exp(−t/τ)` — pressure-relief blowdown.
- **Hawkes** `λ(t) = μ + Σ α exp(−β(t−t_i))` — self-exciting; explicit branching, captures event-driven clustering.
- **Log-Gaussian Cox process** `log λ(t) ~ GP(μ, K)` — stochastic intensity; captures environmentally-driven clustering (compressor cycles, weather windows) without the branching structure of Hawkes.

13 deterministic / Hawkes kernels currently implemented in [`methane_pod.intensity`](projects/methane_pod/src/methane_pod/intensity.py); LGCP is the v1.5 next kernel — it's the natural model when clustering is environmental rather than self-exciting.

Each kernel is an `equinox.Module` exposing the same `__call__(t) → λ` and `sample_priors()` interface. Adding a new kernel is a one-file PR.

### Marks — `f(Q)` (probability density on kg/s)

The mark distribution gives the size of an event conditional on it happening.

| Family | Form | When to use |
|--------|------|-------------|
| **Lognormal** | `Q ~ LN(μ, σ²)` | single-class facility populations |
| **Pure power-law** | `f(Q) ∝ Q^{-α}` for `Q > Q_min` | baseline only; over-emphasises the tail |
| **Lognormal-Pareto** | `LN(μ, σ²)` body, `Pareto(α, Q_break)` tail | **v1 default** — Cusworth 2021 / Sherwin 2024 operational standard |
| **Mixture-of-lognormals** | `Σ π_k · LN(μ_k, σ_k²)` | multi-class facility populations (wells + tanks + pipelines) |

`Q_min` (power-law) and `Q_break` (lognormal-Pareto) are themselves parameters. **v1 default:** `Q_break ~ LogNormal(log Q_break_published, 0.5²)` — informative prior from instrument-class detection floor, not a hard constraint. Joint inference is cleaner but adds an identifiability concern with `α`.

The mark distribution is what Tier V actually wants to recover — it's the population-scale answer to "how big are the leaks at this kind of facility?"

### Detection thinning — `P_d(Q; satellite)` (probability)

Not every event is observed. Each satellite has a probability of detection that depends on the leak size, viewing geometry, surface, and atmospheric state.

The operational form in the methane literature is the **Hill function** (Cusworth 2021, Sherwin 2024):

```
P_d(Q) = 1 / (1 + (Q_50 / Q)^k)
```

where `Q_50` is the leak size at which detection probability is 0.5 and `k` controls the steepness. **This is *not* the same as a logistic on `log Q`** — Hill is rational in `Q`, logistic is sigmoidal in `log Q`. Doc previously listed "logistic"; the operational convention is Hill. (`methane_pod.pod_functions` currently implements both; the population fitter should default to Hill.)

#### POD calibration uncertainty — hierarchical prior

Per-instrument controlled-release campaigns (Sherwin et al. 2024) deliver a **posterior** on `(Q_50, k)`, not a point. v1 default: hierarchical prior carrying calibration-campaign uncertainty:

```
Q_50,inst ~ LogNormal(log Q_50_published, σ_Q50_published²)
k_inst    ~ LogNormal(log k_published,    σ_k_published²)
```

This is the middle ground between (a) hard-coding published values (biased when those are uncertain) and (b) full joint inference (cleanest but identifiability concern with `λ`). v2 promotes to joint inference when basin data warrants.

10 POD models currently in [`methane_pod.pod_functions`](projects/methane_pod/src/methane_pod/pod_functions.py), described visually in [`05_pod_gallery`](projects/methane_pod/notebooks/05_pod_gallery.ipynb). Variants:

- **Hill** — operational standard.
- **Varying-coefficient Hill** — `Q_50 = g(albedo, SZA, scene_class)`.
- **Spectral-aware** — explicitly carries the SWIR retrieval noise floor as a function of column XCH₄.
- **Full GLM** — generalised linear model with multiple scene covariates.

---

## TMTPP likelihood — canonical form

For a set of detected events with per-event posteriors `{p(Q | observation_i)}_{i=1}^n` and detection times `{t_i}` over a window `[0, T]`:

```
log L(λ, f, P_d | data) = Σ_i [ log λ(t_i)  +  log ∫ P_d(Q) · L_i(Q) · f(Q) dQ ]
                        − ∫₀^T λ(t) · ∫ P_d(Q) · f(Q) dQ dt
```

The first sum scores each detected event under: (a) the temporal intensity at the detection time, and (b) the integrated mark contribution that combines the per-event likelihood with the population mark distribution and the satellite POD. The second integral is the **expected number of events that would have been detected** under the model — subtracts the right amount so the posterior is consistent.

### Practical evaluation

The mark integral `∫ P_d(Q) · L_i(Q) · f(Q) dQ` is computed via the importance-weighted Monte Carlo estimator from [06a § Mark likelihood](06a_instantaneous.md#mark-likelihood--importance-weighted-monte-carlo):

```
∫ P_d(Q) · L_i(Q) · f(Q) dQ  ≈  (1/S) Σ_s  P_d(Q_i^{(s)}) · f(Q_i^{(s)}) / π_per-event(Q_i^{(s)})
```

with samples `Q_i^{(s)} ~ p(Q | observation_i)` and `π_per-event(Q)` the per-event prior used at Tier I–IV. The `1 / π_per-event` factor is the importance weight; without it the population fit double-counts the per-event prior.

### Point-regime simplification

When per-event posteriors are tightly concentrated (CV < 20%) and `f` is smooth on that scale, the importance-weighted MC reduces to the **Point regime** of 06a:

```
log L_point = Σ_i [ log λ(t_i) + log f(Q_i) + log P_d(Q_i) ]
            − ∫₀^T λ(t) · ∫ P_d(Q) · f(Q) dQ dt
```

This is the form currently implemented in [`methane_pod.fitting.pod_powerlaw_model`](projects/methane_pod/src/methane_pod/fitting.py). It's the **simplification**, not the canonical form — explicit regime selection per [06a § Regime selection rule](06a_instantaneous.md#regime-selection-rule) decides when it's safe to use.

### Numerical stability of the integrated thinned-rate term

The integral `∫ P_d(Q) · f(Q) dQ` over heavy-tailed `f` (power-law tail, Pareto) and saturating `P_d` (Hill) **must not** be evaluated by naive quadrature in linear `Q` — the heavy tail underflows. Standard fix: **log-space Gauss-Hermite quadrature** (Hill × LogNormal becomes a tractable polynomial-times-sigmoid in log-space; ≤16 nodes give 4-decimal accuracy). For Pareto tails, switch to importance sampling with a Pareto proposal. Bug-class: any `α > 2` power-law silently underestimates the thinned-rate integral with linear quadrature, biasing `λ` low.

---

## Where it plugs into `plumax`

| Input | Source |
|-------|--------|
| `(t_i, instrument_id_i, per-event payload)` per detection | [`06a_instantaneous.md`](06a_instantaneous.md) — Tier V.A adapter |
| Per-event `L_i(Q)` (samples + `π_per-event_logpdf`) | Tiers I–IV inversion + posterior export |
| Per-instrument POD calibration `(Q_50_pub, σ_Q50_pub, k_pub, σ_k_pub)` | Sherwin 2024 / Cusworth 2021 / Kamdar IMEO controlled-release campaigns; alternatively joint inference with the population |
| Per-instrument overpass coverage (for the integrated rate) | [`06a § Non-detection events`](06a_instantaneous.md#non-detection-events-catalog-gaps) — catalog ingest |

| Output | Consumer |
|--------|----------|
| Posterior `λ(t)` | [`06c_persistency.md`](06c_persistency.md) — wait times, dispatch windows |
| Posterior `f(Q)` | [`06d_total_emission.md`](06d_total_emission.md) — total mass under POD-thinning correction |
| Per-instrument POD posterior `(Q_50, k)_inst` | instrument-design and cross-mission calibration questions; multi-satellite fusion (06d) |
| Joint `(λ, f, P_d)` posterior | sensitivity studies, satellite-tasking optimisation |

---

## Population vs. per-source — the v1 commitment

The TMTPP fits *aggregate* the population. Two distinct framings:

- **Across-population** (v1 default for inventory accounting): fit one `(λ, f, P_d)` over all sources of a class within a basin/region. `Q` means "size of an event drawn from this class".
- **Per-source longitudinal** (v1 for persistency forecasting on a known facility): fit one `(λ_facility(t), f_facility(Q), P_d)` per facility, with hierarchical shrinkage to the population. `Q` means "size of an event from this specific facility".

Both have library support; the choice is driven by the scientific question, not by the methodology. Inventory totals (06d) use across-population; dispatch decisions for a known leak history (06c) use per-source.

---

## Module layout

| Concern | Module | Status |
|---------|--------|--------|
| Intensity registry — deterministic + Hawkes | [`methane_pod.intensity`](projects/methane_pod/src/methane_pod/intensity.py) | ✓ (13 kernels) |
| Intensity registry — log-Gaussian Cox process | `methane_pod.intensity.lgcp` | ☐ — v1.5 |
| Mark registry | `methane_pod.marks` (currently inline in `fitting`) | 🚧 — power-law only; lognormal, lognormal-Pareto, mixture-of-lognormals pending |
| POD models `P_d(·)` (Hill + variants) | [`methane_pod.pod_functions`](projects/methane_pod/src/methane_pod/pod_functions.py) | ✓ (10 models) |
| POD time-of-day binning (v1 time-varying POD) | `methane_pod.pod_functions.tod_binned` | ☐ |
| POD continuous `P_d(Q, t)` (v2) | `methane_pod.pod_functions.continuous_t` | ☐ |
| Hierarchical POD calibration prior | `methane_pod.pod_functions.calibration_prior` | ☐ |
| TMTPP likelihood — point regime | [`methane_pod.fitting.pod_powerlaw_model`](projects/methane_pod/src/methane_pod/fitting.py) | ✓ |
| TMTPP likelihood — full importance-corrected regime | `methane_pod.fitting.tmtpp_iw` | ☐ — consumes [`population.adapter.importance`](06a_instantaneous.md#module-layout) |
| Numerical integration helpers (log-space Gauss-Hermite, Pareto IS) | `methane_pod.fitting.integrate` | ☐ |
| Hawkes / self-exciting kernel | `methane_pod.intensity.hawkes` | ☐ — beyond the existing kernels |
| Spatial extension (Cox process) | `methane_pod.spatial` | ☐ — v2; ties to Tier III's `S(x,t)` |

---

## Validation strategy

- **Likelihood gradient.** `jax.grad` matches finite differences within tolerance. Cheap unit test.
- **Synthetic recovery — Point regime.** Already in [`06_stationary_numpyro_mcmc`](projects/methane_pod/notebooks/06_stationary_numpyro_mcmc.ipynb) for power-law mark.
- **Synthetic recovery — Full regime with importance correction.** Generate per-event posteriors with a known `π_per-event`, fit population, recover `(λ*, f*, P_d*)` within reported posterior. Mirrors the importance-correction round trip from [06a § Validation](06a_instantaneous.md#validation-strategy).
- **SBC — point.** Across 1000 simulated populations, per-parameter rank statistics uniform.
- **SBC — soft observation.** Same SBC but with the soft-observation layer (per-event posteriors as input). Validates the cross-tier inference end-to-end.
- **Identifiability stress test.** Generate data where `λ` is high but `P_d` is low (vs. the opposite). Quantitative target: posterior correlation `corr(λ_0, Q_50) > 0.5` when confounded; `< 0.2` when well-separated. Confirms the model knows what it can't disentangle.
- **Log-space integration test.** Compare log-space Gauss-Hermite to Pareto importance sampling for `∫ Hill(Q) · LogNormal-Pareto(Q) dQ` across `α ∈ [1.5, 4]`. Linear quadrature should fail loudly past `α > 2`. Catches the silent thinned-rate underflow bug.
- **Hierarchical POD coverage.** With known controlled-release calibration injected as `(Q_50_pub, σ_Q50_pub)`, the hierarchical POD prior should produce `Q_50` posteriors that contain the true value at 95% CI ~95% of the time across simulated basins.

---

## Open questions

- **Cox vs. Hawkes for clustering.** Both are v1.5 candidates. Pick by basin diagnostic: event-driven clustering (compressor cycles) → Hawkes; environmentally-driven (weather-window persistence) → LGCP.
- **Mark / temporal coupling.** TMTPP currently assumes marks are i.i.d. given the temporal process. Tier IV's `Q(t)` stochastic process per source is the right object to lift here — a per-source `Q(t)` becomes a per-source contribution to the spatial population intensity. Promote when Tier IV `Q(t)` lands.
- **Per-instrument POD prior depth.** Hierarchical with calibration uncertainty (v1 default) vs. full joint inference (v2). When does the basin data warrant promotion? Probably when the calibration-campaign sample size is too small for the instrument-condition (e.g. high-AOD scenes for EMIT).
- **Continuous time-varying POD.** v1 uses time-of-day bins. When do diurnal cloud / glint patterns warrant continuous `P_d(Q, t)`? Likely needed for sun-glint-sensitive instruments over coastal scenes.
- **`Q_break` / `Q_min` joint inference.** Currently informative prior from detection floor. Open: when basin data warrants joint inference with the power-law `α`, does identifiability hold? Likely fine when the catalog spans ≥ 1.5 decades in `Q`.
- **Numerical integration choice.** Log-space Gauss-Hermite is the v1 default. Open: when does adaptive quadrature pay off (highly-variable `f` shapes)?
