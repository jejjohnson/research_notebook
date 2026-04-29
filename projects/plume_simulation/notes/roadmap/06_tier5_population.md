# Tier V — Source population & forecasting

**Forward model:** thinned marked temporal point process (TMTPP) over emission events, with per-event marks drawn from Tier I–IV posteriors and per-event detection thinning by per-satellite POD models.

This tier sits **above** the per-event physics tiers. Tiers I–IV answer *"what's the emission rate from this plume right now?"* — single overpass, single source. Tier V answers a different family of questions:

- *Inventory accounting:* "Given a population of detected plumes (and the ones we missed), what's the true total emitted mass?"
- *Forecasting:* "For a given facility class, when will the next emission event happen, and how big will it be?"
- *Bias diagnosis:* "How biased is the per-overpass average rate when satellites only see the big leaks?"

Inventory and forecasting are **co-equal products** of Tier V — not just totals. The inverted intensity `λ(t)` directly powers operational forecasts (dispatch windows, occurrence probabilities); see [Persistency](06c_persistency.md).

Sub-pages:

- [Instantaneous emission estimation](06a_instantaneous.md) — single-overpass `Q`; the cross-tier interface that turns per-event posteriors into mark likelihoods.
- [Point process model (TMTPP)](06b_point_process.md) — the generative foundation: temporal intensity `λ(t)`, marks `f(Q)`, and detection thinning `P_d(·)`.
- [Persistency](06c_persistency.md) — operational forecasts from inverted `λ(t)`: wait times, dispatch windows, occurrence probabilities.
- [Total emission estimation](06d_total_emission.md) — the missing-mass paradox and POD-corrected regional/national totals.

---

## TMTPP foundations — the three-term log-likelihood

The full population log-likelihood has three terms (derived in [06b](06b_point_process.md)):

```
log L = Σ_{i detected} log p(detected_i | f, λ, P_d)        mark contribution
      + Σ_{i detected} log λ(t_i)                            detection-time intensity
      − ∫₀^T λ(t) · [∫ P_d(Q) f(Q) dQ] dt                    integrated thinned rate
```

The third term is what makes `λ` and `P_d` jointly identifiable — without it the two trade off.

### Mark contribution and the soft-observation framing

The per-event posterior from Tiers I–IV is a **soft observation** of the (unknown) true mark `Q_i`. This is the same Bayesian-deconvolution / errors-in-variables structure used in measurement-error regression. The per-event mark contribution is:

```
p(detected_i | f, λ, P_d) = ∫ P_d(Q) · L_i(Q) · f(Q) dQ
```

where `L_i(Q) = p(observation_i | Q)` is the per-event **likelihood**, not the posterior. In sample-based practice (per-event posterior samples `Q_i^{(s)} ~ p(Q | observation_i)`):

```
p(detected_i | f, λ, P_d)  ≈  (1/S) Σ_s  P_d(Q_i^{(s)}) · f(Q_i^{(s)}) / π_per-event(Q_i^{(s)})
```

with `π_per-event(Q)` the **per-event prior** used at Tier I–IV. The ratio `f / π_per-event` is the importance weight that re-points the per-event posterior at the population mark distribution. **Without this re-weighting the population fit double-counts the per-event prior** — biased posterior on `f`, biased total-mass estimate.

This is the central math of cross-tier inference. Currently the prototype in [`methane_pod.fitting`](projects/methane_pod/src/methane_pod/fitting.py) summarises per-event posteriors to point estimates before the population fit, side-stepping the importance correction. Formalising this is the v1 deliverable for [`06a_instantaneous.md`](06a_instantaneous.md).

---

## How the cycle adapts at population scale

The six-step cycle still applies, but the objects change:

| Step | Tier I–IV (per event) | Tier V (population) |
|------|-----------------------|----------------------|
| 1 — Simple model | Forward physics (plume / PDE / RTM) | Generative TMTPP: `λ(t)` + mark `f(Q)` + POD `P_d(·)` |
| 2 — Model-based inference | MAP / MCMC over source params | NumPyro NUTS over `(λ params, mark params, POD params)`. Cheap at `O(10⁴)` events (minutes); hours-to-days at `O(10⁶)` events (national catalog) |
| 3 — Model emulator | FNO / neural ODE on the PDE | Skip when NUTS fits in budget. Optionally a normalising flow over the population posterior for repeated re-fits or sensitivity studies |
| 4 — Emulator-based inference | PDE-free 4D-Var | Variational fit (`numpyro.infer.SVI`) or flow-based posterior approximation; required at national catalog scale |
| 5 — Amortized predictor | Per-overpass `Q` predictor | (basin tile, history window) → posterior over `(λ, f(Q), total mass, next-event time)` conditioned on per-event evidence and met-region context |
| 6 — Improve | Better physics | Spatial point process (links to Tier III); multi-satellite fusion; varying-coefficient POD (per-(basin, season, scene class) hierarchy); non-Poisson clustering (Hawkes / Cox) |

**Tile definition:** an H3 hex-resolution-7 cell (~5 km²) for sub-basin work, or a basin polygon for inventory accounting. History window: 30–365 days, hierarchical prior on the cutoff.

**Varying-coefficient POD:** `P_d` parameters indexed by `(basin, season, scene class)` with hierarchical shrinkage to the global POD. Captures regional / seasonal detection differences without inflating parameter count.

**Key difference from lower tiers:** the Tier V "forward model" is a *generative process for events*, not a PDE for fields. Mass conservation, advection, etc. are inherited from Tiers I–IV through the per-event posteriors — Tier V does not re-derive them.

---

## Cross-tier interface — the load-bearing contract

### Payload schema

Every per-event posterior consumed by Tier V must carry:

| Field | Type | Notes |
|-------|------|-------|
| `posterior_samples` | `(S,)` array of `Q` draws | OR `posterior_summary` for Gaussian shorthand |
| `posterior_summary` | `(mu_logQ, sigma_logQ)` | lognormal quick form when full samples are too heavy |
| `per_event_prior_logpdf` | callable `Q → log π(Q)` | **required** for the importance correction; without it the population fit is biased |
| `instrument_id` | str | dispatch into per-instrument POD |
| `t_detection` | float (UTC seconds) | for `λ(t)` |
| `x0_posterior` | `(mu_xy, Cov_xy)` | for spatial Cox-process upgrade |
| `quality` | dict | confidence flags from the Tier I–IV quality bitmask |

### Independence assumption — the v1 caveat

The factorised likelihood above assumes detections at different overpasses are independent. Two overpasses of the *same physical leak* (e.g. GHGSat then TROPOMI two days later) violate this. **Bias direction:** ignoring the dependence inflates effective sample size → posterior on `f` is over-concentrated.

v1 assumes independence and screens at the catalog stage (collapse near-coincident detections to one event by spatial-temporal clustering). v2 promotes to a hierarchical model with per-source latent state shared across overpasses — same machinery as Tier IV's `Q(t)` stochastic process but at the population level.

---

## Validation strategy

- **Population SBC.** Generate `(λ*, f*, P_d*)`, simulate the full thinned-and-marked catalog with a synthetic per-event posterior layer, fit, check rank statistics across all hyperparameters. The Tier V analogue of Tier-I synthetic recovery.
- **Importance-weight ESS diagnostic.** Per detection `i`, the IS estimator's effective sample size (`ESS_i = (Σ w_s)² / Σ w_s²`) is a health metric. Low ESS (e.g. `< S/10`) signals that the per-event posterior is far from the population mark distribution; the population fit is unreliable for that event. Report the ESS distribution as a fit diagnostic.
- **Per-event-prior swap-out.** Refit the population using a different per-event prior at Tier I–IV (re-run Tiers I–IV with `π_per-event = LogNormal(0, 2)` instead of the inventory-anchored prior). The population posterior on `f` should not move beyond IS noise. If it does, the importance correction is mis-implemented.
- **Real-data benchmark.** Compare corrected total emission for a well-studied basin (Permian) to published bottom-up inventories (EPA GHGI, GHGRP) and top-down inverse-modelling estimates (Lu et al., Sherwin et al.) — see [06d](06d_total_emission.md).

---

## Module layout — depend on `methane_pod`, don't absorb it

`plumax` depends on the standalone [`methane_pod`](projects/methane_pod/) package (pinned `methane_pod >= 0.1, < 0.2` for v1); the population-scale code is not re-implemented. Rationale:

- `methane_pod` has its own audience (point-process methodologists), test suite, release cadence.
- `plumax` consumes it through a thin adapter that materialises Tier I–IV posteriors as inputs to `methane_pod.fitting`.
- Versioning stays clean — when methane_pod releases v0.X.Y, plumax pins to a known-good version.

| Concern | Module | Status |
|---------|--------|--------|
| Intensity registry `λ(t)` | [`methane_pod.intensity`](projects/methane_pod/src/methane_pod/intensity.py) | library ✓ (13 kernels) |
| POD registry `P_d(·)` | [`methane_pod.pod_functions`](projects/methane_pod/src/methane_pod/pod_functions.py) | library ✓ (10 models) |
| Missing-mass MC simulator | [`methane_pod.paradox`](projects/methane_pod/src/methane_pod/paradox.py) | library ✓ |
| NUTS fitter | [`methane_pod.fitting`](projects/methane_pod/src/methane_pod/fitting.py) | library ✓; **importance-correction integration ☐** |
| Per-event posterior summariser | `plume_simulation.population.adapter.summariser` | ☐ |
| Per-event prior recall (`π_per-event` lookup) | `plume_simulation.population.adapter.prior_recall` | ☐ — required for importance weighting |
| Importance-weight calculator | `plume_simulation.population.adapter.importance` | ☐ |
| Multi-satellite POD union | `plume_simulation.population.adapter.pod_union` | ☐ |
| Catalog schema (CSV / parquet) | `plume_simulation.population.adapter.schema` | ☐ |
| Real-data CSV ingestion | `plume_simulation.population.ingest` | ☐ (placeholder in [`07_pod_fitting_mcmc.md`](projects/methane_pod/notebooks/07_pod_fitting_mcmc.md)) |
| Population SBC harness | `plume_simulation.population.validation.sbc` | ☐ |
| Importance-weight ESS diagnostic | `plume_simulation.population.validation.iw_ess` | ☐ |
| Per-event-prior swap-out test | `plume_simulation.population.validation.prior_swap` | ☐ |
| Spatial Cox-process extension (v2) | `plume_simulation.population.spatial` | ☐ |

A `plume_simulation.population` subpackage doesn't exist yet; this is the proposed shape.

---

## Connection to Tier III — spatial structure

Tier III's distributed source field `S(x,t)` is **exactly a spatial inhomogeneous Poisson rate** at the population level — temporally aggregated, this *is* the spatial intensity of a Cox process over emission events. The v2 spatial extension of Tier V is the same mathematical object Tier III already inverts at the per-event timescale, just averaged over a longer horizon. The two tiers should share the parameterisation: a Matern GP prior on `log S(x,t)` plays the role of both Tier III's source-field prior and Tier V.v2's spatial Cox-process intensity.

This isn't a coincidence — it's why `plumax`'s tier structure works: the same mathematical objects appear at different scales.

---

## Status snapshot

- **Theory.** TMTPP foundations and the missing-mass paradox are written up in [`methane_pod/notebooks/01_mttpp_theory`](projects/methane_pod/notebooks/01_mttpp_theory.md) and [`03_missing_mass_paradox`](projects/methane_pod/notebooks/03_missing_mass_paradox.ipynb).
- **`methane_pod` library:** ✓ — intensity, POD, paradox simulator, NUTS fitter all implemented.
- **Cross-tier integration:** ☐ — per-event posteriors enter the population fit as point estimates (no importance correction). Tier V's main code deliverable.
- **Synthetic validation.** [`06_stationary_numpyro_mcmc`](projects/methane_pod/notebooks/06_stationary_numpyro_mcmc.ipynb) recovers POD parameters on synthetic data without the soft-observation layer.
- **Real-data fit.** [`07_pod_fitting_mcmc`](projects/methane_pod/notebooks/07_pod_fitting_mcmc.md) is a placeholder; needs IMEO + Tanager CSV ingestion.

---

## Open questions

- **Per-event independence — quantitative bias.** v1 assumption. Bias direction is known (over-concentration on `f`). Open: order-of-magnitude — for a basin with ~30% multi-instrument coincidences, how much does the posterior over-contract? Pilot study needed.
- **Multi-satellite POD aggregation.** Either `P_d^union(Q) = 1 − ∏(1 − P_d^k(Q))` (independent detection chances) or marginalise over which satellite actually looked (categorical mark). The two are not equivalent. v1: union; v2: categorical when per-satellite attribution matters for inventory.
- **Non-Poisson clustering.** Real super-emitters cluster (compressor cycles, equipment lifecycle). Two upgrade paths: (a) **Hawkes / self-exciting kernel** when clustering is event-driven; (b) **Cox process with stochastic intensity** (latent OU on `log λ`) when clustering is environmentally driven. Pick by basin diagnostic.
- **Spatial structure.** Currently temporal-only. Spatial extension via Cox process over wells is the natural v2 (and ties to Tier III, see above). Open: H3 hex-resolution-7 vs. continuous Matern-GP intensity — operational vs. physical fidelity trade-off.
- **Forecasting horizon.** [Persistency](06c_persistency.md) gives wait-time forecasts. Open: how far out does the forecast remain useful? Likely instrument-cadence-dependent (TROPOMI daily vs. GHGSat tasked).
- **Hierarchy depth for varying-coefficient POD.** Three levels (global → basin → season) is tractable; four (+ scene class) starts to over-parameterise. Open: which factor matters most, by ELPD / WAIC.
