---
title: "Tier V — Source population & forecasting"
short_title: "Tier V — Population"
subject: "plumax — population-scale TMTPP layer"
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [TMTPP, marked point process, methane inventory, missing-mass paradox, importance weights, Cox process, Hawkes process, forecasting, methane_pod]
---

# Tier V — Source population & forecasting

**Forward model:** thinned marked temporal point process (TMTPP) over emission events, with per-event marks drawn from Tier I–IV posteriors and per-event detection thinning by per-satellite POD models.

This tier sits **above** the per-event physics tiers. Tiers I–IV answer *"what's the emission rate from this plume right now?"* — single overpass, single source. Tier V answers a different family of questions:

- *Inventory accounting:* "Given a population of detected plumes (and the ones we missed), what's the true total emitted mass?"
- *Forecasting:* "For a given facility class, when will the next emission event happen, and how big will it be?"
- *Bias diagnosis:* "How biased is the per-overpass average rate when satellites only see the big leaks?"

Inventory and forecasting are **co-equal products** of Tier V — not just totals. The inverted intensity $\lambda(t)$ directly powers operational forecasts (dispatch windows, occurrence probabilities); see [Persistency](06c_persistency.md).

Sub-pages:

- [Instantaneous emission estimation](06a_instantaneous.md) — single-overpass $Q$; the cross-tier interface that turns per-event posteriors into mark likelihoods.
- [Point process model (TMTPP)](06b_point_process.md) — the generative foundation: temporal intensity $\lambda(t)$, marks $f(Q)$, and detection thinning $P_d(\cdot)$.
- [Persistency](06c_persistency.md) — operational forecasts from inverted $\lambda(t)$: wait times, dispatch windows, occurrence probabilities.
- [Total emission estimation](06d_total_emission.md) — the missing-mass paradox and POD-corrected regional/national totals.

---

(tier5-loglik)=
## TMTPP foundations — the three-term log-likelihood

The full population log-likelihood has three terms (derived in [06b](06b_point_process.md); foundations in {cite:p}`daley2003,daleyVereJones2008`):

```{math}
:label: eq-tier5-loglik
\log L \;=\; \underbrace{\sum_{i \in \mathcal{D}} \log p(\text{detected}_i \mid f, \lambda, P_d)}_{\text{mark contribution}}
\;+\; \underbrace{\sum_{i \in \mathcal{D}} \log \lambda(t_i)}_{\text{detection-time intensity}}
\;-\; \underbrace{\int_{0}^{T} \lambda(t) \left[ \int P_d(Q)\, f(Q)\, \mathrm{d}Q \right] \mathrm{d}t}_{\text{integrated thinned rate}}
```

The third term is what makes $\lambda$ and $P_d$ jointly identifiable — without it the two trade off.

(tier5-mark-contribution)=
### Mark contribution and the soft-observation framing

The per-event posterior from Tiers I–IV is a **soft observation** of the (unknown) true mark $Q_i$. This is the same Bayesian-deconvolution / errors-in-variables structure used in measurement-error regression. The per-event mark contribution is:

```{math}
:label: eq-tier5-mark
p(\text{detected}_i \mid f, \lambda, P_d) \;=\; \int P_d(Q)\, L_i(Q)\, f(Q)\, \mathrm{d}Q
```

where $L_i(Q) = p(\text{observation}_i \mid Q)$ is the per-event **likelihood**, not the posterior. In sample-based practice (per-event posterior samples $Q_i^{(s)} \sim p(Q \mid \text{observation}_i)$):

```{math}
:label: eq-tier5-importance-weights
p(\text{detected}_i \mid f, \lambda, P_d) \;\approx\; \frac{1}{S}\, \sum_{s=1}^{S}\, P_d(Q_i^{(s)})\, \frac{f(Q_i^{(s)})}{\pi_\text{per-event}(Q_i^{(s)})}
```

with $\pi_\text{per-event}(Q)$ the **per-event prior** used at Tier I–IV. The ratio $f / \pi_\text{per-event}$ is the importance weight that re-points the per-event posterior at the population mark distribution.

:::{important} Importance correction is mandatory
**Without this re-weighting the population fit double-counts the per-event prior** — biased posterior on $f$, biased total-mass estimate.

**Why:** the per-event posterior already absorbs $\pi_\text{per-event}$; using its samples directly under $f$ multiplies the prior in. The IS ratio is the standard fix.
**How to apply:** every per-event posterior consumed by the population fit must carry its prior log-density; the importance weight is computed at fit time.
:::

This is the central math of cross-tier inference. Currently the prototype in [`methane_pod.fitting`](../../../methane_pod/src/methane_pod/fitting.py) summarises per-event posteriors to point estimates before the population fit, side-stepping the importance correction. Formalising this is the v1 deliverable for [`06a_instantaneous.md`](06a_instantaneous.md).

---

(tier5-cycle)=
## How the cycle adapts at population scale

The six-step cycle still applies, but the objects change:

```{list-table} Six-step cycle adaptation: Tier I–IV (per event) vs. Tier V (population).
:label: tbl-tier5-cycle
:header-rows: 1

* - Step
  - Tier I–IV (per event)
  - Tier V (population)
* - 1 — Simple model
  - Forward physics (plume / PDE / RTM)
  - Generative TMTPP: $\lambda(t)$ + mark $f(Q)$ + POD $P_d(\cdot)$
* - 2 — Model-based inference
  - MAP / MCMC over source params
  - NumPyro NUTS over $(\lambda \text{ params}, \text{mark params}, \text{POD params})$. Cheap at $O(10^{4})$ events (minutes); hours-to-days at $O(10^{6})$ events (national catalog)
* - 3 — Model emulator
  - FNO / neural ODE on the PDE
  - Skip when NUTS fits in budget. Optionally a normalising flow over the population posterior for repeated re-fits or sensitivity studies
* - 4 — Emulator-based inference
  - PDE-free 4D-Var
  - Variational fit (`numpyro.infer.SVI`) or flow-based posterior approximation; required at national catalog scale
* - 5 — Amortized predictor
  - Per-overpass $Q$ predictor
  - $(\text{basin tile}, \text{history window}) \to$ posterior over $(\lambda, f(Q), \text{total mass}, \text{next-event time})$ conditioned on per-event evidence and met-region context
* - 6 — Improve
  - Better physics
  - Spatial point process (links to Tier III); multi-satellite fusion; varying-coefficient POD (per-(basin, season, scene class) hierarchy); non-Poisson clustering (Hawkes / Cox)
```

**Tile definition:** an H3 hex-resolution-7 cell (~5 km²) for sub-basin work, or a basin polygon for inventory accounting. History window: 30–365 days, hierarchical prior on the cutoff.

**Varying-coefficient POD:** $P_d$ parameters indexed by $(\text{basin}, \text{season}, \text{scene class})$ with hierarchical shrinkage to the global POD. Captures regional / seasonal detection differences without inflating parameter count.

:::{tip} Tier V doesn't re-derive physics
The Tier V "forward model" is a *generative process for events*, not a PDE for fields. Mass conservation, advection, etc. are inherited from Tiers I–IV through the per-event posteriors — Tier V does not re-derive them.
:::

---

(tier5-cross-tier)=
## Cross-tier interface — the load-bearing contract

(tier5-payload-schema)=
### Payload schema

Every per-event posterior consumed by Tier V must carry:

```{list-table} Per-event posterior payload — fields, types, notes.
:label: tbl-tier5-payload
:header-rows: 1

* - Field
  - Type
  - Notes
* - `posterior_samples`
  - `(S,)` array of $Q$ draws
  - OR `posterior_summary` for Gaussian shorthand
* - `posterior_summary`
  - $(\mu_{\log Q}, \sigma_{\log Q})$
  - lognormal quick form when full samples are too heavy
* - `per_event_prior_logpdf`
  - callable $Q \to \log \pi(Q)$
  - **required** for the importance correction; without it the population fit is biased
* - `instrument_id`
  - str
  - dispatch into per-instrument POD
* - `t_detection`
  - float (UTC seconds)
  - for $\lambda(t)$
* - `x0_posterior`
  - $(\boldsymbol{\mu}_{xy}, \boldsymbol{\Sigma}_{xy})$
  - for spatial Cox-process upgrade
* - `quality`
  - dict
  - confidence flags from the Tier I–IV quality bitmask
```

(tier5-independence-caveat)=
### Independence assumption — the v1 caveat

The factorised likelihood above assumes detections at different overpasses are independent. Two overpasses of the *same physical leak* (e.g. GHGSat then TROPOMI two days later) violate this.

:::{caution} v1 caveat — known bias direction
**Bias direction:** ignoring the dependence inflates effective sample size → posterior on $f$ is over-concentrated.

**v1:** assumes independence and screens at the catalog stage (collapse near-coincident detections to one event by spatial-temporal clustering).
**v2:** promotes to a hierarchical model with per-source latent state shared across overpasses — same machinery as Tier IV's $Q(t)$ stochastic process but at the population level.
:::

---

(tier5-validation)=
## Validation strategy

- **Population SBC.** Generate $(\lambda^{*}, f^{*}, P_d^{*})$, simulate the full thinned-and-marked catalog with a synthetic per-event posterior layer, fit, check rank statistics across all hyperparameters. The Tier V analogue of Tier-I synthetic recovery.
- **Importance-weight ESS diagnostic.** Per detection $i$, the IS estimator's effective sample size $\operatorname{ESS}_i = (\sum_s w_s)^{2} / \sum_s w_s^{2}$ is a health metric. Low ESS (e.g. $< S/10$) signals that the per-event posterior is far from the population mark distribution; the population fit is unreliable for that event. Report the ESS distribution as a fit diagnostic.
- **Per-event-prior swap-out.** Refit the population using a different per-event prior at Tier I–IV (re-run Tiers I–IV with $\pi_\text{per-event} = \operatorname{LogNormal}(0, 2)$ instead of the inventory-anchored prior). The population posterior on $f$ should not move beyond IS noise. If it does, the importance correction is mis-implemented.
- **Real-data benchmark.** Compare corrected total emission for a well-studied basin (Permian) to published bottom-up inventories ({cite:p}`epa_ghgi,scarpelli2020sectoral`, GHGRP) and top-down inverse-modelling estimates ({cite:p}`maasakkers2023ghgi,jacob2022quantifying`, Sherwin et al.) — see [06d](06d_total_emission.md).

---

(tier5-modules)=
## Module layout — depend on `methane_pod`, don't absorb it

`plumax` depends on the standalone [`methane_pod`](../../../methane_pod/) package (pinned `methane_pod >= 0.1, < 0.2` for v1); the population-scale code is not re-implemented. Rationale:

- `methane_pod` has its own audience (point-process methodologists), test suite, release cadence.
- `plumax` consumes it through a thin adapter that materialises Tier I–IV posteriors as inputs to `methane_pod.fitting`.
- Versioning stays clean — when `methane_pod` releases v0.X.Y, `plumax` pins to a known-good version.

```{list-table} Tier V module layout — concern, target module, status.
:label: tbl-tier5-modules
:header-rows: 1

* - Concern
  - Module
  - Status
* - Intensity registry $\lambda(t)$
  - [`methane_pod.intensity`](../../../methane_pod/src/methane_pod/intensity.py)
  - library ✓ (13 kernels)
* - POD registry $P_d(\cdot)$
  - [`methane_pod.pod_functions`](../../../methane_pod/src/methane_pod/pod_functions.py)
  - library ✓ (10 models)
* - Missing-mass MC simulator
  - [`methane_pod.paradox`](../../../methane_pod/src/methane_pod/paradox.py)
  - library ✓
* - NUTS fitter
  - [`methane_pod.fitting`](../../../methane_pod/src/methane_pod/fitting.py)
  - library ✓; **importance-correction integration ☐**
* - Per-event posterior summariser
  - `plume_simulation.population.adapter.summariser`
  - ☐
* - Per-event prior recall ($\pi_\text{per-event}$ lookup)
  - `plume_simulation.population.adapter.prior_recall`
  - ☐ — required for importance weighting
* - Importance-weight calculator
  - `plume_simulation.population.adapter.importance`
  - ☐
* - Multi-satellite POD union
  - `plume_simulation.population.adapter.pod_union`
  - ☐
* - Catalog schema (CSV / parquet)
  - `plume_simulation.population.adapter.schema`
  - ☐
* - Real-data CSV ingestion
  - `plume_simulation.population.ingest`
  - ☐ (placeholder in [`07_pod_fitting_mcmc.md`](../../../methane_pod/notebooks/07_pod_fitting_mcmc.md))
* - Population SBC harness
  - `plume_simulation.population.validation.sbc`
  - ☐
* - Importance-weight ESS diagnostic
  - `plume_simulation.population.validation.iw_ess`
  - ☐
* - Per-event-prior swap-out test
  - `plume_simulation.population.validation.prior_swap`
  - ☐
* - Spatial Cox-process extension (v2)
  - `plume_simulation.population.spatial`
  - ☐
```

A `plume_simulation.population` subpackage doesn't exist yet; this is the proposed shape.

---

(tier5-tier3-link)=
## Connection to Tier III — spatial structure

Tier III's distributed source field $S(\mathbf{x},t)$ is **exactly a spatial inhomogeneous Poisson rate** at the population level — temporally aggregated, this *is* the spatial intensity of a Cox process over emission events. The v2 spatial extension of Tier V is the same mathematical object Tier III already inverts at the per-event timescale, just averaged over a longer horizon. The two tiers should share the parameterisation: a Matérn GP prior on $\log S(\mathbf{x},t)$ plays the role of both Tier III's source-field prior and Tier V.v2's spatial Cox-process intensity.

This isn't a coincidence — it's why `plumax`'s tier structure works: the same mathematical objects appear at different scales.

---

(tier5-status)=
## Status snapshot

- **Theory.** TMTPP foundations and the missing-mass paradox are written up in [`methane_pod/notebooks/01_mttpp_theory`](../../../methane_pod/notebooks/01_mttpp_theory.md) and [`03_missing_mass_paradox`](../../../methane_pod/notebooks/03_missing_mass_paradox.ipynb).
- **`methane_pod` library:** ✓ — intensity, POD, paradox simulator, NUTS fitter all implemented.
- **`plumax.population` subpackage (upstream):** 🚧 — the v1 cross-tier catalog adapter (`event_from_posterior` over Gaussian / lognormal / fusion posteriors), the V.A hierarchical lognormal size-distribution fit with per-event uncertainty propagation, and the V.B point-process core (closed-form Gamma–Poisson rate + log-linear inhomogeneous intensity) have landed. Not yet ported into this repo.
- **Cross-tier integration:** 🚧 — per-event posteriors now enter the population fit as Gaussian summaries `(emission_rate, emission_std)` with uncertainty propagation; the importance-corrected full-sample path (carrying $\pi_\text{per-event}$) remains Tier V's main outstanding code deliverable.
- **Synthetic validation.** [`06_stationary_numpyro_mcmc`](../../../methane_pod/notebooks/06_stationary_numpyro_mcmc.ipynb) recovers POD parameters on synthetic data without the soft-observation layer.
- **Real-data fit.** [`07_pod_fitting_mcmc`](../../../methane_pod/notebooks/07_pod_fitting_mcmc.md) is a placeholder; needs IMEO + Tanager CSV ingestion.

---

(tier5-open-questions)=
## Open questions

:::{attention} Per-event independence — quantitative bias
v1 assumption. Bias direction is known (over-concentration on $f$). Open: order-of-magnitude — for a basin with ~30% multi-instrument coincidences, how much does the posterior over-contract? Pilot study needed.
:::

:::{attention} Multi-satellite POD aggregation
Either $P_d^{\cup}(Q) = 1 - \prod_k (1 - P_d^{k}(Q))$ (independent detection chances) or marginalise over which satellite actually looked (categorical mark). The two are not equivalent. v1: union; v2: categorical when per-satellite attribution matters for inventory.
:::

:::{attention} Non-Poisson clustering
Real super-emitters cluster (compressor cycles, equipment lifecycle). Two upgrade paths: (a) **Hawkes / self-exciting kernel** when clustering is event-driven; (b) **Cox process with stochastic intensity** (latent OU on $\log \lambda$) when clustering is environmentally driven. Pick by basin diagnostic.
:::

:::{attention} Spatial structure
Currently temporal-only. Spatial extension via Cox process over wells is the natural v2 (and ties to Tier III, see above). Open: H3 hex-resolution-7 vs. continuous Matérn-GP intensity — operational vs. physical fidelity trade-off.
:::

:::{attention} Forecasting horizon
[Persistency](06c_persistency.md) gives wait-time forecasts. Open: how far out does the forecast remain useful? Likely instrument-cadence-dependent (TROPOMI daily vs. GHGSat tasked).
:::

:::{attention} Hierarchy depth for varying-coefficient POD
Three levels (global → basin → season) is tractable; four (+ scene class) starts to over-parameterise. Open: which factor matters most, by ELPD / WAIC.
:::
