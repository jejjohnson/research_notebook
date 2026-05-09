---
title: "Tier V.A — Instantaneous emission estimation"
short_title: "Tier V.A — Instantaneous"
subject: "plumax — per-event → population glue"
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [instantaneous emission, IME, importance-weighted MC, mark likelihood, catalog ingestion, wind-source rescaling, de-duplication, payload schema]
---

# Tier V.A — Instantaneous emission estimation

**Question:** Given one satellite overpass with a detected plume, what is the posterior over the source's instantaneous emission rate $Q$?

This sub-page is the **glue** between the per-event physics tiers (I–IV) and the population tier (V). It defines the formal interface that turns a per-event posterior into evidence for the population-scale fit. The cross-tier payload schema is pinned in the [Tier V index](06_tier5_population.md#tier5-cross-tier); this page formalises how each payload is converted into a mark-likelihood contribution.

---

(va-two-regimes)=
## Two regimes of "instantaneous Q"

(va-regime-fullphysics)=
### A. Full-physics posterior (Tier I / II / III output)

When you have radiance / column data and the wind field, run the per-event inversion described in Tier I (or II / III for richer transport):

```{math}
:label: eq-va-full-posterior
p(Q \mid \text{overpass}_i) \;\propto\; L_i(Q) \cdot \pi_\text{per-event}(Q), \qquad L_i(Q) \;=\; p(\text{observation}_i \mid Q)
```

This is the **gold-standard per-event evidence** — calibrated, with proper UQ, accounting for wind, transport, and instrument noise. Both $L_i(Q)$ and the prior $\pi_\text{per-event}(Q)$ are exposed downstream because the population fit needs to **divide out** the prior (see [§ Mark likelihood](#va-mark-likelihood) below).

(va-regime-catalog)=
### B. Catalog $Q$ with wind-source consistency rescaling

For published plume catalogs (IMEO, Carbon Mapper alerts {cite:p}`carbon_mapper`, Tanager monthly reports, GHGSat releases {cite:p}`ghgsat,jervis2021ghgsat`), what's available is typically a single point estimate of $Q$ (mass flux, t/h or kg/s) per detection plus a per-event uncertainty estimate.

Catalog $Q$ is **not** $Q/U$ — it's already wind-multiplied. Operational catalogs use the **IME (Integrated Mass Enhancement) method** {cite:p}`varon2018quantifying`: $Q_\text{catalog} \approx \text{IME} \cdot U_\text{catalog} / L_\text{plume}$, where $U_\text{catalog}$ is the wind value the catalog producer used at retrieval time. When fusing catalogs that used different wind sources (IMEO uses GEOS-FP, Tanager uses HRRR, Carbon Mapper uses ECMWF, GHGSat uses its 1 km downscaled product), inconsistent winds give inconsistent $Q$. The fix is **wind-source consistency rescaling**:

```{math}
:label: eq-va-wind-rescale
Q_\text{corrected} \;=\; Q_\text{catalog} \cdot \frac{U_\text{target}}{U_\text{catalog}}
```

with $U_\text{target}$ from a single agreed-upon reanalysis (default: ERA5).

:::{caution} Re-multiplying by $U_\text{target}$ is *not* a proxy for missing physics
It's a consistency correction across heterogeneous catalogs. Don't claim it adds information; it removes inconsistency.
:::

**Per-event uncertainty** comes from controlled-release calibration. Sherwin et al. (2024) report 1σ log-scale errors per instrument:

```{list-table} Per-instrument 1σ (log-scale) per-event errors from Sherwin et al. 2024 controlled-release flights.
:label: tbl-va-per-event-sigma
:header-rows: 1

* - Instrument
  - Per-event 1σ (log scale)
* - TROPOMI ({cite:p}`s5p_tropomi`)
  - ~0.50 (≈ ±50%)
* - GHGSat ({cite:p}`ghgsat`)
  - ~0.25 (≈ ±25%)
* - EMIT ({cite:p}`emit`)
  - ~0.30
* - Tanager ({cite:p}`carbon_mapper`)
  - ~0.30
```

:::{caution} Don't reuse the TROPOMI default
Catalogs that report ±50% are reporting the TROPOMI default; don't assume it for other instruments.
:::

---

(va-mark-likelihood)=
## Mark likelihood — importance-weighted Monte Carlo

The TMTPP mark-likelihood contribution at detection $i$ is (see [Tier V index § TMTPP foundations](06_tier5_population.md#tier5-loglik)):

```{math}
:label: eq-va-mark-integral
p(\text{detected}_i \mid f, \lambda, P_d) \;=\; \int P_d(Q)\, L_i(Q)\, f(Q)\, \mathrm{d}Q
```

with $L_i(Q) = p(\text{observation}_i \mid Q)$ the per-event **likelihood**, not the posterior. In sample-based practice (per-event posterior samples $Q_i^{(s)} \sim p(Q \mid \text{observation}_i)$):

```{math}
:label: eq-va-importance-mc
p(\text{detected}_i \mid f, \lambda, P_d) \;\approx\; \frac{1}{S}\, \sum_{s=1}^{S}\, P_d(Q_i^{(s)})\, \frac{f(Q_i^{(s)})}{\pi_\text{per-event}(Q_i^{(s)})}
```

The ratio $f / \pi_\text{per-event}$ is the importance weight that re-points the per-event posterior at the population mark distribution.

:::{important} Without re-weighting, the population fit double-counts the prior
Posterior on $f$ becomes biased; downstream total-mass estimates inherit the bias.
:::

(va-three-regimes)=
### Three implementation regimes — with importance correction in each

```{list-table} Mark-integration regimes by per-event posterior representation.
:label: tbl-va-regimes
:header-rows: 1

* - Regime
  - Per-event input
  - Mark integration
* - **Point**
  - $\hat{Q}_i$ (MAP / median)
  - $P_d(\hat{Q}_i) \cdot f(\hat{Q}_i) / \pi_\text{per-event}(\hat{Q}_i)$; ignores per-event uncertainty
* - **Gaussian summary**
  - $(\mu_{\log Q}, \sigma^{2}_{\log Q})$ (lognormal)
  - Closed form when $f$ is a power-law: $\int Q^{-\alpha}\, \operatorname{LogNormal}(Q \mid \mu, \sigma^{2})\, \mathrm{d}Q = \exp(-\alpha\mu + \tfrac{1}{2}\alpha^{2}\sigma^{2})$. The full integrand requires a $P_d$ model — for sigmoidal POD with logistic form, the integral is tractable via Gauss–Hermite quadrature (≤ 10 nodes for 4-decimal accuracy)
* - **Full posterior**
  - sample set $\{Q_i^{(s)}\}$ from Tier I–IV MCMC
  - $(1/S)\, \sum_s P_d(Q_i^{(s)}) \cdot f(Q_i^{(s)}) / \pi_\text{per-event}(Q_i^{(s)})$ — the importance-weighted MC estimator
```

Each row evaluates the **same** $P_d \cdot f / \pi$ integrand; the only difference is the per-event posterior representation.

(va-regime-rule)=
### Regime selection rule

```python
def pick_regime(per_event: PosteriorPayload, mark_class: type[MarkDistribution]) -> Regime:
    cv = per_event.coefficient_of_variation()
    if cv < 0.20 and mark_class.is_smooth_on_scale(cv):
        return "point"
    elif cv < 0.50 and mark_class in CONJUGATE_FAMILIES:
        return "gaussian"
    else:
        return "full"      # CV > 0.50 (detection-floor events), multimodal posteriors
```

Operational rule:

- **Point** acceptable only when per-event $\operatorname{CV} < 20\%$ AND $f$ is approximately constant on that scale.
- **Gaussian summary** suffices when $\operatorname{CV} < 50\%$ AND $f \in \{\text{power-law}, \text{lognormal}, \text{gamma}\}$.
- **Full posterior** required when $\operatorname{CV} > 50\%$ (typical for detection-floor, very-low-$Q$ events), when $f$ has structure on the per-event uncertainty scale, or when the per-event posterior is multimodal.

---

(va-detection-floor)=
## Detection-floor and non-detection — explicit handling

### Detected events with posterior mass below the threshold

The importance-weighted MC handles this **automatically** — $P_d(Q^{(s)})$ is small for small $Q^{(s)}$ so the contribution is naturally downweighted. No special code path. **Just don't point-summarise** these events; promote to Full-posterior regime.

### Non-detection events (catalog gaps)

Non-detections do **not** flow through the per-event payload — there's no posterior to ingest. They contribute through the integrated thinned-rate term in the TMTPP likelihood:

```{math}
:label: eq-va-thinned-rate
- \int_{0}^{T} \lambda(t) \left[\int P_d(Q)\, f(Q)\, \mathrm{d}Q\right] \mathrm{d}t
```

What the cross-tier interface needs from the catalog: **per-instrument overpass coverage** (which times each instrument was looking at the basin), so the integral can be computed correctly.

The catalog ingestion module owns the distinction between "non-detect" and "noisy-detect-near-floor" — it's a catalog/ingestion concern, not a per-event-payload concern.

---

(va-multi-source)=
## Multi-source per overpass

Tier I Step 6 (RJMCMC), Tier IV §1 ($K = n_\text{sources}$ first-class), and Tier V index all assume $K > 1$ is supported. The mark-likelihood contribution at a multi-source overpass is a product over sources, assuming **within-overpass independence**:

```{math}
:label: eq-va-multisource
p(\text{detections}_i \mid f, \dots) \;\approx\; \prod_{k} \frac{1}{S}\sum_{s=1}^{S}\, P_d(Q_k^{(s)})\, \frac{f(Q_k^{(s)})}{\pi_\text{per-event}(Q_k^{(s)})}
```

Per-event payload for $K_i > 1$ is a list of $K_i$ sub-payloads, each with its own samples / summary / prior. v2 relaxes within-overpass independence — sources sharing a met realisation have correlated marks.

(va-payload-schema)=
### Per-event payload schema (full)

Pinned in the [Tier V index](06_tier5_population.md#tier5-payload-schema); reproduced here for the implementation cycle:

```python
@dataclass
class PerEventPayload:
    sources: list[SourcePayload]               # K_i entries; K_i ≥ 1
    instrument_id: str                         # per-satellite POD dispatch
    t_detection: float                         # UTC seconds, for λ(t)
    quality: dict                              # bitmask from Tier I–IV

@dataclass
class SourcePayload:
    posterior_samples: jax.Array | None        # (S,) draws of Q
    posterior_summary: tuple[float, float] | None   # (μ_logQ, σ_logQ) lognormal shorthand
    per_event_prior_logpdf: Callable[[float], float]  # required for importance correction
    x0_posterior: tuple[jax.Array, jax.Array]  # (mu_xy, Cov_xy) — for spatial Cox v2 + de-dup
```

`per_event_prior_logpdf` is the load-bearing field — without it the importance correction can't be done.

---

(va-catalog-ingest)=
## Catalog ingestion — heterogeneous sources

Per-source ingestion adapters because schemas, units, wind sources, and quality conventions differ:

```{list-table} Catalog ingestion — schema, units, wind source, quality flags per provider.
:label: tbl-va-catalogs
:header-rows: 1

* - Catalog
  - Format
  - Units
  - Wind source
  - Quality flags
* - IMEO (UNEP)
  - CSV
  - t/h
  - GEOS-FP
  - provider-supplied
* - Tanager monthly ({cite:p}`carbon_mapper`)
  - parquet
  - kg/s
  - HRRR
  - confidence tier
* - Carbon Mapper alerts ({cite:p}`carbon_mapper`)
  - JSON
  - kg/h
  - ECMWF
  - per-pixel mask
* - GHGSat releases ({cite:p}`ghgsat`)
  - CSV
  - t/h
  - 1-km downscaled GHGSat product
  - binary detection
```

Each ingestion adapter normalises to the internal `PerEventPayload`, applies wind-source consistency rescaling against $U_\text{ERA5}$, and emits a unified catalog with explicit provenance fields.

(va-dedup)=
### De-duplication

Same physical leak detected by multiple satellites → multiple catalog rows. v1's independence assumption (see [Tier V index § Independence assumption](06_tier5_population.md#tier5-independence-caveat)) requires **de-duplication before the population fit**. Default rule: spatial-temporal clustering with thresholds $(\Delta d \leq 5\text{ km}, \Delta t \leq 12\text{ h})$. Multi-instrument detections of the same cluster either collapse to one event with a fused per-event payload (preferred, when posteriors are compatible) or to the highest-confidence detection (fallback).

---

(va-modules)=
## Module layout

```{list-table} Tier V.A module layout — concern, target module, status.
:label: tbl-va-modules
:header-rows: 1

* - Concern
  - Module
  - Status
* - Per-event posterior (Tier I)
  - [`gauss_plume.inference`](../../src/plume_simulation/gauss_plume/inference.py)
  - ✓ — needs to emit `per_event_prior_logpdf`
* - Per-event posterior (Tier II/III)
  - [`assimilation.solve`](../../src/plume_simulation/assimilation/solve.py)
  - 🚧
* - Per-event posterior export adapter
  - tier-specific `posterior_export` modules
  - ☐
* - Per-event payload summariser
  - `plume_simulation.population.adapter.summariser`
  - ☐
* - Per-event prior recall
  - `plume_simulation.population.adapter.prior_recall`
  - ☐
* - Importance-weight calculator
  - `plume_simulation.population.adapter.importance`
  - ☐
* - Regime selector
  - `plume_simulation.population.adapter.regime`
  - ☐
* - De-duplication / spatial-temporal clustering
  - `plume_simulation.population.adapter.dedup`
  - ☐
* - Wind-source consistency rescaling
  - `plume_simulation.population.proxy.wind_rescale`
  - ☐
* - Catalog ingest — IMEO
  - `plume_simulation.population.ingest.imeo`
  - ☐
* - Catalog ingest — Tanager
  - `plume_simulation.population.ingest.tanager`
  - ☐
* - Catalog ingest — Carbon Mapper
  - `plume_simulation.population.ingest.carbon_mapper`
  - ☐
* - Catalog ingest — GHGSat
  - `plume_simulation.population.ingest.ghgsat`
  - ☐
* - Per-instrument overpass coverage (for non-detection integral)
  - `plume_simulation.population.ingest.coverage`
  - ☐
```

---

(va-validation)=
## Validation strategy

- **Round trip on synthetic releases.** Generate a known $Q^{*}$, run Tier I forward → noisy observation → Tier I inversion → check $Q^{*}$ sits in the reported credible region. Standard sanity check.
- **Importance-correction round trip.** Generate per-event posteriors using one prior $\pi_\text{per-event} = \operatorname{LogNormal}(0, 1.5)$; run the population fit. Re-run with $\pi_\text{per-event} = \operatorname{LogNormal}(0, 0.5)$ (informative). The recovered population posterior on $f$ should not move beyond IS noise. **Failure here means the importance correction is mis-implemented** — single most diagnostic test.
- **Importance-weight ESS diagnostic.** Synthetic scenario where $f$ is a wide power-law and per-event posteriors are tight lognormals far from the bulk of $f$. The ESS-per-event report should warn ($\operatorname{ESS} \ll S$); the population fit should not silently absorb biased estimates.
- **Regime-selector consistency.** All three regimes (Point, Gaussian, Full) on the same per-event posterior should give the same population posterior modulo regime-appropriate noise. Catches importance-correction bugs in the Gaussian closed-form.
- **Wind-source consistency rescaling.** Synthesize a catalog with $U_\text{catalog} = 1.5 \cdot U_\text{target}$ for half the rows; before rescaling, the population fit on $f$ should be biased; after rescaling, bias gone.
- **De-duplication test.** Two synthetic catalogs that are identical up to instrument label; correct de-duplication should collapse them to a single-instrument result.
- **Catalog-vs-full-inversion bias.** Run the full Tier I inversion on a synthetic radiance, compute the $Q$ you'd get from the IME method ({cite:p}`varon2018quantifying`) on the same data. Quantify systematic bias direction (Sherwin et al. 2024 documents this; replicate).
- **Proxy idempotency.** Apply the per-event → mark-likelihood adapter on a degenerate case (point posterior, identity POD, uniform $f$): the population fit should reduce exactly to a uniform-weight max-likelihood fit on the point estimates. Catches indexing / weighting bugs.

---

(va-open-questions)=
## Open questions

:::{attention} Within-overpass multi-source dependence
v1 assumes independence within an overpass; v2 needs per-overpass shared met latent. When does the bias matter? Probably only when multiple sources are close enough that the same wind realisation drives both plumes.
:::

:::{attention} De-duplication thresholds
$(5\text{ km}, 12\text{ h})$ is a starting point. Open: tune empirically per basin (Permian wells are denser than Marcellus); per-instrument footprint sets the spatial floor.
:::

:::{attention} Gaussian-summary closed form for non-power-law $f$
Power-law + lognormal evidence is closed-form via Gauss–Hermite. Other operational mark families (Pareto, gamma, mixture-of-lognormals) need either tabulated quadrature or numerical integration. Open: which families are in the v1 mark catalog?
:::

:::{attention} Per-event payload storage cost
Full posterior samples at $S = 10^{4} \times 8\text{ bytes} \times 4\text{ fields} \times 10^{6}\text{ events} \approx 320$ GB national catalog. Is the Gaussian summary good enough at population scale, or do we need on-the-fly resampling from per-event flow representations?
:::

:::{attention} Catalog provenance audit
Each catalog row should carry the wind source, the IME-method variant, and the retrieval algorithm. Currently most catalogs are sparse on this. Open: a strict ingestion mode that rejects rows without provenance vs. a permissive mode that fills with defaults.
:::
