# Tier V.A — Instantaneous emission estimation

**Question:** Given one satellite overpass with a detected plume, what is the posterior over the source's instantaneous emission rate `Q`?

This sub-page is the **glue** between the per-event physics tiers (I–IV) and the population tier (V). It defines the formal interface that turns a per-event posterior into evidence for the population-scale fit. The cross-tier payload schema is pinned in the [Tier V index](06_tier5_population.md#cross-tier-interface--the-load-bearing-contract); this page formalises how each payload is converted into a mark-likelihood contribution.

---

## Two regimes of "instantaneous Q"

### A. Full-physics posterior (Tier I / II / III output)

When you have radiance / column data and the wind field, run the per-event inversion described in Tier I (or II / III for richer transport):

```
p(Q | overpass_i) ∝ L_i(Q) · π_per-event(Q),     L_i(Q) = p(observation_i | Q)
```

This is the **gold-standard per-event evidence** — calibrated, with proper UQ, accounting for wind, transport, and instrument noise. Both `L_i(Q)` and the prior `π_per-event(Q)` are exposed downstream because the population fit needs to **divide out** the prior (see [§ Mark likelihood](#mark-likelihood--importance-weighted-monte-carlo) below).

### B. Catalog `Q` with wind-source consistency rescaling

For published plume catalogs (IMEO, Carbon Mapper alerts, Tanager monthly reports, GHGSat releases), what's available is typically a single point estimate of `Q` (mass flux, t/h or kg/s) per detection plus a per-event uncertainty estimate.

Catalog `Q` is **not** `Q/U` — it's already wind-multiplied. Operational catalogs use the **IME (Integrated Mass Enhancement) method**: `Q_catalog ≈ IME × U_catalog / L_plume`, where `U_catalog` is the wind value the catalog producer used at retrieval time. When fusing catalogs that used different wind sources (IMEO uses GEOS-FP, Tanager uses HRRR, Carbon Mapper uses ECMWF, GHGSat uses its 1 km downscaled product), inconsistent winds give inconsistent `Q`. The fix is **wind-source consistency rescaling**:

```
Q_corrected = Q_catalog · (U_target / U_catalog)
```

with `U_target` from a single agreed-upon reanalysis (default: ERA5). Re-multiplying by `U_target` is *not* a proxy for missing physics — it's a consistency correction across heterogeneous catalogs.

**Per-event uncertainty** comes from controlled-release calibration. Sherwin et al. (2024) report 1σ log-scale errors per instrument:

| Instrument | Per-event 1σ (log scale) |
|------------|--------------------------|
| TROPOMI | ~0.50 (≈ ±50%) |
| GHGSat | ~0.25 (≈ ±25%) |
| EMIT | ~0.30 |
| Tanager | ~0.30 |

Catalogs that report `±50%` are reporting the TROPOMI default; don't assume it for other instruments.

---

## Mark likelihood — importance-weighted Monte Carlo

The TMTPP mark-likelihood contribution at detection `i` is (see [Tier V index § TMTPP foundations](06_tier5_population.md#tmtpp-foundations--the-three-term-log-likelihood)):

```
p(detected_i | f, λ, P_d) = ∫ P_d(Q) · L_i(Q) · f(Q) dQ
```

with `L_i(Q) = p(observation_i | Q)` the per-event **likelihood**, not the posterior. In sample-based practice (per-event posterior samples `Q_i^{(s)} ~ p(Q | observation_i)`):

```
p(detected_i | f, λ, P_d)  ≈  (1/S) Σ_s  P_d(Q_i^{(s)}) · f(Q_i^{(s)}) / π_per-event(Q_i^{(s)})
```

The ratio `f / π_per-event` is the importance weight that re-points the per-event posterior at the population mark distribution. **Without this re-weighting the population fit double-counts the per-event prior** — biased posterior on `f`.

### Three implementation regimes — with importance correction in each

| Regime | Per-event input | Mark integration |
|--------|-----------------|------------------|
| **Point** | `Q̂_i` (MAP / median) | `P_d(Q̂_i) · f(Q̂_i) / π_per-event(Q̂_i)`; ignores per-event uncertainty |
| **Gaussian summary** | `(μ_logQ, σ²_logQ)` (lognormal) | Closed form when `f` is a power-law: `∫ Q^{−α} LogNormal(Q\|μ,σ²) dQ = exp(−αμ + ½α²σ²)`. The full integrand requires a `P_d` model — for sigmoidal POD with logistic form, the integral is tractable via Gauss-Hermite quadrature (≤ 10 nodes for 4-decimal accuracy) |
| **Full posterior** | sample set `{Q_i^{(s)}}` from Tier I–IV MCMC | `(1/S) Σ_s P_d(Q_i^{(s)}) · f(Q_i^{(s)}) / π_per-event(Q_i^{(s)})` — the importance-weighted MC estimator |

Each row evaluates the **same** `P_d · f / π` integrand; the only difference is the per-event posterior representation.

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
- **Point** acceptable only when per-event CV < 20% AND `f` is approximately constant on that scale.
- **Gaussian summary** suffices when CV < 50% AND `f ∈ {power-law, lognormal, gamma}`.
- **Full posterior** required when CV > 50% (typical for detection-floor, very-low-Q events), when `f` has structure on the per-event uncertainty scale, or when the per-event posterior is multimodal.

---

## Detection-floor and non-detection — explicit handling

### Detected events with posterior mass below the threshold

The importance-weighted MC handles this **automatically** — `P_d(Q^{(s)})` is small for small `Q^{(s)}` so the contribution is naturally downweighted. No special code path. **Just don't point-summarise** these events; promote to Full-posterior regime.

### Non-detection events (catalog gaps)

Non-detections do **not** flow through the per-event payload — there's no posterior to ingest. They contribute through the integrated thinned-rate term in the TMTPP likelihood:

```
−∫₀^T λ(t) · [∫ P_d(Q) f(Q) dQ] dt
```

What the cross-tier interface needs from the catalog: **per-instrument overpass coverage** (which times each instrument was looking at the basin), so the integral can be computed correctly.

The catalog ingestion module owns the distinction between "non-detect" and "noisy-detect-near-floor" — it's a catalog/ingestion concern, not a per-event-payload concern.

---

## Multi-source per overpass

Tier I Step 6 (RJMCMC), Tier IV §1 (`K = n_sources` first-class), and Tier V index all assume `K > 1` is supported. The mark-likelihood contribution at a multi-source overpass is a product over sources, assuming **within-overpass independence**:

```
p(detections_i | f, ...) ≈ Π_k  (1/S) Σ_s  [ P_d(Q_k^{(s)}) · f(Q_k^{(s)}) / π_per-event(Q_k^{(s)}) ]
```

Per-event payload for `K_i > 1` is a list of `K_i` sub-payloads, each with its own samples / summary / prior. v2 relaxes within-overpass independence — sources sharing a met realisation have correlated marks.

### Per-event payload schema (full)

Pinned in the [Tier V index](06_tier5_population.md#payload-schema); reproduced here for the implementation cycle:

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

## Catalog ingestion — heterogeneous sources

Per-source ingestion adapters because schemas, units, wind sources, and quality conventions differ:

| Catalog | Format | Units | Wind source | Quality flags |
|---------|--------|-------|-------------|---------------|
| IMEO (UNEP) | CSV | t/h | GEOS-FP | provider-supplied |
| Tanager monthly | parquet | kg/s | HRRR | confidence tier |
| Carbon Mapper alerts | JSON | kg/h | ECMWF | per-pixel mask |
| GHGSat releases | CSV | t/h | 1-km downscaled GHGSat product | binary detection |

Each ingestion adapter normalises to the internal `PerEventPayload`, applies wind-source consistency rescaling against `U_ERA5`, and emits a unified catalog with explicit provenance fields.

### De-duplication

Same physical leak detected by multiple satellites → multiple catalog rows. v1's independence assumption (see [Tier V index § Independence assumption](06_tier5_population.md#independence-assumption--the-v1-caveat)) requires **de-duplication before the population fit**. Default rule: spatial-temporal clustering with thresholds `(Δd ≤ 5 km, Δt ≤ 12 h)`. Multi-instrument detections of the same cluster either collapse to one event with a fused per-event payload (preferred, when posteriors are compatible) or to the highest-confidence detection (fallback).

---

## Module layout

| Concern | Module | Status |
|---------|--------|--------|
| Per-event posterior (Tier I) | [`gauss_plume.inference`](src/plume_simulation/gauss_plume/inference.py) | ✓ — needs to emit `per_event_prior_logpdf` |
| Per-event posterior (Tier II/III) | [`assimilation.solve`](src/plume_simulation/assimilation/solve.py) | 🚧 |
| Per-event posterior export adapter | tier-specific `posterior_export` modules | ☐ |
| Per-event payload summariser | `plume_simulation.population.adapter.summariser` | ☐ |
| Per-event prior recall | `plume_simulation.population.adapter.prior_recall` | ☐ |
| Importance-weight calculator | `plume_simulation.population.adapter.importance` | ☐ |
| Regime selector | `plume_simulation.population.adapter.regime` | ☐ |
| De-duplication / spatial-temporal clustering | `plume_simulation.population.adapter.dedup` | ☐ |
| Wind-source consistency rescaling | `plume_simulation.population.proxy.wind_rescale` | ☐ |
| Catalog ingest — IMEO | `plume_simulation.population.ingest.imeo` | ☐ |
| Catalog ingest — Tanager | `plume_simulation.population.ingest.tanager` | ☐ |
| Catalog ingest — Carbon Mapper | `plume_simulation.population.ingest.carbon_mapper` | ☐ |
| Catalog ingest — GHGSat | `plume_simulation.population.ingest.ghgsat` | ☐ |
| Per-instrument overpass coverage (for non-detection integral) | `plume_simulation.population.ingest.coverage` | ☐ |

---

## Validation strategy

- **Round trip on synthetic releases.** Generate a known `Q*`, run Tier I forward → noisy observation → Tier I inversion → check `Q*` sits in the reported credible region. Standard sanity check.
- **Importance-correction round trip.** Generate per-event posteriors using one prior `π_per-event = LogNormal(0, 1.5)`; run the population fit. Re-run with `π_per-event = LogNormal(0, 0.5)` (informative). The recovered population posterior on `f` should not move beyond IS noise. **Failure here means the importance correction is mis-implemented** — single most diagnostic test.
- **Importance-weight ESS diagnostic.** Synthetic scenario where `f` is a wide power-law and per-event posteriors are tight lognormals far from the bulk of `f`. The ESS-per-event report should warn (ESS ≪ S); the population fit should not silently absorb biased estimates.
- **Regime-selector consistency.** All three regimes (Point, Gaussian, Full) on the same per-event posterior should give the same population posterior modulo regime-appropriate noise. Catches importance-correction bugs in the Gaussian closed-form.
- **Wind-source consistency rescaling.** Synthesize a catalog with `U_catalog = 1.5 · U_target` for half the rows; before rescaling, the population fit on `f` should be biased; after rescaling, bias gone.
- **De-duplication test.** Two synthetic catalogs that are identical up to instrument label; correct de-duplication should collapse them to a single-instrument result.
- **Catalog-vs-full-inversion bias.** Run the full Tier I inversion on a synthetic radiance, compute the `Q` you'd get from the IME method on the same data. Quantify systematic bias direction (Sherwin et al. 2024 documents this; replicate).
- **Proxy idempotency.** Apply the per-event → mark-likelihood adapter on a degenerate case (point posterior, identity POD, uniform `f`): the population fit should reduce exactly to a uniform-weight max-likelihood fit on the point estimates. Catches indexing / weighting bugs.

---

## Open questions

- **Within-overpass multi-source dependence.** v1 assumes independence within an overpass; v2 needs per-overpass shared met latent. When does the bias matter? Probably only when multiple sources are close enough that the same wind realisation drives both plumes.
- **De-duplication thresholds.** `(5 km, 12 h)` is a starting point. Open: tune empirically per basin (Permian wells are denser than Marcellus); per-instrument footprint sets the spatial floor.
- **Gaussian-summary closed form for non-power-law `f`.** Power-law + lognormal evidence is closed-form via Gauss-Hermite. Other operational mark families (Pareto, gamma, mixture-of-lognormals) need either tabulated quadrature or numerical integration. Open: which families are in the v1 mark catalog?
- **Per-event payload storage cost.** Full posterior samples at `S = 10⁴` × 8 bytes × 4 fields × 10⁶ events ≈ 320 GB national catalog. Is the Gaussian summary good enough at population scale, or do we need on-the-fly resampling from per-event flow representations?
- **Catalog provenance audit.** Each catalog row should carry the wind source, the IME-method variant, and the retrieval algorithm. Currently most catalogs are sparse on this. Open: a strict ingestion mode that rejects rows without provenance vs. a permissive mode that fills with defaults.
