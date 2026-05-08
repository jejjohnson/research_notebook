---
title: Operational attribution on a unified stack
subject: Operational attribution on a unified stack
subtitle: A forward plan for MARS-style methane attribution rebuilt on `georeader` + `GeoCatalog` + `geotoolz` + `plumax`
short_title: Operational attribution
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: methane, attribution, plume, MARS, geotoolz, plumax, operational, satellite
---

> **Status.** Forward-looking plan. Projected pipeline assuming the unified stack lands. Treats everything except the current `georeader` package as future work.
>
> **Scope.** MARS-style methane source attribution from satellite observations. Three instruments (TROPOMI + GHGSat + EMIT). Covers **exploration, preprocessing, inference, and serving** — explicitly **not** model training.
>
> **Audience.** Three layers in one doc: an executive summary for funders/talks, an architectural plan for engineers, a validation + migration story for paper/blog framing.

---

## TL;DR

Methane attribution from satellite imagery — *"this scene shows a plume; what is the source rate, where is the source, with what uncertainty?"* — is what UNEP-IMEO's **Methane Alert and Response System (MARS)** does operationally today. It works. It's also rebuilt as bespoke glue code in nearly every research project that touches the same problem. *Across the institutions I've worked at — FIT, RIT, Universidad de Valencia, CNRS, CSIC, UNEP-IMEO, MARS — the same research-to-production gap appears every time, with the storage and transport layers getting steadily better while the **factory layer** (the science composition) stays hand-rolled.*

This doc projects what an operational attribution pipeline looks like once the unified stack lands: **`georeader` 2.0 (modernised reader Protocol), `GeoCatalog` (multi-satellite metadata index), `geotoolz`/`xrtoolz` (operator algebra), `plumax` (forward models)**. The same operator graph runs in:

1. **A research notebook** — exploration, single-event analysis, MAP via NumPyro.
2. **A batch pipeline** — last-week's-overpasses fanned out across detections, posteriors written to GeoParquet.
3. **An alert service** — a FastAPI handler that takes a detection event and returns a posterior in seconds.

One graph, three orchestration modes, zero rewrites between research and production. *That's* the substrate gap this stack is meant to close.

---

## 1. Why this doc exists

MARS-style attribution is a real, deployed operational system at UNEP-IMEO. **This is not a proposal to rebuild MARS.** The question this doc answers is different: *what would the right substrate look like for this entire class of operational attribution work — not just MARS, but every research-to-production translation of a satellite-driven Bayesian inversion?*

Two motivations sit underneath:

1. **The same pipeline shape recurs everywhere.** Methane plume attribution, NO₂ inversion, CO₂ flux estimation, oil-spill backtracking, wildfire-emission attribution — all are: *(satellite L1/L2 data) → (forward model with met forcing) → (Bayesian inversion) → (posterior on source parameters)*. The science differs; the **plumbing is identical**. Today every project rebuilds the plumbing.
2. **MARS is a proof point that the operational form exists.** The constraint isn't "can this be done"; it's "can this be done with a *unified, composable, reproducible* substrate that other groups can pick up and extend without re-implementing the IO layer". Plumax + geotoolz is that substrate.

This doc projects how the v1 plumax target — **Tier I + AK + multi-satellite L2 fusion across {TROPOMI, GHGSat, EMIT}** — would be implemented end-to-end on the unified stack, with three orchestration phases demonstrating the research-to-prod arc.

---

## 2. What MARS-style attribution does today (and where the seams are)

Operational attribution today is the composition of half a dozen well-understood pieces, each typically implemented as project-specific glue:

```text
                ┌──────────────────────────────────────────────────────┐
                │  Detection trigger                                   │
                │  (operator alert · matched-filter scan · TROPOMI Q4) │
                └──────────────────────┬───────────────────────────────┘
                                       │ event (lat, lon, time, instrument)
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Multi-instrument data assembly                             │
        │  • Pull EMIT scene for the event time + AOI                 │
        │  • Pull TROPOMI overpass(es) within the event window        │
        │  • Pull GHGSat target acquisition if available              │
        │  • Pull met forcing (WRF / ERA5) for the AOI + window       │
        └─────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Per-instrument preprocessing                               │
        │  • EMIT: matched-filter detection on radiance cube          │
        │  • TROPOMI: L2 XCH₄ + averaging-kernel + retrieval-prior    │
        │  • GHGSat: L2 XCH₄ + footprint                              │
        │  • Cloud + QA masks, native-resolution preserved            │
        └─────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Forward model + likelihood + inference                     │
        │  • Tier I Gaussian plume forward (Q, x₀, ū, θ_wind, …)      │
        │  • Per-instrument AK operator + bias correction             │
        │  • Joint multi-instrument likelihood at native resolution   │
        │  • MAP / MCMC / variational posterior                       │
        └─────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Reporting                                                  │
        │  • Posterior mean + uncertainty                             │
        │  • Provenance (which instruments contributed, met source)   │
        │  • Alert payload to operational responders                  │
        └─────────────────────────────────────────────────────────────┘
```

**Where the seams show up today:**

| Seam | Today's reality | Cost |
|---|---|---|
| Satellite ingest | Per-instrument scripts, often re-implemented per project; auth via env-var soup; no shared cache / metadata index | Multi-week ramp-up per new project + non-reproducible IO |
| Multi-instrument catalog | Per-project CSV / pandas notebooks; bespoke schema; no spatiotemporal index | Cross-project discovery is impossible; same data is re-curated 10× |
| Forward model | Per-paper code; matched filter often a research script; no operator surface | No reuse across projects; small bug fixes don't propagate |
| Inference loop | NumPyro / TensorFlow Probability / Stan, hand-wired to the forward; switching requires rewriting | Slow methods iteration |
| Operational delivery | Inference moved into a FastAPI / Cloud Function via cut-and-paste from the research notebook; metadata round-trips lost | Production code drifts from research code; can't validate prod against research |

The unified stack closes each of these seams with a typed, composable surface. The next sections show how.

---

## 3. The three-instrument scope

The v1 target is **TROPOMI + GHGSat + EMIT** — the same set the [Tier IV plumax design](../plume_simulation/notes/roadmap/05_tier4_coupled.md) targets for v1 multi-instrument fusion. Each instrument contributes a different role; the demo's value is precisely in the *fusion*, not in any single instrument.

| Instrument | Role | Resolution | Cadence | Native product | What it constrains |
|---|---|---|---|---|---|
| **[TROPOMI](https://sentinel.esa.int/web/sentinel/missions/sentinel-5p)** (S5P) | Wide-swath screening | ~5.5 × 7 km nadir | Daily global | L2 XCH₄ NetCDF | Regional context; rough source localisation; persistent emitters |
| **[GHGSat](https://www.ghgsat.com/)** (commercial constellation) | Targeted high-resolution | ~25 m | Tasked | L2 XCH₄ HDF5 | Single-source rate; tight localisation; intermittent leaks |
| **[EMIT](https://earth.jpl.nasa.gov/emit/)** (NASA / ISS) | Hyperspectral imaging spectrometer | ~60 m | ISS overpass cadence | L1B radiance NetCDF | Scene-context plume detection via matched filter; fingerprint validation |

**Why these three, why this fusion:**

- **Independent verification.** Three independent instruments observing the same plume with different physics (TROPOMI's full-disk averaging-kernel, GHGSat's pointed retrieval, EMIT's hyperspectral matched-filter) is the strongest possible evidence basis for an attribution claim. Single-instrument attributions are routinely contested.
- **Different latencies.** TROPOMI is ~near-real-time global; GHGSat is tasked (latency ~hours); EMIT is ISS-cadence (latency ~days). The fusion is *temporally heterogeneous* — `Q(t)` as a stochastic process (per [Tier IV §3](../plume_simulation/notes/roadmap/05_tier4_coupled.md)) handles this naturally.
- **Different state-vector constraints.** TROPOMI constrains regional background; GHGSat constrains source rate; EMIT validates the plume signature spectrally. Together they solve a system that no single instrument solves alone.
- **MARS pattern.** This three-instrument fusion is the operational reality at UNEP-IMEO. The unified stack should match that reality without forcing a re-architecture.

**What's explicitly out of scope for v1:** Sentinel-2 / Landsat (cloud-shadow vegetation-index methane proxies — useful but different physics); PRISMA / Tanager (similar to EMIT, additive once the EMIT pipeline works); CarbonMapper (commercial, schema TBD).

---

## 4. Mapping the unified stack to the attribution pipeline

Each layer of the [geostack](geostack_notes.md) maps onto a concrete piece of the attribution pipeline. The point of the table below is that **none of this is novel science** — the science modules already have well-understood mathematical structure; the substrate work is making them compose cleanly.

```text
        ┌─────────────────────────────────────────────────────────────┐
   USER │  notebook · alert API · operational dashboard               │
        └─────────────────────────▲───────────────────────────────────┘
                                  │
        ┌─────────────────────────┴───────────────────────────────────┐
  LOGIC │  ★ geotoolz operators (preprocessing, AK, masking, fusion)  │
        │  ★ plumax operators (transport, RTM, likelihood, inversion) │
        │  ★ xrtoolz operators (met-field reductions, climatology)    │
        └─────────────────────────▲───────────────────────────────────┘
                                  │
        ┌─────────────────────────┴───────────────────────────────────┐
   DISC │  ★ GeoCatalog: multi-satellite overpass + footprint index   │
        │  Phase 1: GeoPandas + IntervalIndex (≤10⁵ overpasses/basin) │
        │  Phase 2: DuckDB + GeoParquet (global, multi-year)          │
        └─────────────────────────▲───────────────────────────────────┘
                                  │
        ┌─────────────────────────┴───────────────────────────────────┐
   SUBS │  ★ georeader 2.0: GeoTensor carrier + per-sensor readers    │
        │  EMIT_L1B · TROPOMI_L2 · GHGSat_L2 · WRF · ERA5             │
        └─────────────────────────▲───────────────────────────────────┘
                                  │
        ┌─────────────────────────┴───────────────────────────────────┐
  TRANS │  obstore · GDAL/VSI                                         │
        └─────────────────────────▲───────────────────────────────────┘
                                  │
        ┌─────────────────────────┴───────────────────────────────────┐
   STOR │  cloud buckets · COG · NetCDF · GeoParquet · Zarr (met)     │
        └─────────────────────────────────────────────────────────────┘
```

Layer by layer, what each contributes to the attribution pipeline:

### 4.1 Storage

Already exists. EMIT lives on AWS / DAAC; TROPOMI on Copernicus / GES DISC; GHGSat on the vendor's portal; ERA5 on Copernicus CDS; WRF outputs on local compute. **No new storage** — the unified stack is about access patterns, not new repositories.

### 4.2 Transport — `obstore`

Single byte mover for all cloud buckets. Three concrete payoffs over the current per-project mix of `s3fs` / `boto3` / `gcsfs`:

- Concurrent range reads make per-event AOI extraction (e.g. EMIT 1 km × 1 km AOI from a 100 km × 50 km scene) feasible without downloading the whole scene.
- One credential model across S3 / GCS / Azure (where MARS partner data lives) — the [`Credential`](plans/types/credentials.md) Protocol abstracts the auth dance.
- Reproducibility — `obstore` calls are pure functions of `(url, range, credential)`. No global env-var state.

### 4.3 Substrate — modernised `georeader` + per-sensor readers

The most concrete user-visible payoff. Today's per-project ingest replaced with five typed readers conforming to one Protocol:

| Reader | Status today | What lands in the unified stack |
|---|---|---|
| `EMIT_L1B` (NetCDF radiance) | Per-project hand-rolled; matched-filter often a fork of the JPL reference code | Sensor-specific reader emitting `GeoTensor[*batch, bands, H, W]` with a `wavelength` attribute; matched-filter operator consumes it |
| `TROPOMI_L2` (NetCDF L2 XCH₄) | Per-project, often via `harp` or hand-coded `xarray` | Sensor reader emitting `GeoTensor[*time, H, W]` with averaging-kernel sidecar |
| `GHGSat_L2` (HDF5 L2 XCH₄) | Per-project, schema differs by product version | Sensor reader emitting `GeoTensor` with footprint + retrieval-prior metadata |
| `WRF` / `ERA5` (NetCDF / GRIB met) | `xarray.open_dataset(...)` per project | `xrtoolz` reader emitting `MetField` as a `coordax.Dataset` (per [`plumax/00_prerequisites.md`](../plume_simulation/notes/roadmap/00_prerequisites.md#metfield-schema)) |
| `inventory` (EDGAR / GFEI / EPA) | Per-project, often a single CSV import | Vector reader emitting a `GeoDataFrame`-typed prior on `q_a` |

Cross-cutting: `Credential` and `ByteStore` Protocols (per [`plans/types/`](plans/types/)) handle auth and transport for all of them.

### 4.4 Discovery — `GeoCatalog`

The single biggest leap from "hand-rolled per project" to "shared substrate". The catalog is a **multi-satellite spatiotemporal index** keyed on `(instrument, time, footprint, quality, native_url)`.

**Phase 1 (in-memory, per-basin / per-event):** a `geopandas.GeoDataFrame` plus `pd.IntervalIndex`, built once per session by crawling instrument-specific URL conventions. Sufficient for ≤10⁵ overpasses (a basin × multi-year window). Persisted as GeoParquet via `to_geoparquet()` so the same catalog feeds research and batch.

**Phase 2 (DuckDB on GeoParquet):** when the catalog goes operational and global, the same API but DuckDB-backed with predicate pushdown via the GeoParquet 1.1 bbox column. **Most of MARS's catalog needs probably stay in Phase 1** — operational attribution is event-driven, with each event filtering to a small AOI × time window where Phase 1 is more than fast enough.

**The query that anchors everything:**

```python
# "All overpasses within 2 hours of this detection, 50 km bbox around the source,
#  any of the three instruments, quality flag >= 'usable'."
catalog = GeoCatalog.from_geoparquet("s3://mars/catalog/overpasses_2026.parquet")
candidates = catalog.query(
    geometry=event.bbox(50_000),
    interval=event.window(hours=2),
    instruments=["EMIT", "TROPOMI", "GHGSat"],
    quality_min="usable",
)
```

That single query replaces the per-project glob+filter+join logic that today every attribution paper re-implements.

### 4.5 Compute — `geotoolz` × `plumax` × `xrtoolz`

The **factory layer** is where the science composition lives. Three libraries with a clean separation of concerns:

| Library | What it owns in the attribution pipeline |
|---|---|
| **`geotoolz`** | Generic raster operators: `MatchedFilter` (EMIT methane fingerprint), `ApplyAK` (averaging-kernel application), `Mask` (cloud + QA), `Crop` (AOI extraction), `Reproject`, `Stitch`, `Sequential`/`Graph` composition. |
| **`plumax`** | Domain-specific science operators: `GaussianPlume` (Tier I forward), `MetField` loader, `BiasCorrect` per instrument, `JointLikelihood` over instruments, `MAPInverter` / `NUTSInverter` wrappers. *Operators in the geotoolz sense — pickleable, Hydra-friendly, carrier-aware.* |
| **`xrtoolz`** | Met-field operators: `TimeInterpolate` (snapshot / piecewise / advected), `PBLDiagnostic`, `StabilityClassifier`, `RegridToAnalysisGrid`. Operates on `coordax.Dataset` carrier (per [`plumax` adoption](../plume_simulation/notes/roadmap/00_prerequisites.md#open-questions)). |

**The pipeline as one operator graph:**

```python
# Build once, call from notebook / batch / API
attribution = Graph(
    inputs={
        "emit": Input(GeoTensor),       # EMIT L1B radiance cube
        "tropomi": Input(GeoTensor),    # TROPOMI L2 XCH₄
        "ghgsat": Input(GeoTensor),     # GHGSat L2 XCH₄
        "met": Input(MetField),         # WRF or ERA5 met
        "prior": Input(GeoDataFrame),   # EDGAR / GFEI prior
    },

    # Per-instrument preprocessing (geotoolz operators)
    detect_emit       = MatchedFilter(target=ch4_signature) >> "emit",
    masked_tropomi    = Mask(qa="qa_value", min=0.5) >> "tropomi",
    masked_ghgsat     = Mask(qa="quality_flag", min=2) >> "ghgsat",

    # Met-field preparation (xrtoolz operators)
    pbl               = PBLDiagnostic() >> "met",
    stability         = StabilityClassifier() >> "met",
    met_interp        = TimeInterpolate(policy="piecewise") >> "met",

    # Forward model (plumax operators)
    forward_emit      = GaussianPlume(instrument="EMIT") >> ("met_interp", "stability"),
    forward_tropomi   = GaussianPlume(instrument="TROPOMI") >> ("met_interp", "stability"),
    forward_ghgsat    = GaussianPlume(instrument="GHGSat") >> ("met_interp", "stability"),

    # Per-instrument observation operator (geotoolz × plumax)
    sim_emit_radiance = ApplyRTM() >> ("forward_emit", "stability"),
    sim_tropomi       = ApplyAK(instrument="TROPOMI") >> "forward_tropomi",
    sim_ghgsat        = ApplyAK(instrument="GHGSat") >> "forward_ghgsat",

    # Bias correction (plumax)
    sim_emit_bc       = BiasCorrect(instrument="EMIT") >> "sim_emit_radiance",
    sim_tropomi_bc    = BiasCorrect(instrument="TROPOMI") >> "sim_tropomi",
    sim_ghgsat_bc     = BiasCorrect(instrument="GHGSat") >> "sim_ghgsat",

    # Joint multi-instrument likelihood (plumax)
    posterior         = JointLikelihood(instruments=["EMIT","TROPOMI","GHGSat"]) >> (
        ("detect_emit",     "sim_emit_bc"),
        ("masked_tropomi",  "sim_tropomi_bc"),
        ("masked_ghgsat",   "sim_ghgsat_bc"),
        "prior",
    ),
)
```

That's **the** graph. Notebook, batch worker, and FastAPI handler all instantiate this same `Graph` and call it. The orchestration changes around it; the graph itself is identical.

### 4.6 Serving

Three orchestration modes share the graph above. See §5 for each.

---

## 5. The three-phase demo plan

Each phase uses the **identical** operator graph from §4.5. What changes is the orchestration around it: how inputs are resolved, where outputs go, what the latency budget is.

### 5.1 Phase 1 — Research notebook

**Goal.** Validate that the unified-stack operator graph runs and produces a posterior consistent with hand-rolled MARS-style outputs on a known event.

**Anchor event.** A single Permian basin methane release with all three instruments observing within a 24-hour window. (Pick a published event from the literature so we have a published posterior to bench against.)

**Notebook flow:**

```python
import geotoolz as gz
import plumax as px
import xrtoolz as xr_t
from geocatalog import GeoCatalog

# 1. Build / load a small catalog around the event
catalog = GeoCatalog.from_basin_crawl(basin="permian", year=2024)
overpasses = catalog.query(
    geometry=event.bbox(50_000),
    interval=event.window(hours=12),
    instruments=["EMIT", "TROPOMI", "GHGSat"],
)

# 2. Resolve inputs via georeader 2.0
emit_gt    = read_emit_l1b(overpasses["EMIT"][0].url, slice=event.geoslice(50_000))
tropomi_gt = read_tropomi_l2(overpasses["TROPOMI"][0].url, slice=event.geoslice(50_000))
ghgsat_gt  = read_ghgsat_l2(overpasses["GHGSat"][0].url, slice=event.geoslice(50_000))
met        = read_wrf_metfield(event.window(hours=12), event.bbox(100_000))
prior      = read_edgar(year=2024, basin="permian")

# 3. Run the same graph as §4.5
attribution = build_attribution_graph()  # the §4.5 Graph
posterior_callable = attribution.compile()  # MAP via NumPyro under the hood

# 4. Inference
posterior = posterior_callable(
    emit=emit_gt, tropomi=tropomi_gt, ghgsat=ghgsat_gt,
    met=met, prior=prior,
)

# 5. Visualisation (matplotlib + lonboard for the geo bits)
plot_posterior(posterior, event)
plot_observations_vs_simulated(posterior, emit_gt, tropomi_gt, ghgsat_gt)
```

**What this validates:**

- Operator graph composes and runs end-to-end.
- Carrier-aware metadata propagates: the posterior carries `event.location` provenance back to the input AOI.
- Multi-instrument fusion produces a tighter posterior than any single instrument (the differentiated payoff vs current per-instrument MARS practice).
- Bench against published posterior for the chosen event — within published uncertainty.

**Estimated effort.** ~2–3 weeks of focused work after georeader 2.0 + the v0.1 geotoolz core land. Most of the work is in `plumax` operator wrappers (`GaussianPlume`, `JointLikelihood`, `BiasCorrect`) — the plumax tier-I math already exists in the roadmap; the work is wrapping it as Operators with `get_config` for Hydra serialisation.

### 5.2 Phase 2 — Batch pipeline

**Goal.** Demonstrate that the **same** operator graph runs at orchestrated scale: last week's detections fanned out across compute, posteriors written to a queryable artifact.

**Pipeline shape:**

```text
GeoCatalog (phase 1) ── query(week_window) ──▶ list of detection events
        │
        ▼
  for each event:                          (parallelism via Dask / Ray / Modal)
    ┌──────────────────────────┐
    │  resolve inputs (Φ.1)    │
    │  compile attribution     │
    │  run posterior           │
    │  write to GeoParquet     │
    └──────────────────────────┘
        │
        ▼
posteriors.parquet  (per-event posterior + provenance + alert score)
```

**Concretely:**

```python
import dask.distributed as dd
from plumax.batch import attribute_event

client = dd.Client()  # or Ray / Modal

events = catalog.query(
    interval=last_week,
    quality_min="usable",
    detection_channel=["matched_filter", "tropomi_q4"],
).to_events()

# attribute_event is a thin wrapper around the §5.1 logic — same operator graph
futures = [client.submit(attribute_event, event) for event in events]
posteriors = client.gather(futures)

# Persist
write_posteriors_geoparquet(posteriors, "s3://mars/posteriors/2026-W19.parquet")
```

**What this validates:**

- The graph is **pickleable** — Dask / Ray serialisation works without per-Operator special-casing. (CI test from [`geotoolz.md` §11.2](plans/geotoolz/geotoolz.md#112-implementation-gotchas-test-these-in-ci) directly applies.)
- Per-event parallelism scales linearly with worker count (no shared global state).
- Posteriors round-trip to GeoParquet with full provenance — the resulting file is a queryable record of every attribution the pipeline produced, hashable, reproducible.
- Catalog-driven dispatch (Phase 1 catalog → list of events → graph per event) replaces today's bespoke `glob() + groupby + apply` notebooks.

**Estimated effort.** ~1 week after Phase 1. Most of the work is the `attribute_event` adapter and the GeoParquet posterior schema; the operator graph is unchanged.

### 5.3 Phase 3 — Alert service (FastAPI)

**Goal.** Demonstrate research-to-prod with **zero rewrite**: the same graph, served as a request handler.

**Service shape:**

```python
from fastapi import FastAPI
from plumax.attribution import build_attribution_graph, AttributionRequest, AttributionResponse

app = FastAPI()
attribution_graph = build_attribution_graph().compile()  # same Graph as Phase 1 + 2

@app.post("/attribute", response_model=AttributionResponse)
async def attribute(req: AttributionRequest) -> AttributionResponse:
    # 1. Catalog query for overpasses near the event
    overpasses = catalog.query(
        geometry=req.bbox(50_000),
        interval=req.window(hours=12),
        instruments=req.instruments,
    )

    # 2. Resolve inputs (async via the AsyncReader Protocol)
    inputs = await resolve_inputs_async(overpasses, req)

    # 3. Same graph, called as the request handler
    posterior = attribution_graph(**inputs)

    # 4. Build alert payload
    return AttributionResponse(
        event_id=req.event_id,
        posterior_summary=summarize_posterior(posterior),
        provenance=overpasses.provenance(),
        alert_score=compute_alert_score(posterior),
        latency_ms=...,
    )
```

**Operational expectations:**

- Cold-start ≤ 30 s (model + graph load).
- Warm latency ≤ 5 s end-to-end for a typical event (3 instruments, 50 km bbox, MAP inversion).
- Pickleable graph means the service can shard across workers transparently.
- Catalog query is the long pole — depends on Phase 1 vs Phase 2 catalog and AOI size.

**What this validates:**

- The "marquee pitch" — *the same operator graph runs in research and production* — actually delivers. The `attribution_graph` object in the FastAPI handler is the *same Python object* you'd import in a notebook.
- Async readers (`AsyncReader` Protocol from [`plans/georeader/`](plans/georeader/)) integrate cleanly with FastAPI's async stack.
- Response payload includes complete provenance — every overpass URL, met source, prior version, operator graph hash — making each attribution audit-able.

**Estimated effort.** ~1 week after Phase 2. Most of the work is the FastAPI scaffolding and async-reader plumbing; the graph and inference are unchanged.

---

## 6. Operator catalog — what the `Sequential` actually contains

For engineers: the concrete operators each library would contribute. This is the *first-pass* surface — exact signatures land in each library's plans.

### 6.1 `geotoolz` operators (existing per [`plans/geotoolz/`](plans/geotoolz/), with one addition)

- `MatchedFilter(target: spectrum)` — hyperspectral plume detection. Hyperspectral module.
- `ApplyAK(instrument: str)` — averaging-kernel application. New for v0.x — currently lives in plumax; promote when stable.
- `Mask(qa_band: str, min: int|float)` — cloud + QA masking.
- `Crop(geoslice: GeoSlice)` — AOI extraction.
- `Reproject(target_crs: CRS, target_transform: Affine)` — Case-3 (shape-changing) operator.
- `Stitch(method: str)` — inverse of `GridSampler`.
- `Sequential([...])`, `Graph({...})`, `Tap`, `ApplyToEach` — composition core.

### 6.2 `plumax` operators (projected — see [plumax roadmap](../plume_simulation/notes/roadmap/README.md))

- `GaussianPlume(instrument, axis_convention)` — Tier I forward, `(Q, x₀, ū, θ_wind, c_bg) → concentration_field → column_via_AK → simulated_observation`. Wraps the existing tier-I math from [`01_tier1_gaussian.md`](../plume_simulation/notes/roadmap/01_tier1_gaussian.md) as an Operator.
- `LagrangianPlume(...)` — Tier II forward. v2+.
- `EulerianPlume(...)` — Tier III forward. v3+.
- `BiasCorrect(instrument: str)` — per-instrument additive bias as a learnable Operator (compute via the split-object pattern from [§4.3 of geotoolz.md](plans/geotoolz/geotoolz.md)).
- `JointLikelihood(instruments: list[str])` — multi-instrument fusion at native resolution. Returns a callable likelihood; downstream `MAPInverter` / `NUTSInverter` consumes it.
- `MAPInverter(prior, init)` — wraps NumPyro's MAP. Output: `Posterior` PyTree.
- `NUTSInverter(prior, n_warmup, n_samples)` — wraps NumPyro's NUTS. Output: `Posterior` PyTree with full chain.
- `MetFieldLoader(source: str)` — wraps WRF / ERA5 readers from [`prerequisites.md`](../plume_simulation/notes/roadmap/00_prerequisites.md), emits `MetField`.

### 6.3 `xrtoolz` operators (projected — see [github.com/jejjohnson/xr_toolz](https://github.com/jejjohnson/xr_toolz))

- `TimeInterpolate(policy: str)` — `snapshot` / `piecewise` / `advected` per [prerequisites §3](../plume_simulation/notes/roadmap/00_prerequisites.md#open-questions).
- `PBLDiagnostic()` — derive PBL height from met fields.
- `StabilityClassifier(scheme: str)` — Pasquill–Gifford or MO similarity.
- `RegridToAnalysisGrid(target_grid)` — Case-3 op on `coordax.Dataset`.

### 6.4 What lives in `plumax`, not `geotoolz`

The substrate-split rule from [`geotoolz.md` §10.1](plans/geotoolz/geotoolz.md): operators that are *domain-specific* (Gaussian plume math, AK application, methane likelihoods) live in `plumax`. Operators that are *generic* (matched filter, masking, cropping, reprojection) live in `geotoolz`. The split is enforced by import direction — `plumax` imports `geotoolz`, never the reverse.

---

## 7. Out of scope (explicit)

For all three demo phases:

- **Model training.** `geotoolz`, `xrtoolz`, and `plumax` cover *exploration*, *preprocessing*, *inference*, and *serving*. Training (e.g. an emulator surrogate for the forward model, per [plumax Step 3](../plume_simulation/notes/roadmap/README.md#the-data-driven-modeling-cycle)) stays in the user's framework (PyTorch, JAX, equinox). `ModelOp` wraps a *trained* callable for inference; nothing in this stack abstracts over training loops.
- **Tier IV neural RTM.** The current RTM stack ([`04_rtm_stack.md`](../plume_simulation/notes/roadmap/04_rtm_stack.md)) is matrix-based. A neural-network surrogate for the RTM is a v2+ optional upgrade.
- **Tier V population / forecasting.** The TMTPP intensity-process work in [`06_tier5_population.md`](../plume_simulation/notes/roadmap/06_tier5_population.md) is downstream of single-event posteriors; it consumes Phase 2's GeoParquet output but isn't part of v1.
- **Distributed-SQL execution.** Sedona / Spark are not on the path. If catalog scale outgrows DuckDB Phase 2, the next step is sharded GeoParquet, not SQL emission from operator graphs.
- **Streaming pipelines.** Flink / Beam / Kafka Streams are not modelled. Operator graphs run per-event, not per-byte-stream.
- **Edge inference.** No mobile / TFLite / ONNX-runtime targets. The amortised predictor (plumax Step 5) could ship as ONNX, but operator-graph orchestration around it remains server-side.

These exclusions are deliberate, per the [scope-honesty discussion in geostack_notes.md](geostack_notes.md#honest-research-to-prod-scope) — keeping the in-scope list deliverable rather than the out-of-scope list aspirational.

---

## 8. Validation plan

Validation is layered to match the [plumax architectural principle 2](../plume_simulation/notes/roadmap/README.md#architectural-principles): *each step validates the next*.

### 8.1 Per-operator (geotoolz / plumax / xrtoolz)

- **Tier A primitive correctness.** Analytic ground truth on numpy arrays. NDVI, matched filter, AK application — each has a textbook-canonical reference output for known inputs.
- **Tier B Operator metadata round-trip.** Transform / CRS / fill-value preserved through every Operator type (the three cases from [§1.1 of geotoolz.md](plans/geotoolz/geotoolz.md)).
- **Operator pickleability.** Every Operator in §6 round-trips through `pickle.dumps` / `pickle.loads`. Already a CI test per [§11.2](plans/geotoolz/geotoolz.md#112-implementation-gotchas-test-these-in-ci).

### 8.2 Per-graph (the §4.5 attribution `Graph`)

- **Identity-input test.** For a synthetic "no-plume" scene, posterior on `Q` peaks at 0 with credibility-interval covering 0.
- **Recovery test.** For a synthetic plume of known `Q`, posterior recovers `Q` within the simulated observation noise.
- **Multi-instrument tightening test.** Posterior with 3 instruments has tighter credibility intervals than with 1 — quantify the tightening per added instrument.

### 8.3 End-to-end (against published events)

- **Anchor event.** Pick one published Permian methane release with documented MARS-style attribution. The unified-stack posterior must be consistent with the published posterior.
- **Cross-instrument independence.** Posterior obtained from {EMIT only}, {TROPOMI only}, {GHGSat only}, and {all three jointly} — the joint posterior should be the precision-weighted combination of the individual posteriors (within model misspecification).

### 8.4 Operational continuity

- **Phase 3 vs Phase 1 parity.** The FastAPI service, given the same inputs as the Phase 1 notebook, returns the same posterior to within numerical noise. *This is the marquee parity test* — if it fails, the "same graph in research and prod" claim fails.

---

## 9. Migration path from current MARS-style work

For audiences asking *"what changes for the people doing this work today?"* — a concrete mapping.

| Today's piece | Maps to | What changes |
|---|---|---|
| Per-project EMIT / TROPOMI / GHGSat ingest scripts | `georeader` 2.0 sensor readers | One library replaces N scripts; auth + transport unified |
| Project-specific overpass CSVs | `GeoCatalog` Phase 1 | One queryable object replaces ad-hoc CSVs; round-trips to GeoParquet |
| `matched_filter.py` per project | `geotoolz.hyperspectral.MatchedFilter` | Library-quality, tested, version-pinnable |
| Per-project `gauss_plume.py` | `plumax.tier1.GaussianPlume` (Operator-wrapped) | Shared math; bug fixes propagate; Hydra-config-friendly |
| Per-project NumPyro model | `plumax.JointLikelihood` + `MAPInverter` | Likelihood structure becomes a typed, reusable operator |
| Hand-rolled FastAPI handler | `app.post("/attribute", ...)` calling the same graph | Zero rewrite from research to prod; full provenance for free |
| Per-project repository / archival | GeoParquet posterior artifact + operator-graph hash | Each attribution is reproducible from artifact alone |

**What does *not* change for current MARS workflows:**

- The science. Tier I math, AK conventions, multi-instrument fusion logic — all stay the same. The substrate changes; the equations don't.
- The operational reality. UNEP-IMEO partners, alert-response protocols, public reporting cadence — all unchanged.
- The data sources. EMIT on AWS, TROPOMI on Copernicus, GHGSat on the vendor portal — same buckets, same auth, same products.

---

## 10. Risks and open questions

The full risks tracker for the underlying libraries lives in their own design docs ([`plans/geotoolz/geotoolz.md` §11](plans/geotoolz/geotoolz.md), [`plans/geodatabase/`](plans/geodatabase/), [`plans/georeader/`](plans/georeader/)). The risks specific to *this* attribution pipeline:

### 10.1 Critical-path

- **`georeader 2.0` lands.** Every reader in §4.3 depends on the `feature/geotensor_npapi` branch merging upstream. If it stalls, vendor a `GeoTensor` in `plumax` until upstream catches up — but the contingency cost is one library duplication.
- **Geocatalog Phase 1 ships.** Without it, Phase 2 / Phase 3 fall back to per-script glob-and-filter — the demo still works for one event, but the *catalog-driven* pitch evaporates.

### 10.2 Per-instrument

- **GHGSat schema stability.** Commercial product, schema changes per release. Pin reader to a documented product version; bump `_v2` per breaking change. Same convention as plumax sensor presets.
- **EMIT product latency.** EMIT L1B can be hours to days behind acquisition. Phase 3 alert latency claims are bounded by this — no operational fixes available.
- **TROPOMI L2 averaging-kernel format heterogeneity.** Different processors emit different AK schemas. The `Instrument` registry from [prerequisites §5](../plume_simulation/notes/roadmap/00_prerequisites.md#l1--l2-ingest) absorbs this — but the registry must stay maintained, or the unified pipeline breaks per-product-bump.

### 10.3 Inference

- **Three-instrument MAP convergence.** With heterogeneous noise models and a non-trivial `Q(t)` prior, MAP can be hard to converge. v1 ships with NumPyro-default optimizers; v2 may need basin-hopping or sequential Monte Carlo for hard events.
- **Posterior representation.** A `Posterior` PyTree on output of `MAPInverter` / `NUTSInverter` needs a stable schema. Pin in v0.1 of `plumax`; bump the artifact `schema_version` (per [`geocatalog.md` §10.4](plans/geodatabase/geocatalog.md)) when changes happen.

### 10.4 Operational

- **Alert-payload schema is operator-defined.** What goes in `AttributionResponse` matters for downstream responders. v1 should mirror MARS's existing alert schema where possible, with extensions for provenance.
- **Provenance-as-data.** Each posterior must carry: catalog query → resolved overpass URLs → met source + version → operator-graph hash → library versions. Without this, the "auditable" claim doesn't hold. Worth a dedicated `Provenance` PyTree in `plumax`.
- **Phase 3 cold-start budget.** 30 s is plausible if `attribution_graph.compile()` is cheap; if NumPyro JIT-compiles per request, the budget breaks. Pin compilation to import-time, not request-time.

### 10.5 Open questions to settle before v1

- **Where does `ApplyAK` live — `geotoolz` or `plumax`?** Current proposal: in `geotoolz` (it's a generic operator over averaging-kernel-bearing observations), with the per-instrument AK schema registry in `plumax`. Could go the other way.
- **Async at the graph boundary or per-Operator?** Phase 3 uses async I/O via `AsyncReader`; the graph itself is sync. Open per [`geotoolz.md` §11.2](plans/geotoolz/geotoolz.md#112-implementation-gotchas-test-these-in-ci). Decision affects FastAPI throughput.
- **`coordax` for `MetField` from day one?** The plumax roadmap leans yes ([prerequisites open questions](../plume_simulation/notes/roadmap/00_prerequisites.md#open-questions)). If `coordax` proves unstable per [geotoolz.md §11.1](plans/geotoolz/geotoolz.md#111-strategic-risks), fall back to plain xarray.
- **Global vs basin catalogs.** Phase 1 fits ≤10⁵ overpasses, which covers a basin × multi-year window. A *global* catalog (every TROPOMI / EMIT / GHGSat overpass, multi-year) is Phase 2 territory. Decide per-deployment.

---

## 11. References & cross-links

### Plumax internals

- [Plumax roadmap index](../plume_simulation/notes/roadmap/README.md) — full tier structure and the data-driven modeling cycle.
- [Prerequisites — `MetField`, observation interface](../plume_simulation/notes/roadmap/00_prerequisites.md) — the substrate this pipeline reads from.
- [Tier I — Gaussian family](../plume_simulation/notes/roadmap/01_tier1_gaussian.md) — the v1 forward model used in §4.5.
- [Tier IV — Coupled end-to-end](../plume_simulation/notes/roadmap/05_tier4_coupled.md) — the multi-instrument fusion that this demo's v1 target maps onto.
- [RTM stack](../plume_simulation/notes/roadmap/04_rtm_stack.md) — the AK / radiative-transfer pieces.
- [Satellite catalog notes](../plume_simulation/notes/satellites.ipynb) — the per-instrument target list this demo selects from.

### Unified-stack design docs

- [Geostack motivation (`geostack_notes.md`)](geostack_notes.md) — the public motivation doc this pipeline serves as the success story for.
- [Geotoolz design](plans/geotoolz/) — operator algebra, tier model, sharp edges.
- [Geocatalog design](plans/geodatabase/) — Phase 1 GeoPandas + Phase 2 DuckDB.
- [Georeader reconciliation](plans/georeader/) — `Reader` Protocol, async, sensor-specific readers.
- [Per-sensor reader designs](plans/readers/) — geostationary + polar-orbiting.
- [Cross-cutting types](plans/types/) — `GeoSlice`, `Credential`, `ByteStore`.

### External

- [MARS at UNEP-IMEO](https://methane.unep.org/) — the operational system this pipeline is patterned on.
- [EMIT mission](https://earth.jpl.nasa.gov/emit/) — NASA hyperspectral imager.
- [TROPOMI / Sentinel-5P](https://sentinel.esa.int/web/sentinel/missions/sentinel-5p) — ESA/EU.
- [GHGSat](https://www.ghgsat.com/) — commercial constellation.
- [`coordax`](https://github.com/neuralgcm/coordax) — the JAX-native carrier candidate for `MetField`.
