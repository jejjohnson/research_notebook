---
title: "v4 — Coverage planner (available / acquired / gap)"
short_title: "v4 coverage planner"
---

# v4 — Coverage planner

> *A heatmap dashboard that overlays three views of the same global grid —
> what imagery is **available**, what **we have acquired**, and where the
> **gap** is (and, eventually, what to task) — sliceable by sensor, time
> window, and AOI, viewable globally or zoomed to a single location.*

This is the planning/operations layer on top of the
[`satellite_climatology`](README.md) substrate. v1–v3 answer *"how
observable is each pixel?"*; v4 reframes those numbers as a **coverage
ledger**: availability minus our holdings equals the gap we might act on.

## Questions answered

- *"Across the globe, where is imagery **available** for this sensor and
  time window?"* (and how cloud-free / how frequent)
- *"Where have **we** actually acquired data — and where are our holdings
  thin or stale?"*
- *"Where is the **gap** largest, and (future) where could we **task** a
  satellite to close it?"*
- All three, **clipped to an AOI** and **constrained to a date range**,
  with a **global** view as the default.

## The three layers

The dashboard is organised as three toggleable layer-groups over the same
`(sensor, time, lat, lon)` grid. The user picks **layer → metric →
sensor → time window**, then **global or AOI**.

| Layer | Meaning | Built from | Status |
|-------|---------|------------|--------|
| 🟦 **Available** | What could / did exist to be observed | v1 overpasses + v2 scene scan + v3 clear-fraction | reuses v1–v3 |
| 🟩 **Acquired** | What *we* hold, as two sub-layers (see below) | catalog scans | **new in v4** |
| 🟥 **Gap** | `Available − Acquired`, weighted into a priority | derived | **new in v4** |

### Available (reuses v1–v3)

No new compute — these are the bands v1/v2/v3 already write:
`overpasses`, `scenes_count`, `mean_scene_cloud_pct`,
`cloud_free_scene_count`, `clear_fraction`, … The Available layer is a
re-labelling of the climatology product as "supply".

### Acquired — two sub-layers

Per the design decision, "what we've seen" is tracked as **two distinct,
independently toggleable sub-layers**:

1. **Public clear observations** — the cloud-free *usable* supply from the
   public catalog (i.e. v2's `cloud_free_scene_count` / v3's
   `clear_obs_count`). This is "what anyone could have gotten."
2. **Our holdings** — what *this project* has actually ingested, sourced
   from an **external PostGIS holdings table** (see *Holdings source*
   below). It is an already-populated catalog of owned tiles with footprint,
   acquisition date, sensor, cloud %, and (optionally) processing status — so
   we *query* it, we don't rebuild it. Gridded into the same cells to produce
   `held_count`, `held_clear_count`, `days_since_last_held`,
   `held_max_gap_days`.

Keeping them separate lets the map distinguish *"the data exists and is
clear, we just don't have it"* (a pure download decision) from *"no clear
data exists at all"* (a tasking decision).

If the holdings table carries a processing-status column, "held" can be
refined into **held / processed / validated** — a third toggle showing not
just *"do we have the tile"* but *"have we finished
processing it"*.

### Gap — derived, future-flagged tasking

Computed cell-wise from the layers above, no new I/O:

```
deficit            = max(0, desired_clear_obs − held_clear_count)
staleness          = days_since_last_held                     # recency pressure
unmet_supply       = cloud_free_scene_count − held_clear_count  # exists-but-unowned
priority_score     = w_d·norm(deficit)
                   + w_s·norm(staleness)
                   + w_u·norm(unmet_supply)
                   + w_r·revisit_need(aoi)                    # user-supplied weight
taskable           = SENSOR_TASKABILITY[sensor]               # bool, mostly False
```

- `desired_clear_obs` is a **per-sensor target** (clear looks / month),
  adjustable per selected sensor — see *Desired cadence* below. The deficit
  is relative to it.
- **Tasking is future-flagged, not built.** The schema carries a
  `taskable` flag and the UI shows a *"request"* affordance on
  high-priority cells, but it is a no-op placeholder. See *Tasking* below.

#### Desired cadence — per sensor

The deficit needs a target to subtract from, and a useful cadence differs
by an order of magnitude across sensors (MODIS is ~daily, Landsat is
~biweekly). So the target is **per sensor**, expressed as *desired clear
observations per month*:

- Each **selected** sensor gets its own compact target input — you only see
  a control for a sensor you've chosen. Pick 2 sensors → 2 inputs. (This is
  the right read of the "won't that be a lot of sliders?" worry: the count
  is bounded by your sensor selection, not the full registry.)
- Each is **pre-filled from the sensor's nominal revisit** (e.g. S2 ≈ 6/mo,
  Landsat ≈ 2/mo, MODIS ≈ 30/mo), so the defaults already yield a
  meaningful gap and you rarely touch them.
- The gap is computed **per sensor** against its own target. A combined
  *"any clear look"* view (sum of held vs. sum of targets, or a single
  cross-sensor target) is an optional toggle.

## Why two apps (the 2 km question)

A single location of interest is ~2 km; a global heatmap cell is ~11 km
(0.1°). These are **different scales, not a single tunable**:

- **2 km globally** = ~162 M cells — heavy to compute and invisible when
  zoomed to the whole Earth.
- **0.1° globally** = 6.48 M cells — the existing shared grid; fast; the
  right granularity for a world map.

So v4 ships **two surfaces**, mirroring the v2 (global) / v2.5 (per-AOI)
split that already exists:

### App A — Global Coverage Heatmap (the overview)

- Reads the precomputed **Zarr** at the shared **0.1°** grid (H3 res-5 as
  an equal-area alternative — see v2).
- Controls: layer-group, metric, sensor(s), time-window slider, global vs.
  AOI clip.
- Renders as raster/choropleth tile layers on the basemap (the
  satellite_viewer basemap switcher carries straight over).
- Compute: **v2 scene-level scan** for the MVP (laptop-tractable), with
  **v1 overpasses** as the cheap theoretical-available underlay. v3
  pixel-level bands appear in the same dropdowns as they get computed.

### App B — AOI Coverage Drill-down (the detail)

- On-demand for a single AOI (up to the ~50 km cap from v2.5), at **native
  / pixel resolution** (down to ~2 km cells or true per-pixel).
- Reuses **`satellite_viewer.search()`** verbatim for discovery (the
  v0→v2.5 contract) and the **v2.5** windowed-QA path for true
  clear-fraction.
- Renders the per-AOI **time series** of available / acquired / gap, plus a
  fine-grained local heatmap and the per-scene table.
- This is where the 2 km scale lives and makes sense.

A nice-to-have bridge: clicking a cell in App A pre-fills App B's AOI.

## Heatmap metric menu

Per cell × sensor × time-bin. The dropdown is grouped by layer:

| Layer | Metrics |
|-------|---------|
| Available | `overpasses` (theoretical), `scenes_count`, `mean_scene_cloud_pct`, `cloud_free_scene_count`, `clear_fraction` (pixel), `pixel_max_gap_days` |
| Acquired | `held_count`, `held_clear_count`, `days_since_last_held`, `held_max_gap_days` |
| Gap | `deficit`, `unmet_supply`, `priority_score`, `taskable` (mask) |

Diverging colormaps for difference/gap metrics; sequential for counts;
"days since" gets a perceptually-reversed ramp (fresh = cool, stale = hot).

## Time-window aggregation

When the time slider spans several monthly bins, the cell value must be
reduced over the window. This is an explicit **aggregation selector**, not
a hidden choice:

| Mode | Cell value | Best for | Caveat |
|------|-----------|----------|--------|
| **Rate (mean/month)** ⭐ | average per-month over the window | comparing coverage *cadence*; comparable across window sizes | hides within-window variation |
| **Total (sum)** | summed count over the window | "how much did we get in total" | a longer window always looks bigger |
| **Most-recent** | value of the latest bin (or *days since last*) | freshness / "is it stale" | ignores history |
| **Worst month (min/max)** | the limiting bin (fewest clear obs / longest gap) | gap & tasking — the worst month drives the decision | pessimistic by design |
| **Scrub / animate** | no reduction — step through bins | seeing seasonality | not a single map |

**Smart per-metric defaults** so the selector rarely needs touching:
availability counts → *Rate*; `days_since_last_held` → *Most-recent*;
`deficit` / `priority_score` → *Worst month*. The user can override.

## Data model

Extends the **shared Zarr** ([README](README.md)) with two new band-groups.
Same dims `(sensor, time, lat, lon)`, same partial-population contract (the
UI reads whatever bands exist):

```
# v4 — acquired (our holdings, gridded from the holdings catalog)
held_count             (sensor, time, lat, lon) int16
held_clear_count       (sensor, time, lat, lon) int16
days_since_last_held   (sensor, time, lat, lon) float32
held_max_gap_days      (sensor, time, lat, lon) float32

# v4 — gap (derived; can be recomputed on read, or materialised)
deficit                (sensor, time, lat, lon) int16
unmet_supply           (sensor, time, lat, lon) int16
priority_score         (sensor, time, lat, lon) float32
```

### Holdings source — an external PostGIS table

The holdings layer is **not a new catalog and not part of this repo** — it
is an existing, externally-maintained PostGIS table of tiles we own, kept in
a separate private repo/service. v4 just *reads* it over a standard
PostgreSQL/PostGIS connection. The only columns it needs (mapped from
whatever the table actually calls them):

```
holdings (PostgreSQL + PostGIS): one row per owned tile
  datetime   : timestamp        # acquisition time   -> time binning
  sensor     : text             # platform/sensor    -> per-sensor held bands
  geometry   : geometry(4326)   # footprint          -> grid cells
  cloud      : numeric|null     # cloud %            -> clear/cloudy split
  (optional)                    # a processing-status column -> held/processed/validated
```

**Access — credentials and the column mapping live in a gitignored `.env`**
(never committed), read by a generic reader. Nothing project-specific is
imported; it's a plain SQLAlchemy + GeoPandas read:

```python
# .env (gitignored): COVERAGE_DB_* for the connection, COVERAGE_TILES_* to map
# the generic column names above to the table's real columns.
from satellite_climatology.holdings import fetch_holdings   # generic PostGIS reader
gdf = fetch_holdings(aoi=aoi, start=t0, end=t1)
# -> GeoDataFrame[datetime, sensor, cloud, geometry]
```

Gridding the holdings into `held_*` bands reuses v2's exact
`footprint_to_cells` reducer (or the precomputed cell/H3 index if the table
already carries one). **No ingest to write** — the source of truth already
exists and is maintained outside this project.

> Note: where a held sensor also has a public Available source (e.g. EMIT via
> NASA CMR, EnMAP/Landsat via STAC), v4 shows both layers and can compute the
> gap; sensors present on only one side show only that layer.

## Precompute pipeline (geocatalog → DuckDB → Zarr)

The dashboard never hits a STAC API live — App A reads only the
precomputed Zarr. The build pipeline maps onto `geocatalog`'s
**Source → Bundle → Catalog → GeoSlice** flow using its real API:

1. **Scan → GeoParquet.** `geocatalog.from_stac_search(...)` queries each
   sensor's STAC endpoint over the build window into a `GeoCatalog` (one
   `CatalogRow` per item: footprint + interval + `eo:cloud_cover` + asset
   hrefs), persisted with `to_geoparquet(...)` partitioned by
   `(sensor, year, month)` so the DuckDB backend prunes.
2. **Aggregate in DuckDB.** Open the GeoParquet as a `DuckDBGeoCatalog` and
   run the gridding + stats as **DuckDB SQL** (spatial extension does the
   footprint→cell join; `GROUP BY (sensor, time_bin, cell)`) →
   `scenes_count`, `mean_scene_cloud_pct`, `cloud_free_scene_count`. The
   `held_*` bands come from the **external holdings table** instead:
   `holdings.fetch_holdings(...)` returns the owned tiles (footprint +
   datetime + sensor + cloud), fed through the same `footprint_to_cells`
   reducer. DuckDB is out-of-core, so a sensor-year stays laptop-tractable.
3. **Write Zarr.** Pivot the aggregated table into the dense
   `(sensor, time, lat, lon)` arrays (via geocatalog's `xarray_backend` →
   `xr.Dataset`) and write the shared Zarr.
4. **Gap is cheap.** `deficit` / `unmet_supply` / `priority_score` derive
   from the bands above — materialise in the same Zarr or compute on read.

App B's per-AOI discovery is the *same library queried live* instead of
precomputed: `query(catalog, bounds=aoi.bounds, time=(t0, t1))`
(equivalently `satellite_viewer.search`, the v0→v2.5 contract) feeding the
v2.5 windowed-QA path. See [`v4_coverage_planner_api.md`](v4_coverage_planner_api.md)
for concrete signatures + demos.

Heavy lifting stays in columnar/SQL land (DuckDB, out-of-core); Zarr holds
only the final dense product the UI streams. v1 overpass and v3 pixel bands
write into the same Zarr by their own paths — v4 owns the v2 scene-level +
holdings bands.

## Architecture (recommended: inside `satellite_climatology`)

```
projects/satellite_climatology/
└── src/satellite_climatology/
    ├── grid.py            # (shared v1+) global grid + footprint_to_cells
    ├── sensors.py         # (shared)
    ├── catalog.py         # (v2) availability DuckDBGeoCatalog scan
    ├── holdings.py        # (v4 NEW) generic external PostGIS reader (.env creds) + grid
    ├── coverage.py        # (v4 NEW) available/acquired/gap band assembly
    ├── tasking.py         # (v4 NEW, stub) taskability table + request hook
    └── apps/
        ├── global_heatmap_{panel,streamlit}.py   # App A
        └── aoi_drilldown_{panel,streamlit}.py    # App B (reuses satellite_viewer.search)
```

Rationale for living here rather than a standalone project or a
satellite_viewer tab:

- The grid, Zarr schema, sensor list, time-binning, and `geocatalog` scan
  are **already designed** for this project — v4 adds bands and an ingest,
  not a new substrate.
- The v1/v2/v3 dashboard is **already milestone M8**; App A *is* that
  dashboard with the Acquired/Gap layers added.
- App B reuses `satellite_viewer.search` — the v0→v2.5 contract the docs
  already commit to — so it doesn't fork discovery logic.

Library + thin UIs (Panel / Streamlit / notebook) matches how
`satellite_viewer` is already structured.

**External dependencies:** `geocatalog` (availability scan, GeoParquet,
DuckDB, Zarr write) and a standard PostgreSQL/PostGIS client (SQLAlchemy +
GeoPandas) for the holdings table. The holdings DB is **external and
private** — its credentials and column mapping come from a gitignored `.env`
(never committed). `holdings.py` isolates it behind a small interface
(`fetch_holdings(bbox|aoi, start, end) -> GeoDataFrame`), so the apps stay
generic and the holdings layer degrades gracefully to "off" when the DB is
unconfigured or unreachable.

## Tasking (future-flagged)

First, a distinction the Gap layer depends on: **"request" means two
different actions.** If a scene already *exists* over a target but we don't
hold it (`unmet_supply > 0`), the action is **ingest** — download + process
it (the available-but-unheld backlog, which applies to *every* sensor with
an Available source, Landsat and Sentinel-2 included). Only when *no scene
exists* does **tasking** — commanding a new
acquisition — come into play, and only for pointable instruments. See
[`v4_coverage_planner_api.md`](v4_coverage_planner_api.md) §"Request
semantics".

Public systematic sensors — **Sentinel-2, Landsat, MODIS, VIIRS — cannot be
tasked**; they image on a fixed schedule (so their gap is closed by
*ingest*, never tasking). Tasking only applies to:

- **Commercial** constellations (Planet, Maxar/SecureWatch, Capella SAR,
  ICEYE, Airbus) via their ordering/tasking APIs.
- A few **targeted** instruments (e.g. EMIT's target-request list).

So in v4 the tasking layer is **designed but inert**:

- `SENSOR_TASKABILITY` maps each sensor → bool; the Gap layer's `taskable`
  mask greys out cells where tasking is impossible (most of them).
- The UI exposes a *"flag for request"* button on high-`priority_score`
  cells/AOIs that currently just records the intent to a local table.
- `tasking.py` defines the `submit_request(provider, geometry, window)`
  signature with a `NotImplementedError` body, so wiring a real provider
  later is a localised change, not a redesign.

This keeps the "what to request" story coherent end-to-end without
pretending we can task Sentinel-2.

## Compute & scale

- **App A MVP** = v2 scene-level scan. Per v2's budget this is a catalog
  query + grid reduce — minutes-to-hours for a year of one sensor,
  laptop-tractable; no pixel reads.
- **v1 overpasses** = ~100-line skyfield script, negligible cost; gives the
  theoretical-available underlay immediately.
- **Holdings gridding** = trivial (we own ≪ the public catalog).
- **v3 pixel-level global** = the cluster job from
  [v3](v3_pixel_level_global.md); appears in App A's dropdowns as those
  bands fill in. Not on the v4 critical path.
- **App B** = on-demand v2.5 windowed reads for one AOI; seconds.

## Milestones (suggested)

1. **M4.0 — Available-only global heatmap.** App A over the v2 Zarr
   (single sensor, 1 year, 0.1°). Just the supply layer. Validates the
   tile-rendering + time-slider + AOI-clip UI.
2. **M4.1 — Holdings from the external DB.** `holdings.py` reads the
   external PostGIS table (creds via gitignored `.env`); grid
   owned tiles into `held_*` bands. Add the Acquired layer (both sub-layers
   + the held/processed/validated toggle) to App A.
3. **M4.2 — Gap layer.** `coverage.py` deficit/priority; the `desired`
   knob and weight sliders; diverging colormaps.
4. **M4.3 — App B (AOI drill-down).** Reuse `satellite_viewer.search` +
   v2.5 path; per-AOI time series of all three layers; cell→AOI handoff
   from App A.
5. **M4.4 — Tasking hook (inert).** `SENSOR_TASKABILITY`, the `taskable`
   mask, the flag-for-request table, the stubbed `submit_request`.
6. **M4.5 — v3 bands wired in** as they complete (no new UI work).

## Risks & open questions

- **Defining "desired" — resolved: per sensor.** Per-sensor target (clear
  obs / month), pre-filled from nominal revisit, one control per *selected*
  sensor (see *Desired cadence*). Per-AOI override is a later refinement.
- **Holdings provenance — the external DB is the source of truth.** No
  manual ingest to keep in sync — the Acquired layer reads the externally
  maintained PostGIS table directly. The residual risk is the *opposite*:
  the v4 dashboard is read-only against a DB it doesn't own, so its
  freshness tracks that external pipeline.
- **Private-data hygiene + DB access.** Credentials and the table/column
  mapping live only in a gitignored `.env` (never committed); the code is a
  generic PostGIS reader with no private imports. The DB needs network
  reachability — isolate behind `holdings.fetch_holdings(...)` so the
  Acquired layer degrades to "unavailable" (not crash) when it's
  unconfigured or unreachable.
- **Sensor-set mismatch.** The holdings sensors may differ from the v1–v3
  Available sensors (which lean optical: S2/Landsat/MODIS/VIIRS). The
  Acquired and Available layers only overlap where a sensor is in *both* —
  e.g. EMIT. Document per-sensor which layers exist so the Gap isn't
  computed across a mismatched pair.
- **Double-counting at tile edges.** Inherits v3's MGRS-overlap semantics —
  document that overlapping scenes are independent observations.
- **Equal-area vs. lat/lon.** Coverage *counts* are biased by cell area in
  EPSG:4326 (polar cells are tiny). Offer the H3 option for any
  area-normalised metric.
- **Colormap honesty.** "Gap" maps can imply urgency where a sensor simply
  doesn't overpass (poles for S2). Always show `taskable` / availability
  alongside gap so a gap isn't mistaken for an actionable one.
- **Time-window semantics — resolved: an aggregation selector.** See
  *Time-window aggregation* — default is rate (mean/month), with
  most-recent for freshness and worst-month for gap/priority.

## Out of scope (v4)

- Real tasking-API integration (future; hook only).
- Automated download orchestration (v4 *recommends*; it doesn't fetch).
- Pixel-level global compute (that's v3; v4 just surfaces it).
- Cost/quota modelling for commercial tasking.
- ML gap-prediction — v4 reports observed/derived gaps, not forecasts.
