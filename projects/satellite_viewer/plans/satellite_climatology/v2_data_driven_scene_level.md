---
title: "v2 — Data-driven, scene-level cloud climatology"
short_title: "v2 scene-level"
---

# v2 — Data-driven, scene-level cloud climatology

> *Of the scenes that were actually acquired and ingested into the
> Planetary Computer / Earth Search catalogs, how many cover each
> pixel and what was their scene-wide cloud cover?*

## Goal

For every pixel of the shared 0.1° global grid and each sensor, scan
the real STAC catalog and aggregate per-cell statistics from the
**item metadata** (footprint + `eo:cloud_cover` + datetime). No pixel
reads — this stage stays in metadata-land.

## Question answered

> "Given the catalog as it actually exists, how many scenes covered
> this pixel in $[t_0, t_1]$, and what fraction were nominally
> cloud-free at the scene level?"

The user-facing pitch is the **"reality gap" map**: subtract v2's
`scenes_count` from v1's `overpasses` and you visualise where the
catalog is incomplete vs. theoretical max — useful for spotting
processing-pipeline outages, off-nadir tasking effects, or regions
where a sensor is simply not acquired (e.g., open ocean for L8/9
WRS-2 land grid).

## Scope

- **Sensors**: same five as v1, but each maps to one or more STAC
  collections (e.g., `sentinel-2-l2a` for sentinel-2).
- **Time window**: configurable. Default: rolling 12 months. Longer
  is feasible (24 → 36 months) but linearly more compute.
- **Resolution**: 0.1° (shared). H3 res-5 alternative discussed below.
- **STAC source**: Microsoft Planetary Computer (anonymous, no auth)
  for all five sensors. Element84 Earth Search as a fallback.

## Algorithm

Two-pass design — neither pass queries per cell:

### Pass 1 — catalog ingestion

For each sensor's collection:

1. Issue a **paginated STAC search** over the global bbox + time
   window with no spatial filter (or a coarse one to chunk).
2. Stream items into a **DuckDB-backed GeoCatalog** (`geocatalog`'s
   `DuckDBGeoCatalog` — GeoParquet 1.1 with bbox-column predicate
   pushdown). Persists as `data/catalogs/<sensor>.parquet`.
3. Item attributes kept: `id`, `datetime`, `geometry` (footprint),
   `eo:cloud_cover`, `platform`.

Why DuckDB and not InMemory: 1 year of S2 over the globe is ~1M items.
DuckDB handles this on a laptop; an InMemory GeoDataFrame doesn't.

### Pass 2 — per-cell aggregation

The key insight is **inverting the loop**. Don't iterate cells and
query the catalog. Iterate **items** and stamp them into cells:

1. For each item, compute the set of grid cells its footprint
   intersects (rasterise the footprint polygon onto the 0.1° grid).
2. For each cell, push the item's `(datetime, eo:cloud_cover)` into
   that cell's accumulator.
3. After all items: each cell's accumulator yields:
   - `scenes_count` = `len(items)`
   - `mean_scene_cloud_pct` = `mean(item.cloud_cover for item in items)`
   - `cloud_free_scene_count` = `sum(c < 10 for c in cloud_covers)`
   - `max_gap_days` = `max(diff(sorted(datetimes)))`

Either pass can be windowed by `geopatcher` to keep memory bounded:
process the global grid in N-cell tiles, aggregate per-tile, concat.

## Architecture

This is where your repo stack lights up:

```
projects/satellite_climatology/
└── src/satellite_climatology/
    ├── grid.py             # (shared with v1)
    ├── sensors.py          # (shared with v1) + STAC collection mapping
    ├── catalog.py          # build_global_catalog(sensor, t0, t1) -> DuckDBGeoCatalog
    ├── aggregate.py        # bin items into cells -> per-cell stats
    └── operators.py        # geotoolz Operators wrapping the reducers
```

Repo wiring:

- **`geocatalog`** is the substrate.
  - `from_stac_search(STACSource(...))` ingests the items into a
    `DuckDBGeoCatalog` per sensor.
  - The catalog persists between runs (it's GeoParquet on disk), so
    re-aggregating with different thresholds is free.

- **`geopatcher`** drives the grid iteration.
  - `SpatialPatcher(SpatialRectangular(size=(0.1°, 0.1°)),
    SpatialRegularStride(...))` over a `RasterField` covering the
    global bbox.
  - Per-tile (not per-cell) work is run via `runners.parallel_map`
    across processes.

- **`geotoolz`** holds the reducer as a composable graph:
  ```
  per_cell = (
      Fanout({
          "scenes_count":           Count(),
          "mean_scene_cloud_pct":   Mean(field="cloud_cover"),
          "cloud_free_scene_count": CountWhere(field="cloud_cover", op="<", value=10),
          "max_gap_days":           MaxGap(field="datetime"),
      })
  )
  ```
  The `Fanout` shape means one item-iteration computes all four stats;
  `get_config()` serialises the reducer for the run manifest.

- **`georeader`** is **not** used in v2. (That's v2.5.)

## Output

Adds these four bands to the shared Zarr:

```
scenes_count           (sensor, time, lat, lon) int16
mean_scene_cloud_pct   (sensor, time, lat, lon) float32
cloud_free_scene_count (sensor, time, lat, lon) int16
max_gap_days           (sensor, time, lat, lon) float32
```

Same grid as v1, so:

```python
ds = xr.open_zarr("data/satellite_climatology.zarr")
gap_v1 = ds["overpasses"]                    # theoretical
got_v2 = ds["scenes_count"]                  # observed
catalog_gap = (gap_v1 - got_v2).clip(min=0)  # the "reality gap" map
```

## Compute budget

Pass 1 (catalog scan):

- S2 1 yr globally: ~1M items, MPC paginated search at ~500 items/page
  → ~2000 paginated requests, ~30 min wall-clock (mostly network).
- × 5 sensors with very different volumes: MODIS daily × 8 platforms
  is the heaviest at ~10M items/yr.
- → ~2–6 hours total. Run once, persist.

Pass 2 (aggregation):

- 1M items × ~100 cells/item = ~100M (cell, item) stamps.
- Vectorised by tile, this is ~5–15 min per sensor on a laptop.
- Parallelised over patches: linearly faster.

Storage:

- Per-sensor GeoParquet catalogs: a few GB total.
- Output Zarr (with v1+v2 bands, 5 sensors, 12 months): ~2–3 GB.

## UI integration

Adds new stats to the dashboard dropdown:

- `scenes_count` — observed catalog density
- `cloud_free_scene_count` — observed cloud-free density
- `mean_scene_cloud_pct` — observed mean cloud
- `max_gap_days` — longest dry spell
- `reality_gap` = v1.overpasses − v2.scenes_count — derived band

Same tile-server approach as v1.

## H3 alternative (equal-area)

For latitudes > 60° the 0.1° lat/lon cell shrinks to ~5 km width while
staying 11 km tall — heavy distortion. If equal-area matters
(area-weighted statistics, comparing high vs low latitudes):

- Replace the grid with **H3 hex bins at resolution 5** (~250 km² each,
  ~2.0M cells globally).
- All `geopatcher`-iteration logic is the same; replace
  `SpatialRectangular` with a `PolygonIntersection` geometry over the
  H3 polygon set.
- Zarr → Parquet (`pandas` with H3 index column), or sparse Zarr.

Decide this at M3 once we see the latitude-distortion in the v1 maps.

## Risks & open questions

- **Catalog completeness varies by sensor.** MPC's Landsat goes back
  to 2013, MODIS to 2000+, S2 to 2015. The product window must be
  clipped per-sensor or padded with nulls.
- **`eo:cloud_cover` is scene-wide.** A 60%-cloud scene can be 100%
  clear over your pixel (or vice versa). v2 is documented as a
  *scene-level* statistic; v2.5 is the per-pixel answer.
- **Footprints aren't pixel-exact** — STAC item geometry is the scene
  outline, which can be slightly larger than actual valid-data extent
  (especially S2 swath edges). Acceptable for cell-scale stats.
- **Re-ingestion drift** — collections get reprocessed (new baseline,
  republished items). Pin the run date and store it in the Zarr
  metadata so users know what catalog snapshot they're looking at.
- **Rate limits** — MPC is generous but not infinite. Throttle the
  scan with `pystac-client`'s `max_items` per call + a backoff
  decorator.

## Acceptance

- All four data vars are written for at least one sensor over a
  12-month window at 0.1°.
- Sanity checks:
  - `scenes_count` ≤ v1 `overpasses` cell-wise (catalog ≤ theoretical).
  - `mean_scene_cloud_pct` map qualitatively matches a published cloud
    climatology (e.g., MODIS MOD08 monthly mean) — wet-season Amazonas
    > Atacama Desert.
  - `cloud_free_scene_count` is zero for cells with `scenes_count = 0`.
- Catalog Parquet files are reproducible (re-running the pipeline
  yields the same row count given the same date pin).
- Reality-gap map renders in the dashboard.

## Out of scope

- Per-pixel cloud masks (v2.5).
- L1C / L2A choice — we use L2A everywhere it exists; document the
  collection per sensor in `sensors.py`.
- Off-nadir tasking modelling.
- Sub-monthly time bins. (Easy to add later — the schema supports it,
  just blow up the time axis.)
