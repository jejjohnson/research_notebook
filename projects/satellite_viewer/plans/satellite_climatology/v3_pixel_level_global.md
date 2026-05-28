---
title: "v3 — Global pixel-level cloud climatology"
short_title: "v3 global pixel-level"
---

# v3 — Global pixel-level cloud climatology

> *v2.5's per-AOI clear-fraction algorithm, batched over the whole
> globe and written into the shared Zarr.*

This is the capstone: the same per-scene operator graph from v2.5
applied to every cell of the global 0.1° grid for every scene in v2's
catalog, producing the truthful per-pixel clear-observation statistics
at planetary scale.

## Goal

For every cell of the shared 0.1° global grid, each sensor, and each
monthly time bin, compute the **true pixel-level clear-observation
statistics** by reading each intersecting scene's QA band, masking to
clear pixels, and aggregating into the cell.

## Question answered

> "How often was the surface inside this exact cell *actually* visible
> to the sensor over $[t_0, t_1]$?"

The truthful version of v2's question, answered globally. The
v2-vs-v3 difference map is the headline product: it shows where
scene-level cloud is a *biased* estimate of pixel-level
observability — coastal, mountain, partly-cloudy regions.

## Scope

- **Sensors**: subset of v2. Start with `sentinel-2` (validated by
  v2.5 first). Add `landsat-8-9` once its v2.5 decoder ships.
  MODIS / VIIRS deferred to v3.1.
- **Time window**: same as v2 by default (12 months rolling). Longer
  is feasible but linearly more compute.
- **Resolution**: 0.1° (shared default). Coarser is an easy first run.
- **Compute target**: batch job. Single laptop is feasible only for
  continental subsets at coarser grid; full global × all-sensors
  needs a cluster.

## Algorithm

The same `per_scene` operator graph from v2.5, run with the loop
inverted to keep compute tractable:

```
1. For each (sensor, time_bin):
     items = v2_catalog.query(global_bbox, time_bin)
2. For each item (in parallel across workers):
     qa_band = georeader.read_full(item.assets[qa_band_key])   # one full SCL/QA read
     clear_pixel_mask = decoder(qa_band, clear_classes=cfg.clear_classes)
     # Reduce to grid cells immediately — don't accumulate full-res masks.
     cells_touched = footprint_to_cells(item.geometry, grid_resolution=0.1°)
     for cell in cells_touched:
         window = clear_pixel_mask.window(cell)                # ~10×10 pixels at 0.1°
         emit (cell, item.datetime, window.mean())             # one float per (cell, item)
3. For each cell, aggregate the per-item floats:
     clear_obs_count    = sum(frac > clear_threshold)
     clear_fraction     = mean(frac)
     pixel_max_gap_days = max(gap between frac > threshold)
4. Write into shared Zarr at (sensor, time, lat, lon).
```

**Why loop-inversion is essential**: the alternative ("for each cell,
loop over items") opens each scene N times (N = cells per scene ~ 100).
Inversion opens each scene **once**. Same total work, 100× less I/O.

## Architecture

```
projects/satellite_climatology/
└── src/satellite_climatology/
    ├── grid.py             # (shared)
    ├── sensors.py          # (shared)
    ├── qa_decoders/        # (shared with v2.5)
    ├── operators.py        # (shared with v2.5) — the same per_scene graph
    ├── catalog.py          # (shared with v2) — DuckDBGeoCatalog scan
    ├── aggregate.py        # (extends v2) — bin per-item clear floats into cells
    └── global_pipeline.py  # the v3 entry point — runs the inverted loop
```

The four-repo capstone in one diagram:

```
            +---- DuckDBGeoCatalog (geocatalog) ----+
            | one row per STAC item                  |
            | bbox + datetime + asset hrefs          |
            +---------------------------------------+
                          |
                          v
            +---- geopatcher.SpatialPatcher ---------+
            | iterates monthly batches of items      |
            | (parallelisation unit = N items)       |
            +---------------------------------------+
                          |
                          v
            +---- geotoolz Operator graph -----------+   <- IDENTICAL to v2.5
            | ResolveAsset                           |
            | | ReadQA(reader="georeader")  <-+      |
            | | DecodeQA(decoder=...)        |      |  georeader does the
            | | CellClearFrac (per-cell mean)|      |  windowed read
            +--------------------------------|------+
                          |                  |
                          v                  v
            +---- accumulator (xarray Zarr) --------+
            | bands: clear_obs_count, clear_frac,   |
            |        pixel_max_gap_days             |
            +---------------------------------------+
```

## Output

Writes the last three bands of the shared Zarr:

```
clear_obs_count        (sensor, time, lat, lon) int16
clear_fraction         (sensor, time, lat, lon) float32
pixel_max_gap_days     (sensor, time, lat, lon) float32
```

After v3 the Zarr is complete and the dashboard shows the full ten-band
set, including the v2-vs-v3 difference map.

## Compute budget

Per S2 item (the worst case):

- SCL band is 20 m, single 109×109 km scene = ~5500×5500 pixels =
  60 MB uncompressed (12 MB after COG-deflate).
- Read full SCL via `georeader`: ~5–15 s per scene (network-bound).
- Decode + per-cell reduce: ~1–2 s.
- → ~10 s/scene wall-clock.
- 1 year of S2 globally ≈ 1M scenes.
- **3000 CPU-hours / year of S2.**

Sized down:

- 0.5° grid × 1 year × S2 only × CONUS: ~50k scenes × 10 s ÷ 32
  workers = ~4 hours. **Laptop-tractable.**
- 0.1° grid × 1 year × S2 only × CONUS: same scenes, same per-scene
  cost (resolution doesn't change the read). ~4 hours.
- 0.1° grid × 1 year × S2 only × global: ~3000 worker-hours →
  ~4 days on 32-worker cluster, ~half-day on 256-worker.

Memory: per-worker peak is one SCL band read (~150 MB live). Trivial.

Storage:

- Output Zarr (all ten bands, 5 sensors, 12 months, 0.1°): ~3 GB.
- Intermediate (per-scene clear-frac records): ~1 KB/scene × 1M
  scenes = ~1 GB per sensor-year.

## Sizing path

1. **M6.1** — single sensor (S2), continental AOI (CONUS), 0.5° grid,
   1 year. Validates the pipeline end-to-end. Laptop-tractable.
2. **M6.2** — S2 only, CONUS, 0.1° grid, 1 year. Same scenes, finer
   binning.
3. **M6.3** — S2 only, **global**, 0.1° grid, 1 year. First cluster
   job. Decide on infra (Azure Batch, Dask cluster, etc.).
4. **M6.4** — Add Landsat 8/9.
5. **M6.5** — Add MODIS / VIIRS once their decoders ship.

Ship the dashboard with v2 + v2.5 at M5; v3 progressively fills in the
global pixel-level bands as each milestone completes.

## UI integration

The v3 Zarr is rendered as additional tile layers in the same
dashboard from v1/v2. New dropdowns:

- `clear_obs_count` — observed pixel-clear scene count
- `clear_fraction` — the headline pixel-level cloud-free fraction
- `pixel_max_gap_days` — longest "no clear look" interval
- `scene_vs_pixel_diff` = `v2.cloud_free_scene_count` −
  `v3.clear_obs_count` — derived band; the v2/v3 disagreement map

The v2.5 AOI tab is unchanged — it remains the user-facing "drill
into a single AOI" tool, with v3 being its global archive.

## Risks & open questions

- **Compute is the headline risk.** Loop inversion is non-negotiable.
  Even with it, going global × all-sensors needs a cluster. Decide
  infra at M6.3 start.
- **`georeader` performance at scale** — micro-bench at M6.1 against
  `rioxarray` / `odc-stac` / direct rasterio with explicit windowed
  reads. If the API doesn't expose batched async reads, swap in
  whichever does — the `Operator` abstraction makes this transparent.
- **Re-runs are expensive.** Pin the catalog snapshot date in the
  Zarr metadata. Re-runs are full re-runs; incremental update is
  v3.1.
- **MGRS edge effects** — S2 scenes overlap their tile neighbours by
  ~10 km. A given cell near an MGRS edge can appear in 2–4 scenes
  per overpass. Double-counting? Not really: each is an independent
  observation and clear-fraction averages over them. Document the
  semantics so users know.
- **HDF reads for MODIS** — MODIS QA lives inside MOD09GA HDFs, not
  COGs. Separate `georeader` adapter or fallback to `pyhdf` /
  `h5py` + grid lookup. Defer to v3.1.
- **Cost** — if the cluster is on Azure (same region as MPC),
  egress is free and S2 reads are cheap (< $100 for 1 year × global).
  Cross-region or cross-cloud will dominate the bill.

## Acceptance

- All three pixel-level bands populated for at least Sentinel-2 over
  a 12-month window at 0.1° globally, written into the shared Zarr.
- Sanity (continues from v2.5):
  - `clear_obs_count` ≤ `v2.scenes_count` cell-wise (clear ≤ total).
  - `clear_fraction` map qualitatively matches MODIS MOD09CMG annual
    cloud-fraction climatology.
  - For a random sample of cells, the v3 Zarr value at the cell
    centre matches the v2.5 dashboard answer for an AOI = the
    cell polygon (within 2% — the only source of difference is
    binning).
- The v2-vs-v3 difference map renders in the dashboard and matches
  intuition (coastal / mountain / monsoon regions are the largest
  disagreement).
- Pipeline is reproducible from the run manifest
  (`get_config()` of every operator + the catalog-snapshot date).

## Out of scope

- Real-time / incremental updates — batch only.
- Adaptive resolution (denser grid where it matters) — uniform 0.1°.
- ML-based gap-filling — we report the raw observations.
- Atmospheric / BRDF correction quality flags.
- Above-MGRS-extent polar coverage for S2 (data simply doesn't exist).
