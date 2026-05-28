---
title: "v2.5 — Per-AOI pixel-level cloud climatology"
short_title: "v2.5 AOI pixel-level"
---

# v2.5 — Per-AOI pixel-level cloud climatology

> *Given an AOI (point with radius, polygon, or bbox up to ~50 km), use
> the actual QA bands of every intersecting scene to compute the
> truthful clear-fraction time series — on demand, in seconds.*

This is the **single-AOI, on-demand** version of the pixel-level problem.
It shares the AOI flow with `satellite_viewer` (the same input shape —
draw a polygon, pick a sensor, pick a date range) but the output is the
**clear-fraction climatology** for that AOI, not a thumbnail preview.

## Goal

For one user-supplied AOI and time window:

1. Discover intersecting scenes (reuse `satellite_viewer.search`).
2. For each scene, lazily read the QA band windowed to the AOI via
   `georeader`.
3. Decode the QA band to a per-pixel clear mask using a per-sensor
   decoder.
4. Reduce the mask to a single number: **clear-pixel fraction** inside
   the AOI for this acquisition.
5. Return the (datetime, clear_fraction, scene_id) time series plus
   aggregates (`clear_obs_count`, `mean_clear_fraction`,
   `longest_clear_gap_days`).

## Question answered

> "Across all scenes that touched this AOI in $[t_0, t_1]$, what
> fraction of the AOI's pixels were *actually* clear in each scene, and
> what does the resulting clear-fraction time series look like?"

Validates v3's algorithm at user-facing scale before committing to the
global compute. Also a **standalone product**: ecologists, agronomists,
plume hunters all care about "how often can I see this specific field /
hotspot / region clearly?" — exactly the v2.5 deliverable.

## Scope

- **AOI**: single shapely geometry, **size-capped** (default 50 km bbox
  side; user knob). Above the cap the dashboard refuses the request
  with a "use v3 / batch mode" hint.
- **Sensors**: starts with `sentinel-2-l2a` (SCL band). Add
  `landsat-c2-l2` (QA_PIXEL) next. MODIS / VIIRS deferred.
- **Time window**: user-supplied. Multi-year is fine — the work is
  proportional to scenes, not to area.
- **Compute target**: 1–30 s per request on a laptop / single
  notebook server.

## Algorithm

```
1. items = satellite_viewer.search(sensor, aoi, t0, t1, cloud_lt=None, max_items=1000)
2. for item in items:                                 # → parallelisable
       qa = georeader.read_window(item.assets[qa_band_key], aoi)
       clear_mask = decoder(qa, clear_classes=cfg.clear_classes)
       clear_frac = clear_mask.mean()                 # float in [0, 1]
       record (item.datetime, item.id, clear_frac)
3. emit:
   - per-scene time series: [(datetime, scene_id, clear_frac, eo:cloud_cover)]
   - aggregates: clear_obs_count = sum(frac > clear_threshold),
                 mean_clear_fraction = mean(frac),
                 longest_clear_gap_days = max(gap between frac > threshold)
```

The per-item work is the same code that v3 will run in its inner loop —
the abstraction split is "v2.5 = one AOI, run interactively" vs.
"v3 = every cell of the global grid, run as a batch."

## Architecture

```
projects/satellite_climatology/
└── src/satellite_climatology/
    ├── grid.py             # (shared with v1/v2/v3) — not used by v2.5
    ├── sensors.py          # (shared) — adds qa_band_key, clear_classes per sensor
    ├── qa_decoders/        # (shared with v3)
    │   ├── s2_scl.py
    │   ├── landsat_qapixel.py
    │   └── modis_state1km.py
    ├── operators.py        # (shared with v3) — ReadQA, DecodeQA, CellClearFrac
    └── aoi_pipeline.py     # the v2.5 entry point — runs the per-scene op over an AOI
```

Repo wiring:

- **`satellite_viewer.search`** — reused unchanged for discovery. The
  same `(sensor, aoi, t0, t1) -> GeoDataFrame[id, datetime, geometry, …]`
  surface is what v2.5 consumes.
- **`geotoolz`** holds the per-scene pipeline as a serialisable
  `Operator` graph:
  ```
  per_scene = (
      ResolveAsset(asset_key=cfg.qa_band)        # item -> signed COG URL
      | ReadQA(reader="georeader", window=aoi)    # URL -> ndarray
      | DecodeQA(decoder=cfg.qa_decoder, clear_classes=cfg.clear_classes)
      | CellClearFrac()                           # ndarray -> float in [0, 1]
  )
  ```
  The same graph is reused in v3. `get_config()` serialises the whole
  thing so the v3 batch job and the v2.5 interactive call produce
  bit-identical numbers.
- **`georeader`** is the heavy lifting: lazy windowed reads of the
  COG / HDF / Zarr asset, clipped to the AOI bbox at the asset's CRS,
  no full-scene download.
- **`geocatalog`** is **not** used at v2.5 scale — we don't need a
  persistent index for a single AOI's worth of items. v3 brings it
  back when the catalog scan goes global.
- **`geopatcher`** is **not** used at v2.5 (single AOI, no tiling).
  v3 brings it back.

## Output

A pandas DataFrame, not a Zarr band:

```
columns: ["datetime", "scene_id", "sensor", "scene_cloud_pct", "clear_fraction"]
shape:   (n_scenes, 5)
index:   reset
crs:     no geometry (the AOI is shared across rows)
```

Plus the three scalar aggregates returned alongside.

## UI integration

A new tab in the `satellite_viewer` dashboard, or a sibling subapp.
Same AOI input shape as the preview tool. Output panes:

1. **Clear-fraction time series** — scatter or line of `clear_fraction`
   vs. `datetime`, colour-coded by sensor.
2. **Climatology stats card** — `clear_obs_count`, `mean_clear_fraction`,
   `longest_clear_gap_days`, `n_scenes`.
3. **Side-by-side scenes** — for each acquisition, RGB thumbnail (from
   STAC `rendered_preview`, reused from `satellite_viewer`) next to the
   QA-derived clear mask preview at AOI scale.
4. **Comparison with scene-level** — overlay `eo:cloud_cover` (the v2
   answer) vs. `1 − clear_fraction` (the v2.5 answer). Where they
   diverge is the entire point of v2.5.

## Compute budget

Per AOI, default 50 km bbox cap:

- 1 year of S2 over a 50 km AOI ≈ 50–150 items.
- Per item: one windowed COG read of the SCL band, AOI-clipped → ~50
  KB to ~5 MB raw bytes, dominated by TCP / TLS / signing.
- → ~100 items × 100–300 ms (parallelised with `asyncio` / thread
  pool) = **3–10 s wall-clock**.
- Memory: peak ~50 MB for the largest single SCL window.

This is the dashboard-tractable target. Above the size cap or beyond
~5-year windows the user gets routed to v3.

## Risks & open questions

- **Per-sensor QA decoder is real work** — S2 SCL is the easiest (single
  class enum, well-documented). Landsat QA_PIXEL is bit-packed.
  MODIS state_1km is bit-packed and version-dependent. **Ship S2 only
  for M5.**
- **"Clear" definition is a user knob.** Default for S2 SCL: classes
  {4, 5, 6, 11} (veg, bare, water, snow) treated as clear; {3, 8, 9, 10}
  (shadow, cloud_med, cloud_high, thin_cirrus) treated as not clear;
  {7} (unclassified) treated as not clear by default. Expose
  `clear_classes` in the dashboard so users can flip e.g. snow on/off.
- **CRS handling** — S2 SCL is per-MGRS-tile UTM; AOI is EPSG:4326.
  Reprojection on read (or AOI reprojection to the item's CRS) is
  `georeader`'s job. Verify it does the right thing for a multi-tile
  AOI in M5's first day.
- **AOI larger than one MGRS tile** — request is split into per-tile
  reads, merged at AOI scale. `georeader` already handles this for
  S2; verify and bench.
- **Out-of-MGRS AOIs** (e.g., polar > 80°) need explicit handling
  since S2 doesn't tile there. Just exclude with a clear error.
- **Async vs. thread pool** — `georeader`'s API is sync (last I
  checked). Wrap with `concurrent.futures.ThreadPoolExecutor` for
  per-scene parallelism. Re-evaluate if `georeader` adds an async API.

## Acceptance

- Single notebook + Panel/Streamlit subapp answering "draw an AOI,
  pick a date range, get a clear-fraction time series" in < 15 s for
  the default 50 km × 1 year case.
- Sanity:
  - `clear_fraction` ≤ 1 in every row.
  - For cells fully inside a single scene of low `eo:cloud_cover`:
    `clear_fraction ≈ 1 − eo:cloud_cover / 100` (within ~10%).
  - For cells inside the cloudy half of a heterogeneous scene:
    `clear_fraction` ≪ `1 − eo:cloud_cover / 100` (the *point* of v2.5).
- The per-scene operator graph runs unchanged in a v3 prototype over
  a 1° × 1° patch (same numbers as the dashboard).

## Out of scope

- Global compute (that's v3).
- Multi-sensor fusion (different sensors have different "clear"
  semantics; defer until each sensor's decoder ships).
- Atmospheric correction quality flags (only "clear or not").
- Sub-pixel cloud masking — we accept the QA band's spatial resolution.
- Real-time / incremental updates — every request re-fetches.
