---
title: "Satellite climatology — design docs"
---

# Satellite climatology — design docs

A global product showing **temporal cadence and cloud-free observability**
of satellite imagery per pixel of the Earth, sliced by sensor and time
window. Built sequentially in five stages of increasing fidelity:

| Stage    | Scale             | What it answers                                              | Data source              | New repo introduced |
|----------|-------------------|--------------------------------------------------------------|--------------------------|---------------------|
| **v0**   | one AOI on demand | *Which scenes touched my AOI?* (footprints + thumbnails)    | STAC item metadata       | `pystac-client` + `planetary-computer` |
| **v1**   | global            | *Theoretical* overpass cadence per pixel per sensor          | TLEs / orbit propagation | (skyfield)          |
| **v2**   | global            | *Observed* scene count + scene-level cloud cover per pixel   | STAC `eo:cloud_cover`    | `geocatalog`, `geopatcher`, `geotoolz` |
| **v2.5** | one AOI on demand | *True per-pixel* clear-observation fraction inside an AOI    | STAC + windowed reads    | `georeader`         |
| **v3**   | global            | v2.5 scaled to the whole globe — *true per-pixel*, batched  | STAC + windowed reads    | (same as v2.5; cluster) |
| **v4**   | global + AOI      | Coverage ledger: *available* vs *acquired* vs *gap* (+ tasking hook) | v1–v3 bands + external holdings DB | external PostGIS holdings table (`.env` creds); reuses satellite_viewer.search |

Each global stage writes into the **same Zarr product** below (except
v0 and v2.5, which are per-AOI tools returning a DataFrame, not a
global grid). The dashboard reads whichever bands exist.

## Why staged

- **v0** is the satellite_viewer AOI preview tool — already shipped in
  this PR. *"What scenes are available here?"* with footprints,
  timestamps, scene-wide cloud cover, and preview thumbnails. The
  upstream-of-download inspection step.
- **v1** is a ~100-line orbit-mechanics script with **no catalog scan**.
  It gives the *theoretical ceiling* (how often *could* you image here?)
  and is the baseline the data-driven stages get compared against.
- **v2** adds the catalog scan but stays at scene-level metadata. ~100×
  cheaper than v3. Answers "how many actual scenes per pixel, and what
  was the *scene-wide* cloud cover?" Good enough for many users.
- **v2.5** is the **single-AOI, on-demand** variant of the pixel-level
  problem. Inherits v0's AOI flow: pick an AOI (up to a size cap,
  ~50 km), the system reads QA bands for the intersecting scenes and
  returns the **truthful** clear-fraction time series for that AOI.
  Validates QA decoders + `georeader` integration at user-facing scale
  before committing to global compute.
- **v3** is v2.5 scaled to the whole globe — same per-scene operator
  graph, batched per tile, written into the global Zarr. Cluster-grade
  compute.

## Shared substrate

### Global grid

```
crs        : EPSG:4326
resolution : 0.1°   (3600 × 1800 = 6.48M cells)  -- tunable knob
extent     : [-180, -90, 180, 90]
indexing   : (lat ascending, lon ascending)
```

The 0.1° default puts each cell at ~11 km at the equator, ~6 km at 50°.
That matches the scene-footprint scale of Landsat/S2 (~110×110 km) ÷ ~10
so a typical scene covers ~100 cells — fine for revisit stats, coarse
enough that 6.48M cells × ~5 sensors × ~24 months × ~6 bands fits in a
few GB Zarr.

Switch to **H3 hex bins** (res 5: ~250 km² cells) if equal-area is
important — discussed in v2.

### Sensor list

| key                  | platforms                                  | nominal revisit      |
|----------------------|--------------------------------------------|----------------------|
| `sentinel-2`         | Sentinel-2A + 2B + 2C                      | 5 days @ equator     |
| `landsat-8-9`        | Landsat 8 + 9 (combined)                   | 8 days               |
| `modis-terra`        | Terra MODIS                                | ~daily               |
| `modis-aqua`         | Aqua MODIS                                 | ~daily               |
| `viirs-jpss`         | NPP + JPSS-1 + JPSS-2 VIIRS                | ~daily               |

Sensor key is a dimension coordinate in the Zarr.

### Output product (Zarr schema)

```
Dims     : (sensor, time, lat, lon)
Coords   :
  sensor : ["sentinel-2", "landsat-8-9", "modis-terra", "modis-aqua", "viirs-jpss"]
  time   : pandas.PeriodIndex(freq="M", start=...)   # monthly bins
  lat    : np.arange(-90, 90, 0.1)
  lon    : np.arange(-180, 180, 0.1)
Data vars:
  # v1 — analytical
  overpasses             : int16     # count of overpasses in this month
  mean_gap_days          : float32   # mean gap between consecutive overpasses
  p95_gap_days           : float32   # 95th percentile gap (long-gap stat)

  # v2 — data-driven, scene-level
  scenes_count           : int16     # number of catalog items intersecting this cell
  mean_scene_cloud_pct   : float32   # mean of eo:cloud_cover across items
  cloud_free_scene_count : int16     # items where eo:cloud_cover < 10%

  # v3 — pixel-level QA (global)
  clear_obs_count        : int16     # pixel actually clear in QA mask
  clear_fraction         : float32   # clear_obs_count / scenes_count
  pixel_max_gap_days     : float32   # longest gap between clear observations
```

A given Zarr can be partially populated — v1 only writes the first three
bands, v2 adds the next three, v3 adds the last three. The UI reads
what's there.

**v2.5 does *not* write into this Zarr.** It produces a per-AOI time-series
DataFrame and is rendered live in the dashboard. The same `ReadQA → DecodeQA
→ CellClearFrac` pipeline is reused unchanged in v3 — that's the contract
the staging guarantees.

### Time binning

Monthly is the default — captures seasonality (cloud climatology has
strong monsoon / wet-season signal) without exploding the time axis.
Yearly is a one-line `.resample("Y").sum()` collapse for users who just
want long-term averages.

## Repos & how they slot in

| Stage  | geocatalog  | geopatcher                  | geotoolz                | georeader        |
|--------|-------------|-----------------------------|-------------------------|------------------|
| v0     | —           | —                           | —                       | —                |
| v1     | —           | global grid iteration (opt) | —                       | —                |
| v2     | catalog scan + bbox queries | grid iteration over cells | `Fanout` of per-cell reducers | — |
| v2.5   | (reuses satellite_viewer.search) | — (single AOI) | per-scene `Operator` graph for QA mask | windowed SCL/QA reads |
| v3     | (as v2)     | (as v2)                     | (as v2.5)               | (as v2.5)        |

## Milestones (suggested)

0. **M0 — v0 (shipped)**: the satellite_viewer AOI preview tool in this PR.
   `satellite_viewer.search` + Panel / Streamlit / Jupyter subapps.
1. **M1 — design**: this folder. Five docs reviewed before more code.
2. **M2 — v1 prototype**: skyfield + numpy, single notebook, 0.5° grid,
   one sensor. Validates the Zarr schema and the UI hook.
3. **M3 — v1 full**: full sensor list, 0.1° grid, monthly bins.
4. **M4 — v2 scene-level**: catalog scan over 1 year, S2 only, 0.1°.
5. **M5 — v2 full**: extend to all sensors and 3 years.
6. **M6 — v2.5 AOI pixel-level**: per-AOI dashboard reusing
   `satellite_viewer.search` for discovery, `georeader` for QA reads.
   Single sensor first (S2).
7. **M7 — v3 global pixel-level**: same operators batched over the
   global grid; sized for a cluster job.
8. **M8 — UI**: notebook + Panel/Streamlit dashboard reading the Zarr,
   with the v0 / v2.5 AOI panels as separate tabs.

## Files in this folder

- [`v0_aoi_preview.md`](v0_aoi_preview.md) — AOI preview tool (shipped).
- [`v1_analytical.md`](v1_analytical.md) — orbit-mechanics version.
- [`v2_data_driven_scene_level.md`](v2_data_driven_scene_level.md) — catalog + scene-level cloud.
- [`v2_5_pixel_level.md`](v2_5_pixel_level.md) — per-AOI pixel-level via georeader.
- [`v3_pixel_level_global.md`](v3_pixel_level_global.md) — v2.5 scaled to the whole globe.
- [`v4_coverage_planner.md`](v4_coverage_planner.md) — available / acquired / gap coverage dashboard (global heatmap + AOI drill-down), tasking hook future-flagged.
- [`v4_coverage_planner_api.md`](v4_coverage_planner_api.md) — v4 implementation spec: module API, type signatures, and demos (build pipeline + both apps) grounded in the real geocatalog API, with the Acquired layer as a generic PostGIS read (`.env` creds).
