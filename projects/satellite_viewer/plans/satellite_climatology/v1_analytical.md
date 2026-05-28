# v1 — Analytical revisit climatology

> *Where on Earth, and how often, could each sensor have imaged the
> surface — independent of what's actually in the catalog?*

## Goal

For every pixel of a 0.1° global grid and each sensor in scope, compute
the **theoretical overpass cadence** purely from orbit propagation +
swath geometry. No catalog scan, no STAC, no pixel reads.

## Question answered

> "How many times in $[t_0, t_1]$ does a satellite with this orbit and
> swath cross over this pixel?"

Useful as:

1. The **ceiling** the data-driven stages get compared against —
   missing scenes in v2 = catalog gaps, processing failures, off-nadir
   tasking limits, etc.
2. A **standalone product** for users who need "max possible cadence"
   regardless of cloud / catalog status (mission planning,
   feasibility studies).

## Scope

- **Sensors**: full list from the shared README — `sentinel-2`,
  `landsat-8-9`, `modis-terra`, `modis-aqua`, `viirs-jpss`.
- **Time window**: configurable. Default: rolling 24 months ending now.
- **Resolution**: 0.1° (shared default).
- **Grid**: WGS84 regular lat/lon (shared default).

## Algorithm

For each sensor:

1. **Acquire TLEs** for every platform in the constellation, sampled at
   weekly cadence over the window (orbits drift; one TLE per year is
   too coarse). Source: CelesTrak or Space-Track.
2. **Propagate** with `skyfield` (or `sgp4`) at a fixed time step Δt
   (60 s is fine — at 7 km/s the satellite moves ~420 km/step, finer
   than the swath width for any of these sensors).
3. **Project ground track + swath** onto the WGS84 grid:
   - Compute nadir lat/lon at each step.
   - Inflate to a cross-track swath polygon using the sensor's swath
     width (S2: 290 km, L8/9: 185 km, MODIS: 2330 km, VIIRS: 3060 km).
   - Rasterise the swath polygon to the grid → boolean mask of cells
     touched at this timestep.
4. **Tally per cell**: increment `overpasses[lat, lon, sensor, month]`
   for every cell touched.
5. **Gap statistics**: for each cell, derive `mean_gap_days` and
   `p95_gap_days` from the timestamps of overpasses in that cell.

## Architecture

This stage is intentionally **library-light**. No `geocatalog`, no
`geotoolz`. Single notebook + small helper module:

```
projects/satellite_climatology/
└── src/satellite_climatology/
    ├── grid.py        # global Zarr schema + grid helpers (shared)
    ├── sensors.py     # platform → swath / inclination / TLE source
    └── analytical.py  # propagate_and_rasterise(sensor, t0, t1) -> xr.Dataset
```

The one place a repo *could* be used:

- **`geopatcher`** could chunk the global grid into N tiles for
  parallel rasterisation. Useful only when you push to 0.01° or want
  multi-node compute. **For 0.1°, skip it** — a single-process loop
  finishes in minutes.

## Output

Writes these three bands of the shared Zarr:

```
overpasses             (sensor, time, lat, lon) int16
mean_gap_days          (sensor, time, lat, lon) float32
p95_gap_days           (sensor, time, lat, lon) float32
```

## Compute budget

Per sensor:

- 24 months × 30 d × 1440 min/d ≈ 1M timesteps.
- Per timestep: rasterise one swath polygon into the global grid.
  Vectorised numpy / shapely-prepared polygon → ~ms per step.
- → ~15–30 min per sensor, single-process.
- × 5 sensors = ~2 hours wall-clock on a laptop.

Memory: peak ~1 GB (grid + accumulator).

## UI integration

The Zarr is loaded by the dashboard / notebook directly. UI controls:

- Sensor dropdown (single-select)
- Stat dropdown: `overpasses` | `mean_gap_days` | `p95_gap_days`
- Time-window slider (start / end month)
- Aggregation: monthly (raw) | yearly mean | window total

The whole-globe raster is rendered as a tile layer (leafmap `add_raster`
or deck.gl `BitmapLayer` after a quick TiTiler tile).

## Risks & open questions

- **TLE quality drift** — TLE accuracy degrades over weeks. Mitigate
  by pulling one TLE per platform per week and using each TLE only
  for ±3 days around its epoch.
- **Tasked vs. nominal acquisition** — EMIT and (partly) S2 only
  acquire over tasked targets. The "theoretical max" overstates
  reality for these. **Document this caveat in the UI**; don't try
  to model tasking here.
- **Sensor decommissioning** — Landsat 7 / Aqua / Terra are end-of-life
  and may stop acquiring during the window. Pull each platform's status
  from CelesTrak and respect end-dates.
- **Swath roll** — some sensors point off-nadir (SPOT, Pleiades). For
  our list this isn't an issue (all nadir-only), so the simple
  inflate-by-swath-width approximation is fine.
- **Day/night** — should we count only daytime passes? Most optical
  users want this; SAR users want both. Add a `daylight_only` band
  (computed from sub-satellite solar zenith) so the UI can flip
  between the two.

## Acceptance

- Zarr written to `data/satellite_climatology.zarr` matches the
  schema in the shared README.
- Sanity checks pass:
  - Equatorial cell has S2 `overpasses` ≈ `(t1 - t0) / 5 days` (within ±20%).
  - Polar cell has > equatorial cell for all polar-orbiting sensors.
  - MODIS `overpasses` > S2 `overpasses` everywhere.
- Notebook reproduces a published figure (e.g., the equator → pole
  revisit curve for S2 from the ESA Sentinel-2 user guide).
- UI tile layer renders without serverside compute.

## Out of scope

- Cloud cover (that's v2).
- Tasked acquisition modelling (that's a research problem).
- Sub-pixel ground sample distance — we're per-grid-cell only.
- Pointing / agility (off-nadir constellations).
