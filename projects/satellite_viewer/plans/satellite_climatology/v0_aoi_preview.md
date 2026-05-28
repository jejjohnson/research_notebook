---
title: "v0 — AOI preview tool (shipped)"
short_title: "v0 AOI preview"
---

# v0 — AOI preview tool

> *Given an AOI, a date range, and a list of sensors, return the
> intersecting STAC items with their footprints, timestamps, scene-wide
> cloud cover, and preview thumbnail URLs — before downloading anything.*

**Status:** shipped. The code referenced by this doc is the contents of
this PR (`projects/satellite_viewer/`). This page exists so the
staging arc starts from the thing already built, not from a blank
slate.

## Goal

A user-facing tool that answers *"what scenes are available here?"*
without paying the bandwidth bill of opening any of them. It's the
upstream-of-download step: inspect first, choose, then commit to a
download (or a `geocatalog.from_stac_items(...)` ingest, or a v2.5
clear-fraction request).

## Question answered

> "Across these sensors, between these dates, which scenes touched my
> AOI, when, and how cloudy were they at the scene level?"

This is **scene-level metadata only**. v2.5 is the same input shape
but the output is the *pixel-level truthful clear-fraction* for that
AOI — see `v2_5_pixel_level.md`.

## Scope

Polar-orbiting / tasked sensors only, all via Microsoft Planetary
Computer STAC (anonymous read, no auth):

| key              | description                                               |
|------------------|-----------------------------------------------------------|
| `sentinel-2-l2a` | Sentinel-2 L2A surface reflectance (10-60 m, ~5 d revisit)|
| `sentinel-1-grd` | Sentinel-1 C-band SAR GRD (10 m, all-weather)             |
| `landsat-c2-l2`  | Landsat C2 L2 surface reflectance (30 m, ~16 d revisit)   |
| `modis-09a1`     | MODIS Terra/Aqua 8-day surface reflectance (500 m)        |
| `emit-l2a-rfl`   | EMIT L2A imaging spectrometer reflectance (60 m, tasked)  |

Geostationary platforms (GOES, Himawari, Meteosat) are deliberately
out of scope — their ~5-15 minute cadence makes "is there imagery
over my AOI?" trivially yes inside any scan sector.

## Algorithm

```
def search(sensor, aoi, t0, t1, *, cloud_lt=None, max_items=100):
    cfg = SENSORS[sensor]
    client = pystac_client.Client.open(
        cfg.stac_endpoint,
        modifier=planetary_computer.sign_inplace if cfg.requires_pc_signing else None,
    )
    items = client.search(
        collections=[cfg.collection_id],
        intersects=aoi.__geo_interface__,
        datetime=f"{t0.isoformat()}/{t1.isoformat()}",
        query={cfg.cloud_field: {"lt": cloud_lt}} if cloud_lt is not None else None,
        max_items=max_items,
    ).items()
    return gpd.GeoDataFrame.from_records(
        {
            "id": it.id,
            "datetime": ...,
            "geometry": shape(it.geometry),
            "sensor": cfg.name,
            "cloud_cover": it.properties.get(cfg.cloud_field),
            "preview_url": it.assets[cfg.preview_asset].href,
        }
        for it in items
    )
```

One STAC call per sensor; signed `rendered_preview` hrefs let the UI
display thumbnails without fetching pixel data.

## Architecture

```
projects/satellite_viewer/
├── src/satellite_viewer/
│   ├── __init__.py          # exports search, SENSORS
│   ├── sensors.py           # the SENSORS registry (5 entries)
│   └── search.py            # the search() function above
├── apps/
│   ├── panel_app.py         # Panel subapp (leafmap + holoviews + tabulator)
│   └── streamlit_app.py     # Streamlit subapp (folium + altair)
├── notebooks/
│   └── viewer.py            # jupytext py:percent notebook (ipywidgets + leafmap + matplotlib)
└── tests/
    └── test_sensors.py      # offline registry sanity checks (4 tests)
```

Repos & how they slot in:

- **`pystac-client`** — STAC search and item iteration.
- **`planetary-computer`** — signs the asset hrefs.
- **None of jejjohnson/* yet.** v0 is deliberately library-light; the
  later stages bring `geocatalog`, `geopatcher`, `geotoolz`, and
  (spaceml-org's) `georeader` in turn.

## Output

A single `GeoDataFrame`, schema stable across sensors:

```
columns:
  id                   : string         # STAC item id
  datetime             : datetime[UTC]  # acquisition timestamp
  geometry             : Polygon        # scene footprint (WGS84)
  sensor               : string         # registry key
  cloud_cover          : float | NaN    # eo:cloud_cover (scene-wide)
  preview_url          : string | None  # signed rendered_preview href
crs:  EPSG:4326
```

## UI integration

Three subapps share this DataFrame as the data layer:

1. **Panel** (`apps/panel_app.py`) — sidebar widgets, leafmap map,
   holoviews timeline, FlexBox thumbnail strip, tabulator results.
2. **Streamlit** (`apps/streamlit_app.py`) — sidebar widgets, folium
   map with Draw plugin, altair timeline, st.columns thumbnail grid,
   st.dataframe results.
3. **Jupyter** (`notebooks/viewer.py`) — ipywidgets controls, leafmap
   map with DrawControl, matplotlib timeline, ipywidgets.Image grid.

All three call `satellite_viewer.search(...)` for discovery; they only
differ in the presentation layer.

## Compute budget

Per request:

- One STAC search per selected sensor, paginated by `max_items`
  (default 50). ~500 ms - 3 s per sensor depending on AOI size and
  result density.
- No pixel reads, no downloads. Preview thumbnails are fetched
  lazily by the UI when they render.
- Memory: trivial (≤ a few MB of GeoDataFrame + ≤ 50 thumbnail PNGs
  of ~30 KB each).

## Risks & open questions

- **STAC endpoint quirks** — `query={"eo:cloud_cover": {"lt": ...}}`
  only filters where the field exists. SAR (S1) has no cloud_cover;
  the search code skips the query for sensors with `cloud_field=None`.
- **Preview asset key isn't standard.** MPC uses `rendered_preview`;
  Earth Search uses `thumbnail`. The registry keeps this per-sensor
  to stay portable.
- **Date range > sensor lifetime** — the apps don't clip to per-sensor
  availability windows (S2 starts 2015, Landsat 8 starts 2013, etc.).
  Result is just "0 items" instead of an explicit error.
- **Live UI smoke-test still pending** — the test suite covers the
  registry only. Panel / Streamlit / notebook subapps haven't been
  clicked through against live MPC yet (test plan checklist in the
  draft PR).

## Acceptance

- `pixi run -e satellite-viewer test-satellite-viewer` passes (4/4
  registry checks).
- `ruff check` and `ruff format --check` pass on
  `projects/satellite_viewer/`.
- Each of the three subapps starts and renders the default Lake Tahoe
  AOI without errors. Clicking *Search* with default Sentinel-2
  yields scenes on the map + timeline + thumbnails.

## Where v0 ends, the climatology starts

v0 returns *which scenes intersect my AOI*. Three follow-on questions
are answered by the climatology stages:

- *How often **could** I have imaged here?* → v1
- *How often **was** here imaged, and how cloudy were those scenes?*
  → v2 (globally), already-shipped v0 (per-AOI scene-level)
- *How often was **this exact pixel** clear?* → v2.5 (per-AOI),
  v3 (globally)

The v2.5 AOI subapp will reuse `satellite_viewer.search` verbatim for
its discovery step — that's the v0 → v2.5 contract.

## Out of scope

- Pixel reads of any kind (v2.5 / v3).
- Catalog persistence (v2 / v3 brings `geocatalog.DuckDBGeoCatalog`).
- Cloud-free time series for a single AOI (v2.5).
- Global cadence / coverage maps (v1 / v2 / v3).
- Geostationary sensors (out of scope project-wide).
