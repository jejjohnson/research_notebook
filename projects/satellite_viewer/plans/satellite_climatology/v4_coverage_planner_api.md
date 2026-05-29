---
title: "v4 — Coverage planner: API & demos"
short_title: "v4 API & demos"
---

# v4 — Coverage planner: API & demos

> Implementation spec for [`v4_coverage_planner.md`](v4_coverage_planner.md).
> Concrete module layout, type signatures, and runnable-looking demos for
> the **library**, the **build pipeline**, and the **two apps**. Signatures
> for `geocatalog` calls are verified against that package; the Acquired
> layer is a generic PostGIS read (no project-specific package); the
> `satellite_climatology` API below is *proposed* (this doc is the contract
> we build to).

All examples assume:

```python
import datetime as dt
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
```

## Package layout

```
projects/satellite_climatology/
├── pyproject.toml                 # deps: geocatalog[duckdb], xarray, zarr, sqlalchemy,
│                                  #       psycopg2, python-dotenv, skyfield (v1), folium, ...
├── config/
│   └── build.toml                 # which sensors / window / grid / dest
├── .env.example                   # template for the holdings DB creds (.env is gitignored)
└── src/satellite_climatology/
    ├── grid.py                    # GridSpec + footprint_to_cells
    ├── sensors.py                 # COVERAGE_SENSORS registry (avail + held + taskable)
    ├── catalog.py                 # availability scan via geocatalog (multi-source)
    ├── locations.py               # optional target AOIs (source-agnostic loader)
    ├── holdings.py                # our holdings via an external PostGIS table (.env creds)
    ├── coverage.py                # build Zarr, compute gap, open + select for the UI
    ├── tasking.py                 # taskability table + inert request hook
    ├── render.py                  # DataArray -> map layer (folium / geoviews)
    ├── _cli.py                    # `python -m satellite_climatology ...`
    └── apps/
        ├── global_heatmap_streamlit.py    # App A
        ├── global_heatmap_panel.py        # App A
        └── aoi_drilldown_streamlit.py     # App B
```

## Configuration

Holdings-DB credentials are **not** in `build.toml` and **not** in the repo —
they live in a **gitignored `.env`** (see `.env.example`: `COVERAGE_DB_*` for
the connection, `COVERAGE_TILES_*` to map column names), read by `holdings.py`.
`build.toml` only holds the build knobs:

```toml
# config/build.toml
[grid]
resolution = 0.1            # degrees; or set h3_res for hex bins
crs = "EPSG:4326"

[window]
start = "2025-01-01"
end   = "2025-12-31"
freq  = "MS"                # monthly bins (pandas offset alias)

[sensors]
keys = ["emit", "sentinel-2", "landsat-8-9"]

[output]
zarr = "data/coverage.zarr"
parquet_dir = "data/catalog"   # availability GeoParquet store
```

## Core types

```python
# grid.py
from dataclasses import dataclass
import numpy as np
from shapely.geometry.base import BaseGeometry

@dataclass(frozen=True)
class GridSpec:
    resolution: float = 0.1          # degrees (ignored if h3_res set)
    crs: str = "EPSG:4326"
    h3_res: int | None = None        # if set, hex bins instead of lat/lon raster

    @property
    def shape(self) -> tuple[int, int]:          # (n_lat, n_lon)
        ...
    @property
    def lats(self) -> np.ndarray: ...            # cell-centre latitudes  (ascending)
    @property
    def lons(self) -> np.ndarray: ...            # cell-centre longitudes (ascending)

def footprint_to_cells(geom: BaseGeometry, grid: GridSpec) -> list[tuple[int, int]]:
    """Grid cells a footprint touches, as (lat_idx, lon_idx). Reused by v2."""

def bbox_to_index(bbox: tuple[float, float, float, float], grid: GridSpec
                  ) -> tuple[slice, slice]:
    """(minx,miny,maxx,maxy) -> (lat_slice, lon_slice) for clipping a DataArray."""
```

```python
# sensors.py — Available sources (often several per sensor) + Acquired name
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class AvailableSource:
    kind: Literal["stac", "cmr", "earthaccess"]   # -> geocatalog.from_stac_search / from_cmr / ...
    endpoint: str
    collection_id: str
    cloud_field: str | None = None            # eo:cloud_cover, or None (HSI/SAR without it)

@dataclass(frozen=True)
class CoverageSensor:
    key: str                                  # our stable key, e.g. "emit"
    available_sources: tuple[AvailableSource, ...]   # primary first; () = no public source
    held_satellite: str | None                # holdings "sensor" value, or None if not held
    nominal_revisit_days: float
    default_desired_per_month: float          # pre-fill for the per-sensor target
    taskable: bool = False                    # SATELLITE TASKING (point the sensor at a target).
                                              # Distinct from *ingest-requestable*, which holds
                                              # whenever available_sources is non-empty — see
                                              # "Request semantics" below.

# Most of these are served by more than one provider; list a primary + alternates
# (failover / cross-check). Each AvailableSource.kind maps to a geocatalog from_* call.
_PC           = "https://planetarycomputer.microsoft.com/api/stac/v1"
_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
_CDSE         = "https://catalogue.dataspace.copernicus.eu/stac"
_CMR_LPCLOUD  = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
_DLR          = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"

COVERAGE_SENSORS: dict[str, CoverageSensor] = {
    "emit": CoverageSensor(
        key="emit",
        available_sources=(
            AvailableSource("cmr", _CMR_LPCLOUD, "EMITL2ARFL"),
            AvailableSource("earthaccess", _CMR_LPCLOUD, "EMITL2ARFL"),  # same archive, auth dl
        ),
        held_satellite="EMIT",                # holdings "sensor" == "EMIT"
        nominal_revisit_days=16, default_desired_per_month=2,
        taskable=True,                        # EMIT target-request list (tasked instrument)
    ),
    "enmap": CoverageSensor(
        key="enmap",
        available_sources=(
            AvailableSource("stac", _DLR, "ENMAP_HSI_L2A", "eo:cloud_cover"),
        ),
        held_satellite="EnMAP",               # holdings "sensor" == "EnMAP"
        nominal_revisit_days=27, default_desired_per_month=2,
        taskable=True,                        # DLR accepts acquisition requests
    ),
    "prisma": CoverageSensor(
        key="prisma",
        available_sources=(),                 # NO public STAC — ASI email/FTP ingest only
        held_satellite="PRISMA",              # holdings "sensor" == "PRISMA"
        nominal_revisit_days=7, default_desired_per_month=2,
        taskable=True,                        # ASI tasking
    ),
    "landsat-8-9": CoverageSensor(
        key="landsat-8-9",
        available_sources=(
            AvailableSource("stac", _PC, "landsat-c2-l2", "eo:cloud_cover"),
            AvailableSource("stac", _EARTH_SEARCH, "landsat-c2-l2", "eo:cloud_cover"),
        ),
        held_satellite="Landsat",             # holdings "sensor" == "Landsat"
        nominal_revisit_days=8, default_desired_per_month=3,
        taskable=False,                       # systematic survey — nothing to task
    ),
    "sentinel-2": CoverageSensor(
        key="sentinel-2",
        available_sources=(
            AvailableSource("stac", _PC, "sentinel-2-l2a", "eo:cloud_cover"),
            AvailableSource("stac", _EARTH_SEARCH, "sentinel-2-l2a", "eo:cloud_cover"),
            AvailableSource("stac", _CDSE, "SENTINEL-2", "eo:cloud_cover"),  # Copernicus DataSpace
        ),
        held_satellite=None,                  # not in holdings -> Available-only
        nominal_revisit_days=5, default_desired_per_month=6,
        taskable=False,                       # systematic survey — nothing to task
    ),
}

# Per-sensor layer + request matrix (drives the "sensor-set mismatch" guard):
#   sensor       Available (sources)            Acquired   ingest-req?  taskable?
#   emit         CMR + earthaccess              EMIT       yes          yes
#   enmap        STAC(DLR)                      EnMAP      yes          yes
#   prisma       — (ASI email/FTP, no STAC)     PRISMA     n/a*         yes
#   landsat-8-9  STAC(PC / Earth Search)        Landsat    yes          NO  (systematic)
#   sentinel-2   STAC(PC / Earth Search / CDSE) —          yes          NO  (systematic)
# *PRISMA has no queryable public Available layer yet, so its ingest backlog
#  can't be computed from a catalog — only its Acquired holdings are shown.
# The dashboard shows only the layers/actions a sensor actually supports.
```

```python
# coverage.py — query + gap types
from dataclasses import dataclass, field
from typing import Literal

Layer = Literal["available", "acquired", "gap"]
Aggregation = Literal["rate", "total", "recent", "worst", "scrub"]

@dataclass(frozen=True)
class DesiredCadence:
    per_sensor: dict[str, float] = field(default_factory=dict)   # clear obs / month
    def for_sensor(self, key: str) -> float:
        return self.per_sensor.get(key, COVERAGE_SENSORS[key].default_desired_per_month)

@dataclass(frozen=True)
class CoverageQuery:
    layer: Layer
    metric: str                              # a data_var name, e.g. "cloud_free_scene_count"
    sensors: list[str]
    t0: dt.date
    t1: dt.date
    aggregation: Aggregation = "rate"
    bbox: tuple[float, float, float, float] | None = None    # None -> global
```

## Module API

```python
# catalog.py — Available layer (geocatalog)
from pathlib import Path
import geocatalog

def build_availability_catalog(
    sensor: CoverageSensor, start: str, end: str, *,
    bbox: tuple[float, float, float, float] | None = None,
    dest: Path, source_index: int = 0,
) -> Path | None:
    """Scan one of sensor.available_sources into a GeoParquet store. Picks
    available_sources[source_index] (0 = primary; bump to fail over to an
    alternate provider). Dispatches on src.kind:
        stac/cmr     -> geocatalog.from_stac_search(src.endpoint, ...)
        earthaccess  -> geocatalog.from_cmr / earthaccess source
    with backend='duckdb', out_path=dest,
    extra_properties=[src.cloud_field] if src.cloud_field else ().
    Returns the parquet path, or None if sensor.available_sources is empty
    (e.g. PRISMA) — caller then shows Available as 'n/a' for that sensor."""

def availability_stats(
    parquet: Path, grid: GridSpec, *, freq: str = "MS",
    cloud_lt: float = 10.0,
) -> xr.Dataset:
    """Open the GeoParquet as a DuckDBGeoCatalog and run the grid GROUP BY
    (DuckDB spatial join footprint->cell). Returns a Dataset with
    scenes_count, mean_scene_cloud_pct, cloud_free_scene_count on
    (sensor, time, lat, lon)."""
```

```python
# holdings.py — Acquired layer: a generic external PostGIS holdings table.
# Knows nothing project-specific: connection + table/column names come from a
# gitignored .env (see .env.example). Any private schema lives only in .env.
import geopandas as gpd

def fetch_holdings(
    *, bbox: tuple[float, float, float, float] | None = None, aoi=None,
    start: str, end: str,
) -> gpd.GeoDataFrame:
    """Query the holdings table -> GeoDataFrame[datetime, sensor, cloud, geometry]
    for rows intersecting the area within the range. Connection + the
    COVERAGE_TILES_* column mapping are read from environment (.env). Raises
    HoldingsUnavailable if unconfigured / driver missing / DB unreachable, so
    the caller can show the Acquired layer as 'off'."""

def holdings_stats(holdings: gpd.GeoDataFrame, start: str, end: str,
                   grid: GridSpec) -> xr.Dataset:
    """Grid the holdings -> held_count, held_clear_count on (time, lat, lon)."""

class HoldingsUnavailable(RuntimeError): ...
```

```python
# locations.py (optional) — target AOIs the gap is evaluated against.
# Coverage is often only interesting over a set of sites, not a uniform globe.
# This is a thin, source-agnostic loader (a GeoJSON/Parquet of AOIs, or any
# external table) — kept generic so no private site list lives in this repo.
import geopandas as gpd
from shapely.geometry.base import BaseGeometry

def fetch_targets(source: str | None = None) -> gpd.GeoDataFrame:
    """Load target AOIs -> GeoDataFrame[id, name, geometry]. `source` is a path
    or URI (defaults to a TARGETS env var); None disables target-clipping and
    the gap is computed over the whole queried area."""

def targets_union(targets: gpd.GeoDataFrame) -> BaseGeometry:
    """targets.geometry.union_all() — clip 'available'/'acquired' to the AOIs."""
```

```python
# coverage.py — assembly, gap, reader, slice
from pathlib import Path

def build_coverage_zarr(
    sensors: list[CoverageSensor], start: str, end: str, grid: GridSpec, *,
    dest: Path, parquet_dir: Path, include_holdings: bool = True,
) -> Path:
    """Run catalog.* for each sensor's Available bands + holdings.* for the
    Acquired bands, merge into one (sensor,time,lat,lon) Dataset, write Zarr."""

def compute_gap(ds: xr.Dataset, desired: DesiredCadence, *,
                weights: dict[str, float] | None = None) -> xr.Dataset:
    """Add deficit, unmet_supply, priority_score, taskable (vectorised over
    sensor/time/lat/lon). Pure xarray, cheap; call on read if not materialised."""

def open_coverage(zarr_path: Path) -> xr.Dataset:
    """xr.open_zarr(zarr_path) — the dashboards' single read entry point."""

def select(ds: xr.Dataset, q: CoverageQuery) -> xr.DataArray:
    """Slice sensors+time(+bbox), then reduce the time axis per q.aggregation:
       rate -> mean over months, total -> sum, recent -> isel(time=-1),
       worst -> min/max depending on metric, scrub -> returns the (time, y, x)
       cube for the caller to animate. Returns the 2-D (lat, lon) DataArray."""
```

```python
# tasking.py — request routing; tasking integration designed but inert
from typing import Literal
RequestKind = Literal["ingest", "task", "none"]

def request_kind(sensor: str, *, unmet_supply: int) -> RequestKind:
    """Which action closes the gap for this cell/location?
        unmet_supply > 0           -> 'ingest'  (scene exists & unowned: download it)
        else if sensor is taskable -> 'task'    (no scene exists: command an acquisition)
        else                       -> 'none'    (systematic sensor, nothing available)
    Landsat/Sentinel-2 are never 'task' (systematic survey); they are 'ingest'."""

def taskable_mask(ds: xr.Dataset, sensors: list[str]) -> xr.DataArray:
    """Per-sensor boolean broadcast from COVERAGE_SENSORS[*].taskable."""

def flag_for_request(geometry, window: tuple[dt.date, dt.date], sensor: str,
                     kind: RequestKind, *, store: Path) -> None:
    """Record the intent (ingest or task) to a local parquet/sqlite — the
    'ingest' backlog is the available-but-unheld set. No external call yet."""

def submit_request(provider: str, geometry, window) -> None:
    raise NotImplementedError(
        "Tasking-API integration is future work; see v4 doc §Tasking.")
```

## Request semantics: ingest vs tasking

Intersect the Available scenes with the Acquired holdings over the AOIs and
you get two sets: **held** (available *and* in our holdings = Acquired) and
**available-but-unheld** (exists over a target, we don't have it). That
unheld set is the actionable backlog. So "what to request" is **two
different actions**, and which one applies depends on the sensor:

| Situation (per cell/AOI) | Action | Applies to |
|------------------------------|--------|------------|
| Scene **exists** but we don't hold it (`unmet_supply > 0`) | **Ingest** — download + process | any sensor with an Available source — incl. Landsat & Sentinel-2 |
| Scene **doesn't exist** (`deficit > 0`, `unmet_supply == 0`) and sensor is pointable | **Task** — command a new acquisition | EMIT, EnMAP, PRISMA only |
| Scene doesn't exist and sensor is systematic | **none** — just wait for the next overpass | Landsat, Sentinel-2 |

So **Landsat and Sentinel-2 are `taskable=False` on purpose**: they're
systematic survey missions on fixed orbits that image everything
continuously — there is nothing to *task*. But they are fully
**ingest-requestable**: their gap is closed by downloading the scenes that
already exist. The pointable instruments (EMIT/EnMAP/PRISMA) are the only
ones where a *tasking* request makes sense — and only when no scene is
already available to ingest.

## Zarr schema (concrete)

```python
>>> ds = open_coverage("data/coverage.zarr"); ds
<xarray.Dataset>
Dimensions:                 (sensor: 3, time: 12, lat: 1800, lon: 3600)
Coordinates:
  * sensor                  ('emit', 'sentinel-2', 'landsat-8-9')
  * time                    datetime64[ns]  2025-01-01 ... 2025-12-01  (monthly)
  * lat                     float64  -89.95 ... 89.95
  * lon                     float64  -179.95 ... 179.95
Data variables:
    # Available (v2 scene-level; v1/v3 bands optional)
    scenes_count            (sensor, time, lat, lon) int16
    mean_scene_cloud_pct    (sensor, time, lat, lon) float32
    cloud_free_scene_count  (sensor, time, lat, lon) int16
    # Acquired (from the holdings DB; NaN where sensor has no held_satellite)
    held_count              (sensor, time, lat, lon) int16
    held_clear_count        (sensor, time, lat, lon) int16
    days_since_last_held    (sensor, time, lat, lon) float32
    # Gap (derived; materialised or computed on read)
    deficit                 (sensor, time, lat, lon) int16
    unmet_supply            (sensor, time, lat, lon) int16
    priority_score          (sensor, time, lat, lon) float32
Attributes:
    catalog_snapshot: "2026-05-29"
    grid_resolution: 0.1
```

## Demo 1 — build the coverage Zarr

```python
from satellite_climatology import grid, coverage
from satellite_climatology.sensors import COVERAGE_SENSORS

g = grid.GridSpec(resolution=0.1)
coverage.build_coverage_zarr(
    sensors=[COVERAGE_SENSORS[k] for k in ("emit", "sentinel-2")],
    start="2025-01-01", end="2025-12-31", grid=g,
    dest="data/coverage.zarr", parquet_dir="data/catalog",
    include_holdings=True,                 # pulls the holdings DB (.env creds)
)
```

What `build_coverage_zarr` does internally for the Available half (real
`geocatalog` API):

```python
import geocatalog
src = s.available_sources[0]                  # primary provider; [1:] are failovers
# scan -> GeoParquet (DuckDB backend, cloud preserved as a column)
geocatalog.from_stac_search(
    src.endpoint, collections=[src.collection_id],
    datetime=f"{start}/{end}", backend="duckdb",
    out_path=f"{parquet_dir}/{s.key}.parquet",
    extra_properties=[src.cloud_field] if src.cloud_field else (),
)
# (src.kind == "cmr"/"earthaccess" would dispatch to geocatalog's CMR source;
#  s.available_sources == () — e.g. PRISMA — means skip: no Available layer.)
# aggregate: open the same artifact lazily and group into grid cells
cat = geocatalog.open_catalog(f"{parquet_dir}/{s.key}.parquet", engine="duckdb")
# grid GROUP BY runs as DuckDB SQL over the parquet (spatial ext);
# emitted into the (time, lat, lon) arrays for this sensor.
```

And the Acquired half (generic PostGIS read, inside `holdings.fetch_holdings`
— connection + column mapping from a gitignored `.env`):

```python
# COVERAGE_DB_* + COVERAGE_TILES_* come from the env (.env); nothing
# project-specific is hard-coded here.
gdf = fetch_holdings(bbox=bbox, start=start, end=end)
# -> GeoDataFrame[datetime, sensor, cloud, geometry]; raises HoldingsUnavailable
#    if the DB isn't configured/reachable so the layer just shows as 'off'.
```

> The holdings table is an **external, private** dataset (kept in a separate
> repo/service); this project reads it generically over a standard
> PostgreSQL/PostGIS connection and never imports any private package.

## Demo 2 — App A: global heatmap (Streamlit)

```python
import streamlit as st, datetime as dt
from satellite_climatology import coverage
from satellite_climatology.coverage import CoverageQuery
from satellite_climatology.sensors import COVERAGE_SENSORS
from satellite_climatology import render

ds = coverage.open_coverage("data/coverage.zarr")

layer = st.sidebar.radio("Layer", ["available", "acquired", "gap"])
METRICS = {
    "available": ["scenes_count", "cloud_free_scene_count", "mean_scene_cloud_pct"],
    "acquired":  ["held_count", "held_clear_count", "days_since_last_held"],
    "gap":       ["deficit", "unmet_supply", "priority_score"],
}
metric = st.sidebar.selectbox("Metric", METRICS[layer])
sensors = st.sidebar.multiselect("Sensors", list(COVERAGE_SENSORS), ["emit"])
t0, t1 = st.sidebar.slider("Window", dt.date(2025, 1, 1), dt.date(2025, 12, 31),
                           (dt.date(2025, 1, 1), dt.date(2025, 6, 30)))
agg_default = {"days_since_last_held": "recent", "deficit": "worst",
               "priority_score": "worst"}.get(metric, "rate")
agg = st.sidebar.selectbox("Aggregation",
                           ["rate", "total", "recent", "worst", "scrub"],
                           index=["rate","total","recent","worst","scrub"].index(agg_default))

q = CoverageQuery(layer=layer, metric=metric, sensors=sensors,
                  t0=t0, t1=t1, aggregation=agg)
da = coverage.select(ds, q)                       # 2-D (lat, lon)

m = render.to_folium(da, basemaps=True)           # ImageOverlay + LayerControl
from streamlit_folium import st_folium
st_folium(m, height=560)
st.caption(f"{layer} · {metric} · {agg} · {', '.join(sensors)}")
```

`render.to_folium` colormaps the array to RGBA and adds an
`ImageOverlay(bounds=[[s,w],[n,e]])` over the basemap switcher we already
built; the Panel variant uses `da.hvplot.image(geo=True, rasterize=True)`
(geoviews + datashader) for the same result natively from xarray.

## Demo 3 — App B: AOI drill-down (available vs acquired vs gap)

```python
from shapely.geometry import box
from satellite_viewer import search                       # v0 -> Available (per-AOI)
from satellite_climatology.holdings import fetch_holdings  # Acquired
import pandas as pd

aoi = box(-104.0, 31.9, -103.96, 31.94)        # ~2 km Permian site
t0, t1 = dt.datetime(2025, 1, 1), dt.datetime(2025, 12, 31)

available = search("emit-l2a-rfl", aoi, t0, t1)            # GeoDataFrame
held = fetch_holdings(bbox=aoi.bounds, start=t0, end=t1, satellites=["EMIT"])

# monthly time series: available vs held vs gap, for this one AOI
avail_m = available.set_index("datetime").resample("MS").size()
held_m  = held.set_index("tile_date").resample("MS").size()
ts = pd.DataFrame({"available": avail_m, "held": held_m}).fillna(0)
ts["gap"] = (ts["available"] - ts["held"]).clip(lower=0)   # exists-but-unowned
```

Clicking a cell in App A pre-fills `aoi` here (the cell→AOI handoff). The
"available via STAC/CMR" call is `satellite_viewer.search` — the v0→v2.5
contract — so discovery logic isn't forked.

## Demo 4 — available ↔ held matchup with geocatalog

`geocatalog.intersect` gives the "what we own vs. what exists" join for
free, without hand-written SQL:

```python
import geocatalog
avail = geocatalog.open_catalog("data/catalog/emit.parquet", engine="duckdb")
held  = geocatalog.from_stac_items(...)            # or from a holdings parquet
owned = geocatalog.intersect(avail, held)          # rows present in both
# unmet_supply (per cell) = availability_stats(avail) - availability_stats(owned)
```

## CLI

```bash
# Build / refresh the product
python -m satellite_climatology build --config config/build.toml
# or ad-hoc:
python -m satellite_climatology build \
    --sensors emit,sentinel-2 --start 2025-01-01 --end 2025-12-31 \
    --grid 0.1 --out data/coverage.zarr

# Launch an app (uv, mirroring satellite_viewer)
streamlit run src/satellite_climatology/apps/global_heatmap_streamlit.py
panel serve  src/satellite_climatology/apps/global_heatmap_panel.py --show
```

## Rendering options (render.py)

| Stack | Approach | Notes |
|-------|----------|-------|
| **Streamlit / folium** | colormap `DataArray` → RGBA → `folium.raster_layers.ImageOverlay` | reuses the v0 basemap switcher; cheap; good to ~few M cells |
| **Panel / geoviews** | `da.hvplot.image(geo=True, rasterize=True, cmap=...)` | native xarray; datashader handles the 6.48 M cells |
| **either, large/zoomed** | write a COG, serve via `leafmap`/titiler | only if ImageOverlay gets sluggish |

Diverging cmaps for `deficit`/`unmet_supply`; sequential for counts;
reversed ramp for `days_since_last_held`.

## Test plan / acceptance

- **Unit**: `GridSpec.shape`/`lats`/`lons`; `footprint_to_cells` on a known
  polygon; `compute_gap` arithmetic on a tiny synthetic Dataset;
  `select` aggregations (rate/total/recent/worst) on a 3-month fixture.
- **holdings.py offline**: monkeypatch `fetch_holdings` to return
  fixtures → assert the GeoDataFrame schema; assert `HoldingsUnavailable`
  on a connection error (so the app degrades, not crashes).
- **Integration (live, marked `slow`)**: build a 1-sensor × 1-month ×
  continental Zarr; assert `held_count ≤ scenes_count` cell-wise where both
  exist; assert App A renders each layer for the default AOI.
- **Acceptance (M4.0)**: `build` produces a Zarr with the Available bands;
  App A renders a global `cloud_free_scene_count` heatmap, the time slider
  reslices it, and the AOI box clips it — for one sensor, one year, 0.1°.

## Grounded API references

Verified against the installed packages (so the demos above are real):

- `geocatalog.from_stac_search(client_or_url, *, collections, bbox, datetime, asset_key, backend, max_items, target_crs, out_path, extra_properties)`
- `geocatalog.from_stac_items(items, *, asset_key, backend, target_crs, out_path, extra_properties)`
- `geocatalog.open_catalog(source, *, backend, engine="auto"|"duckdb"|"memory", crs, storage_options)`
- `geocatalog.query(catalog, slice_=None, *, bounds, crs, time)` ; `intersect(left, right, *, spatial_only)` ; `union(left, right)`
- `geocatalog.to_geoparquet(catalog, path, *, partition_by=...)` ; `from_geoparquet(path, ...)`
- `geocatalog.GeoSlice(bounds=(xmin,ymin,xmax,ymax), interval=pd.Interval(t0,t1,closed="both"), resolution=(rx,ry), crs)`
- `DuckDBGeoCatalog.open/query/intersect/to_geoparquet/sql(where)/get_config`
- Acquired layer: a standard PostgreSQL/PostGIS read (SQLAlchemy + GeoPandas
  `read_postgis` with `ST_Intersects`), configured entirely by `.env`
  (`COVERAGE_DB_*` connection, `COVERAGE_TILES_*` column mapping). No private
  package is imported; the external holdings table lives in a separate repo.
