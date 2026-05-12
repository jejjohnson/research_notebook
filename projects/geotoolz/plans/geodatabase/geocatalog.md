---
title: GeoCatalog (Phase 1)
subject: geodatabase design
subtitle: In-memory GeoDataFrame with R-tree + IntervalIndex
short_title: Phase 1
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, geodatabase, catalog, geopandas
---

> **Scope:** evaluating the `remote_sensing/dataset*.py`, `sampler.py`, `operations.py`, and `rasterio_utils.py` files in [`jejjohnson/jej_vc_snippets`](https://github.com/jejjohnson/jej_vc_snippets) for promotion into [`jejjohnson/georeader`](https://github.com/jejjohnson/georeader).
> **Status:** design proposal, no code changed yet.

---

## 0. What I read

I worked from the public-API view (signatures + docstrings) plus the algorithmic core of every function.
The six files in `jej_vc_snippets/remote_sensing/` that are dataset-builder material are:

| File | Lines | Top-level surface |
| --- | ---: | --- |
| `dataset.py` | 1835 | `build_spatiotemporal_index`, `load_and_merge_rasters_spatial`, `load_and_merge_spatiotemporal` |
| `dataset_xarray.py` | 1758 | `build_xarray_spatiotemporal_index`, `load_xarray_datasets`, `merge_xarray_datasets`, `load_xarray_datasets_spatiotemporal`, `merge_xarray_datasets_spatiotemporal` |
| `dataset_vector.py` | 1888 | `build_vector_spatiotemporal_index`, `load_and_rasterize_vectors_spatial`, `load_and_rasterize_vectors_spatiotemporal` (+ `_convert_poly_coords`) |
| `sampler.py` | 1378 | `GeoSlice` dataclass + `random_geo_sampler`, `grid_geo_sampler`, `run_inference_with_grid_sampler`, `stitch_predictions`, helpers |
| `operations.py` | ~1000 | `query_spatiotemporal_index`, `intersect_spatiotemporal_indexes`, `union_spatiotemporal_indexes` |
| `rasterio_utils.py` | ~250 | `update_metadata`, `read_metadata`, `filter_by_metadata`, `get_tags`, `print_all_metadata`, `save_image_with_tags`, `append_tags_to_existing` |

> ⚠️ Found two literal syntax errors in `dataset_vector.py` (`return_meta bool = True` missing colon at line ~416; `if return_meta` missing colon at line ~1076).
> The file has never been imported successfully.
> Tests are therefore sparse to non-existent.

---

## 1. Inventory: what these snippets actually are

The six files form a small **catalog → query → load → sample → stitch** stack, with three storage backends (raster / xarray / vector) sharing one index format.

### 1.1 The shared index format

Every `build_*_spatiotemporal_index` function emits the *same* object: a `geopandas.GeoDataFrame` whose row label is a `pd.IntervalIndex(name='datetime', closed='both')` and whose `geometry` column holds reprojected file footprints (bounding-box `Polygon`s in a uniform target CRS).
One row per file.
Extra columns differ by backend (`filepath` always; for xarray, also `data_vars`, `time_var`, `time_resolution`, `n_timesteps`; for vector, `layer`).

This is the central design choice and it's a good one: **pandas + geopandas already give you `O(log n)` interval-tree temporal lookup and `O(log n)` R-tree spatial lookup for free.** No custom data structures needed.

### 1.2 The three backends

- **Raster (`dataset.py`)** — file footprints come from `rasterio.open(...).bounds`, reprojected to `target_crs` via a lazy `WarpedVRT`.
  Loaders use `rasterio.merge.merge()` driven by query bounds and a target resolution; output is a numpy array `(bands, height, width)` plus a validity mask, plus an `Affine`.
- **Xarray (`dataset_xarray.py`)** — opens each file with `xr.open_dataset(...)` (engine inferred from extension: `.zarr` → zarr, else netcdf4/h5netcdf), resolves CRS via `rio.crs` or a fallback, derives bounds from coordinate min/max, parses time from a `time_var` (default `'time'`).
  Loaders return `xr.Dataset` (or per-var numpy arrays); merging uses `rioxarray.merge.merge_datasets`.
- **Vector (`dataset_vector.py`)** — opens each file with `geopandas.read_file(...)`, reprojects to target CRS, footprint = `total_bounds`.
  Loaders rasterize with `rasterio.features.rasterize` and return `torch.long` tensors keyed by ML task (`semantic_segmentation`, `object_detection`, `instance_segmentation`).

### 1.3 The set algebra

`operations.py` adds:

- `query_spatiotemporal_index` — spatial + temporal filter on one catalog.
- `intersect_spatiotemporal_indexes` — cross-catalog AND.
- `union_spatiotemporal_indexes` — cross-catalog OR (a glorified `pd.concat`).

All operate at the index level, never opening a file.

### 1.4 The sampler / inference layer

`sampler.py` is the ML-glue: a `GeoSlice` dataclass + `random_geo_sampler` / `grid_geo_sampler` / `stitch_predictions` factories.
The catalog is what the samplers iterate; the slices flow downstream into loaders and operators.
Detailed design — dataclass invariants, sampling math, stitch reductions — lives in [`types/geoslice.md`](../types/geoslice.md).
This document only covers how the catalog feeds the samplers.

### 1.5 Auxiliary

`rasterio_utils.py` is *not* a dataset builder — it's GDAL tag read/write helpers.
It does not belong in this migration; fold the read helpers into georeader's existing `rasterio_reader.py` and drop the rest.

---

## 2. User story

**Persona.** A scientist or ML engineer with a folder (or bucket) of N satellite or model-output files spanning years and tiles.
Possibly heterogeneous: S2 mixed with Landsat, NetCDF reanalysis mixed with ad-hoc Zarr stores, raster imagery alongside vector labels.

**Goal arc:**

1. *"I have 10 000 files.
   Make them queryable."* → `build_*_index(filepaths, regex, target_crs)` → one GeoDataFrame.
2. *"Give me the data for this bbox + this date range."* → `query_index(...)` → filtered subset → `load_and_merge_*(...)` → mosaicked numpy/xarray.
3. *"Stream me ML training chips."* → `random_geo_sampler(index, chip_size).__call__()` → iterator of `GeoSlice` → call loader per slice → batch.
4. *"Run my model over the whole AOI for the whole year and write a georeferenced raster."* → `run_inference_with_grid_sampler(index, model, output_path, chip_size, stride)` → stitched output.
5. *"My imagery is in catalog A, my labels are in catalog B; pair them."* → `intersect_spatiotemporal_indexes(A, B)` → joint catalog, queries return only spatiotemporally-paired tiles.
6. *"Combine Landsat 7 + Landsat 8 into one virtual dataset."* → `union_spatiotemporal_indexes(L7, L8)`.

The stack is therefore not "yet another dataloader" — it's a **lightweight, file-based, Pythonic STAC**.
No JSON catalog format, no service, no pgSTAC. Just a GeoDataFrame you can pickle.

---

## 3. Motivation

### 3.1 The gap this fills in georeader

`georeader` today is excellent at the *single-file* level: open a Sentinel-2 SAFE, read a bbox, get a `GeoTensor`, reproject, save COG. But it has **no story for collections of files**.
If the user has ten years of daily files across many tiles, they're on their own to write the catalog, the query, the mosaic, and the temporal stack.
These snippets are exactly that missing layer.

The georeader modules these would build on are:

| Snippet capability | Existing georeader module to extend / call into |
| --- | --- |
| Per-file footprint + bounds extraction | `rasterio_reader.py`, `abstract_reader.py` |
| Spatial windowing into queried bounds | `window_utils.py`, `read.read_from_bounds` |
| Multi-file spatial mosaic | `mosaic.py` |
| Vector → raster | `rasterize.py` |
| Output container | `geotensor.GeoTensor` |
| Per-band reflectance, etc. (out of scope) | `reflectance.py` |

### 3.2 Compared with alternatives

- **TorchGeo `RasterDataset`** has the same idea (regex-parsed filename, R-tree, `BoundingBox` query).
  The snippets are a **lighter, GeoDataFrame-native reimagining** of that pattern, without the torch dependency in the index layer.
  That's a feature, not a bug — it lets non-ML users (analysis, mosaicking, inference) use the same catalog.
- **STAC / pystac-client** is heavier: requires a JSON spec, an API, network fetches.
  The snippets work on local files and return a Python object you can ship around.
- **xbatcher / xarray-spatial / odc-geo** overlap with the *xarray* loader but not with the catalog/intersection logic.

The concrete value-add over TorchGeo is therefore:

1. GeoDataFrame instead of a custom R-tree wrapper.
2. Backend-agnostic same-shape index for raster / xarray / vector.
3. Explicit set algebra (intersect / union) at the catalog level.
4. No torch in the core path.

---

## 3.3 Primer for newcomers

> **ELI5.** An R-tree is a **russian-doll of bounding boxes** — each big box knows what smaller boxes it contains.
> To find files in your area, you only open the boxes that overlap your area, never the rest.
> Combined with a similar trick for time, queries answer in milliseconds even over thousands of files.

### `gpd.GeoDataFrame` (geopandas)

**What it is.** A `pandas.DataFrame` subclass with a `geometry` column holding Shapely geometries (Polygon, MultiPolygon, Point, LineString).
Each row is a feature; rows behave like dataframe rows; the geometry column gets spatial-aware operations.

**How it works.** `gpd.GeoDataFrame({"path": [...], "date": [...], "geometry": [Polygon(...), ...]})` — same constructor pattern as a regular DataFrame, but the `geometry` column is special-cased.
Geopandas wraps Shapely (Python bindings to GEOS, the C library) plus pyproj (CRS handling) plus rtree (spatial indexing) plus fiona (file I/O).
All the heavy lifting is in C; the Python layer is convenient bookkeeping.

**What this means for us.** Phase 1's catalog is *literally* a GeoDataFrame plus an `IntervalIndex` for time.
Builders read each file's bounds via `WarpedVRT` (lazy reprojection that doesn't read pixels), assemble those into geometries, and stick the result in a gdf.
Queries leverage geopandas's existing R-tree + Shapely operations.
No new spatial-data-structure code.

### R-tree + IntervalIndex (the indices)

**What it is.** Two indices stacked: an **R-tree** for "find rows whose `geometry` overlaps this bbox" and an **IntervalIndex** for "find rows whose `interval` overlaps this date range." Combined, they answer spatiotemporal queries in `O(log n + k)`.

**How it works.** Geopandas builds the R-tree on first access via the `gdf.sindex` property — backed by libspatialindex's R-tree implementation in C. Pandas builds the IntervalIndex when you attach intervals via `gdf = gdf.set_index(pd.IntervalIndex.from_arrays(start, end, closed='both'))`.
A combined query is `gdf[gdf.index.overlaps(query_interval)].cx[xmin:xmax, ymin:ymax]` — first the time filter, then the spatial filter.

**What this means for us.** A catalog of 10k tile×date rows answers "files overlapping this AOI in June 2023" in sub-millisecond.
The index is built lazily (first query pays the construction cost), so importing a Phase 1 catalog is fast even if you never query it.
Beyond ~10⁵ rows the gdf's row-iteration overhead dominates and you should switch to Phase 2 (DuckDB).

```{mermaid}
flowchart TD
    Root[Root bbox: world]
    Root --> N1[Node A: Europe]
    Root --> N2[Node B: Africa]
    Root --> N3[Node C: Americas]
    N1 --> L1[file 1.tif]
    N1 --> L2[file 2.tif]
    N2 --> L3[file 3.tif]
    N2 --> L4[file 4.tif]
    N3 --> L5[file 5.tif]

    Q[Query bbox in Europe] -.->|overlaps| N1
    Q -.->|skipped| N2
    Q -.->|skipped| N3
```

### Set algebra over catalogs

**What it is.** Catalogs support `query`, `intersect`, and `union` operations that produce new catalogs — like set operations on indexed file collections.
Lets you compose "data I have" with "data I want" without writing custom join code per workflow.

**How it works.** `intersect(catalog_A, catalog_B)` walks pairs of geometries (using the R-tree to skip non-intersecting pairs), computes per-pair `(geom_A ∩ geom_B, max(start_A, start_B), min(end_A, end_B))`, drops empty intersections.
`union` is the opposite — concatenate, then optionally dedupe.
`query` is "intersect with a single-row catalog" — the AOI is a one-row catalog with one geometry and one interval.

**What this means for us.** Pairing imagery with labels (raster × vector across two backends), building cross-sensor catalogs (S2 + Landsat for change detection), filtering to "files that have all three of (S2, EnMAP, ERA5)" — all expressible as catalog set algebra, not bespoke pandas joins each time.
Same shape carries to Phase 2 where it's SQL `INTERSECT` / `UNION` under the hood.

```{mermaid}
sequenceDiagram
    participant User
    participant A as catalog_A (S2)
    participant B as catalog_B (labels)
    participant Out as result

    User->>A: intersect(B)
    A->>A: gpd.overlay(A, B, intersection)<br/>spatial filter via R-tree
    A->>A: temporal: max(starts), min(ends)
    A->>A: drop empty intersections
    A-->>Out: paired catalog
    Note over Out: rows where (geom_A ∩ geom_B)<br/>and time intervals overlap
```

---

## 4. Mathematics

### 4.1 Footprint construction

For each file, the spatial footprint is the rectangle of its bounds in the target CRS:

```python
with rasterio.open(filepath) as src:
    with WarpedVRT(src, crs=target_crs) as vrt:
        xmin, ymin, xmax, ymax = vrt.bounds
        polygon = shapely.geometry.box(xmin, ymin, xmax, ymax)
```

`WarpedVRT` is a *lazy* virtual reprojection: bounds are computed analytically from the source's affine + CRS without resampling pixels.
This is what makes indexing 10 000 files fast.

### 4.2 Temporal interval construction

For a single-date capture filename, the file is treated as covering the full UTC day:

```python
mint = pd.Timestamp(date_str).floor('D')          # 00:00:00.000000
maxt = pd.Timestamp(date_str).ceil('D') - 1e-6    # 23:59:59.999999
```

For a `(start, stop)` filename, the interval is `[floor(start), ceil(stop) - 1µs]`.
For files with no date, `(pd.Timestamp.min, pd.Timestamp.max)` — a point of contention; see [§7](#7-sharp-edges-to-fix-on-the-way-in).

### 4.3 Combined query

Spatial + temporal lookup is the composition of two index probes:

```python
query_interval = pd.Interval(tmin, tmax, closed='both')
filtered = index[index.index.overlaps(query_interval)]   # interval tree, O(log n + k)
filtered = filtered.cx[xmin:xmax, ymin:ymax]              # R-tree, O(log n + k)
```

`IntervalIndex.overlaps` uses scipy / sortedcontainers under the hood; `GeoDataFrame.cx` uses the rtree spatial index that geopandas builds on first access.

### 4.4 Index intersection (`operations.py`)

For two indexes A and B sharing a CRS, the spatiotemporal intersection is computed in two stages:

**Spatial.** `gpd.overlay(A, B, how='intersection', keep_geom_type=True)` — for each pair of polygons `(a, b)`, produces the polygon `a ∩ b` (or drops the pair if empty).

**Temporal.** For each surviving pair, the temporal intersection of intervals `[a₁, b₁]` and `[a₂, b₂]` is

$$
[\max(a_1, a_2),\ \min(b_1, b_2)]
$$

discarded if `min(b₁, b₂) < max(a₁, a₂)`.
Vectorised:

```python
mint  = np.maximum(datetime_1.left,  datetime_2.left)
maxt  = np.minimum(datetime_1.right, datetime_2.right)
valid = maxt >= mint
```

The result's `IntervalIndex` is built from those two arrays with `closed='both'`.

### 4.5 Sampler and stitch math

Random-sampler weighting (area-weighted tile selection + uniform chip placement), grid-sampler stride math, and the four stitch-reduction modes (`average` / `max` / `first` / `last`) are specified in [`types/geoslice.md`](../types/geoslice.md).
Those primitives consume what this catalog produces but aren't catalog-specific — see that document for invariants and edge cases.

---

## 5. Coupling with georeader

| Snippet | What in georeader it duplicates / extends |
| --- | --- |
| `build_spatiotemporal_index` | New capability — closest analog is nothing in georeader today. The per-file `WarpedVRT` bounds extraction is *not* in georeader; could leverage `rasterio_reader.RasterioReader` instead of opening raw rasterio. |
| `load_and_merge_rasters_spatial` | Heavy overlap with `mosaic.py` — both do windowed reads + `rasterio.merge`. The snippet adds the index-driven query layer on top. |
| `load_xarray_datasets*` | New territory; georeader doesn't really do xarray reads end-to-end. `dataarray.py` exists as a thin GeoTensor↔DataArray bridge; this would extend that. |
| `load_and_rasterize_vectors*` | Overlaps `rasterize.py`. The snippet adds index-driven file selection and per-task label conventions. |
| `sampler.py` / `GeoSlice` | Promoted to its own design — see [`types/geoslice.md`](../types/geoslice.md), which reconciles `GeoSlice` with `rasterio.windows.Window` and `slices.py`. |
| `run_inference_with_grid_sampler` | Cut. The model loop belongs at the operator layer ([`geotoolz.inference.ApplyToChips`](../geotoolz/geotoolz.md)), not in georeader. |
| `stitch_predictions` | Specified in [`types/geoslice.md`](../types/geoslice.md). |
| `query` / `intersect` / `union` | New. Set algebra over a GeoDataFrame catalog — clean fit as a new module. |
| `rasterio_utils.py` | Keep only `get_tags` / `print_all_metadata` and fold into existing readers. The xarray-tagging half should not live here; it's ad-hoc rioxarray plumbing. |

---

## 6. Proposed API surface in georeader

A clean, opinionated public API. Three goals:

1. One shape across backends.
2. `GeoTensor` / `xr.Dataset` on output — never `dict[str, np.ndarray]`.
3. ML deps stay optional.

### 6.1 Module layout

```text
georeader/
├── catalog/                    # NEW
│   ├── __init__.py
│   ├── base.py                 # GeoCatalog (wraps the GeoDataFrame index)
│   ├── raster.py               # build_raster_catalog + raster loaders
│   ├── xarray.py               # build_xarray_catalog + xarray loaders   [extra: xarray]
│   ├── vector.py               # build_vector_catalog + rasterizers
│   └── ops.py                  # query, intersect, union
├── samplers/                   # NEW — see types/geoslice.md design
│   ├── __init__.py
│   ├── geoslice.py             # GeoSlice dataclass
│   ├── random.py               # random_sampler
│   ├── grid.py                 # grid_sampler
│   └── stitch.py               # stitch
└── ...                         # existing modules unchanged
```

### 6.2 Core types

`GeoSlice` is specified in [`types/geoslice.md`](../types/geoslice.md).
The catalog imports and re-exports it for convenience.

```python
class GeoCatalog:
    """Thin wrapper around a GeoDataFrame with IntervalIndex + geometry."""
    gdf: gpd.GeoDataFrame
    backend: Literal["raster", "xarray", "vector"]

    # set algebra
    def query(self, slice: GeoSlice, *, t_step: int | None = None) -> "GeoCatalog": ...
    def intersect(self, other: "GeoCatalog", *, spatial_only: bool = False) -> "GeoCatalog": ...
    def union(self, other: "GeoCatalog") -> "GeoCatalog": ...

    # introspection
    @property
    def total_bounds(self) -> tuple[float, float, float, float]: ...
    @property
    def temporal_extent(self) -> pd.Interval: ...
    def __len__(self) -> int: ...
```

### 6.3 Builders (one per backend)

```python
def build_raster_catalog(
    filepaths: Sequence[str | Path],
    *,
    filename_regex: str,
    date_format: str = "%Y%m%d",
    target_crs: CRS | str | None = None,        # default: first file's CRS
    target_resolution: tuple[float, float] | None = None,
) -> GeoCatalog: ...


def build_xarray_catalog(
    filepaths: Sequence[str | Path],
    *,
    target_crs: CRS | str | None = None,
    target_resolution: tuple[float, float] | float | None = None,
    data_vars: Sequence[str] | None = None,
    time_var: str = "time",
) -> GeoCatalog: ...


def build_vector_catalog(
    filepaths: Sequence[str | Path],
    *,
    filename_regex: str,
    date_format: str = "%Y%m%d",
    target_crs: CRS | str | None = None,
    layer: str | int | None = None,
) -> GeoCatalog: ...
```

### 6.4 Loaders return `GeoTensor` / `xr.Dataset`, not dicts

```python
def load_raster(
    catalog: GeoCatalog,
    slice: GeoSlice,
    *,
    band_indexes: Sequence[int] | None = None,
    resampling: Resampling = Resampling.bilinear,
    merge_method: Literal["first", "last", "min", "max", "sum", "count"] = "last",
) -> GeoTensor: ...                          # (bands, h, w) + transform + crs


def load_raster_timeseries(
    catalog: GeoCatalog,
    slice: GeoSlice,
    *,
    t_step: int = 1,
    temporal_aggregation: Literal["nearest", "mean", "median", "first", "last"] = "nearest",
    ...,
) -> GeoTensor: ...                          # (time, bands, h, w)


def load_xarray(catalog: GeoCatalog, slice: GeoSlice, ...) -> xr.Dataset: ...


def load_vector(
    catalog: GeoCatalog,
    slice: GeoSlice,
    *,
    task: Literal["semantic_segmentation", "object_detection", "instance_segmentation"],
    label_field: str | None = None,
) -> GeoTensor: ...
```

The shift from `dict[str, np.ndarray]` to `GeoTensor` is the single biggest cleanup — it carries CRS + transform around so users don't lose georeferencing.

### 6.5 Samplers

`random_sampler`, `grid_sampler`, and `stitch` are specified in [`types/geoslice.md`](../types/geoslice.md) — including signatures, area-weighting math, stride math, and the four stitch reduction modes.
From the catalog's perspective the only API touchpoint is that all three accept a `GeoCatalog` and consume `iter_rows()` / `query()` to find tiles.

### 6.6 What I'd cut

- `run_inference_with_grid_sampler` — too opinionated.
  Replace with a 5-line cookbook example.
  Library shouldn't own the model loop.
- The `dict[str, np.ndarray]` return contract.
- Hardcoded `torch.long` in the vector loaders; make torch optional, default to numpy with `dtype=np.int64`.
- `loguru` as a hard dependency — switch to stdlib `logging`.
- `_convert_poly_coords` private helper — `rasterio.features.rasterize` already accepts `transform=`; this helper is redundant.

---

## 7. Sharp edges to fix on the way in

1. **Two literal syntax errors in `dataset_vector.py`** — the file has never run.
   Tells you tests are sparse.
2. **`pd.Timestamp.min` / `Timestamp.max` as fallback intervals** for date-less files will dominate `IntervalIndex` ranges and make logs misleading.
   Either require a date or carry an explicit `static` flag.
3. **Bounds in unprojected lat/lon get distorted by `WarpedVRT`** — for files spanning the antimeridian or poles, `.bounds` is the *reprojected* envelope, not a great-circle hull.
   Document this; consider exposing a `densify=N` option that samples the boundary before reprojection.
4. **`t_sample = interval.left.value / 1e9`** uses `Timestamp.value` (nanoseconds).
   Fine, but it'll silently break for `Timestamp.min` / `.max` (overflow on the float).
   Skip random temporal sampling when the interval is "infinite".
5. **`gpd.overlay` is `O(n × m)`** worst-case despite the R-tree.
   For `10k × 10k` catalogs this is slow; document the cost or add a chunked variant.
6. **Hardcoded `nodata=0`** in the xarray merge call.
   Should respect the source's `_FillValue` / `nodata`.
7. **`merge_method='count'`** is listed in the raster loader signature but isn't a valid `rasterio.merge` method.
   Either implement or remove.
8. **`use_cache=True` + `lru_cache`** on file handles in long-running processes — at fork time these break under multiprocessing dataloaders.
   Document or disable in worker contexts.
9. **No tests, no examples** that exercise multi-CRS catalogs (e.g. mixing UTM zones).
   The most likely real-world failure mode.
10. **`loguru` everywhere** — gives nice-looking logs but is a hard dep.
    Bury behind stdlib `logging` so georeader users aren't forced into it.

---

## 8. End-to-end examples

A varied gallery, organized by intent.
Each example uses only the proposed API; small extensions beyond §6 are flagged inline as `# extension:` comments.
Imports are kept minimal per snippet — assume `numpy as np`, `pandas as pd` everywhere.

| § | Example | Pattern |
| --- | --- | --- |
| 8.1 | A. Cloud-free monthly composite | mask + temporal-last reduction |
| 8.1 | B. Daily NDVI series at one point | xarray, 1-pixel slice over long TOI |
| 8.2 | C. Atmospheric correction ETL → Zarr | tiled load + per-chip transform + write |
| 8.2 | D. Hyperspectral → S2-like band binning | per-chip spectral matrix product |
| 8.2 | E. Uncertainty maps → vector anomalies | stitched postprocess + raster→vector |
| 8.3 | F. Change detection across two epochs | siamese on `intersect(spatial_only)` |
| 8.3 | G. Cloud detection write-back | per-tile inference, aligned QA output |
| 8.4 | H. Image-label pairing (raster × vector) | cross-backend `intersect` |
| 8.4 | I. SAR + optical fusion | multi-sensor pairing with interval padding |
| 8.4 | J. Multi-resolution S2 + MODIS | one slice, two native resolutions |
| 8.4 | K. Optical + DEM (static) + ERA5 (coarse) | three backends, mixed CRS |
| 8.5 | L. Random chips for a torch `DataLoader` | basic ML adapter |
| 8.5 | M. Self-supervised temporal-pair sampler | SimCLR-style positives |
| 8.5 | N. Spatial-block cross-validation | leakage-safe folds via centroids |
| 8.5 | O. Foundation-model embedding store | grid sampler → frozen encoder → Zarr |

### 8.1 Simple pipelines

#### A. Cloud-free monthly composite over an AOI

Query a month + AOI, mask cloud bits from QA60, take the most recent valid pixel per band.

```python
from glob import glob
from georeader.catalog import build_raster_catalog, GeoSlice, load_raster
from georeader.save import save_cog

catalog = build_raster_catalog(
    filepaths=glob("/data/s2/T29SND_*.tif"),
    filename_regex=r"T29SND_(?P<date>\d{8})_.*\.tif",
    target_crs="EPSG:32629",
    target_resolution=(10.0, 10.0),
)

aoi = GeoSlice(
    bounds=(500_000, 4_000_000, 540_000, 4_040_000),
    interval=pd.Interval(pd.Timestamp("2023-06-01"), pd.Timestamp("2023-06-30"), closed="both"),
    resolution=(10.0, 10.0),
    crs="EPSG:32629",
)

rgbn = load_raster(catalog, aoi, band_indexes=[2, 3, 4, 8], merge_method="last")
qa   = load_raster(catalog, aoi, band_indexes=["QA60"],     merge_method="last")
cloud = (qa.values.astype(np.uint16) & ((1 << 10) | (1 << 11))) > 0   # opaque + cirrus
rgbn.values[:, cloud[0]] = 0
save_cog(rgbn, "/out/s2_2023-06_composite.tif", descriptions=["B02", "B03", "B04", "B08"])
```

#### B. Daily NDVI series at one point (xarray archive)

A 1-pixel `GeoSlice` plus a long TOI degenerates to a 1-D series.

```python
from glob import glob
from georeader.catalog import build_xarray_catalog, GeoSlice, load_xarray

catalog = build_xarray_catalog(
    filepaths=glob("/data/modis/MOD13A2_*.nc"),
    target_crs="EPSG:4326",
    data_vars=["NDVI"],
    time_var="time",
)

madrid = GeoSlice(
    bounds=(-3.7038, 40.4168, -3.7028, 40.4178),
    interval=pd.Interval(pd.Timestamp("2010-01-01"), pd.Timestamp("2023-12-31"), closed="both"),
    resolution=(0.005, 0.005),
    crs="EPSG:4326",
)

ds     = load_xarray(catalog, madrid)
series = ds["NDVI"].mean(("x", "y")).to_pandas()
series.to_csv("/out/madrid_ndvi.csv")
```

### 8.2 Heavy preprocessing & postprocessing

#### C. Atmospheric correction → cloud mask → monthly composite → Zarr

A full ETL: tiled load of a multi-year archive, georeader's reflectance helpers, monthly aggregation, sharded Zarr output.

```python
import xarray as xr
from georeader.catalog import build_raster_catalog, load_raster
from georeader.samplers import grid_sampler
from georeader import reflectance

catalog = build_raster_catalog(...)
chips = list(grid_sampler(catalog, chip_size=(2048, 2048), stride=(2048, 2048)))

monthly: dict[pd.Period, list] = {}
for sl in chips:
    toa = load_raster(catalog, sl, band_indexes=list(range(1, 14)))
    boa = reflectance.toa_to_boa(toa, ...)                              # GeoTensor
    qa  = load_raster(catalog, sl, band_indexes=["QA60"])
    bad = (qa.values & ((1 << 10) | (1 << 11))) > 0
    boa.values[:, bad[0]] = np.nan
    monthly.setdefault(sl.interval.left.to_period("M"), []).append(boa)

for month, parts in monthly.items():
    da = xr.concat([p.to_xarray() for p in parts], dim="chip").mean("chip", skipna=True)
    da.to_zarr(f"/out/s2_{month}.zarr", mode="w", consolidated=True)
```

#### D. Hyperspectral → S2-like band binning (per chip)

Apply Sentinel-2 spectral response functions (SRFs) on each EnMAP / PRISMA cube to produce a 13-band multispectral output that drops straight into models trained for S2.

```python
from georeader.catalog import build_raster_catalog, load_raster
from georeader.geotensor import GeoTensor
from georeader.samplers import grid_sampler

srf = np.load("/data/s2a_srf.npy")                  # (13 s2 bands, 224 enmap bands)

enmap = build_raster_catalog(
    enmap_files,
    filename_regex=r"ENMAP_L2A_(?P<date>\d{8}).*\.tif",
    target_crs="EPSG:32629",
)

for sl in grid_sampler(enmap, chip_size=(512, 512)):
    cube  = load_raster(enmap, sl)                          # GeoTensor (224, 512, 512)
    multi = np.einsum("ij,jhw->ihw", srf, cube.values)      # (13, 512, 512)
    out   = GeoTensor(values=multi, transform=cube.transform, crs=cube.crs)
    out.save(f"/out/enmap_as_s2/{sl.interval.left.date()}_{sl.bounds}.tif")
```

#### E. Postprocess: stitched uncertainty maps → vector anomalies

Run a model with an uncertainty head, stitch, threshold on `μ` *and* `σ`, vectorise.

```python
from georeader.catalog import build_raster_catalog, load_raster
from georeader.samplers import grid_sampler, stitch
from georeader.vectorize import polygons_from_mask

catalog = build_raster_catalog(...)
slices  = list(grid_sampler(catalog, chip_size=(256, 256), stride=(192, 192)))

mu_chips, sigma_chips = [], []
for sl in slices:
    x = load_raster(catalog, sl, band_indexes=[1, 2, 3, 4]).values
    mu, sigma = model_with_uncertainty(x)
    mu_chips.append(mu); sigma_chips.append(sigma)

mu    = stitch(mu_chips,    slices, method="average")
sigma = stitch(sigma_chips, slices, method="average")
flag  = (mu.values[0] > 0.7) & (sigma.values[0] > 0.2)

gdf = polygons_from_mask(flag, transform=mu.transform, crs=mu.crs)
gdf.to_file("/out/anomalies.geojson", driver="GeoJSON")
```

### 8.3 Inference patterns

#### F. Change detection across two epochs (siamese on intersected catalogs)

`intersect(spatial_only=True)` keeps only tiles whose footprints overlap; you then evaluate each chip's "before" and "after" load with explicit time windows.

```python
import dataclasses
from georeader.catalog import build_raster_catalog, load_raster
from georeader.samplers import grid_sampler, stitch

before = build_raster_catalog(pre_event_files,  filename_regex=PRE_REGEX)
after  = build_raster_catalog(post_event_files, filename_regex=POST_REGEX)
shared = before.intersect(after, spatial_only=True)

chips = list(grid_sampler(shared, chip_size=(256, 256), stride=(256, 256)))
deltas = []
for sl in chips:
    sl_b = dataclasses.replace(sl, interval=pd.Interval(pd.Timestamp("2023-08-15"),
                                                         pd.Timestamp("2023-08-25"), closed="both"))
    sl_a = dataclasses.replace(sl, interval=pd.Interval(pd.Timestamp("2023-09-15"),
                                                         pd.Timestamp("2023-09-25"), closed="both"))
    deltas.append(siamese_change_model(load_raster(before, sl_b).values,
                                       load_raster(after,  sl_a).values))

stitch(deltas, chips, method="max").save("/out/change.tif")
```

#### G. Cloud detection: write aligned QA bands per source tile

For each tile in the catalog, predict a cloud mask and write a sidecar `.cloud.tif` with the *same* affine and shape as the input.

```python
from pathlib import Path
from georeader.catalog import build_raster_catalog, GeoSlice, load_raster
from georeader.geotensor import GeoTensor

catalog = build_raster_catalog(s2_files, filename_regex=S2_REGEX)

for row in catalog.gdf.itertuples():
    sl = GeoSlice(
        bounds=row.geometry.bounds,
        interval=row.Index,
        resolution=(10.0, 10.0),
        crs=catalog.gdf.crs,
    )
    rgbn  = load_raster(catalog, sl, band_indexes=[2, 3, 4, 8])
    cloud = cloud_model(rgbn.values)                                       # (1, H, W) prob
    out   = GeoTensor(values=cloud, transform=rgbn.transform, crs=rgbn.crs,
                      fill_value_default=255)
    out.save(Path(row.filepath).with_suffix(".cloud.tif"))                 # COG aligned to input
```

### 8.4 Heterogeneous / multi-sensor pairing

#### H. Image-label pairing across two backends (raster × vector)

```python
from georeader.catalog import build_raster_catalog, build_vector_catalog, load_raster, load_vector
from georeader.samplers import random_sampler

imagery = build_raster_catalog(image_files, filename_regex=IMG_REGEX)
labels  = build_vector_catalog(label_files, filename_regex=LBL_REGEX)
paired  = imagery.intersect(labels)               # spatially AND temporally aligned

for sl in random_sampler(paired, chip_size=(512, 512)):
    x = load_raster(imagery, sl, band_indexes=[1, 2, 3])
    y = load_vector(labels,  sl, task="semantic_segmentation", label_field="class_id")
    train_step(x.values, y.values)
```

#### I. SAR + optical fusion (Sentinel-1 GRD + Sentinel-2 L2A)

Different acquisition cadences.
Pad each S1 row's interval by ±3 days so `intersect` allows loose temporal pairing, then fuse on load.

```python
from georeader.catalog import build_raster_catalog
from georeader.samplers import random_sampler

opt = build_raster_catalog(s2_files, filename_regex=S2_REGEX, target_crs="EPSG:32629")
sar = build_raster_catalog(s1_files, filename_regex=S1_REGEX, target_crs="EPSG:32629")

# extension: pad each row's interval by ±3 days
sar.gdf.index = pd.IntervalIndex.from_arrays(
    sar.gdf.index.left  - pd.Timedelta(days=3),
    sar.gdf.index.right + pd.Timedelta(days=3),
    closed="both", name="datetime",
)

paired = opt.intersect(sar)

for sl in random_sampler(paired, chip_size=(256, 256), length=20_000, seed=0):
    rgbn = load_raster(opt, sl, band_indexes=[2, 3, 4, 8]).values        # (4, 256, 256)
    vvvh = load_raster(sar, sl, band_indexes=[1, 2]).values              # (2, 256, 256) linear
    sar_db = 10 * np.log10(np.clip(vvvh, 1e-6, None))
    yield np.concatenate([rgbn, sar_db], axis=0)                          # (6, 256, 256)
```

#### J. Multi-resolution: Sentinel-2 (10 m) + MODIS (500 m) at one slice

The slice declares the *target* resolution; coarser data is upsampled, finer is downsampled — both arrive with identical `(H, W)` and transform.

```python
from rasterio.enums import Resampling
from georeader.catalog import build_raster_catalog, GeoSlice, load_raster

s2    = build_raster_catalog(s2_files,    filename_regex=S2_REGEX,    target_crs="EPSG:32629")
modis = build_raster_catalog(modis_files, filename_regex=MODIS_REGEX, target_crs="EPSG:32629")

sl = GeoSlice(bounds=AOI, interval=DAY_INTERVAL,
              resolution=(10.0, 10.0), crs="EPSG:32629")

x_high = load_raster(s2,    sl, band_indexes=[1, 2, 3, 4])                              # native
x_low  = load_raster(modis, sl, band_indexes=[1, 2], resampling=Resampling.bilinear)    # upsampled
combined = np.concatenate([x_high.values, x_low.values], axis=0)                         # same H, W
```

#### K. Multi-modal stack: optical + DEM (static) + ERA5 (coarse, lat/lon)

Three backends and two CRSs.
DEM has no time → `spatial_only=True`.
ERA5 is xarray in EPSG:4326 → reproject the slice on the fly.

```python
from georeader.catalog import build_raster_catalog, build_xarray_catalog, load_raster, load_xarray
from georeader.samplers import random_sampler

opt  = build_raster_catalog(s2_files,  filename_regex=S2_REGEX,  target_crs="EPSG:32629")
dem  = build_raster_catalog(dem_files, filename_regex=DEM_REGEX, target_crs="EPSG:32629")
era5 = build_xarray_catalog(era5_files, target_crs="EPSG:4326",
                            data_vars=["t2m", "tp"], time_var="time")

opt_dem = opt.intersect(dem, spatial_only=True)

for sl in random_sampler(opt_dem, chip_size=(128, 128), length=5_000):
    rgbn   = load_raster(opt, sl, band_indexes=[2, 3, 4, 8]).values    # (4, 128, 128)
    elev   = load_raster(dem, sl, band_indexes=[1]).values              # (1, 128, 128)
    sl_ll  = sl.to_crs("EPSG:4326")                                     # extension on GeoSlice
    weather = load_xarray(era5, sl_ll)                                  # very coarse cube
    yield {
        "rgbn":    rgbn,
        "dem":     elev,
        "weather": weather[["t2m", "tp"]].mean(("x", "y", "time")).to_array().values,  # (2,)
    }
```

### 8.5 Special ML patterns

#### L. Random chips for a torch `DataLoader` (basic adapter)

```python
import torch
from georeader.catalog import build_raster_catalog, load_raster
from georeader.samplers import random_sampler

catalog = build_raster_catalog(...)


class GeoDataset(torch.utils.data.IterableDataset):
    def __init__(self, catalog, chip_size, length, seed=0):
        self.iter_factory = lambda: random_sampler(catalog, chip_size, length=length, seed=seed)
        self.catalog = catalog

    def __iter__(self):
        for sl in self.iter_factory():
            yield load_raster(self.catalog, sl, band_indexes=[1, 2, 3, 4]).values


loader = torch.utils.data.DataLoader(GeoDataset(catalog, (256, 256), 10_000),
                                     batch_size=32, num_workers=4)
```

#### M. Self-supervised: temporal-pair sampler (SimCLR-style positives)

Anchor and positive are the *same* spatial chip at *different* dates within the same tile interval.

```python
import dataclasses
import shapely.geometry
from georeader.samplers import random_sampler

def temporal_pair_sampler(catalog, chip_size, length, *, min_gap_days=10, seed=0):
    """Yield (anchor, positive) GeoSlice pairs from the same tile, far apart in time."""
    rng  = np.random.default_rng(seed)
    base = random_sampler(catalog, chip_size, length=length * 2, seed=seed)
    for anchor in base:
        chip_box = shapely.geometry.box(*anchor.bounds)
        tile = catalog.gdf[catalog.gdf.geometry.contains(chip_box)].iloc[0]
        lo, hi = tile.name.left.value, tile.name.right.value
        t_pos = pd.Timestamp(rng.integers(lo, hi), unit="ns")
        if abs((t_pos - anchor.interval.left).days) < min_gap_days:
            continue
        positive = dataclasses.replace(
            anchor, interval=pd.Interval(t_pos, t_pos, closed="both"),
        )
        yield anchor, positive


for a, p in temporal_pair_sampler(catalog, (224, 224), 100_000):
    x_a = load_raster(catalog, a).values
    x_p = load_raster(catalog, p).values
    loss = simclr_loss(encoder(x_a), encoder(x_p))
```

#### N. Spatial-block cross-validation (no spatial leakage)

Cluster tile centroids into K spatial blocks; fold by block, never by row.
Critical for honest generalization estimates on geospatial models.

```python
from sklearn.cluster import KMeans

cents = np.column_stack([catalog.gdf.geometry.centroid.x,
                         catalog.gdf.geometry.centroid.y])
catalog.gdf["block"] = KMeans(n_clusters=5, random_state=0).fit_predict(cents)

for fold in range(5):
    train = catalog.where("block != @fold")     # extension: pandas-like filter on gdf
    val   = catalog.where("block == @fold")
    train_one_fold(model, make_loader(train), make_loader(val))
```

#### O. Foundation-model embedding store

Encode every chip in the catalog with a frozen vision encoder; persist `(N, D)` embeddings + chip metadata to a Zarr store for downstream retrieval, clustering, or weak supervision.

```python
import zarr
from georeader.catalog import build_raster_catalog, load_raster
from georeader.samplers import grid_sampler

catalog = build_raster_catalog(...)
slices  = list(grid_sampler(catalog, chip_size=(224, 224), stride=(224, 224)))
N, D    = len(slices), 1024

z   = zarr.open("/out/embeddings.zarr", mode="w")
emb = z.zeros("emb", shape=(N, D), chunks=(1024, D), dtype="float32")
meta = z.create_dataset("meta", shape=(N,), dtype=[
    ("xmin", "f8"), ("ymin", "f8"), ("xmax", "f8"), ("ymax", "f8"),
    ("tmin", "M8[ns]"), ("tmax", "M8[ns]"),
])

for i, sl in enumerate(slices):
    x = load_raster(catalog, sl, band_indexes=[2, 3, 4, 8]).values
    emb[i]  = frozen_encoder(x)
    meta[i] = (*sl.bounds,
               np.datetime64(sl.interval.left), np.datetime64(sl.interval.right))
```

### 8.6 API extensions implied by these examples

The new examples lean on a small set of conveniences that aren't in §6's minimal surface but are natural to add:

- `GeoSlice.to_crs(target_crs)` — reproject the bbox + resolution into another CRS (used in K).
- `GeoCatalog.where(query: str)` — pandas-`.query()` passthrough on `gdf` (used in N).
- `GeoCatalog.with_interval_padding(days=N)` — vectorised interval shift; in I we inlined it for clarity.
- `GeoTensor.to_xarray()` — already implied by `georeader/dataarray.py`; used in C.
- `polygons_from_mask(mask, transform, crs)` — likely already lives near `georeader/vectorize.py`; used in E.
- Mask-aware reductions in `load_raster` (e.g. drop pixels where a paired QA band sets a bit) — for now we load QA explicitly and apply the mask in user code (A, C).

---

## 9. Verdict

The snippets are the **right idea, executed roughly**.
The shared GeoDataFrame index is genuinely good — the algebra (`intersect` / `union` / `query`) falls out for free, three backends share one shape, and the math is sound (modulo the sharp edges in [§7](#7-sharp-edges-to-fix-on-the-way-in)).
The samplers and stitching are textbook but correct.

What needs work before promotion to a public georeader API:

- [ ] Replace dict returns with `GeoTensor` / `xr.Dataset` so georeferencing is preserved.
- [ ] Wrap the GeoDataFrame in a `GeoCatalog` class — it makes `intersect` / `union` / `query` discoverable and lets us evolve the index format later.
- [ ] Make `torch` and `loguru` optional. Keep the core import surface to numpy + pandas + geopandas + shapely + rasterio (already georeader's deps) + pyproj.
- [ ] Fix the two syntax errors and add at least smoke tests for build → query → load on each backend, and for cross-CRS catalogs.
- [ ] Drop `run_inference_with_grid_sampler` from the library surface; ship it as a docs example instead.
- [ ] Decide if `GeoSlice` should reuse / extend `georeader.slices.py` or be a new sibling — I'd reuse and extend.

If you do those six things, this becomes a real `georeader.catalog` module — the dataset-collection layer that georeader is currently missing — without bloating the dependency footprint.

---

## 10. Open questions, gotchas, and warnings

The Phase 1 / Phase 2 design is sound; several execution-level concerns deserve flags.

### 10.1 Cross-CRS query footgun

The catalog stores all geometries in a uniform target CRS (Phase 1 default: source CRS preserved per-row but reprojected for query; Phase 2 convention from `geoduckdb.md` §4: write all GeoParquet in EPSG:4326).
Users querying `WHERE ST_Intersects(geom, AOI)` with AOI in a non-target CRS get **silently empty results** — no error, just no rows.
**Mitigation:** provide a `query` helper that takes `(aoi: shapely.Geometry, aoi_crs: pyproj.CRS)` and projects internally before passing to the underlying engine.
Make this the canonical path; the raw `gdf.cx[...]` / SQL paths assume the AOI is already in catalog CRS.

### 10.2 GeoParquet 1.1 writer adoption is uneven

`geopandas.to_parquet` defaults to GeoParquet 1.0; the per-row bbox column for predicate pushdown requires **GeoParquet 1.1**, which means an explicit `version="1.1"` flag (or whatever the geopandas writer uses) and a `geopandas` version pin.
DuckDB's GeoParquet *write* path is newer than its read path.
**Mitigation:** pin `geopandas >= 0.14` (or whatever first-shipped 1.1 writer) and `duckdb >= 1.1` explicitly in the package metadata; add a CI test that round-trips a small catalog through `to_geoparquet` → disk → `from_geoparquet` and verifies the bbox column survives.

### 10.3 Phase 2 risk of bit-rot

Most users will stay on Phase 1 (GeoPandas + IntervalIndex) for the 90% case.
Phase 2 (DuckDB) won't get exercised unless someone hits the 10⁶+-row threshold.
**Risk:** untested code paths on the upgrade.
**Mitigation:** gate the Phase 2 release behind one real Phase 2 user (a multi-million-row catalog from your own work, or a community case study); add a multi-million-row CI fixture if no organic user emerges within v0.2.

### 10.4 `schema_version` column is reserved, not used yet

GeoParquet schemas evolve.
Reserve a `schema_version` column from v0.1 so future migrations are tractable; current Phase 1 schema is de facto v0; bump to v1 the first time we change anything substantive (e.g. add a `n_timesteps` column).
Document the migration runbook so this doesn't turn into tribal knowledge.

### 10.5 Concurrent write semantics

Phase 1 (GeoPandas in memory) is single-writer.
Phase 2 (DuckDB on a shared GeoParquet) is **read-safe** across processes; **write-safe only if you treat Parquet as append-only** (new row groups in new files, not in-place mutation).
Document the recommended write pattern; users coordinating multiple crawlers will trip over this.

### 10.6 IntervalIndex edge cases

`pd.IntervalIndex` with `closed="both"` handles overlapping intervals fine, but **back-to-back intervals** (one ends at `t`, next starts at `t`) double-count at the boundary.
For sub-daily satellite data this rarely matters; for daily/hourly products it does.
Decide on `closed="left"` vs `closed="both"` and stick with it; document the choice.

### 10.7 Adapter scope

The plan mentions STAC read/write adapters and a `GeoDataset` torch adapter.
Each adapter is a small library on its own — STAC has its own metadata model that doesn't round-trip cleanly to a GeoDataFrame in all cases (collections vs items, asset-level metadata).
**Scope honestly:** v0.1 ships `to_geoparquet` / `from_geoparquet` round-trip; STAC and torch adapters are v0.2+. Don't promise adapters that aren't built.

