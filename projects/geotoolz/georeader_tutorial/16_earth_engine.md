# Ch. 16 — Earth Engine

> **Modules:**
> - `georeader/readers/ee_image.py` (539 LOC)
> - `georeader/readers/ee_query.py` (589 LOC)
> - `georeader/readers/ee_utils.py` (58 LOC)
> **Role:** export raster data from Google Earth Engine into `GeoTensor`s, handling GEE's request-size limits through recursive tile splitting and parallel downloads.

---

## 1. Why this module exists

Google Earth Engine is the standard cloud platform for petabyte-scale remote-sensing analytics — it hosts every major archive (Landsat, Sentinel, MODIS, dozens more) and provides a JavaScript / Python API for server-side computation. But: **GEE export is constrained**.

- `ee.data.computePixels()` and `ee.data.getPixels()` cap requests at ~32 MB per call.
- A modest AOI at 10 m resolution easily exceeds this.
- Manually splitting and stitching is annoying and the source of countless duct-tape scripts.

This module's job is to take any `ee.Image` and any AOI, and return a `GeoTensor` — splitting and parallelising the GEE calls automatically, no matter how large.

---

## 2. The export workflow

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                  GEE EXPORT WORKFLOW                                     │
│                                                                          │
│   User Request                                                           │
│   ┌──────────────────┐                                                  │
│   │ export_image()   │                                                  │
│   │ - ee.Image       │                                                  │
│   │ - geometry       │                                                  │
│   │ - bands          │                                                  │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────────────────────────────────────────────┐          │
│   │  Try ee.data.computePixels() / ee.data.getPixels()       │          │
│   │  (Limited to ~32MB per request)                          │          │
│   └────────┬─────────────────────────────────┬───────────────┘          │
│            │ Success                         │ "Total request size"     │
│            ▼                                 ▼ error                    │
│   ┌──────────────────┐            ┌──────────────────────────┐          │
│   │ Return GeoTensor │            │ RECURSIVE TILE SPLITTING │          │
│   └──────────────────┘            │                          │          │
│                                   │  ┌────┬────┐             │          │
│                                   │  │ Q1 │ Q2 │ Split into  │          │
│                                   │  ├────┼────┤ 4 quadrants │          │
│                                   │  │ Q3 │ Q4 │             │          │
│                                   │  └────┴────┘             │          │
│                                   │       │                  │          │
│                                   │       ▼                  │          │
│                                   │  Process each in        │          │
│                                   │  parallel (ThreadPool)  │          │
│                                   │       │                  │          │
│                                   │       ▼                  │          │
│                                   │  spatial_mosaic()       │          │
│                                   │  to combine tiles       │          │
│                                   └──────────────────────────┘          │
│                                            │                            │
│                                            ▼                            │
│                                   ┌──────────────────┐                  │
│                                   │ Return GeoTensor │                  │
│                                   └──────────────────┘                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

The recursion logic:

1. Try the full request via `ee.data.computePixels()`.
2. If GEE returns a "Total request size" error, split the bbox into 4 quadrants.
3. Process each quadrant in parallel via a `ThreadPoolExecutor`.
4. Each quadrant may itself need splitting — recurse.
5. Mosaic results back together with [`mosaic.spatial_mosaic`](08_mosaic.md).

This pattern handles arbitrary AOI sizes — the only practical limit is memory (final `GeoTensor` has to fit in RAM) and time (GEE rate-limits requests per project).

---

## 3. The four export functions

| Function | Use for |
|---|---|
| `export_image_fast(image, ...)` | quick single-call export — no splitting; raises if too big |
| `export_image(image_or_asset_id, ...)` | the workhorse — recursive splitting, parallel download |
| `export_image_getpixels(asset_id, ...)` | direct-asset export via `ee.data.getPixels` (when you have the asset ID, not a computed image) |
| `export_cube(query, geometry, ...)` | time-series export — returns `(T, C, H, W)` `GeoTensor` |

`export_image` is the one you'll use 95% of the time. Source: [ee_image.py:223](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/ee_image.py#L223).

`export_cube` ([ee_image.py:392](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/ee_image.py#L392)) is built on top — accepts a GeoDataFrame of (acquisition_date, asset_id) rows, exports each row as a 3D image, and stacks them along a new time axis.

---

## 4. The split helper

`split_bounds(bounds) → list[bounds]` ([ee_image.py:211](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/ee_image.py#L211)) does the quadrant split:

```text
   bounds = (minx, miny, maxx, maxy)
   midx = (minx + maxx) / 2
   midy = (miny + maxy) / 2

   Q1: (minx, midy, midx, maxy)   Q2: (midx, midy, maxx, maxy)
   Q3: (minx, miny, midx, midy)   Q4: (midx, miny, maxx, midy)
```

Always 4-way split (not adaptive based on aspect ratio). Adequate for the typical "moderately large AOI" case; can be inefficient for very long thin AOIs (a 10000:1 aspect ratio strip would need many splits before each chunk fits — but the typical RS AOI is roughly square).

---

## 5. The Sentinel-2 special case: `interpolate_20mbands_s2ee`

When you download S2 from GEE asking for 10 m resolution, GEE upsamples the 20 m and 60 m bands using **nearest neighbour**. That produces ugly blocky bands at 10 m — not what you want.

`interpolate_20mbands_s2ee(geotensor, ...)` ([ee_image.py:476](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/ee_image.py#L476)) post-processes the result by re-resampling the affected bands with **bicubic** interpolation. The result looks like a real 10 m image rather than blocky pixels.

This isn't a bug in this module — it's a quirk of GEE's S2 ingestion. The workaround is the kind of detail that takes you an afternoon to find on the GEE forum; having it as a one-line call makes the module pay for itself.

---

## 6. The query side — `ee_query.py`

`ee_query.py` (589 LOC) handles the **discovery** side: given a polygon and a date range, return a GeoDataFrame of available scenes. Functions like `query_collection(collection_name, polygon, start_date, end_date)` for each major collection (Sentinel-2, Landsat, MODIS, etc.).

The output GeoDataFrame is the input shape that `export_cube` consumes — so the pattern is:

```python
from georeader.readers import ee_query, ee_image

# 1. Find scenes
scenes = ee_query.query_s2_l2a(polygon=aoi, start_date="2024-01-01", end_date="2024-12-31")
# scenes is a GeoDataFrame with one row per S2 scene over the AOI in 2024

# 2. Filter
scenes = scenes[scenes["cloud_cover"] < 20]

# 3. Export as a time-series cube
cube = ee_image.export_cube(scenes, geometry=aoi, bands=["B04", "B03", "B02"], scale=10)
# cube is a (T, 3, H, W) GeoTensor
```

That's the whole "find S2 scenes over Madagascar in 2024 with < 20% cloud cover, get an RGB time series" pipeline in 6 lines of meaningful code.

---

## 7. Authentication

GEE requires authentication. The package uses `ee.Authenticate()` and `ee.Initialize()` from the official `earthengine-api`. Standard setup:

```python
import ee
ee.Authenticate()  # browser flow first time, cached afterwards
ee.Initialize(project="my-gcp-project")
```

Install `georeader-spaceml` with the `[ee]` extra to pull in `earthengine-api`. Without it, importing `readers.ee_image` raises `ImportError`. In this repo the relevant install commands are `pixi add georeader-spaceml[ee]` or `uv add 'georeader-spaceml[ee]'` per the project's [CONTRIBUTING.md](../../../CONTRIBUTING.md) tooling conventions.

The credentials live in `~/.config/earthengine/credentials` (managed by `ee.Authenticate()`); the module doesn't add its own credential file like EMIT does.

---

## 8. Function reference

**Image export**
- `export_image_fast(image, geometry, bands=None, scale=None, ...)` — single call, no splitting
- `export_image(image_or_asset_id, geometry, bands=None, scale=None, crs=None, ...)` — recursive splitting + parallel download
- `export_image_getpixels(asset_id, geometry, bands=None, ...)` — for direct-asset access via `getPixels`
- `export_cube(query, geometry, bands=None, scale=None, ...)` — time-series export from a GeoDataFrame query

**Helpers**
- `split_bounds(bounds) → list[bounds]` — 4-way quadrant split
- `_find_padding(v, divisor=8) → int` — pad to GEE-friendly multiples (internal)
- `interpolate_20mbands_s2ee(geotensor, ...)` — fix S2 nearest-neighbour upsampling

**Query side (separate file `ee_query.py`)**
- `query_collection(collection, polygon, ...)` — generic collection query
- `query_s2_l2a(...)`, `query_s2_l1c(...)`, `query_landsat(...)`, etc. — collection-specific helpers

**Utilities (separate file `ee_utils.py`, 58 LOC)**
- Auth helpers, GEE-version checks, small adapters between `ee.Geometry` and Shapely

---

## 9. Sharp edges

- **GEE rate limits per project.** Recursive splitting can issue many concurrent requests; on a small AOI this is fine, on a continental one you might hit `quota_exceeded` errors. Tune via fewer parallel threads (`max_workers=2`) or stagger requests.
- **`ee.Initialize(project=...)` is required.** GEE migrated to project-scoped quotas in 2024; old code that just called `ee.Initialize()` without a project ID will fail. Use a GCP project you have access to.
- **Authenticated S3-style mirrors are different.** GEE's mirrors of Sentinel-2 are *not* the same as the AWS / GCS public buckets. The `gs://gcp-public-data-sentinel-2` URL works without GEE auth (Chapter 14). GEE access to S2 needs `ee.Initialize` + project quota.
- **Nearest-neighbour upsampling for S2 20m bands.** Always run `interpolate_20mbands_s2ee` after exporting S2 at 10 m or your bands look terrible.
- **`export_cube` aligns to the first scene's grid.** Heterogeneous resolutions across times need a `scale=` arg to be explicit. Mix-and-match Landsat-8 (30 m) + Sentinel-2 (10 m) without `scale=` will silently coerce to whatever the first scene was.
- **Recursive splitting can OOM at the join.** The final `spatial_mosaic` call materialises all quadrants in memory. For petabyte-scale exports, `export_cube` with windowed writes to disk is more appropriate than `export_image` to a single `GeoTensor`.
- **The `ee` import is hard.** `from georeader.readers import ee_image` requires `earthengine-api` to be installed. The package's `[ee]` extra makes this explicit; the import won't be silently skipped.

---

## 10. Connection to `geotoolz`

GEE doesn't have a dedicated wrapper in [`geotoolz.md`](../plans/geotoolz/geotoolz.md), but it shows up as a substrate alternative:

- **Catalog discovery via `ee_query`** is a sibling to `geotoolz.catalog_ops.CatalogPipeline(catalog, op).run()` from the geotoolz plan. The two could converge: `ee_query.query_*` produces a GeoDataFrame; `CatalogPipeline` consumes a `GeoCatalog`. A small adapter would let `geotoolz.catalog_ops` operate on GEE-discovered scenes directly.
- **`export_cube` is the GEE analogue of `RasterioReader([paths], stack=True)`** — both produce `(T, C, H, W)`. Operators that consume time-stacks (e.g., `geotoolz.compositing.MedianComposite`) work on either substrate without modification.

For users who already live on GEE, this module is the bridge that lets them move pipelines into the georeader / geotoolz operator world without abandoning the GEE archive.

Next chapter: [17_legacy_sensors.md](17_legacy_sensors.md) — the legacy operational sensor readers (SPOT VGT, Proba-V).
