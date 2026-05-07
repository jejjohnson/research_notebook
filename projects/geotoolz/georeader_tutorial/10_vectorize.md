# Chapter 10 — `vectorize.py`: rasters → vectors

> **Module:** `georeader/vectorize.py` (370 LOC)
> **Role:** the inverse of [Chapter 9](09_rasterize.md). Extract polygon geometries from binary raster masks. Standard tool for converting segmentation outputs and classification rasters back to GIS-friendly vector formats.

---

## 1. The job

You have a binary (or thresholded) raster — typically the output of a CNN segmentation, a cloud detector, or a classified scene. You want a list of Shapely `Polygon`s in CRS coordinates, suitable for:

- Writing to GeoJSON / Shapefile / GeoParquet for downstream GIS work.
- Computing per-region statistics (area, perimeter, intersection with other vectors).
- Counting / filtering objects (e.g., "how many flood polygons larger than 1 km²?").
- Overlaying on a basemap with `lonboard` / Leaflet / QGIS.

Underneath this is `rasterio.features.shapes` (which delegates to GDAL's polygonization). This module wraps it with three on-by-default ergonomics: small-polygon filtering, vertex simplification, and polygon buffering.

---

## 2. The vectorization process

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    VECTORIZATION PROCESS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Raster (Binary Mask)                Vector (Polygons)                  │
│  ────────────────────                ─────────────────                  │
│                                                                          │
│  ┌─┬─┬─┬─┬─┬─┬─┬─┐                       ╔═══════════╗                  │
│  │0│0│0│1│1│1│0│0│                      ╔╝           ╚╗                 │
│  ├─┼─┼─┼─┼─┼─┼─┼─┤                     ╔╝             ╚╗                │
│  │0│0│1│1│1│1│1│0│   ═══════════►     ╔╝               ╚╗               │
│  ├─┼─┼─┼─┼─┼─┼─┼─┤   Vectorize        ║    Polygon 1    ║               │
│  │0│1│1│1│1│1│1│0│                    ╚╗               ╔╝               │
│  ├─┼─┼─┼─┼─┼─┼─┼─┤                     ╚╗             ╔╝                │
│  │0│0│1│1│1│1│0│0│                      ╚╗           ╔╝                 │
│  └─┴─┴─┴─┴─┴─┴─┴─┘                       ╚═══════════╝                  │
│                                                                          │
│  1 = foreground (vectorized)                                            │
│  0 = background (ignored)                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

Every connected component of `1`-valued pixels becomes one polygon. The polygon traces the **outside** edge of the foreground pixels — so a 3×3 patch of `1`s becomes a polygon enclosing 9 pixel squares (perimeter ≈ 12 pixels), not a single point.

For multi-class rasters, threshold or mask first: `vectorize` is fundamentally a binary operation. To extract per-class polygons from a label raster, loop:

```python
polygons_per_class = {
    c: get_polygons(labels == c, ...) for c in np.unique(labels)
}
```

---

## 3. Polygon simplification

```text
┌─────────────────────────────────────────────────────────────────────────┐
│              POLYGON SIMPLIFICATION (tolerance parameter)                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Raw (pixelated)              Simplified (tolerance=1)                  │
│  ────────────────              ──────────────────────                   │
│                                                                          │
│  ┌─┐                                ╭───────╮                            │
│  │ └─┐                             ╱         ╲                           │
│  │   └─┐                          ╱           ╲                          │
│  │     └─┐   ────────────►       │             │    Fewer vertices,     │
│  │       │   simplify            │             │    smoother edges      │
│  │     ┌─┘                        ╲           ╱                          │
│  │   ┌─┘                           ╲         ╱                           │
│  └───┘                              ╰───────╯                            │
│                                                                          │
│  tolerance=0: Keep all vertices (staircase pattern)                     │
│  tolerance=1: Simplify ~1 pixel tolerance (DEFAULT)                     │
│  tolerance>1: More aggressive simplification                            │
└─────────────────────────────────────────────────────────────────────────┘
```

Without simplification, a polygon traces every pixel boundary — a 100×100 region produces ~400 vertices forming a staircase pattern. That's both visually ugly and storage-expensive.

`tolerance` is the Douglas-Peucker simplification distance, in **pixels**. Default `1.0` collapses single-pixel jaggies into smooth edges while preserving features ≥ 1 pixel. Larger values smooth more aggressively at the cost of accuracy:

| `tolerance` | What you get |
|---|---|
| `0` | Exact pixel boundaries (staircase). Use when downstream consumers expect rasterise round-trip equality. |
| `1.0` (default) | One-pixel smoothing. Visually clean while preserving ≥ 1 pixel features. |
| `2.0–5.0` | Aggressive smoothing for visualisation. Loses fine-grained shape detail. |
| `> 10` | Bounding-box-like simplification. Suitable only for index/preview purposes. |

The simplification happens after vectorization, in pixel coordinates, before applying the affine transform. So `tolerance=1` always means "1 pixel" regardless of CRS or resolution.

---

## 4. Polygon filtering

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    POLYGON FILTERING                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Parameters:                                                             │
│  ───────────                                                             │
│                                                                          │
│  min_area=25.5 (default)    Remove polygons smaller than ~5x5 pixels    │
│                             Helps filter noise and artifacts             │
│                                                                          │
│  polygon_buffer=0           Buffer/erode polygons by N pixels           │
│                             Positive: expand                             │
│                             Negative: shrink (erode)                     │
│                                                                          │
│  Before (min_area=0):                After (min_area=25):               │
│  ┌────────────────────┐              ┌────────────────────┐             │
│  │  ■   ┌───────┐     │              │      ┌───────┐     │             │
│  │ ■ ■  │       │  ■  │   ═══════►   │      │       │     │             │
│  │      │       │     │   Filter     │      │       │     │             │
│  │ ■    └───────┘     │              │      └───────┘     │             │
│  └────────────────────┘              └────────────────────┘             │
│     ↑ small polygons removed                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

Two filters layered onto the basic extraction:

- **`min_area`** — drop any polygon whose area is below this threshold. Units are **pixels** (so `25.5` ≈ a 5×5 pixel square; the `.5` accounts for the typical pixel-centre adjustment). Crucial for noise removal: CNN segmentations often emit single-pixel false positives, and the default `25.5` filters them at very low cost. For large-scale work where every commission error matters, raise to 100 or 1000.
- **`polygon_buffer`** — apply a shapely buffer in pixel units. Positive value expands each polygon (closes small gaps, merges nearby objects); negative value erodes (shrinks; can split a polygon into pieces or eliminate thin features). `polygon_buffer=-1` is the standard "shrink by one pixel to avoid edge inclusivity" trick.

Order of operations: vectorize → buffer → simplify → filter by min_area. So the area filter applies to the *final* simplified geometry, not the raw extraction.

---

## 5. The two functions

| Function | Returns | What it does |
|---|---|---|
| `get_polygons(binary_mask, min_area=25.5, polygon_buffer=0, tolerance=1.0, transform=None)` | `list[Polygon]` | the workhorse — see below |
| `transform_polygon(polygon, transform=None, src_crs=None, dst_crs=None)` | `Polygon` / `MultiPolygon` | apply an affine transform and/or reproject between CRSs |

### `get_polygons(binary_mask, ...)`

Accepts either a numpy array or a `GeoData` (`GeoTensor` / `RasterioReader`).

- **If `binary_mask` is a `GeoData`** — the function reads `.transform` and uses it to convert pixel-coordinate polygons to CRS-coordinate polygons. The output Polygons are georeferenced.
- **If `binary_mask` is a plain ndarray** — output Polygons are in pixel coordinates, **unless** you pass `transform=` explicitly. In which case they're in CRS coords.

This duck-typing on input is the killer convenience: pipe a CNN's `(H, W)` numpy output and a transform together, get georeferenced polygons. No manual `polygon_to_crs` round-trip.

### `transform_polygon(polygon, ...)`

The post-processing helper for when you already have a polygon and want to:

- **Apply an affine** (e.g., the polygon came from `get_polygons(..., transform=None)` and you now have a transform).
- **Reproject between CRSs** (e.g., raster is UTM, you want WGS84 polygons for a GeoJSON).

Source: [vectorize.py:271](../../../../tmp/georeader_src/georeader/vectorize.py#L271).

---

## 6. The standard CNN-output-to-vectors recipe

```python
import numpy as np
from georeader import vectorize

# 1. CNN prediction on a GeoTensor input — you get back logits or probs
probs = model(s2_gt.values)               # (H, W) float32, ndarray
mask = probs > 0.5                        # (H, W) bool

# 2. Wrap in a GeoTensor so transform tags along
mask_gt = s2_gt.array_as_geotensor(mask)  # (H, W) bool GeoTensor

# 3. Vectorize with sensible defaults
polygons = vectorize.get_polygons(
    mask_gt,
    min_area=100,            # 10×10 pixel min — drops noise
    tolerance=1.0,           # one-pixel smoothing
    polygon_buffer=0,
)

# 4. Persist
import geopandas as gpd
gdf = gpd.GeoDataFrame(geometry=polygons, crs=s2_gt.crs)
gdf.to_file("flood_polygons.geojson", driver="GeoJSON")
```

Five lines for the whole "CNN prediction → georeferenced vector layer" pipeline. The `array_as_geotensor` step (Chapter 1 §10) is what makes the final polygons georeferenced — without it you'd be in pixel coords.

---

## 7. The rasterize ↔ vectorize round trip

`rasterize` and `vectorize` are conceptually inverses, but **they are not exactly invertible**. Three sources of asymmetry:

1. **Rasterization quantises** — every pixel either is or isn't in the polygon. After `rasterize → vectorize`, you get a polygon traced along pixel boundaries (a staircase) rather than the original smooth shape.
2. **`tolerance > 0` smooths**. Round-tripping through `vectorize(tolerance=1)` then `rasterize` will not exactly match the original raster.
3. **`min_area` drops content**. Small features below the threshold are gone for good after `vectorize → rasterize`.

For applications that need exact round-trips (e.g., test fixtures), pass `tolerance=0, min_area=0, polygon_buffer=0`. The output will have full vertex counts (large) and exact pixel boundaries.

---

## 8. Sharp edges

- **`min_area` units are pixels, not CRS area.** A `min_area=25` means 25 pixels regardless of resolution. If you want "1 km² minimum" on a 10m raster, that's `100*100 = 10000` pixels.
- **`polygon_buffer` units are pixels too.** Same caveat.
- **`tolerance=0` produces large polygons.** Every pixel boundary becomes a vertex. A flood-mapping output across a city can produce polygons with 100k+ vertices each; downstream tools (especially leaflet) will choke. Default `tolerance=1` is correct in nearly all cases.
- **The function only vectorizes `True` pixels.** For multi-class rasters, threshold or split per-class first.
- **A `MultiPolygon` is returned as multiple `Polygon`s.** `get_polygons` doesn't preserve multipolygon grouping — disconnected components are separate list entries even if the source mask had them as part of the same logical region.
- **Plain ndarray + no `transform` → pixel-coordinate output.** Easy to forget; the polygons look "off-by-1000-orders-of-magnitude wrong" when you plot them on a map. Either pass `transform=` or wrap as a `GeoTensor` first.
- **GeoJSON expects WGS84 (`EPSG:4326`).** The polygons come out in the source raster's CRS. Use `transform_polygon(p, src_crs=..., dst_crs="EPSG:4326")` before exporting if your raster wasn't already WGS84.

---

## 9. Connection to `geotoolz`

Two operators in [`geotoolz.md`](../plans/geotoolz.md) wrap this module:

- **`postprocess.PolygonsFromMask(min_area=..., tolerance=...)`** — a terminal operator that converts the final `(H, W)` boolean output of a `Sequential` to a list of polygons. Useful as the last step of a `[Sequential(model + threshold + PolygonsFromMask)]` pipeline.
- **`catalog_ops.WriteGeoJSON(...)`** — write polygons (with optional attributes) to disk per-tile during catalog processing. Internal call: `get_polygons(...)` + `gpd.GeoDataFrame(...)` + `to_file(...)`.

These complete the "raster ML pipeline → GIS-ready output" loop without users touching rasterio or geopandas directly.

Next chapter: [11_reflectance.md](11_reflectance.md) — the radiometry / spectral-response-function module (971 LOC, 97 box-drawing chars; the third densest in the package).
