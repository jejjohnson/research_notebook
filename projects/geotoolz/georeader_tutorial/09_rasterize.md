---
title: rasterize
subject: georeader tutorial
subtitle: Vectors → rasters
short_title: Ch. 9 — Rasterize
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: tutorial, georeader, rasterize
---

> **Module:** `georeader/rasterize.py` (438 LOC)
> **Role:** burn vector geometries (polygons, lines, GeoDataFrames) into raster grids aligned to an existing `GeoData`. The standard tool for building masks, segmentation labels, and ROI maps from GIS-flavoured data.

---

## 1. The job

You have:

- One or more vector geometries (`Polygon`, `MultiPolygon`, `LineString`, or a `GeoDataFrame` with attribute values).
- A reference raster — `GeoTensor` or `RasterioReader` — whose grid you want the output aligned to.

You want a raster of the same shape and georeferencing as the reference, with **inside-the-geometry** pixels set to a chosen value and outside-the-geometry pixels set to a fill value.

This is what GIS calls "rasterization" or "burning vectors." The implementation underneath is `rasterio.features.rasterize` (which delegates to GDAL); this module wraps it with `GeoData`-aware ergonomics so you don't manually pass transform / shape / dtype every time.

---

## 2. The rasterization process

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    RASTERIZATION PROCESS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Vector (Polygon)                    Raster (Grid)                      │
│  ─────────────────                   ─────────────                      │
│                                                                          │
│       ╔═══════════╗                  ┌─┬─┬─┬─┬─┬─┬─┬─┐                  │
│      ╔╝           ╚╗                 │░│░│░│▓│▓│▓│░│░│                  │
│     ╔╝             ╚╗                ├─┼─┼─┼─┼─┼─┼─┼─┤                  │
│    ╔╝               ╚╗   ═══════►   │░│░│▓│▓│▓│▓│▓│░│                  │
│    ║     Polygon     ║   Rasterize  ├─┼─┼─┼─┼─┼─┼─┼─┤                  │
│    ╚╗               ╔╝               │░│▓│▓│▓│▓│▓│▓│░│                  │
│     ╚╗             ╔╝                ├─┼─┼─┼─┼─┼─┼─┼─┤                  │
│      ╚╗           ╔╝                 │░│░│▓│▓│▓│▓│░│░│                  │
│       ╚═══════════╝                  └─┴─┴─┴─┴─┴─┴─┴─┘                  │
│                                                                          │
│  ░ = fill value (outside polygon)                                       │
│  ▓ = burn value (inside polygon)                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

Two values control the output: `value` (the "burn" value, default `1`) and `fill` (the "outside" value, default `0`). The default produces a binary mask — pass non-binary `value` for class labels (e.g., `value=3` for "this polygon is class 3").

---

## 3. The `all_touched` decision

```text
┌─────────────────────────────────────────────────────────────────────────┐
│              all_touched=False vs all_touched=True                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  all_touched=False (default)         all_touched=True                   │
│  ───────────────────────────         ──────────────────                 │
│                                                                          │
│  Only pixels with CENTER inside      ALL pixels that TOUCH the polygon  │
│                                                                          │
│  ┌─┬─┬─┬─┬─┬─┐                       ┌─┬─┬─┬─┬─┬─┐                      │
│  │░│░│░│░│░│░│  ╱ polygon edge       │░│▓│▓│▓│▓│░│                      │
│  ├─┼─┼─┼─┼─┼─┤ ╱                     ├─┼─┼─┼─┼─┼─┤                      │
│  │░│░│·│▓│▓│░│╱                      │░│▓│▓│▓│▓│▓│                      │
│  ├─┼─┼─┼─┼─┼─┤                       ├─┼─┼─┼─┼─┼─┤                      │
│  │░│░│▓│▓│▓│░│    · = center         │░│▓│▓│▓│▓│░│                      │
│  └─┴─┴─┴─┴─┴─┘    ▓ = included       └─┴─┴─┴─┴─┴─┘                      │
│                                                                          │
│  Best for:                           Best for:                          │
│  • Area calculations                 • Inclusive masks                   │
│  • Avoiding edge pixels              • Conservative estimates            │
│  • Conservative estimates            • Complete coverage                 │
└─────────────────────────────────────────────────────────────────────────┘
```

A pixel that's **half** inside the polygon: include it or not?

- **`all_touched=False`** (default) — include only if the pixel **centre** is inside. Conservative; preserves area; matches GDAL's default. The right choice for area computations and for ML labels where you don't want bordering pixels labelled as belonging to a class their centre isn't in.
- **`all_touched=True`** — include any pixel the polygon **touches**. Inclusive; over-estimates area; never gaps at boundaries. The right choice for masks where you want **no missing data inside the AOI** — e.g., reading a polygon AOI before doing per-pixel work, where missing edge pixels would bias statistics.

The two modes can produce visibly different masks for thin geometries (lines, narrow polygons). For a `LineString`, `all_touched=False` produces an effectively empty raster (lines have zero area, so no pixel centre is "inside") — you almost always want `all_touched=True` for line rasterization.

---

## 4. GeoDataFrame rasterization with attributes

```text
┌─────────────────────────────────────────────────────────────────────────┐
│              GEODATAFRAME RASTERIZATION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  GeoDataFrame:                       Output Raster:                     │
│  ┌──────────────────────┐            ┌─┬─┬─┬─┬─┬─┬─┬─┐                  │
│  │ geometry │ class_id  │            │0│0│0│0│0│0│0│0│                  │
│  ├──────────┼───────────┤            ├─┼─┼─┼─┼─┼─┼─┼─┤                  │
│  │ Poly A   │    1      │  ══════►   │0│1│1│1│0│2│2│0│                  │
│  │ Poly B   │    2      │            ├─┼─┼─┼─┼─┼─┼─┼─┤                  │
│  │ Poly C   │    3      │            │0│1│1│0│0│0│3│0│                  │
│  └──────────┴───────────┘            └─┴─┴─┴─┴─┴─┴─┴─┘                  │
│                                                                          │
│  Usage:                                                                  │
│    rasterize_geodataframe(gdf, data_like, attribute="class_id")         │
└─────────────────────────────────────────────────────────────────────────┘
```

Each row contributes a different burn value, taken from the named attribute column. The standard pattern for building **segmentation labels** from GIS-curated training data:

- Column = `class_id` integer per row.
- `fill=0` so background is class 0.
- Output dtype determined by the column's dtype (cast to `uint8` if you have ≤ 256 classes).

For overlapping polygons, **last-row wins** — the rasterio default. If your classes have a meaningful precedence (water > vegetation > bare), sort the GeoDataFrame in priority order before passing.

---

## 5. The four functions

The module exports four entry points distinguished by **input** (single geometry vs DataFrame) and **grid spec** (matching another raster vs explicit transform/shape):

| Function | Input | Grid spec | Use for |
|---|---|---|---|
| `rasterize_geometry_like(geometry, data_like, value=1, fill=0, ...)` | one Shapely geometry | another `GeoData`'s grid | the common case — make a mask aligned to a raster |
| `rasterize_from_geometry(geometry, transform, shape, value=1, fill=0, ...)` | one Shapely geometry | explicit `(transform, shape)` | when you don't have a reference raster, only an output grid spec |
| `rasterize_geopandas_like(dataframe, data_like, column, fill=0, ...)` | `GeoDataFrame` + attribute | another `GeoData`'s grid | building segmentation labels |
| `rasterize_from_geopandas(dataframe, transform, shape, column, fill=0, ...)` | `GeoDataFrame` + attribute | explicit `(transform, shape)` | DataFrame on a designed grid |

Common kwargs across all four:

- **`crs_geometry` / `crs_dataframe`** — CRS of the input geometry. If different from the raster's CRS, it's reprojected to match before rasterising. Required when crossing CRSs; the function doesn't guess.
- **`all_touched`** — section 3 above.
- **`dtype`** — output dtype. Default `uint8` for the binary case; for class-label rasterization, defaults to the column's dtype.

The "`_like`" forms wrap the explicit forms by reading `data_like.transform`, `data_like.crs`, `data_like.shape`, `data_like.dtype`. So in practice you'll use the `_like` forms 95% of the time.

---

## 6. Two idiomatic uses

**Polygon AOI mask, applied to a raster:**

```python
from shapely.geometry import box
from georeader import rasterize

aoi_mask = rasterize.rasterize_geometry_like(
    geometry=box(-122.5, 37.0, -122.0, 37.5),
    data_like=s2_reader,
    crs_geometry="EPSG:4326",
    all_touched=True,             # don't drop edge pixels
)
gt = read.read_from_polygon(s2_reader, aoi)
masked = gt * aoi_mask            # both are GeoTensors with same extent
```

This is how you go from "I have a Shapely polygon" to "I have a polygon-shaped mask in pixel space" without wiring transform/CRS/shape by hand.

**Segmentation label raster from a labelled GeoJSON:**

```python
import geopandas as gpd
gdf = gpd.read_file("labels.geojson")     # has class_id column
labels = rasterize.rasterize_geopandas_like(
    dataframe=gdf,
    data_like=s2_reader,
    column="class_id",
    fill=0,
    all_touched=False,            # match centre-only convention used in training
)
# labels.shape == s2_reader.shape[-2:], dtype=uint8
```

This pairs with whatever `data_like` you read for X — `(s2_image, labels)` is now a coregistered training pair.

---

## 7. The relationship to `window_utils.exterior_pixel_coords`

`rasterize.py` could in principle be implemented purely on top of `window_utils.exterior_pixel_coords` (Chapter 4 §7) — convert polygon vertices to pixel coords, then fill via `cv2.fillPoly` or `skimage.draw.polygon`. The reason this module instead delegates to `rasterio.features.rasterize`:

1. **GDAL handles topology correctly.** Multipolygons with holes, self-intersecting rings, and degenerate edge cases all just work.
2. **`all_touched` is built in.** Reimplementing the centre-vs-touched distinction correctly across edge cases is annoying; GDAL has done it.
3. **Multi-geometry batching is natural.** `rasterio.features.rasterize` takes an iterable of `(geom, value)` pairs and burns them in one pass.

The `exterior_pixel_coords` function still has its uses (custom drawing, vertex-level analysis), but for **filling a polygon to a binary or class raster**, this module is the right tool.

---

## 8. Sharp edges

- **`all_touched=False` is the default.** For boolean masks where you want to read everything inside the AOI, you almost always want `all_touched=True`. The default is conservative because it matches GDAL.
- **Overlapping polygons: last-row wins.** Sort your GeoDataFrame in priority order if classes overlap.
- **`crs_geometry` is required when CRSs differ.** No "auto-detect from rows" — pass it explicitly. A common bug: WGS84 polygons against a UTM raster, no `crs_geometry`, output is empty because the polygon coordinates are nonsensical in UTM space.
- **Lines with `all_touched=False` are mostly empty.** Use `True` for `LineString` rasterization.
- **Dtype matters.** Default `uint8` clips at 255; for segmentation tasks with > 256 classes, pass `dtype=np.uint16` or your column's dtype is silently truncated.
- **Output is a `GeoTensor`, not a numpy array.** It comes with the reference raster's transform / CRS / fill_value. Ready for arithmetic with sibling `GeoTensor`s without coordinate fiddling.
- **The shape matches the spatial dims of `data_like`.** If `data_like.shape` is `(C, H, W)`, the rasterised output is `(H, W)` (2D). Multiply with `data_like` and broadcasting handles the band axis.

---

## 9. Connection to `geotoolz`

Two operators in [`geotoolz.md`](../plans/geotoolz/geotoolz.md) lean on this module:

- **`cloud.ApplyMask(mask)`** — when the mask is geometry-shaped (e.g., AOI polygon), `ApplyMask` rasterises before applying. The user passes a `Polygon`, the operator handles the burn.
- **`catalog_ops.WriteCOG(write_polygon=...)`** — clipping output to a polygon footprint at write time. Rasterise the polygon to a mask, multiply, write.

Beyond those, anywhere users pass a Shapely geometry and need a per-pixel decision, this module is the bridge.

Next chapter: [10_vectorize.md](10_vectorize.md) — the inverse: extracting polygon geometries from binary raster masks (segmentation outputs → GIS-ready vectors).
