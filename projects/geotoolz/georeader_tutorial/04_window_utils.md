---
title: window_utils
subject: georeader tutorial
subtitle: Pixel ↔ geographic coordinate math
short_title: Ch. 4 — Windows
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: tutorial, georeader, window
---

> **Module:** `georeader/window_utils.py` (1471 LOC, the second-densest in diagrams) **Role:** the math underneath everything else in the package.
> Windows, bounds, transforms, rounding, padding, polygon reprojection.
> Reading these utilities once is the cheapest way to understand why the higher-level `read.py` API is shaped the way it is.

---

## 1. What this module owns

Three responsibilities in one file:

1. **Windows ↔ bounds.** Convert between pixel-space rectangles (`rasterio.windows.Window`) and geographic rectangles `(minx, miny, maxx, maxy)` — both directions, both signs of CRS mismatch.
2. **Padding & rounding.** Grow / shrink / round windows so they align with pixel boundaries, fit a fixed CNN input size, or include all partial pixels intersected by a query.
3. **Polygon ↔ pixel.** Reproject Shapely geometries between CRSs and convert their vertices to pixel-coordinate paths (used by `rasterize` and `vectorize` downstream).

A single design decision sits underneath all of it: `PIXEL_PRECISION = 3` decimal places.
Coordinate transforms drift through reprojection round-trips, so the module deliberately tolerates ≤ 0.001-pixel error before deciding "this is or isn't an integer offset."

---

## 2. Window anatomy

A `rasterio.windows.Window` is a rectangle in *pixel* space — not geographic space.
The constructor argument order is the source of most bugs that touch this module.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    RASTERIO WINDOW ANATOMY                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    Full Raster (0,0 at top-left)                                        │
│    ↓                                                                     │
│    ┌────────────────────────────────────────────────────────┐           │
│    │ (0,0)                                        cols →    │           │
│    │     ┌────────────────────┐                             │           │
│    │     │← col_off →│        │                             │           │
│    │     │    (row_off, col_off) ← Window origin            │           │
│    │     │           ·─────────────────┐                    │           │
│  r │     │           │    WINDOW       │                    │           │
│  o │     │           │                 │ height             │           │
│  w │     │           │    width        │                    │           │
│  s │     │           └─────────────────┘                    │           │
│    │     │                                                  │           │
│  ↓ │     │                                                  │           │
│    └────────────────────────────────────────────────────────┘           │
│                                                                          │
│    Window = rasterio.windows.Window(col_off, row_off, width, height)    │
│                                     ───────  ───────  ─────  ──────     │
│                                     column   row      cols   rows       │
│                                     offset   offset                     │
└─────────────────────────────────────────────────────────────────────────┘
```

> **NOTE:** Window constructor order is `(col_off, row_off)` but most geospatial operations use `(row, col)` or `(y, x)` order.
> **Be careful!**

That note is in the source verbatim and it earns its caps.
Three places this trips people up:

- `np.zeros(shape)` is `(rows, cols)` = `(height, width)`.
  A `Window` is `(col_off, row_off, width, height)`.
  Mismatched axis order between the two has cost more than one afternoon.
- `pad_window(window, pad_size)` takes `(pad_rows, pad_cols)` — numpy order, **not** Window order.
  The docstring warns about this.
- `pad_window_to_size(window, size)` takes `(height, width)` — numpy order, **not** `(width, height)`.

The module commits to numpy order for axis-tuple arguments because that's what users coming from `np.pad` expect.
The Window constructor order is fixed by rasterio.

---

## 3. Window ↔ bounds — the core conversion

```text
┌─────────────────────────────────────────────────────────────────────────┐
│              WINDOW ↔ BOUNDS TRANSFORMATION                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    WINDOW (pixels)              AFFINE TRANSFORM           BOUNDS       │
│    ┌─────────────┐                    ║                ┌─────────────┐  │
│    │ col_off=100 │                    ║                │ minx=-122.5 │  │
│    │ row_off=200 │   ──────────────►  ║  ──────────►   │ miny=37.0   │  │
│    │ width=256   │   window_bounds()  ║                │ maxx=-122.0 │  │
│    │ height=256  │                    ║                │ maxy=37.5   │  │
│    └─────────────┘                    ║                └─────────────┘  │
│                                       ║                                  │
│                                       ║   Affine(a, b, c,               │
│                      ◄────────────────║          d, e, f)               │
│                      bounds_to_windows()                                │
│                                       ║                                  │
│    Affine Transform encodes:          ║                                  │
│    • Pixel resolution (a, e)          ║                                  │
│    • Origin coordinates (c, f)        ║                                  │
│    • Rotation/shear (b, d)            ║                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

Two named functions handle the round-trip:

- **`window_bounds(window, transform) → (minx, miny, maxx, maxy)`** — apply `transform` to the four window corners, return their axis-aligned bounding box.
  Fast, deterministic, no CRS involved.
- **`bounds_to_windows(data, bounds_dst, crs_dst) → list[Window]`** — the more interesting one.
  Reprojects `bounds_dst` from `crs_dst` into `data.crs`, then inverts the transform.
  Returns a **list** because a query that crosses the antimeridian splits into two windows in the source CRS. Most calls return a length-1 list; AOI bboxes that wrap longitude need length-2 handling.

The asymmetry — one direction is geometry math, the other is CRS-aware — is why this module is bigger than it might first seem.

---

## 4. Rounding: outer vs inner

Bounds rarely land on integer pixel boundaries.
You have to choose which way to round.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    WINDOW ROUNDING STRATEGIES                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Exact bounds (before rounding):                                       │
│   ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐                                │
│   │           Desired area             │                                │
│   └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘                                │
│                                                                          │
│   round_outer_window():                 round_inner_window():           │
│   ┌─────────────────────────────┐      ┌─────────────────────┐         │
│   │ ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │      │  ┌ ─ ─ ─ ─ ─ ─ ─ ┐ │         │
│   │ │                       │  │      │  │               │  │         │
│   │ │   Expands outward     │  │      │  │ Shrinks inward │  │         │
│   │ │   to include all      │  │      │  │ to only fully  │  │         │
│   │ │   partial pixels      │  │      │  │ covered pixels │  │         │
│   │ └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │      │  └ ─ ─ ─ ─ ─ ─ ─ ┘ │         │
│   └─────────────────────────────┘      └─────────────────────┘         │
│                                                                          │
│   Use outer when: You need all data that intersects the bounds          │
│   Use inner when: You need only data fully within the bounds            │
└─────────────────────────────────────────────────────────────────────────┘
```

Default in georeader: **outer**.
Reasons:

- For a read query, "give me everything that intersects this AOI" is what users mean 99% of the time.
- Combined with `boundless=True` reads, outer rounding plus padding gives you a guaranteed-shape result with no missing data inside the AOI.

`round_inner_window` shows up in two niche cases: when you're cropping to a strict mask (no partial-pixel contamination), and when you're computing the largest *fully covered* sub-window for tile-aligned reads.

Both functions tolerate `PIXEL_PRECISION = 3` slop before rounding — the example in the docstring: `99.9997 → 100`, `100.5 → unchanged`.
That tolerance is what lets reprojection round-trips not slowly drift the window by sub-pixel amounts each pass.

---

## 5. Padding for CNN tiles

Two related functions, both in the "make my tile a fixed size" job.

### `pad_window(window, pad_size=(pad_rows, pad_cols))`

Symmetric expansion on all four sides.
Used to add **context** around an inference tile.

```text
Original window:              Padded window (pad_size=(2, 3)):
┌─────────────┐               ┌─────────────────────┐
│             │               │← 3 cols → ← 3 cols →│
│   100×50    │     ───►      │↑         ↑          │
│   window    │               │2   106×54 window    │
│             │               │↓         ↓          │
└─────────────┘               │← 3 cols → ← 3 cols →│
                              └─────────────────────┘

Output: width = 100 + 2×3 = 106
        height = 50 + 2×2 = 54
        col_off = original - 3
        row_off = original - 2
```

The output window may have **negative offsets** — that's intentional, and you read it via `read_from_window(..., boundless=True)` to pad the off-edge region with `fill_value_default`.
(Chapter 3 §6.)

### `pad_window_to_size(window, size=(height, width))`

Re-centre to a target size.
Both expansion *and* contraction work — passing a size smaller than the window gives you a centre crop.

```text
Expansion (size > window):          Contraction (size < window):

┌───────────────────────┐          ┌───────────────────────┐
│     padded area       │          │                       │
│   ┌─────────────┐     │          │   ┌─────────────┐     │
│   │             │     │          │   │┌───────────┐│     │
│   │   original  │     │    ◄──   │   ││  center   ││     │
│   │    window   │     │          │   ││   crop    ││     │
│   └─────────────┘     │          │   │└───────────┘│     │
│     padded area       │          │   └─────────────┘     │
└───────────────────────┘          └───────────────────────┘

Symmetric expansion             Symmetric contraction
```

Odd differences favour the bottom/right edge (integer division).
When you need exact symmetry, use `pad_window` with explicit `pad_size`.

---

## 6. `figure_out_transform` — building output grids

When mosaicking, reprojecting, or designing an output grid, you usually have *some* of `{transform, bounds, resolution_dst}` and want georeader to derive the rest.

```text
┌────────────┬────────┬──────────────┬─────────────────────────────┐
│ transform  │ bounds │ resolution   │ Result                      │
├────────────┼────────┼──────────────┼─────────────────────────────┤
│ ✓          │ ✗      │ ✗            │ Return unchanged            │
│ ✓          │ ✗      │ ✓            │ Rescale resolution          │
│ ✓          │ ✓      │ ✗            │ Shift origin to bounds      │
│ ✓          │ ✓      │ ✓            │ Rescale + shift             │
│ ✗          │ ✓      │ ✓            │ Create new rectilinear      │
│ ✗          │ ✗      │ any          │ ERROR (need bounds)         │
│ ✗          │ ✓      │ ✗            │ ERROR (need resolution)     │
└────────────┴────────┴──────────────┴─────────────────────────────┘
```

Useful idioms:

- **"Same grid, coarser resolution":** `figure_out_transform(transform=src.transform, resolution_dst=20.0)` — keeps origin and orientation; rescales `a` and `e`.
- **"New AOI, same resolution as source":** `figure_out_transform(transform=src.transform, bounds=aoi)` — keeps `a, b, d, e`; shifts `c, f` to the AOI's upper-left.
- **"Fresh rectilinear from scratch":** `figure_out_transform(bounds=aoi, resolution_dst=10.0)` — north-up, square pixels, `b = d = 0`.

The "ERROR" rows are validation: you can't conjure an origin without bounds, and you can't size a grid without resolution.

This function is the seam between `read.read_from_bounds` (which calls it) and the higher-level mosaic / reprojection workflows in `read.py` ([Chapter 5](05_read.md)).

---

## 7. Polygon ↔ pixel — exterior coordinates

`window_polygon` is the geometry-aware sibling of `window_bounds`.
Returns a Shapely `Polygon` rather than a 4-tuple — important when the transform has rotation/shear (`b ≠ 0` or `d ≠ 0`), where the bounding box and the actual footprint differ.

```text
window_surrounding=False (default):     window_surrounding=True:
Polygon includes full pixels             Polygon passes through pixel centers

┌───┬───┬───┬───┐                       ┌───┬───┬───┬───┐
│ P │ P │ P │ P │ ◄─ Polygon            │ · │ · │ · │ · │
├───┼───┼───┼───┤    edges              ├───○───○───○───┤
│ P │ P │ P │ P │    touch              │ · │ · │ · │ · │ ◄─ Polygon
├───┼───┼───┼───┤    pixel              ├───○───○───○───┤    passes
│ P │ P │ P │ P │    boundaries         │ · │ · │ · │ · │    through ○
└───┴───┴───┴───┘                       └───┴───┴───┴───┘
```

The `window_surrounding` flag picks between two valid pixel-footprint conventions:

- **`False` (default)** — the polygon traces the *outside* of the pixel grid.
  Pixels are areas with non-zero extent.
  Use when you're computing intersections with vector geometries that should agree with `window_bounds`.
- **`True`** — the polygon vertices sit at *pixel centres*.
  Pixels are samples (points) on a regular grid.
  Use when you're matching against point clouds or building irregular-grid representations.

The default matches GIS convention; the alternative is needed when interfacing with point-sampling tools (some scipy/skimage routines treat pixels as samples by default).

Companion functions:

- **`polygon_to_crs(polygon, crs_polygon, dst_crs)`** — reproject a Shapely geometry.
  Works on `Polygon` and `MultiPolygon`; preserves geometry type.
- **`exterior_pixel_coords(transform, crs, polygon, crs_polygon=None)`** — reproject `polygon` into the raster's CRS, then convert each ring of vertices to pixel coordinates.
  Returns `List[List[(col, row)]]` (one inner list per polygon part).
  Used by `rasterize` ([Chapter 9](09_rasterize.md)) to draw filled regions.

---

## 8. The "boundless reading" mechanics — `get_slice_pad`

This is the function that makes `read_from_window(boundless=True)` actually work, and it deserves a callout because the behaviour seems magical until you see it.

Given two windows:

- `window_data` — the actual data extent (e.g., `(0, 0, W, H)` for a file)
- `window_read` — the (possibly out-of-bounds) read request

…it returns:

- A `dict[str, slice]` to extract the part of `window_read` that **does** overlap `window_data`
- A `dict[str, (pad_left, pad_right)]` to pad the result so it ends up the requested shape

The reader then reads the slice from disk and applies `np.pad` (or its own `fill_value_default`-aware equivalent) to produce a full-size array.
CNN inference at scene edges relies entirely on this — every chip comes back the requested shape, with off-edge regions filled with nodata.

Source: [window_utils.py:599](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/window_utils.py#L599).

---

## 9. Tiling-and-stitching: `slice_save_for_pred`

The companion to `pad_window` for inference pipelines: when you've read a tile *with* padded context, run a CNN, and now want to write only the centre back into a global output, this function tells you which slice to extract from the prediction.

```text
1. Read overlapping tiles with padding to avoid edge artifacts
2. Run CNN inference on padded tiles
3. Extract only the center region (removing padding) for final output
4. Stitch extracted regions together to form complete prediction
```

The reference is Huang et al. (2018) — the standard tile-and-stitch recipe.
Used by ml4floods and similar segmentation pipelines built on georeader.

Source: [window_utils.py:1256](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/window_utils.py#L1256).

---

## 10. Function reference

Grouped by purpose.
Every function takes / returns `rasterio.Affine`, `rasterio.windows.Window`, or shapely `Polygon`/`MultiPolygon` — no GeoTensor/Reader anywhere in this module.

**Padding & rounding**
- `pad_window(window, pad_size=(rows, cols))` — symmetric expansion
- `pad_window_to_size(window, size=(h, w))` — re-centre to fixed size
- `round_outer_window(window, precision=3)` — expand to integer pixels
- `round_inner_window(window, precision=3)` — shrink to integer pixels
- `get_slice_pad(window_data, window_read)` — boundless-read decomposition
- `pad_list_numpy(pad_width)` — convert dim-named pad dict to `np.pad` arg list
- `slice_save_for_pred(w_read, w_write)` — tile-and-stitch slice extractor

**Windows ↔ bounds**
- `window_bounds(window, transform)` — pixels → geo bbox
- `bounds_to_windows(data, bounds_dst, crs_dst)` — geo bbox → windows (handles antimeridian)
- `normalize_bounds(bounds, margin_add_if_equal=0.0005)` — fix inverted / degenerate bboxes

**Transforms**
- `figure_out_transform(transform=None, bounds=None, resolution_dst=None)` — flexible factory
- `transform_to_resolution_dst(transform, resolution_dst)` — rescale a transform's pixel size
- `compare_crs(a, b)` — CRS equality across EPSG int / string / WKT / pyproj
- `res(transform)` — `(|a|, |e|)` from a transform

**Polygons**
- `window_polygon(window, transform, window_surrounding=False)` — pixel rect → Shapely polygon
- `polygon_to_crs(polygon, src_crs, dst_crs)` — reproject geometry
- `exterior_pixel_coords(transform, crs, polygon, crs_polygon=None)` — polygon vertices → pixel coords
- `apply_transform_to_pol(pol, transform)` — apply affine to polygon vertices
- `get_valid_mask(...)` — mask of valid (in-polygon) pixels

**Convenience indexes**
- `row_end(window) → int` — `row_off + height`
- `col_end(window) → int` — `col_off + width`

(Functions starting with `_` like `_is_exact_round` are internal precision helpers and aren't part of the public API.)

---

## 11. Sharp edges

- **Window order vs numpy order.** `Window(col_off, row_off, width, height)` — `(x, y, w, h)` in screen-coordinate parlance.
  `pad_window(pad_size)` is `(rows, cols)` — `(y, x)` in numpy parlance.
  `pad_window_to_size(size)` is `(height, width)`.
  The module deliberately uses numpy order for axis-tuple args; the rasterio Window constructor is fixed.
- **`PIXEL_PRECISION = 3` is module-global.** If you have a workflow with sub-millipixel meaningful precision (you don't, but if), you'd need to thread `precision=` through every call.
  In practice, 3 is right.
- **Antimeridian-crossing AOIs return *two* windows.** `bounds_to_windows` returns a list.
  Code that does `windows[0]` will silently lose half the AOI when an Asia/Pacific bbox is queried.
- **Rotated transforms break `window_bounds`.** When `b ≠ 0` or `d ≠ 0`, the four-corner bounding box overestimates the data footprint.
  Use `window_polygon` and intersect against actual geometry.
- **`figure_out_transform` errors are validation, not bugs.** "ERROR (need bounds)" rows in the table are deliberate — there's no sensible default.
- **`exterior_pixel_coords` returns col-row, not row-col.** Matches Shapely (x, y) convention.
  Don't pass it directly to `np.zeros[...]`.

---

## 12. Why this module matters for `geotoolz`

Three concrete things `geotoolz.sampling` and `geotoolz.inference` will lean on:

1. **`pad_window` + `slice_save_for_pred`** is the entire ApplyToChips machinery.
   Read with context, predict, save the centre, stitch.
2. **`bounds_to_windows`** is what a `BoundingBoxSampler` (TorchGeo-style) calls under the hood when chips are specified in geographic coords rather than pixel coords.
3. **`figure_out_transform`** is the "design my output grid" function.
   Every reprojection-aware operator (mosaicking, ensemble averaging across scenes) needs it.

You can build the whole `geotoolz.sampling` module without touching anything in `georeader` *except* this file plus `read.py`.
That's a clean cut for the operator layer.

Next chapter: [05_read.md](05_read.md) — the high-level reading API (`read_from_bounds`, `read_from_polygon`, `read_from_center_coords`, reprojection / resampling).
The single densest module in the package by diagram count, and the one most users touch first.
