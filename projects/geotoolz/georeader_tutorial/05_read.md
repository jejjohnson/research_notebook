# Chapter 5 — `read.py`: the high-level reading API

> **Module:** `georeader/read.py` (1967 LOC, the densest module in the package — 123 box-drawing characters across the docstring)
> **Role:** the public face of georeader. Most users start here. Six "specify the AOI in the form most natural to your problem" entry points, plus reprojection / resampling / grid-matching.

---

## 1. The mental model

You have:

- A **source** that satisfies the `GeoData` protocol (a `RasterioReader`, a `GeoTensor`, or any reader you build) — see [Chapter 2](02_abstract_reader.md).
- An **AOI** described in some natural form — a polygon, a bbox, a centre coordinate, a pixel window, a web tile, or another raster's grid.
- A **destination CRS / resolution / shape** — possibly the same as the source (cheap), possibly different (full reprojection).

`read.py` is the dispatcher: it converts whichever AOI form you supplied into a window in the source's pixel space, asks the source for that window (lazily), reprojects/resamples if needed, and hands you a `GeoTensor`.

The whole module is built on `window_utils` ([Chapter 4](04_window_utils.md)) and on the `GeoData.read_from_window` / `GeoData.load` interface from [Chapter 2](02_abstract_reader.md).

---

## 2. Six ways to specify "what region do I want?"

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    READING WORKFLOW: AREA SPECIFICATION                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input Specification          Function                     Output       │
│  ────────────────────         ─────────────────────        ──────────   │
│                                                                          │
│  Polygon (geometry)     ───►  read_from_polygon()    ───►  GeoTensor   │
│                                                                          │
│  Bounds (minx,miny,     ───►  read_from_bounds()     ───►  GeoTensor   │
│          maxx,maxy)                                                      │
│                                                                          │
│  Center + Shape         ───►  read_from_center_coords() ─► GeoTensor   │
│  (x, y) + (H, W)                                                         │
│                                                                          │
│  Window (row_off,       ───►  read_from_window()     ───►  GeoTensor   │
│          col_off, H, W)                                                  │
│                                                                          │
│  Web Tile (x, y, z)     ───►  read_from_tile()       ───►  GeoTensor   │
│                                                                          │
│  Match another raster   ───►  read_reproject_like()  ───►  GeoTensor   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Quick guide to which one to reach for:

| Your situation | Use |
|---|---|
| You have a Shapely polygon | `read_from_polygon` |
| You have a bbox tuple | `read_from_bounds` |
| You have a click-point + a chip size | `read_from_center_coords` |
| You're already in pixel coordinates | `read_from_window` |
| You're rendering a slippy-map tile | `read_from_tile` |
| You need to match another raster's grid exactly | `read_reproject_like` |

Each `read_from_*` has a sibling `window_from_*` that returns just the `Window` without reading bytes — used internally and useful when you want to inspect "what pixels will I read?" before committing.

---

## 3. Window vs bounds — the semantic split

```text
┌─────────────────────────────────────────────────────────────────────────┐
│              WINDOW (PIXELS) vs BOUNDS (GEOGRAPHIC COORDINATES)          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  WINDOW (pixel space)                BOUNDS (CRS units)                 │
│  ─────────────────────               ──────────────────                 │
│                                                                          │
│  (col_off, row_off)                  (minx, maxy)  ← upper-left         │
│       ↓                                   ↓                              │
│    ┌──────────────┐                  ┌──────────────┐                   │
│    │ width pixels │                  │              │ geographic        │
│    │              │   ◄═══════►      │              │ extent in         │
│    │ height pixels│    transform     │              │ CRS units         │
│    └──────────────┘                  └──────────────┘                   │
│                                           ↑                              │
│                                      (maxx, miny)  ← lower-right        │
│                                                                          │
│  Window: rasterio.windows.Window(col_off, row_off, width, height)       │
│  Bounds: (minx, miny, maxx, maxy) - order matches shapely/rasterio      │
│                                                                          │
│  Conversion:                                                             │
│    bounds = window_utils.window_bounds(window, transform)               │
│    window = window_from_bounds(data, bounds, crs_bounds)                │
└─────────────────────────────────────────────────────────────────────────┘
```

The split matters because:

- **Windows are CRS-free.** Once you have a window in the source's pixel space, no further coordinate math is needed. `read_from_window` is the cheapest entry point.
- **Bounds carry a CRS.** `read_from_bounds(data, bounds, crs_bounds)` *always* takes a separate `crs_bounds` arg — even if it's the same as the data — because there's no way to read the CRS off a tuple. Forgetting `crs_bounds` is a common bug.

The `window_from_*` siblings are the bridge: they normalise any AOI specification into a pixel window in the source's CRS, with antimeridian handling and outer-rounding applied. The `read_from_*` functions then call `data.read_from_window(window, boundless=True).load()`.

---

## 4. Reprojection and resampling

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                     REPROJECTION WORKFLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Source CRS (e.g., EPSG:4326)         Target CRS (e.g., EPSG:32633)    │
│  ┌─────────────────────┐              ┌─────────────────────┐           │
│  │  ╱╲    ╱╲    ╱╲    │              │ □ □ □ □ □ □ □ □ □ │           │
│  │ ╱  ╲  ╱  ╲  ╱  ╲   │    ═════►    │ □ □ □ □ □ □ □ □ □ │           │
│  │╱    ╲╱    ╲╱    ╲  │   Reproject  │ □ □ □ □ □ □ □ □ □ │           │
│  │ Irregular grid     │   + Resample │ Regular UTM grid   │           │
│  └─────────────────────┘              └─────────────────────┘           │
│                                                                          │
│  Resampling Methods (rasterio.warp.Resampling):                         │
│  ┌────────────────┬────────────────────────────────────────────────┐    │
│  │ Method         │ Best for                                       │    │
│  ├────────────────┼────────────────────────────────────────────────┤    │
│  │ nearest        │ Categorical data, masks, classification        │    │
│  │ bilinear       │ Continuous data, fast                          │    │
│  │ cubic          │ Continuous data, smooth                        │    │
│  │ cubic_spline   │ Continuous data, very smooth (DEFAULT)         │    │
│  │ lanczos        │ Downsampling, sharp edges                      │    │
│  │ average        │ Downsampling, area-weighted mean               │    │
│  │ mode           │ Downsampling categorical data                  │    │
│  └────────────────┴────────────────────────────────────────────────┘    │
│                                                                          │
│  Anti-aliasing: Automatic Gaussian blur before downsampling to          │
│                 prevent aliasing artifacts. Controlled by:              │
│                 - anti_aliasing=True (default in resize)                │
│                 - anti_aliasing_sigma (auto-calculated or manual)       │
└─────────────────────────────────────────────────────────────────────────┘
```

Three reprojection-flavoured entry points:

- **`read_to_crs(data, dst_crs, resampling=...)`** — simplest case: same bounds, new CRS. Used for visualisation flips (UTM → Web Mercator).
- **`read_reproject(data, dst_crs=None, bounds=None, resolution_dst_crs=None, dst_transform=None, ...)`** — the workhorse. Choose any combination of CRS / bounds / resolution / explicit transform; the function fills in the rest using `window_utils.figure_out_transform`.
- **`read_reproject_like(data_in, data_like, ...)`** — match another raster's grid exactly. Reads `data_like.transform`, `data_like.crs`, `data_like.shape` and builds the matching read. Standard pattern for stacking heterogeneous sources onto a common grid.

The default resampling is `cubic_spline`. The advice in the table is good: **switch to `nearest` for masks and class labels**, otherwise you'll get fractional class IDs after resampling.

The `resize` function is the resolution-only sibling — same CRS, same origin, new pixel size — with built-in anti-aliasing pre-blur for downsampling.

---

## 5. Boundless reading

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    BOUNDLESS READING (boundless=True)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Requested Window              Result with boundless=True               │
│  ─────────────────              ─────────────────────────               │
│                                                                          │
│       ┌─────────────┐           ┌─────────────┐                         │
│       │ fill │ data │           │  0  │ data │   fill_value_default    │
│       │ ─────┼───── │           │ ────┼───── │   fills out-of-bounds   │
│       │ fill │ data │           │  0  │ data │   pixels                │
│       └─────────────┘           └─────────────┘                         │
│            ↑                                                             │
│     Request extends                                                      │
│     beyond raster bounds                                                 │
│                                                                          │
│  boundless=False: Raises error or clips to valid region                 │
│  boundless=True:  Pads with fill_value_default (default behavior)       │
└─────────────────────────────────────────────────────────────────────────┘
```

`boundless=True` is the default everywhere in `read.py`. This is deliberate: the moment you compose tiled inference with edge tiles, you *want* fixed-size outputs and don't want to special-case "this tile is at the edge."

`boundless=False` is the right choice when you specifically want to detect off-edge requests — e.g., "did my AOI actually fall inside the scene?"

---

## 6. The eight-step `read_reproject` walkthrough

The implementation of `read_reproject` ([read.py:1348](../../../../tmp/georeader_src/georeader/read.py#L1348)) is annotated with eight numbered banner-comments inside the function body. They're not a single ASCII diagram, but they're the clearest map of how reprojection actually works in this package — worth preserving.

```text
─────────────────────────────────────────────────────────────────────────
STEP 1: DETERMINE OUTPUT TRANSFORM
─────────────────────────────────────────────────────────────────────────
The output transform defines the mapping from pixel coordinates to
geographic coordinates in the destination CRS. It can be:
- Provided directly (dst_transform)
- Computed from bounds + resolution
- Computed from bounds with inferred resolution

Shape tracking:
  Input:  named_shape = {'band': C, 'y': H_in, 'x': W_in}
  Output: will have {'band': C, 'y': H_out, 'x': W_out}
─────────────────────────────────────────────────────────────────────────

STEP 2: COMPUTE OUTPUT DIMENSIONS
─────────────────────────────────────────────────────────────────────────
The output window defines pixel dimensions (W_out, H_out). Either:
- Provided directly (window_out)
- Computed from bounds in dst_crs coordinate units

Example: bounds=(0, 0, 1000, 1000) with 10m resolution → (100, 100) pixels
─────────────────────────────────────────────────────────────────────────

STEP 3: CHECK FOR NO-OP OPTIMIZATION
─────────────────────────────────────────────────────────────────────────
If source and destination have:
  - Same CRS
  - Same pixel size (transform.a and transform.e)
  - Grid-aligned origins (integer pixel offset)
Then we can skip reprojection entirely and use a simple window read.
This is ~10-100x faster for aligned data.
─────────────────────────────────────────────────────────────────────────

STEP 4: HANDLE DATA TYPES
─────────────────────────────────────────────────────────────────────────
Boolean arrays need special handling:
  bool → float32 → interpolate → threshold(0.5) → bool
This prevents interpolation artifacts in mask data while still
allowing smooth boundaries (anti-aliasing effect).
─────────────────────────────────────────────────────────────────────────

STEP 5: ALLOCATE OUTPUT ARRAY
─────────────────────────────────────────────────────────────────────────
Pre-allocate with nodata fill. Shape is built by replacing x,y dims
with the new window dimensions while preserving other dims (band, time).

Example shape transformation:
  Input:  (4, 1000, 1000)  → 4 bands, 1000×1000 pixels
  Output: (4, 500, 600)    → 4 bands, 500×600 pixels (new extent)
─────────────────────────────────────────────────────────────────────────

STEP 6: CHECK INTERSECTION
─────────────────────────────────────────────────────────────────────────
If the requested output region doesn't overlap the input data at all,
return early with the nodata-filled array. Saves computation.
─────────────────────────────────────────────────────────────────────────

STEP 7: LOAD SOURCE DATA
─────────────────────────────────────────────────────────────────────────
For lazy data (xarray/dask), read only the region that will contribute
to the output, plus a 3-pixel buffer for interpolation edge handling.
The buffer prevents edge artifacts from bilinear/cubic resampling.
─────────────────────────────────────────────────────────────────────────

STEP 8: ITERATE OVER NON-SPATIAL DIMENSIONS
─────────────────────────────────────────────────────────────────────────
rasterio.warp.reproject operates on 2D arrays. For multi-band or
multi-temporal data, we iterate over all (time, band) combinations.

Example: shape (4, 3, H, W) with dims ('time', 'band', 'y', 'x')
  → iterates over 4×3=12 slices: (0,0), (0,1), (0,2), (1,0), ...
─────────────────────────────────────────────────────────────────────────

CORE REPROJECTION: rasterio.warp.reproject
─────────────────────────────────────────────────────────────────────────
This is where the actual coordinate transformation happens:
1. For each output pixel, compute its geographic coordinates
2. Transform those coords from dst_crs to src_crs
3. Sample the input raster at those locations using resampling
4. Write the result to the output array
─────────────────────────────────────────────────────────────────────────
```

Five things worth highlighting from this walkthrough:

1. **Step 3 is the fast path.** Same CRS + same pixel size + grid-aligned origins → reprojection collapses to a window read. ~10–100× speedup. This is why operators that read coregistered data should reach for `read_from_bounds` (which dispatches via this path) instead of `read_reproject` blindly.
2. **Step 6 short-circuits non-intersecting requests.** Asking for an AOI that doesn't overlap the scene returns a nodata-filled array, not an error. Composing tiled inference across catalogs of partially-overlapping scenes works without try/except.
3. **Step 7 reads with a 3-pixel buffer.** That's the anti-aliasing-of-resampling-kernels reason. Bilinear / cubic kernels need neighbouring pixels at the destination boundary; without the buffer you get a thin nodata stripe along edges.
4. **Step 4's bool dance.** Reprojecting a bool mask via `nearest` is correct semantically but loses smooth boundaries. The float32 → interpolate → threshold trick gives anti-aliased mask edges while preserving bool dtype on output.
5. **Step 8 iterates by non-spatial dim.** A `(T, C, H, W)` GeoTensor reprojects band-by-band, time-by-time. The outer loop is in Python; the inner reprojection is in `rasterio.warp.reproject` (GDAL). For very large stacks this can be a bottleneck — one of the things a JAX-batched reprojection in `geotoolz` could improve later.

---

## 7. Function reference

**Window factories (no I/O)**
- `window_from_polygon(data_in, polygon, crs_polygon=None, ...)`
- `window_from_bounds(data_in, bounds, crs_bounds, ...)`
- `window_from_center_coords(data_in, center, shape, crs_center_coords=None, ...)`
- `window_from_tile(data_in, x, y, z)` — Web Mercator XYZ tile

**Read functions (return `GeoTensor`)**
- `read_from_window(data_in, window, boundless=True, return_only_data=False, trigger_load=True)`
- `read_from_bounds(data_in, bounds, crs_bounds=None, ...)`
- `read_from_polygon(data_in, polygon, crs_polygon=None, pad_add=(0, 0), ...)`
- `read_from_center_coords(data_in, center, shape, crs_center_coords=None, ...)`
- `read_from_tile(data, x, y, z, ...)` — slippy-map tile

**Reprojection / resampling**
- `read_to_crs(data, dst_crs, resampling=...)` — same bounds, new CRS
- `read_reproject(data_in, dst_crs=None, bounds=None, resolution_dst_crs=None, dst_transform=None, window_out=None, dst_nodata=None, dtype_dst=None, resampling=cubic_spline)` — the workhorse
- `read_reproject_like(data_in, data_like, ...)` — match another raster's grid
- `resize(data_in, output_shape=..., resolution_dst=..., anti_aliasing=True, anti_aliasing_sigma=None, ...)` — same CRS / origin, new resolution
- `apply_anti_aliasing(data_in, sigma=...)` — Gaussian pre-blur (used by `resize` internally)
- `calculate_transform_window(...)` — utility for computing matched windows

**Lower-level utilities**
- `read_rpcs(input_npy, ...)` — handle Rational Polynomial Coefficients (sensor-model georeferencing for raw imagery)
- `_round_all`, `_transform_from_crs` — internal helpers; not public API

---

## 8. The `pad_add` argument on `read_from_polygon`

`read_from_polygon` accepts `pad_add=(rows, cols)` — a tuple of pixel padding to grow the read window beyond the polygon's bounding box. This isn't documented as a feature of the others, but it shows up in two contexts:

- **Reprojection edge handling.** Internal calls in `read_reproject` use `pad_add=(3, 3)` so that the resampling kernel has neighbours at the destination boundary (Step 7 above).
- **CNN context windows.** For a segmentation model that needs context around the AOI, `pad_add=(32, 32)` reads a buffer; you pass through to inference, then crop the centre.

For other entry points, achieve the same effect by `pad_window`-ing the result of `window_from_*` and calling `read_from_window` directly.

---

## 9. Sharp edges

- **`crs_bounds` / `crs_polygon` / `crs_center_coords` are required when they differ from the source.** If omitted, the function assumes the AOI is already in the source's CRS — which is silently wrong if it isn't. Pass the CRS explicitly always.
- **Default resampling is `cubic_spline`.** Wrong for masks. Pass `resampling=Resampling.nearest` for class labels / cloud masks.
- **`read_reproject` allocates the output upfront (Step 5).** A misconfigured huge output shape (e.g., units mismatch in `resolution_dst_crs`) tries to allocate a TB-scale array. If allocation seems suspicious, print the inferred output shape before reading.
- **The fast path (Step 3) requires *exact* integer pixel offsets.** A fractional offset of 0.0003 still triggers the full reproject path. The `_is_exact_round` precision check matches `PIXEL_PRECISION = 3` from `window_utils`.
- **`read_from_polygon` reads the polygon's *bounding box*, then masks.** It doesn't do per-pixel polygon membership at the read step — the read returns a rectangle. For pixel-level masking, follow with `rasterize` ([Chapter 9](09_rasterize.md)) or use `data.footprint().intersects(polygon)` checks.
- **Antimeridian-crossing AOIs split into two reads.** Inherited from `bounds_to_windows` ([Chapter 4 §3](04_window_utils.md)); the splitting is automatic but it's two HTTP requests on cloud data.
- **`read_from_tile` returns a Web Mercator tile.** If your source isn't in Web Mercator, this triggers a full reprojection per tile — fine for occasional rendering, expensive for a tile-loop.

---

## 10. Why this module is the operator surface for `geotoolz.sampling`

Three functions from this module are essentially the entire `geotoolz.sampling.GridSampler` / `RandomSampler` / `Stitch` machinery:

- `window_from_bounds` — convert a sampler's geographic chip query to a source window.
- `read_from_window(boundless=True)` — fixed-shape chip retrieval.
- `read_reproject_like` — when sampling from heterogeneous sources, match all chips to a common grid.

A `GridSampler` is mostly a generator of `(x_min, y_min, x_max, y_max)` tuples in the source CRS; the `chip_op` calls `read_from_bounds` per tuple. Add `pad_add=(context, context)` for inference with context, then `slice_save_for_pred` ([Chapter 4 §9](04_window_utils.md)) to crop back when stitching. The whole loop is ~50 lines that wrap functions already in this module.

Next chapter: [06_slices.md](06_slices.md) — tiling strategies (overlap, stride, chunked) for memory-efficient processing of datasets that don't fit in RAM.
