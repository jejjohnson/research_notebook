# Ch. 1 — `geotensor`

> **Module:** `georeader/geotensor.py` (2532 LOC)
> **Branch:** `feature/geotensor_npapi` — this module *is* the headline change of that branch.
> **Role:** the package's central data structure. Every reader in georeader produces a `GeoTensor`; every writer consumes one; every operator in `geotoolz` will be `GeoTensor → GeoTensor`.

---

## 1. What `GeoTensor` is, in one sentence

A `numpy.ndarray` **subclass** that carries `(transform, crs, fill_value_default, attrs)` alongside the buffer, and propagates that metadata through ufuncs, slicing, copies, and views — so `gt + 1`, `np.sqrt(gt)`, and `gt[:, 100:200, 100:200]` all return correctly georeferenced `GeoTensor`s with **zero** boilerplate at the call site.

This is a fundamentally different design from the previous `GeoTensor` (which was a wrapper holding a `.values` array). On this branch:

- `gt` *is* an ndarray. You can pass it to anything that takes one.
- `gt.values` exists as a back-compat property returning `self.view(np.ndarray)`.
- `__array_ufunc__` intercepts ufuncs to re-wrap the result with the original metadata.
- `__array_finalize__` propagates metadata across views/copies/slices.
- Spatial slicing rewrites `transform` so georeferencing stays correct.

---

## 2. Memory layout and dim conventions

`GeoTensor` supports 2D, 3D, and 4D arrays. **The last two dimensions are always spatial** — `(..., y, x)` — and are the only dimensions whose semantics the class is opinionated about.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                     GEOTENSOR DIMENSION CONVENTIONS                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  2D: (H, W)           Single-band raster (e.g., DEM, mask)              │
│                       shape = (height, width)                           │
│                                                                         │
│  3D: (C, H, W)        Multi-band raster (e.g., RGB, multispectral)      │
│                       shape = (channels, height, width)                 │
│                                                                         │
│  4D: (T, C, H, W)     Time-series cube (e.g., satellite time stack)     │
│                       shape = (time, channels, height, width)           │
│                                                                         │
│  Dimension names:  dims = ("y", "x") or ("band", "y", "x") or           │
│                          ("time", "band", "y", "x")                     │
│                                                                         │
│  Note: "y" decreases downward (row index), "x" increases rightward      │
└─────────────────────────────────────────────────────────────────────────┘
```

`dims` is computed from the array's `ndim` — it isn't free-form like xarray's. The constructor raises `ValueError` for `ndim < 2` or `ndim > 4`.

---

## 3. The affine transform — pixel ↔ geographic coordinates

`GeoTensor.transform` is a `rasterio.Affine` (6-tuple) that maps `(col, row)` pixel coordinates to `(x, y)` geographic coordinates in the CRS units (typically metres for projected CRSs, degrees for `EPSG:4326`).

```text
┌─────────────────────────────────────────────────────────────────────────┐
│              AFFINE TRANSFORM: PIXEL ↔ GEOGRAPHIC COORDINATES           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Pixel Space (row, col)              Geographic Space (x, y)            │
│  ┌───┬───┬───┬───┬───┐              ┌─────────────────────────┐         │
│  │0,0│0,1│0,2│0,3│...│              │OriginX ──────────────►  │         │
│  ├───┼───┼───┼───┼───┤     ═══►     │OriginY                  │         │
│  │1,0│1,1│   │   │   │   Transform  │   │                     │         │
│  ├───┼───┼───┼───┼───┤              │   ▼                     │         │
│  │2,0│   │   │   │   │              │        (CRS units)      │         │
│  └───┴───┴───┴───┴───┘              └─────────────────────────┘         │
│                                                                         │
│  Affine Transform:  | a  b  c |    x_geo = a * col + b * row + c        │
│                     | d  e  f |    y_geo = d * col + e * row + f        │
│                                                                         │
│  Typical (north-up):  a = pixel_width (positive)                        │
│                       e = -pixel_height (negative, y decreases down)    │
│                       c = origin_x (upper-left corner)                  │
│                       f = origin_y (upper-left corner)                  │
│                       b, d = 0 (no rotation/shear)                      │
│                                                                         │
│  Resolution:  res = (|a|, |e|) in CRS units (e.g., meters, degrees)     │
└─────────────────────────────────────────────────────────────────────────┘
```

A few things this diagram is doing implicitly:

- **The y-axis is flipped.** Image-space convention is row 0 at the top; geographic-space convention is y increasing northward. The negative `e` reconciles them.
- **`b` and `d` are nonzero only for rotated/sheared rasters.** Most readers in georeader assume north-up; some readers (PRISMA, EnMAP, MODIS swath) deliberately don't and route through `griddata` instead — see [Chapter 7](07_griddata.md).
- **`bounds` is derived, not stored.** `gt.bounds` returns `(minx, miny, maxx, maxy)` computed from `transform` and `shape` on every access. Likewise `gt.res = (|a|, |e|)`.

---

## 4. NumPy operations preserve geospatial info

Because `GeoTensor` is a true ndarray subclass, **all numpy operations work** — and the spatial metadata rides along automatically:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    NUMPY OPERATIONS PRESERVE GEOSPATIAL INFO            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Arithmetic:     gt + 1, gt * 2, gt1 + gt2 (same extent required)       │
│  Comparison:     gt > 0, gt == other                                    │
│  Ufuncs:         np.sqrt(gt), np.log(gt), np.clip(gt, 0, 1)             │
│  Aggregation:    gt.mean(), gt.sum(axis=0)  # Returns scalar/array      │
│  Slicing:        gt[0], gt[:, 10:20, 10:20]  # Updates transform        │
│                                                                         │
│  Important: Operations that change spatial dimensions (slicing)         │
│  automatically update the transform to maintain georeferencing!         │
│                                                                         │
│  Example:                                                               │
│    gt_subset = gt[:, 100:200, 100:200]                                  │
│    # gt_subset.transform origin shifted by (100*res_x, 100*res_y)       │
└─────────────────────────────────────────────────────────────────────────┘
```

The mechanics behind this — three numpy hooks doing the heavy lifting:

| Hook | Where | What it does |
|---|---|---|
| `__array_finalize__` | called on every new view/copy | Copies `transform`, `crs`, `fill_value_default`, `attrs` from the parent. Default behaviour: passthrough. |
| `__array_ufunc__` | called on every ufunc (`np.add`, `np.sqrt`, …) | Strips inputs to plain ndarrays, applies the ufunc, re-wraps the output as a `GeoTensor` with metadata copied from the first `GeoTensor` input. |
| `__getitem__` | called on `gt[...]` | Standard ndarray slice, then **rewrites `transform`** if the spatial axes (last two) were sliced — origin shifts by `start * res`, resolution rescales by `step`. |

The aggregation case (`.mean()`, `.sum(axis=0)`) returns a scalar or lower-dim array; a *spatial* reduction usually returns a plain ndarray because the result no longer has a meaningful transform. (See `_preserved_spatial` at [geotensor.py:385](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/geotensor.py#L385).)

---

## 5. Constructor and core attributes

```python
GeoTensor(values, transform, crs, fill_value_default=0, attrs=None)
```

| Attribute | Type | Notes |
|---|---|---|
| `values` | `np.ndarray` view | back-compat alias for `self.view(np.ndarray)` |
| `transform` | `rasterio.Affine` | pixel → geographic |
| `crs` | EPSG code / WKT / `pyproj.CRS` | passed through to rasterio |
| `fill_value_default` | scalar or `None` | what out-of-bounds reads pad with |
| `attrs` | `dict` | freeform metadata bag (sensor info, acquisition time…) |
| `bounds` | `(minx, miny, maxx, maxy)` | derived from transform + shape |
| `res` | `(x_res, y_res)` | `(|a|, |e|)` from transform |
| `height`, `width`, `count` | `int` | last dim, second-to-last dim, third-from-last (or 1 for 2D) |
| `dims` | `tuple[str, ...]` | `("y","x")` / `("band","y","x")` / `("time","band","y","x")` |

Round-trip helpers: `gt.to_json()` / `GeoTensor.from_json(d)` — useful for catalogs and for Hydra round-trips downstream.

---

## 6. xarray-style indexing: `isel`

`__getitem__` is the numpy-positional path. `isel` is the **named-dimension** path — closer to xarray ergonomics and the recommended API for geotoolz operators.

```text
Standard dimension names for 3D GeoTensor (C, H, W):
┌──────────┬────────────────┬────────────────────┐
│ Dim name │ Array axis     │ Description        │
├──────────┼────────────────┼────────────────────┤
│ "band"   │ axis 0 (C)     │ Spectral bands     │
│ "y"      │ axis 1 (H)     │ Rows (north-south) │
│ "x"      │ axis 2 (W)     │ Cols (east-west)   │
└──────────┴────────────────┴────────────────────┘
```

`isel` accepts a dict mapping dim names to `slice`, `list[int]`, or `int`. Spatial dims (`"x"`, `"y"`) only accept slices — fancy indexing on spatial axes would invalidate the affine transform. Examples (per the docstring):

- `gt.isel({"x": slice(10, 110)})` — columns 10..109
- `gt.isel({"band": [3, 2, 1]})` — pick + reorder bands (RGB from S2 BGR)
- `gt.isel({"x": slice(0, 100, 2), "y": slice(0, 100, 2)})` — 2× downsample (rewrites `res`)

Source: [geotensor.py:1330](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/geotensor.py#L1330).

---

## 7. Padding for fixed-size tiles

`gt.pad(pad_width=...)` extends a `GeoTensor` with the fill value. Useful when CNN inference needs a fixed input size and your tile lands at a scene edge.

```text
pad_width = {"x": (2, 3), "y": (1, 4)}

Original (5×5):           Padded (10×10):
┌─────────────┐           ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
│ █ █ █ █ █ │           │   ← 1 row top           │
│ █ █ █ █ █ │    →      │ ┌─────────────┐          │
│ █ █ █ █ █ │           │ │ █ █ █ █ █ │          │
│ █ █ █ █ █ │           │ │ █ █ █ █ █ │          │
│ █ █ █ █ █ │           │ │ █ █ █ █ █ │          │
└─────────────┘           │ │ █ █ █ █ █ │          │
                          │ │ █ █ █ █ █ │          │
                          │ └─────────────┘          │
                          │   ← 4 rows bottom       │
                          └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                            ↑           ↑
                            2 cols      3 cols
                            left        right
```

Padding shifts `transform.c` (origin x) leftward by `pad_left * res_x` and `transform.f` (origin y) upward by `pad_top * res_y` so the geographic position of the original pixels is unchanged. There's a sister method `pad_array` that pads only the buffer without touching the transform — used internally by readers.

---

## 8. Window-based reads: boundless vs bounded

`gt.read_from_window(window, boundless=True|False)` extracts a sub-region using a `rasterio.windows.Window`. The same interface as rasterio's lazy windowed reads, so you can swap a `RasterioReader` for an in-memory `GeoTensor` without changing call sites — this is the whole point of the [abstract reader protocol](02_abstract_reader.md).

```text
Window extends beyond data:     boundless=True:      boundless=False:
┌───────────────┐                 ┌─────────┐           ┌─────┐
│ ┌─────────┐   │                 │▒▒▒▒▒▒▒▒▒│           │     │
│ │  DATA   │   │                 │▒▒▒███▒▒▒│           │ ███ │
│ │         │   │  ─────────►    │▒▒▒███▒▒▒│           │ ███ │
│ └─────────┘   │                 │▒▒▒▒▒▒▒▒▒│           └─────┘
└───────────────┘                 └─────────┘           Returns only
     Window                   Padded with            intersection
     request                  fill_value
```

`boundless=True` (the default for `RasterioReader` reads in georeader) is the right choice for tiled inference: every chip comes back with the requested shape, so batches stack cleanly. `boundless=False` is useful when you want to *know* you're at an edge.

Source: [geotensor.py:2227](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/geotensor.py#L2227).

---

## 9. Stacking — building time series

`GeoTensor.stack([gt1, gt2, gt3])` is a classmethod that prepends a new axis. All inputs must share `transform`, `crs`, and `shape` (checked via `same_extent`).

```text
Input: 3 GeoTensors each (3, H, W)

gt1: (3, 100, 100)  ─┐
gt2: (3, 100, 100)  ─┼──► stacked: (3, 3, 100, 100)
gt3: (3, 100, 100)  ─┘     ↑
                        new time/batch dim
```

Result `dims = ("time", "band", "y", "x")`. There's also `concatenate(geotensors, axis=0)` for joining along an existing axis (e.g., concatenating bands from two coregistered scenes).

---

## 10. Method reference (the public surface)

Grouped roughly by lifecycle:

**Construction / I/O**
- `GeoTensor(values, transform, crs, fill_value_default=0, attrs=None)` — primary constructor
- `GeoTensor.load_file(path, ...)` — read a whole file into memory ([geotensor.py:2064](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/geotensor.py#L2064))
- `GeoTensor.load_bytes(data, ...)` — read from a bytes buffer ([geotensor.py:2132](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/geotensor.py#L2132))
- `to_json()` / `from_json(d)` — JSON round-trip of metadata

**Properties**
- `values`, `dims`, `res`, `dtype`, `height`, `width`, `count`, `bounds`

**Geometry**
- `meshgrid(dst_crs=None)` — pixel-centre coordinate arrays
- `footprint(crs=None)` — outer polygon in any CRS
- `valid_footprint(...)` — polygon excluding nodata regions
- `same_extent(other, precision=1e-3)` — transform-and-shape equality

**Array ops with georeferencing semantics**
- `isel(sel)` — xarray-style slicing
- `pad(pad_width)` / `pad_array(pad_width)` — fixed-size tile padding
- `resize(...)` — resampling onto a new shape
- `transpose(axes=None)` — for `(C,H,W) ↔ (H,W,C)` plotting flips
- `squeeze`, `expand_dims`, `clip`
- `validmask()` / `invalidmask()` — boolean masks based on `fill_value_default`

**Window I/O**
- `read_from_window(window, boundless=True)` — extract a sub-region
- `write_from_window(data, window)` — write into a sub-region in place

**Multi-tensor**
- `GeoTensor.stack([gts])` — new leading axis (time/batch)
- `GeoTensor.concatenate([gts], axis=0)` — join along existing axis

**Arithmetic / comparison dunders**
- `+`, `-`, `*`, `/`, `&`, `|`, `==`, `!=`, `<`, `<=`, `>`, `>=` — all preserve metadata, reject mismatched extents for binary `GeoTensor`-`GeoTensor` ops

**NumPy plumbing (you rarely call these, but they're how the magic works)**
- `__array_finalize__` — propagates metadata across views/copies
- `__array_ufunc__` — re-wraps ufunc outputs
- `__array__` — explicit cast back to `np.ndarray`
- `array_as_geotensor(arr, ...)` — instance method to wrap a fresh ndarray with `self`'s metadata
- `_preserved_spatial(method, **kwargs)` — internal: does this reduction keep the spatial shape?

---

## 11. Why this redesign matters for `geotoolz`

Three concrete wins for the operator-composition layer:

1. **Operators stop unwrapping.** Pre-redesign code looked like `out = my_func(gt.values); return GeoTensor(out, gt.transform, gt.crs, ...)`. Now it's `return my_func(gt)` — the ufunc protocol carries the metadata for free.
2. **scikit-image / scipy.ndimage interop is half-free.** Anything that goes through ufuncs round-trips automatically. Functions that don't (`skimage.transform.resize`, `scipy.ndimage.uniform_filter`) need a small `preserve_subclass` decorator at the Tier B layer — see [geotoolz.md §5.2](../plans/geotoolz.md).
3. **Time becomes a first-class axis** without inventing a new container. 4D `(T, C, H, W)` is just an ndarray shape; `GeoTensor.stack` builds it; numpy reductions slice it.

The downside to be aware of: spatial reductions (e.g., `gt.sum(axis=(-2,-1))`) drop to a plain ndarray because the result has no transform. Operators that want to keep a `GeoTensor`-shaped result for *non-spatial* reductions must wrap the output explicitly via `gt.array_as_geotensor(...)`.

---

## 12. Sharp edges

- **Same-extent rule for binary ops.** `gt1 + gt2` raises if the transforms or shapes don't match (see `same_extent` and the `__add__` body at [geotensor.py:625](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/geotensor.py#L625)). Reproject first, don't try to "broadcast" across spatial extent.
- **`fill_value_default` is metadata, not a mask.** It's used by `read_from_window(boundless=True)`, `validmask()`, `invalidmask()`, and `pad`. It does *not* turn arithmetic into masked arithmetic — `gt + 1` still adds 1 to nodata pixels.
- **Spatial slicing must use `slice`, not lists.** `gt.isel({"x": [0, 5, 10]})` is rejected; the resulting raster wouldn't be regularly sampled and the transform wouldn't be definable.
- **`crs` is whatever rasterio accepts.** EPSG int, EPSG string, WKT, or `pyproj.CRS`. The class doesn't normalise — be consistent within a pipeline.

Next chapter: [02_abstract_reader.md](02_abstract_reader.md) — the type hierarchy and protocols that make `GeoTensor` and `RasterioReader` interchangeable.
