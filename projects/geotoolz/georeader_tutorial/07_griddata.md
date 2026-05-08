---
title: griddata
subject: georeader tutorial
subtitle: Irregular-grid interpolation and GLT orthorectification
short_title: Ch. 7 — Griddata
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: tutorial, georeader, griddata
---

> **Module:** `georeader/griddata.py` (617 LOC)
> **Role:** the onramp for **curvilinear sensors** — pushbroom imagers, swath scanners, and any sensor that gives you per-pixel `lons` and `lats` rather than a clean affine transform. Where `read.py` ends and EMIT / PRISMA / EnMAP / MODIS / VIIRS begin.

---

## 1. Why this module exists

Rectilinear rasters (S2, Landsat, Planet) come with an `Affine` transform — six numbers that map pixel `(col, row)` to geographic `(x, y)`. Every function in `read.py` assumes this transform exists.

But many sensors **don't ship that way**. EMIT, PRISMA, EnMAP, MODIS, VIIRS, AVHRR all give you:

- A 3D radiance/reflectance array `(H, W, bands)`.
- Two 2D coordinate arrays `lons (H, W)` and `lats (H, W)`, **one per pixel**.

There's no transform. Two adjacent pixels in the array don't necessarily map to adjacent points on Earth. Reading a polygon AOI requires either **interpolating** the irregular grid onto a regular one, or **looking up** which sensor pixel each output pixel came from. This module does both.

---

## 2. Irregular vs regular grids

```text
┌─────────────────────────────────────────────────────────────────────────┐
│   IRREGULAR vs REGULAR GRIDS                                            │
│                                                                         │
│   Irregular (Swath/Sensor)           Regular (Orthorectified)           │
│   ─────────────────────────           ──────────────────────            │
│                                                                         │
│       ●  ●   ●  ●                     ┌──┬──┬──┬──┐                     │
│     ●    ●  ●    ●                    ├──┼──┼──┼──┤                     │
│      ●   ● ●  ●                       ├──┼──┼──┼──┤                     │
│    ●   ●    ●   ●                     ├──┼──┼──┼──┤                     │
│                                       └──┴──┴──┴──┘                     │
│                                                                         │
│   Each pixel has unique (lon, lat)    Fixed transform: pixel → geo      │
│   Spacing varies with scan angle      Uniform spacing, axis-aligned     │
│   Common in: pushbroom sensors        Required for: GIS, web maps       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The job of `griddata.py` is to take the left and produce the right — a `GeoTensor` with a real affine transform that the rest of the package can consume.

There are two strategies for doing this:

1. **Interpolation** (`reproject`, `read_to_crs`, `read_reproject_like`) — sample the scattered points onto a regular grid via `scipy.interpolate.griddata`. Works for any sensor that gives you `(lons, lats)` arrays. Slow for large hyperspectral cubes; smooth output.
2. **GLT lookup** (`georreference`) — apply a precomputed Geolocation Lookup Table that names "for each output pixel, which sensor pixel?" Fast, lossless (no interpolation artefacts), but requires the sensor's product to ship a GLT (NASA EMIT does; PRISMA doesn't).

---

## 3. Interpolation method comparison

```text
┌────────────────────────────────────────────────────────────────────────┐
│  INTERPOLATION METHOD COMPARISON                                       │
│                                                                        │
│  Method      │ Continuity │ Speed  │ Best For                          │
│  ────────────┼────────────┼────────┼─────────────────────────────────  │
│  "nearest"   │ C⁰         │ Fast   │ Categorical data, masks           │
│  "linear"    │ C⁰         │ Medium │ Simple surfaces, quick preview    │
│  "cubic"     │ C²         │ Slow   │ Smooth continuous data (default)  │
│                                                                        │
│  C⁰ = continuous but not differentiable (may have sharp edges)         │
│  C² = smooth, twice differentiable (recommended for radiance/refl)     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

The default is `"cubic"` — Clough-Tocher interpolation on the Delaunay triangulation of the scattered points. C² smoothness matters for hyperspectral retrievals because matched filters and unmixing react badly to discontinuities. The cost: cubic on a ~10⁶-point hyperspectral scene is multi-minute; consider downsampling first if the geometry doesn't need full-res.

`"linear"` is the right default for "I just want to look at it"; `"nearest"` for categorical data (cloud masks, class labels).

---

## 4. GLT — geolocation lookup tables

```text
┌────────────────────────────────────────────────────────────────────────┐
│  GLT-BASED ORTHORECTIFICATION                                          │
│                                                                        │
│  Sensor Array (irregular)              Output Grid (regular)           │
│  ┌───────────────────────┐             ┌──┬──┬──┬──┬──┐                │
│  │  0   1   2   3   ...  │             │  │  │  │  │  │                │
│  │                       │             ├──┼──┼──┼──┼──┤                │
│  │ [r,c] = radiance      │     GLT     │  │██│██│  │  │                │
│  │                       │  ────────►  ├──┼──┼──┼──┼──┤                │
│  │                       │             │  │██│██│██│  │                │
│  └───────────────────────┘             └──┴──┴──┴──┴──┘                │
│                                                                        │
│  GLT[0, i, j] = column in sensor array                                 │
│  GLT[1, i, j] = row in sensor array                                    │
│  output[i, j] = sensor[GLT[1,i,j], GLT[0,i,j]]                         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

A GLT is a `(2, H_out, W_out)` integer array. Channel 0 holds the source column, channel 1 holds the source row, and each output pixel reads from `data[..., glt[1, i, j], glt[0, i, j]]` — a pure index gather. Invalid pixels (output cells with no source coverage) are marked with `fill_value_default` (typically `-1`).

GLTs are produced **once** by the data provider's processing chain (using the full sensor model — orbital ephemeris, look angles, terrain DEM) and shipped alongside the radiance product. NASA EMIT does this; the EnMAP and PRISMA L1 products instead ship raw lons/lats and require interpolation at consumption time. The structural diagram repeats inside the `georreference` docstring:

```text
┌────────────────────────────────────────────────────────────────────┐
│  GLT ARRAY STRUCTURE                                               │
│                                                                    │
│  glt.shape = (2, H_out, W_out)                                     │
│                                                                    │
│  glt[0, i, j] = source column (x-index in sensor array)            │
│  glt[1, i, j] = source row    (y-index in sensor array)            │
│                                                                    │
│  For each output pixel (i, j):                                     │
│    output[..., i, j] = data[..., glt[1,i,j], glt[0,i,j]]           │
│                                                                    │
│  ┌──────────────────────┐      ┌──────────────────────┐            │
│  │    Sensor Array     │       │   Output Grid        │            │
│  │    (raw data)       │       │   (orthorectified)   │            │
│  │   ┌───┬───┬───┐     │       │  ┌──┬──┬──┬──┐       │            │
│  │   │ A │ B │ C │     │       │  │  │ A│ B│  │       │            │
│  │   ├───┼───┼───┤     │ GLT   │  ├──┼──┼──┼──┤       │            │
│  │   │ D │ E │ F │  ───────►   │  │  │ D│ E│ F│       │            │
│  │   ├───┼───┼───┤     │       │  ├──┼──┼──┼──┤       │            │
│  │   │ G │ H │ I │     │       │  │ G│ H│ I│  │       │            │
│  │   └───┴───┴───┘     │       │  └──┴──┴──┴──┘       │            │
│  └──────────────────────┘      └──────────────────────┘            │
│                                                                    │
│  GLT handles: terrain distortion, sensor geometry, Earth curvature │
│  Invalid pixels: glt values = fill_value_default (typically -1)    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

The output grid in the diagram has cells that pull from non-adjacent sensor pixels — that's terrain distortion encoded in the GLT. Because it's a pure gather, GLT orthorectification:

- Preserves **exact** sensor pixel values (no interpolation artefacts).
- Runs at memory-bandwidth speed — a single fancy-index over the band dim.
- Handles arbitrary geometry (rotation, terrain warp, antimeridian) without special cases.

The trade-off: fixed output grid. If you want a different resolution or CRS than the GLT was built for, you either re-ship the GLT or fall back to interpolation.

---

## 5. The interpolation workflow, step by step

The internal walkthrough lives inside `reproject`'s docstring:

```text
┌────────────────────────────────────────────────────────────────────┐
│  INTERPOLATION WORKFLOW                                            │
│                                                                    │
│  1. Flatten inputs                                                 │
│     data: (H, W, C) → (H×W, C)  [or (H, W) → (H×W,)]               │
│     lons/lats: (H, W) → (H×W,)                                     │
│                                                                    │
│  2. Generate output coordinate grid                                │
│     meshgrid(transform, width, height) → (xs, ys)                  │
│     Transform xs, ys from dst_crs to input crs if different        │
│                                                                    │
│  3. Call scipy.interpolate.griddata                                │
│     points = (lons_flat, lats_flat)                                │
│     values = data_flat                                             │
│     xi = (xs_grid, ys_grid)                                        │
│     result = griddata(points, values, xi, method=method)           │
│                                                                    │
│  4. Reshape and handle nodata                                      │
│     Fill NaN regions with fill_value_default                       │
│     Transpose to (C, H, W) if multi-band                           │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

A few details that bite:

- **Step 1 expects `(H, W, C)`, not `(C, H, W)`.** `griddata` is the only entry point in the package that takes channels-last input. The output is transposed back to `(C, H, W)` in step 4. Don't pass a `GeoTensor` directly — strip to `data.values.transpose(1, 2, 0)` first if you have one.
- **Step 2 inverts directions.** The `meshgrid` call uses `source_crs=dst_crs, dst_crs=crs` — the output grid coordinates need to be expressed in the **input's** CRS so they line up with the `(lons, lats)` arrays. This is why the function also accepts `crs="EPSG:4326"` as the input CRS by default.
- **Step 3 is the slow one.** `scipy.interpolate.griddata` builds a Delaunay triangulation over `H × W` scattered points and then samples at `width × height` query locations. For a 1000×1000 EMIT scene at full res, that's ~10⁶ triangulation points and ~10⁶ queries. Cubic adds the Clough-Tocher per-triangle solve. Budget 1–10 minutes per scene per band group.
- **Step 4's NaN handling.** Output pixels outside the convex hull of the input points come back as NaN. The function replaces them with `fill_value_default` (default `-1`). For `boundless`-style behaviour, that's correct; for downstream NaN-aware code, set `fill_value_default=np.nan` explicitly.

---

## 6. The three high-level entry points

All three return a `GeoTensor` with a real affine transform. Pick by what you have on hand:

### `read_to_crs(data, lons, lats, resolution_dst, dst_crs=..., crs="EPSG:4326", method="cubic")`

You have `(data, lons, lats)` and a target resolution. Auto-computes the output bounds (from `footprint(lons, lats)`), the output shape, and a `figure_out_transform`. Default `dst_crs` is the auto-detected UTM zone.

**Use this** for the first orthorectification of a scene with no preferred grid.

### `read_reproject_like(data, lons, lats, data_like, method="cubic", ...)`

You have `(data, lons, lats)` and want the result on **another raster's grid** (e.g., to coregister an EMIT scene with a Sentinel-2 scene). Reads `data_like.transform`, `data_like.crs`, `data_like.shape` and dispatches to `reproject`.

**Use this** when stacking heterogeneous sensors onto a common grid.

### `reproject(data, lons, lats, width, height, transform, dst_crs, crs="EPSG:4326", fill_value_default=-1, method="cubic")`

The full-control entry point. You provide the output grid explicitly. Both wrappers above ultimately call this.

**Use this** for production pipelines where the output grid is fixed by upstream config (e.g., a tile catalog).

There's also `get_shape_transform_crs(lons, lats, resolution_dst, ...)` — the inverse-design helper that `read_to_crs` uses internally. Useful when you want to plan a grid before committing to the slow interpolation step.

---

## 7. The fast path: `georreference(glt, data)`

```python
georreference(glt: GeoTensor, data: NDArray,
              valid_glt: Optional[NDArray] = None,
              fill_value_default: Optional[Union[int, float]] = None) -> GeoTensor
```

`glt` carries the transform and CRS (because it's a `GeoTensor`); `data` is the raw sensor array. The function:

1. Checks the `valid_glt` mask (or computes it from `glt != fill_value_default`).
2. Allocates the output array with shape inferred from `glt.shape[1:]` plus `data`'s leading dims.
3. Issues `output[..., valid_glt] = data[..., glt[1, valid_glt], glt[0, valid_glt]]` — a single fancy-index gather.

Speed scales with output pixel count and band count, not with input scene size. A 285-band EMIT scene orthorectifies in seconds vs minutes for `cubic` interpolation.

Source: [griddata.py:473](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/griddata.py#L473).

---

## 8. Footprint helper

```python
footprint(lons: NDArray, lats: NDArray) -> Polygon
```

Builds a Shapely polygon for an irregular grid. Picks the four extremum points (argmin/argmax of lon and lat in raveled form) and connects them. Approximate — not the actual hull — but fast and good enough for AOI-overlap checks.

For exact hulls use `shapely.MultiPoint(zip(lons.ravel(), lats.ravel())).convex_hull`. The fast version is used inside `read_to_crs` to derive output bounds.

---

## 9. Function reference

| Function | Returns | Use for |
|---|---|---|
| `reproject(data, lons, lats, width, height, transform, dst_crs, ...)` | `GeoTensor` | full-control interpolation |
| `read_to_crs(data, lons, lats, resolution_dst, ...)` | `GeoTensor` | auto-grid interpolation |
| `read_reproject_like(data, lons, lats, data_like, ...)` | `GeoTensor` | match another raster's grid |
| `get_shape_transform_crs(lons, lats, resolution_dst, ...)` | `(shape, transform, crs)` | plan before interpolating |
| `meshgrid(transform, width, height, source_crs=..., dst_crs=...)` | `(xs, ys)` arrays | build query points |
| `georreference(glt, data, valid_glt=..., fill_value_default=...)` | `GeoTensor` | GLT-based orthorectification |
| `footprint(lons, lats)` | `Polygon` | quick irregular-grid hull |

`METHOD_DEFAULT = "cubic"` is the module-level default for all interpolation entry points.

---

## 10. Sharp edges

- **`(H, W, C)` channels-last for input data.** The package convention everywhere else is `(C, H, W)`. `griddata.reproject` is the exception. Transpose before, untranspose after — or rely on the EMIT/PRISMA readers in `georeader.readers` which handle this internally.
- **`fill_value_default = -1` is unusual.** Most of the package uses `0`. The `-1` choice here is to make "outside convex hull" visually obvious. Override if you're feeding the result into something that interprets negative numbers as physical (e.g., reflectance models).
- **The Delaunay triangulation is built per call.** If you're orthorectifying many bands of the same scene, *don't* call `reproject` per band. Pass the multi-band array; `griddata` builds the triangulation once and queries it for each band internally. This is a 100× speedup.
- **`lons` and `lats` must be 2D and same-shape as `data`'s spatial dims.** Some sensor products ship them as 1D `(H,)` and `(W,)` axes (rectilinear lat/lon grids that aren't projected); for those, build `lats[:, None] + 0*lons[None, :]` to broadcast to 2D first, or just use `read.read_reproject` which handles the rectilinear case.
- **NaN inputs propagate through `griddata`.** Mask them out before calling — `griddata` has no `nan_policy`. The function's NaN handling at step 4 is for *out-of-hull* output pixels, not for NaN inputs.
- **The function's name is `georreference` — note the double 'r'.** It's a typo that's now part of the public API (renaming is a breaking change). Don't fix it locally.

---

## 11. How the curvilinear sensor readers use this module

- **EMIT** (NASA, ISS) — ships a GLT in the L1B product. `georeader.readers.emit.load(...)` calls `georreference(glt, radiance)`. Fast.
- **PRISMA** (ASI) — ships per-pixel `(lons, lats)` but no GLT. `georeader.readers.prisma.load(...)` calls `read_to_crs(...)`. Slow but unavoidable.
- **EnMAP** (DLR) — same pattern as PRISMA. Per-pixel coords, interpolation on read.
- **MODIS, VIIRS** (planned in [`modis.md`](../plans/readers/modis.md)) — same pattern as PRISMA, with extra wrinkles (bowtie pixel duplicates, antimeridian crossing for polar orbits).

The `geotoolz` plan is to keep this division: GLT-equipped sensors get the fast path, everyone else gets cubic interpolation. The `griddata` module is the substrate; the readers are the per-sensor wrappers.

---

## 12. Why this matters for `geotoolz`

Three concrete things downstream:

1. **Hyperspectral operators (matched filter, ACE, RX, unmixing) consume orthorectified `GeoTensor`s.** The interpolation step happens at *read* time, inside the sensor reader. By the time `geotoolz.hyperspectral.MatchedFilter` runs, it's just a `GeoTensor` like any other — the curvilinear-ness is invisible.
2. **The cost asymmetry shapes the pipeline.** Interpolating a 285-band EMIT scene cubically is minutes. Operators that re-read the scene (e.g., parameter sweeps) should `load()` once and reuse, not ortho-on-demand.
3. **GLT support is sensor-specific.** A `geotoolz.presets.emit.EMIT_METHANE_MF` preset can rely on the fast GLT path; the equivalent EnMAP/PRISMA preset cannot. This shows up as different default behaviour in those presets — not a bug, a substrate constraint.

Next chapter: [08_mosaic.md](08_mosaic.md) — combining multiple rasters into seamless composites with reprojection and nodata fill.
