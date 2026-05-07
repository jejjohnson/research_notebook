# Ch. 8 — `mosaic`

> **Module:** `georeader/mosaic.py` (450 LOC)
> **Role:** turn N partially-overlapping `GeoData` sources into a single seamless `GeoTensor`. Reprojects, resamples, and fills nodata gaps from later rasters in the list.

---

## 1. The job

You have a list of rasters and one of these problems:

- **Adjacent scenes with gaps.** Two S2 tiles cover an AOI that straddles a swath edge. Each has a triangular nodata zone where the other has data. You want the union with no gaps.
- **Cloudy time series.** Three S2 acquisitions of the same scene over a month, each with different clouds. You want a single cloud-free image — pixel by pixel, take the first valid observation.
- **Heterogeneous mixed sources.** S2 + Landsat covering the same AOI at different resolutions. You want one grid, S2 where available, Landsat to fill gaps.

All three are the same operation underneath: **iterate the list, fill remaining nodata from each subsequent raster.** That's `spatial_mosaic`.

---

## 2. Spatial mosaic — gap filling by precedence

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    SPATIAL MOSAIC CONCEPT                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input Rasters (with gaps)              Output Mosaic                   │
│  ─────────────────────────              ─────────────                   │
│                                                                          │
│   Raster 1         Raster 2                                             │
│  ┌─────────┐      ┌─────────┐           ┌─────────────────┐            │
│  │▓▓▓░░░░░░│      │░░░░░▓▓▓▓│           │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │
│  │▓▓▓▓░░░░░│  +   │░░░░▓▓▓▓▓│    ═══►   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │
│  │▓▓▓▓▓░░░░│      │░░░▓▓▓▓▓▓│           │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │
│  │░░░░░░░░░│      │░░▓▓▓▓▓▓▓│           │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │
│  └─────────┘      └─────────┘           └─────────────────┘            │
│                                                                          │
│   ░ = nodata/gaps                        Gaps filled from               │
│   ▓ = valid data                         overlapping rasters            │
│                                                                          │
│  Processing Order:                                                       │
│  • First raster fills as much as possible                               │
│  • Each subsequent raster fills remaining gaps                          │
│  • Continues until no nodata remains (or list exhausted)                │
└─────────────────────────────────────────────────────────────────────────┘
```

The two rules underneath:

1. **Order = priority.** `[A, B, C]` means "use A wherever A is valid; use B where A is nodata but B is valid; use C only where both A and B are nodata."
2. **Early termination.** If after raster B every pixel is valid, raster C is never read. A user-supplied list of cloud-free fallback scenes pays nothing for the unused tail when the AOI is mostly cloud-free.

This is the `rasterio.merge.merge` shape but with two extras `merge` doesn't have: per-raster validity masks (next section) and arbitrary masking functions (e.g., apply a cloud detector lazily before contributing to the mosaic).

---

## 3. Temporal reduction — many timesteps, one image

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL REDUCTION CONCEPT                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Time Series Input                      Reduction Output                │
│  ─────────────────                      ────────────────                │
│                                                                          │
│   t=1    t=2    t=3                                                     │
│  ┌───┐  ┌───┐  ┌───┐                   ┌───────────────┐               │
│  │ 5 │  │ 7 │  │ 6 │                   │               │               │
│  │   │  │   │  │   │   ─────────────►  │  median = 6   │               │
│  │   │  │   │  │   │   np.nanmedian    │  mean = 6.0   │               │
│  └───┘  └───┘  └───┘   np.nanmean      │  max = 7      │               │
│                                        └───────────────┘               │
│                                                                          │
│  Common Reduction Functions:                                            │
│  • np.nanmedian: Robust to outliers (clouds, shadows)                   │
│  • np.nanmean: Average value                                            │
│  • np.nanmax: Maximum composite (e.g., max NDVI)                        │
│  • np.nanmin: Minimum composite                                         │
│  • np.nanstd: Temporal variability                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

Conceptually different from spatial mosaicking: **every** timestep contributes (subject to nodata), and a reduction (median, mean, max, ...) decides the per-pixel output value.

The module docstring promises a `rasters_reduction(rasters, reducer=np.nanmedian, ...)` function for this. **It isn't implemented in the current module** — only `spatial_mosaic` is exported. See [§8 below](#8-the-implementation-vs-docstring-gap) for what's actually shipped vs documented.

The conceptual reduction recipes are still useful as a reference:

- **`nanmedian`** — robust temporal composite. The standard cloud-free recipe: stack T scenes, mask clouds to NaN, take the per-pixel median across time. Outliers (residual clouds, shadows, snow) bias the mean but barely move the median.
- **`nanmean`** — when you actually want the average (e.g., monthly mean radiance for an energy-budget paper).
- **`nanmax`** — max-NDVI compositing. For each pixel, pick the timestep where NDVI was highest. Standard greenness-of-the-year recipe.
- **`nanmin`** — minimum compositing. Useful for water/wetland detection where the *darkest* observation rejects clouds.
- **`nanstd`** — temporal variability map. Not a composite but a derived product (where does the scene change a lot?).

---

## 4. Mosaic with masks

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    MASKED MOSAIC WORKFLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: (data, mask) tuples                                             │
│  ─────────────────────────                                              │
│                                                                          │
│   Raster 1     Cloud Mask      │    Raster 2     Cloud Mask            │
│  ┌─────────┐  ┌─────────┐      │   ┌─────────┐  ┌─────────┐            │
│  │▓▓▓▓▓▓▓▓▓│  │░░░█████░│      │   │▓▓▓▓▓▓▓▓▓│  │░░░░░░░░░│            │
│  │▓▓▓▓▓▓▓▓▓│  │░░█████░░│      │   │▓▓▓▓▓▓▓▓▓│  │░░░░░░░░░│            │
│  │▓▓▓▓▓▓▓▓▓│  │░██████░░│   +  │   │▓▓▓▓▓▓▓▓▓│  │░░░░░░░░░│            │
│  └─────────┘  └─────────┘      │   └─────────┘  └─────────┘            │
│                   ↑            │                                        │
│               █ = invalid      │   Uses Raster 2 where Raster 1        │
│               (cloud/shadow)   │   is masked as invalid                 │
│                                                                          │
│  Usage:                                                                  │
│    data_list = [(raster1, mask1), (raster2, mask2), ...]               │
│    mosaic = spatial_mosaic(data_list, ...)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

`spatial_mosaic` accepts two list shapes:

```python
spatial_mosaic([reader1, reader2, reader3], ...)              # nodata-only
spatial_mosaic([(reader1, mask1), (reader2, mask2), ...], ...) # explicit masks
```

The mask convention: **`True` means invalid** (cloud, shadow, sensor flag set). Invalid pixels are treated like nodata — fall through to the next raster.

There's also a third form via `masking_function`:

```python
spatial_mosaic([reader1, reader2, reader3], masking_function=detect_clouds, ...)
```

`masking_function` is called with each `GeoData` and returns a `GeoData` of invalid pixels. The wrapper computes the mask lazily, which avoids materialising mask rasters when the spatial precedence rules out their parent (early termination above).

The masking function is the seam where downstream packages plug in cloud detectors (CloudSEN12, s2cloudless, FMask) without the mosaic module taking a hard dep.

---

## 5. The `spatial_mosaic` signature

```python
spatial_mosaic(
    data_list,                    # list[GeoData] | list[(GeoData, GeoData)]
    polygon=None,                 # output AOI as Shapely polygon
    crs_polygon=None,
    dst_transform=None,           # output transform
    bounds=None,                  # output bbox (alternative to polygon)
    dst_crs=None,                 # output CRS (defaults to first raster's)
    dtype_dst=None,               # output dtype (defaults to first raster's)
    window_size=None,             # tile size for chunked processing
    resampling=Resampling.cubic_spline,
    masking_function=None,        # GeoData → invalid-mask GeoData
    dst_nodata=None,              # output nodata (defaults to first raster's)
) -> GeoTensor
```

A few non-obvious points:

- **Output grid is configured via the same triplet as `read_reproject`.** Polygon / bounds + transform / dst_crs + resolution. Pass any consistent subset; `figure_out_transform` ([Chapter 4 §6](04_window_utils.md)) fills in the rest.
- **`window_size=(h, w)` switches to tiled processing.** For large mosaics that don't fit in RAM, tile through the output grid; for each output tile, call `read_reproject(reader, bounds=tile_bounds)` per input raster. Memory cost is `O(tile_size × n_rasters)`, not `O(output_size)`.
- **`resampling=cubic_spline` is the default.** Same caveat as `read.py` — flip to `Resampling.nearest` when mosaicking categorical data (cloud masks, class labels).
- **The "first raster wins" defaults** (CRS, dtype, nodata) make the call short for "give me everything in raster1's coordinate system." Override only when you have a reason — passing inconsistent dtype across rasters can silently truncate values during the first reproject step.

Source: [mosaic.py:159](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/mosaic.py#L159).

---

## 6. Building cloud-free composites — the canonical recipe

The whole module exists to make this idiomatic:

```python
# 1. Acquire scenes from a catalog (e.g., a month of S2 over an AOI)
scenes = [reader_jan_5, reader_jan_12, reader_jan_19, reader_jan_26]
masks  = [cloudsen12_jan_5, cloudsen12_jan_12, cloudsen12_jan_19, cloudsen12_jan_26]

# 2. Order by quality (least cloudy first)
ranked = sorted(zip(scenes, masks), key=lambda sm: sm[1].values.mean())

# 3. Mosaic
composite = mosaic.spatial_mosaic(
    ranked,
    polygon=aoi,
    crs_polygon="EPSG:4326",
    dst_crs="EPSG:32630",          # output UTM
)
```

Key idea: **rank before mosaicking.** The "first raster wins" rule means the order of `data_list` controls per-pixel preference. Ordering by global cloud cover (or by acquisition recency, or by snow score, etc.) makes the composite reflect a quality preference, not just a temporal order.

For temporal reductions that *don't* prefer one timestep — true median compositing — you'd want `rasters_reduction`, which (see next section) isn't actually implemented yet.

---

## 7. Function reference

What's actually exported from this module:

| Function | Status | Notes |
|---|---|---|
| `spatial_mosaic` | ✅ implemented | the only public function |
| `spatial_mosaic_chunked` | ❌ not in source | promised in module docstring |
| `rasters_reduction` | ❌ not in source | promised in module docstring |
| `pad_add_rasters` | ❌ not in source | promised in module docstring |

The `window_size=(h, w)` argument on `spatial_mosaic` covers the chunked-processing case that `spatial_mosaic_chunked` would have provided.

---

## 8. The implementation-vs-docstring gap

The module docstring promises four functions; the source defines one. This is worth flagging because:

- If you read the docstring expecting a `rasters_reduction` for temporal compositing, **it doesn't exist**. You'd implement it yourself using `np.nanmedian` over a stacked-time `GeoTensor` (Chapter 1 §9 covers `GeoTensor.stack`).
- The `pad_add_rasters` helper would have been useful for aligning rasters of different shapes prior to stacking; the workaround is `read_reproject_like` per raster against a common reference.

This is one of the loose ends downstream `geotoolz.compositing` fills in — `MeanComposite`, `MedianComposite`, `MaxNDVIComposite`, `CloudFreeComposite` (the [geotoolz.md plan §1.2](../plans/geotoolz.md)) are the operator-form versions of what the docstring here describes but doesn't implement.

For now, a hand-rolled temporal reduction looks like:

```python
gts = [read.read_from_polygon(reader, aoi, ...) for reader in readers]    # list of (C, H, W)
stacked = GeoTensor.stack(gts)                                            # (T, C, H, W)
median = np.nanmedian(stacked.values, axis=0)                             # (C, H, W) ndarray
out = GeoTensor(median, transform=stacked.transform, crs=stacked.crs)
```

That's six lines you'd otherwise expect to be `mosaic.rasters_reduction(readers, reducer=np.nanmedian)`. Worth knowing it isn't there.

---

## 9. Sharp edges

- **First raster's metadata wins.** `dst_crs`, `dtype_dst`, `dst_nodata` all default to the first raster. Mixing dtypes silently truncates; mixing CRS triggers reprojection (correct, but the cost surprises people who didn't realise their list was heterogeneous).
- **Mask convention is `True = invalid`.** It's the inverse of "valid mask." Easy to get backwards if you're coming from `masked_array` or sklearn.
- **Default `cubic_spline` resampling.** Flip to `nearest` for categorical mosaics (cloud masks, class labels).
- **Tile processing (`window_size`) is per-tile complete.** It doesn't stream — each output tile fully reads and reprojects each contributing input tile. Memory bound is per-tile; total compute is the same.
- **`spatial_mosaic_chunked` / `rasters_reduction` / `pad_add_rasters` aren't implemented.** Don't trust the module docstring's "Module Functions Overview" list. Inspect the source.
- **No deduplication.** Passing the same reader twice in `data_list` reads it twice. The module doesn't try to detect identity.
- **Order matters and is silent.** Two cloud-free scenes in `[scene1, scene2]` vs `[scene2, scene1]` produce different mosaics in their overlap region. There's no "blend" mode in the current implementation.

---

## 10. Connection to `geotoolz.compositing`

The [geotoolz.md plan §1.2](../plans/geotoolz.md) lists `compositing` as one of the v0.2 modules with four operators: `MedianComposite`, `MaxNDVIComposite`, `CloudFreeComposite`, `MeanComposite`. Mapping to this module:

| `geotoolz` operator | Maps to `mosaic.spatial_mosaic` how |
|---|---|
| `CloudFreeComposite` | direct: list of `(scene, cloud_mask)` tuples, ranked, → `spatial_mosaic` |
| `MaxNDVIComposite` | not direct: needs the unimplemented `rasters_reduction` with a custom argmax-of-NDVI reducer |
| `MedianComposite` | not direct: needs `rasters_reduction(reducer=np.nanmedian)` |
| `MeanComposite` | not direct: needs `rasters_reduction(reducer=np.nanmean)` |

So `geotoolz.compositing` is *partly* a thin wrapper over `spatial_mosaic` and *partly* the missing `rasters_reduction` reimplemented at the operator layer. The implementation gap above means `geotoolz.compositing` will probably ship its own temporal-reduction core rather than depending on this module for it.

Next chapter: [09_rasterize.md](09_rasterize.md) — burning vector geometries into raster grids (the inverse of `vectorize`).
