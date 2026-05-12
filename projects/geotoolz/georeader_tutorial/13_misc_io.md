---
title: utilities
subject: georeader tutorial
subtitle: "`io`, `dataarray`, `plot` — the connective tissue"
short_title: Ch. 13 — Utilities
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: tutorial, georeader, io
---

> **Modules:**
>
> - `georeader/io.py` (113 LOC) — NetCDF safe-open
> - `georeader/dataarray.py` (145 LOC) — xarray bridge
> - `georeader/plot.py` (336 LOC) — matplotlib helpers
>
> **Role:** the connective tissue.
> Each module is small, focused, and invisible until you need it.
> **Diagrams:** none.
> These files don't have ASCII art — they're pure utility code.

---

## 1. `io.py` — NetCDF backend roulette

The whole module is one helper plus one URL-detector:

- **`is_url(path) → bool`** — `True` for paths starting with `http://`, `https://`, `ftp://`.
  Used elsewhere in the package to decide between local and VSI access.
- **`safe_open_netcdf(file_path_or_object, cache=False, load=True, group=None, **kwargs) → xr.Dataset`** — open a NetCDF file by trying multiple xarray engines until one succeeds.

### Why "safe" open exists

NetCDF has three on-disk formats (NetCDF3, NetCDF4/HDF5, NetCDF-Java) and xarray has three backends (`scipy`, `h5netcdf`, `netcdf4`) with overlapping but not identical support.
Different sensor providers ship different formats, and a backend that handles one will fail on another with cryptic errors.

`safe_open_netcdf` cycles through engines in a sensible order until one works:

| Input type | Engine order |
|---|---|
| Remote URL (OPeNDAP) | `netcdf4` only (the others don't support remote) |
| Local file or file-like object | `h5netcdf` → `scipy` → `netcdf4` |

`h5netcdf` is tried first for local files because it's typically faster than `netcdf4` and handles HDF5 well; `scipy` second because it's small and self-contained; `netcdf4` last because it's the most comprehensive but heaviest dep.

### Sharp edges

- **`load=True` by default**, meaning the entire dataset is read into memory.
  For multi-GB EMIT NetCDFs you almost always want `load=False` and lazy slicing via xarray.
- **The function eats errors per-engine** and only raises `IOError` if all engines fail.
  Debug with `logging.getLogger("georeader.io").setLevel(logging.DEBUG)`.
- **File-like objects need `.seek(0)`** support — the function tries to rewind between engines, but some streams (e.g., raw HTTP responses) can't be rewound.
  Pass a `BytesIO` wrapper if you have to retry.
- **`group=` is honoured** — for hierarchical NetCDFs (EMIT, EnMAP) where the data lives in a sub-group like `"location"` or `"sensor_band_parameters"`.
  Most readers in `georeader.readers.emit` etc. already pass the right group internally.

This is the function that makes the EMIT / PRISMA / EnMAP readers work cleanly across providers without users worrying about which library wraps which format.

---

## 2. `dataarray.py` — the xarray bridge

Four functions that translate between `GeoTensor` and `xr.DataArray`.
This is the substrate seam between georeader and `xr_toolz` (the climate-side sibling library — see [`geotoolz.md` §10](../plans/geotoolz/geotoolz.md)).

### The four functions

- **`coords_to_transform(coords, x_axis_name="x", y_axis_name="y") → rasterio.Affine`** — given an xarray coordinates object with `x` and `y` axes, infer the affine transform.
  Inverse of `xr.open_rasterio`'s coord computation.
  Useful when reading a NetCDF or Zarr that has lon/lat coords but no transform metadata.
- **`getcoords_from_transform_shape(transform, shape, x_axis_name="x", y_axis_name="y") → dict`** — the forward direction.
  Given a transform and a `(H, W)` shape, build coordinate arrays placed at pixel centres (`+ 0.5` offset).
  Asserts `transform.is_rectilinear` because rotated transforms can't be expressed as simple 1D coord arrays.
- **`toDataArray(x: GeoTensor, x_axis_name="x", y_axis_name="y", extra_coords=None) → xr.DataArray`** — `GeoTensor` → `xr.DataArray`.
  Builds coords from the transform, attaches `attrs={"crs": ..., "fill_value_default": ...}`, optionally adds `extra_coords` for non-spatial dims (e.g., `{"time": dates_array}`).
- **`fromDataArray(x: xr.DataArray, crs=None, ...) → GeoTensor`** — the inverse.
  Reads the `crs` from the DataArray's attrs (or from `crs=` argument as override), computes transform from `coords`, returns a `GeoTensor`.

### Why this matters

The whole point of having both `geotoolz` (RS substrate) and `xr_toolz` (climate substrate) as separate libraries is that **`georeader` is the bridge**.
A workflow that reads with georeader, runs a climate analysis in xr_toolz, and writes back via georeader looks like:

```python
gt = georeader.read_from_bounds(reader, bounds=AOI)
da = georeader.dataarray.toDataArray(gt)            # GeoTensor → DataArray
result_da = xr_toolz.detrend.RemoveClimatology(clim)(da)
result_gt = georeader.dataarray.fromDataArray(result_da)  # back to GeoTensor
georeader.save_cog(result_gt, "/out/result.tif")
```

Five lines, no metadata loss.
The conversion is cheap (it's just rebuilding coord arrays, no data copy if you're careful with `.values`).

### Sharp edges

- **`is_rectilinear` is required.** Rotated transforms can't be expressed as 1D x and y coordinate arrays.
  For rotated rasters you'd have to convert to a regular grid first (Chapter 7, `griddata`).
- **xarray's coord convention is pixel-centre.** The `+ 0.5` in `getcoords_from_transform_shape` is the pixel-centre vs pixel-edge convention.
  Forgetting this in custom code drifts the coord by half a pixel.
- **CRS round-trip via `attrs`.** xarray doesn't have a first-class CRS concept; `toDataArray` stuffs it in `da.attrs["crs"]`.
  Other libraries (rioxarray, cf-xarray) use different conventions.
  If a downstream consumer expects rioxarray's `da.rio.crs`, an extra translation step is needed.
- **`fromDataArray` defaults to `crs=None`.** It pulls from `attrs["crs"]` if present; otherwise raises.
  Pass `crs=` explicitly when reading from sources that don't carry CRS metadata.
- **xarray is an optional dep.** This module fails to import if xarray isn't installed.
  Users without an xarray pipeline never need to touch this module.

---

## 3. `plot.py` — matplotlib helpers

Four functions for visualising geospatial data with matplotlib:

| Function | What it does |
|---|---|
| `show(data, add_colorbar_next_to=False, ...)` | Display a `GeoTensor` / `RasterioReader` on an axis with proper extent + georeferencing |
| `add_shape_to_plot(shape, ax=None, ...)` | Overlay Shapely geometries / GeoDataFrames |
| `plot_segmentation_mask(mask, color_array=None, ...)` | Discrete-class raster with categorical colormap and legend |
| `colorbar_next_to(im, ax, fig=None, ...)` | Attach a colorbar that doesn't squeeze the main axis |

### `show(data, ...)`

The workhorse.
Reads the GeoData's transform and bounds, calls `ax.imshow(data.values.transpose(1, 2, 0))` for RGB or `ax.imshow(data.values)` for single-band, sets the extent so axis ticks display in geographic coordinates, optionally adds a colorbar via `colorbar_next_to`.

Used in the README quickstart:

```python
plot.show((gt_rgb / 3500).clip(0, 1))
```

The `/3500` and `.clip(0, 1)` are because S2 reflectance values are stored as int16 with a scale factor of 10000, and a viewable RGB needs floats in `[0, 1]`.
Note this still uses `GeoTensor` arithmetic — the clipped result has the same transform/CRS, so the axis ticks are correct geo coords.

### `add_shape_to_plot(shape, ax=None, ...)`

Accept `Polygon`, `MultiPolygon`, list of geometries, or `GeoDataFrame`.
Draws on the supplied axis or `plt.gca()`.
Reprojects the geometry to the axis's CRS if needed (via the `crs_geometry=` arg).
The standard "draw the AOI on top of the imagery" pattern.

### `plot_segmentation_mask(mask, color_array=None, ...)`

Class-label rasters need a categorical colormap and a legend, not a continuous colorbar.
This function:

1. Converts the integer label raster to RGB using `color_array` (or a sensible default palette).
2. Adds a legend with one entry per class.
3. Handles nodata transparency.

Useful for showing CNN segmentation outputs in notebooks.

### `colorbar_next_to(im, ax, fig=None, ...)`

The "don't squeeze the main axis" colorbar.
Uses `mpl_toolkits.axes_grid1.make_axes_locatable` to allocate a thin axis to the right of the main one, sized in the same proportion as the figure.
Standard matplotlib idiom that never quite works on the first try; this packages it.

### Sharp edges

- **`show` reads the entire data into memory.** Calls `data.load()` if it's a reader.
  For large rasters, downsample first via `read.resize` or read from an overview level (Chapter 3 §8).
- **RGB ordering is `(C, H, W)` → matplotlib's `(H, W, C)`.** The function does the transpose; user code that bypasses `show` and does `ax.imshow(gt.values)` directly will get the wrong shape.
- **Tick labels are in the source CRS.** If you want lat/lon ticks on a UTM-projected raster, reproject first (Chapter 5 §4).
- **No interactive zoom optimisation.** This is a static-figure module.
  For interactive maps use `lonboard` / `leafmap` against the COG-on-disk ([Chapter 12](12_save.md)).
- **Matplotlib is a hard import.** Unlike xarray, no `try/except`.
  Importing `georeader.plot` requires matplotlib to be installed.

---

## 4. Why these three are grouped here

None of `io`, `dataarray`, `plot` carry the architectural weight of the modules in Parts I–IV. They're small, single-purpose, and you reach for them when you need them rather than designing pipelines around them.
Grouping into one chapter avoids three thin chapters with little to say.

The pattern they share: **delegate to a well-maintained external library, paper over the sharp edges that come up in real RS workflows.** `safe_open_netcdf` papers over the engine-format mismatch; `to/fromDataArray` papers over the xarray-vs-rasterio metadata convention gap; `show` papers over the matplotlib boilerplate for georeferenced display.

---

## 5. Connection to `geotoolz`

These three modules don't have direct `geotoolz` operators wrapping them, but each shows up indirectly:

- **`io.safe_open_netcdf`** — used inside the curvilinear-sensor readers (EMIT, PRISMA, EnMAP).
  When `geotoolz.presets.emit.EMIT_METHANE_MF` runs `georeader.readers.emit.load(scene_path)`, it's `safe_open_netcdf` underneath.
- **`dataarray.toDataArray` / `fromDataArray`** — the `geotoolz`/`xr_toolz` substrate-bridge example in [`geotoolz.md` §10.4](../plans/geotoolz/geotoolz.md) is literally these two functions.
  They keep the libraries decoupled while still letting users compose pipelines across substrates.
- **`plot.show`** — `geotoolz.radiometry` viz operators (`PercentileClip → Gamma → MinMax → S2_L2A_RGB`) produce visualisation-shaped `GeoTensor`s; the natural last step is `plot.show(...)` for inline display in notebooks.

Next chapter: [14_sentinel2.md](14_sentinel2.md) — the Sentinel-2 SAFE reader (1845 LOC, the largest single file in the package).
