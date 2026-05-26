# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "Patching — Field backends (RasterField, XarrayField, RioXarrayField, …)"
# ---
#
# # Backends: one Patcher, five `Field` adapters
#
# `geopatcher` keeps the locality algebra (`Geometry`, `Sampler`,
# `Window`, `Aggregation`) decoupled from the substrate via the `Field` /
# `Domain` Protocols. The same `SpatialPatcher` therefore drives:
#
# | Field | Domain | Backend |
# |---|---|---|
# | `RasterField` | `RasterDomain` (`GeoDataBase`) | `RasterioReader`, `GeoTensor`, `AsyncGeoTIFFReader` |
# | `RioXarrayField` | `RasterDomain` | rioxarray `DataArray` |
# | `XarrayField` | `GridDomain` | `xarray.DataArray` (non-raster) |
# | `GeoPandasField` | `VectorDomain` or `PointDomain` | `geopandas.GeoDataFrame` |
# | `XvecField` | `PointDomain` | `xvec.Dataset` |
#
# Each adapter is gated behind an optional extra (`grid`, `vector`, `point`,
# `xarray-raster`); install with `pip install 'geopatcher[patch-full]'`.

# %%
import html

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor
from IPython.display import HTML


# Collapsed repr helper — wrap every backend's rich display in a <details>
# block so the reader expands the ones they want to inspect rather than
# scrolling past everything inlined. Falls back to the text repr inside a
# <pre> when the object has no `_repr_html_` (e.g. `GeoTensor`).
def collapsed(obj, summary: str = "click to inspect") -> HTML:
    body = (
        obj._repr_html_()
        if hasattr(obj, "_repr_html_")
        else "<pre style='margin:0'>" + html.escape(repr(obj)) + "</pre>"
    )
    return HTML(
        f"<details><summary><b>{html.escape(summary)}</b></summary>{body}</details>"
    )


from geopatcher import (
    Patch,
    RasterField,
    SpatialBoxcar,
    SpatialByIndex,
    SpatialExplicit,
    SpatialKNNGraph,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRadiusGraph,
    SpatialRectangular,
    SpatialRegularStride,
)


# %% [markdown]
# ## 1. `RasterField` — `GeoTensor` / `RasterioReader`
#
# The canonical raster backend. Wraps anything satisfying georeader's
# `GeoData` Protocol — `GeoTensor` (in-memory), `RasterioReader` (lazy
# file-backed), `AsyncGeoTIFFReader` (lazy async). The Patcher consumes them
# identically.

# %%
arr = np.outer(np.linspace(0, 1, 32), np.linspace(0, 1, 32)).astype(np.float32)
print(f"raster arr.shape: {arr.shape}")  # (32, 32)
gt = GeoTensor(values=arr, transform=rasterio.Affine.identity(), crs="EPSG:32630")

# %% [markdown]
# Underlying `GeoTensor` (numpy subclass + CRS/affine):

# %%
collapsed(gt, summary="GeoTensor")

# %%
raster_field = RasterField(gt)
print(f"raster_field.domain.shape: {raster_field.domain.shape}")  # (32, 32)
print(f"raster_field.domain.crs:   {raster_field.domain.crs}")

raster_patcher = SpatialPatcher(
    geometry=SpatialRectangular(size=(8, 8)),
    sampler=SpatialRegularStride(step=8),
    window=SpatialBoxcar(),
    aggregation=SpatialOverlapAdd(),
)
raster_patches = list(raster_patcher.split(raster_field))
print(f"raster: {len(raster_patches)} patches")
print(f"  first.indices: {raster_patches[0].indices}")
print(f"  first.data.shape: {raster_patches[0].data.shape}")

# %% [markdown]
# ## 2. `RioXarrayField` — rioxarray-flavoured `DataArray`
#
# Same raster domain, accessed through the `xarray` surface (chunked Dask
# reads, unified xarray pipelines). The Patcher sees the affine + shape via
# the rioxarray accessor and treats it identically to a `RasterioReader`.

# %%
import xarray as xr


# Default-collapse the inner sections of xarray's HTML repr so users have
# to click into Coordinates / Indexes / Attributes / Data themselves.
xr.set_options(
    display_expand_data=False,
    display_expand_attrs=False,
    display_expand_coords=False,
    display_expand_data_vars=False,
    display_expand_indexes=False,
)

from geopatcher import RioXarrayField


xr_raster = xr.DataArray(
    arr[None, :, :],  # (band, y, x) — rioxarray convention
    dims=("band", "y", "x"),
    coords={
        "band": [1],
        "y": np.linspace(31.5, 0.5, 32),
        "x": np.linspace(0.5, 31.5, 32),
    },
)
xr_raster = xr_raster.rio.write_crs("EPSG:32630")
xr_raster = xr_raster.rio.write_transform(rasterio.Affine.identity())

# %% [markdown]
# Underlying `xarray.DataArray` with the rioxarray `.rio` accessor:

# %%
collapsed(xr_raster, summary="xarray.DataArray (rioxarray-flavoured)")

# %%
rio_xr_field = RioXarrayField(xr_raster)
print(
    f"rio_xr_field.domain.shape: {rio_xr_field.domain.shape}, "
    f"transform: {rio_xr_field.domain.transform}"
)

# %% [markdown]
# ## 3. `XarrayField` — non-raster N-D grid
#
# For dense, labeled cubes that aren't necessarily raster — climate
# reanalyses, model output. The Patcher dispatches `SpatialRectangular`
# on `GridDomain`, slicing through `xarray.DataArray.isel`.

# %%
from geopatcher import XarrayField


# Tiny (time=5, lat=64, lon=64) climate-like cube
cube = xr.DataArray(
    np.random.default_rng(0).standard_normal((5, 64, 64)).astype(np.float32),
    dims=("time", "lat", "lon"),
    coords={
        "time": np.arange(5),
        "lat": np.linspace(-90, 90, 64),
        "lon": np.linspace(-180, 180, 64),
    },
    name="temperature_anomaly",
)

# %% [markdown]
# Underlying `xarray.DataArray` (labeled coords, named dims, rich repr):

# %%
collapsed(cube, summary="xarray.DataArray (time × lat × lon)")

# %%
grid_field = XarrayField(cube)
print(f"grid_field.domain.shape: {grid_field.domain.shape}")
print(f"grid_field.domain.coords keys: {list(grid_field.domain.coords)}")

grid_patcher = SpatialPatcher(
    geometry=SpatialRectangular(size=(5, 16, 16)),  # full time × 16 lat × 16 lon
    sampler=SpatialRegularStride(step=(5, 16, 16)),
    window=SpatialBoxcar(),
    aggregation=SpatialByIndex(),  # ragged-friendly
)
grid_patches = list(grid_patcher.split(grid_field))
print(f"grid: {len(grid_patches)} patches")
print(f"  first.anchor: {grid_patches[0].anchor}")
print(f"  first.indices: {grid_patches[0].indices}")
print(f"  first.data.da.shape: {grid_patches[0].data.da.shape}")

# %% [markdown]
# ## 4. `GeoPandasField` — vector geometries
#
# Polygons (or any vector geometry) over a `VectorDomain`. The `SpatialKNNGraph`
# / `SpatialRadiusGraph` geometries dispatch on the domain's spatial index
# to find geometries near each anchor.

# %%
import geopandas as gpd
import shapely
from geopatcher import GeoPandasField


# Synthetic admin polygons on a 10×10 grid
polygons = []
for i in range(8):
    for j in range(8):
        polygons.append(shapely.box(j * 1.0, i * 1.0, (j + 1) * 1.0, (i + 1) * 1.0))
gdf = gpd.GeoDataFrame(
    {"id": np.arange(len(polygons)), "geometry": polygons},
    crs="EPSG:32630",
)

# %% [markdown]
# Underlying `geopandas.GeoDataFrame` (pandas DataFrame + a geometry column):

# %%
collapsed(gdf.head(8), summary="geopandas.GeoDataFrame (polygons)")

# %%
vector_field = GeoPandasField(gdf)
print(f"vector_field.domain.crs: {vector_field.domain.crs}")
print(f"vector_field.domain.bounds: {vector_field.domain.bounds}")

# Find all polygons within radius 1.5 of the centre of the grid
vector_patcher = SpatialPatcher(
    geometry=SpatialRadiusGraph(radius=1.5),
    sampler=SpatialExplicit(anchors_=[shapely.Point(4.0, 4.0)]),
    window=SpatialBoxcar(),
    aggregation=SpatialByIndex(),
)
vector_patches = list(vector_patcher.split(vector_field))
print(f"vector: {len(vector_patches)} patches")
print(f"  first.indices length: {len(vector_patches[0].indices)}")
print(f"  matching polygon IDs: {vector_patches[0].data.gdf['id'].tolist()}")

# %% [markdown]
# ## 5. `GeoPandasField(as_points=True)` — point cloud via geopandas
#
# When every geometry is a `shapely.Point`, ask for a `PointDomain` view —
# the field builds a `cKDTree` over the point coordinates so `KNNGraph` /
# `RadiusGraph` queries are cheap.

# %%
rng = np.random.default_rng(0)
pts = rng.uniform(-1, 1, size=(80, 2))
pt_gdf = gpd.GeoDataFrame(
    {"value": rng.standard_normal(80)},
    geometry=gpd.points_from_xy(pts[:, 0], pts[:, 1]),
    crs="EPSG:32630",
)

# %% [markdown]
# Underlying point-only `geopandas.GeoDataFrame` (same shape, different
# geometry type — the `as_points=True` flag tells the adapter to expose a
# `PointDomain` with a `cKDTree` instead of a `VectorDomain`):

# %%
collapsed(pt_gdf.head(6), summary="geopandas.GeoDataFrame (points)")

# %%
point_field = GeoPandasField(pt_gdf, as_points=True)
print(f"point_field.domain.coords.shape: {point_field.domain.coords.shape}")
print(f"point_field.domain has kdtree: {point_field.domain.kdtree is not None}")

point_patcher = SpatialPatcher(
    geometry=SpatialKNNGraph(k=5),
    sampler=SpatialExplicit(anchors_=[np.array([0.0, 0.0])]),
    window=SpatialBoxcar(),
    aggregation=SpatialByIndex(),
)
point_patches = list(point_patcher.split(point_field))
print(f"point: {len(point_patches)} patches")
print(f"  k-NN neighbours: {point_patches[0].indices.tolist()}")
print(f"  data type: {type(point_patches[0].data).__name__}")

# %% [markdown]
# ## 6. `XvecField` — xvec data cubes for in-situ multivariate data
#
# `xvec` puts a `shapely.Point` geometry coordinate on an `xarray.Dataset`,
# which is the modern pattern for stations / floats / swath samples with
# multiple variables and times.

# %%
import xvec  # noqa: F401  — registers the .xvec accessor
from geopatcher import XvecField


# Synthetic xvec dataset: 30 stations, 24 hourly measurements each
n_stations, n_hours = 30, 24
station_xy = rng.uniform(-1, 1, size=(n_stations, 2))
station_geoms = gpd.points_from_xy(station_xy[:, 0], station_xy[:, 1])
xvec_ds = xr.Dataset(
    {
        "temperature": (
            ("geometry", "time"),
            rng.standard_normal((n_stations, n_hours)),
        ),
        "pressure": (
            ("geometry", "time"),
            1000 + rng.standard_normal((n_stations, n_hours)),
        ),
    },
    coords={"geometry": station_geoms, "time": np.arange(n_hours)},
).xvec.set_geom_indexes("geometry", crs="EPSG:32630")

# %% [markdown]
# Underlying `xarray.Dataset` with an xvec-registered `geometry` coord
# (multivariate stations × time data cube):

# %%
collapsed(xvec_ds, summary="xarray.Dataset (xvec, stations × time)")

# %%
xvec_field = XvecField(xvec_ds)
print(f"xvec_field.domain.coords.shape: {xvec_field.domain.coords.shape}")

xvec_patcher = SpatialPatcher(
    geometry=SpatialKNNGraph(k=4),
    sampler=SpatialExplicit(anchors_=[np.array([0.0, 0.0])]),
    window=SpatialBoxcar(),
    aggregation=SpatialByIndex(),
)
xvec_patches = list(xvec_patcher.split(xvec_field))
print(f"xvec: {len(xvec_patches)} patches")
print(f"  k-NN neighbours: {xvec_patches[0].indices.tolist()}")
print(f"  data.ds.dims: {dict(xvec_patches[0].data.ds.sizes)}")

# %% [markdown]
# ## A common operator
#
# Same `SpatialPatcher` shape, different backends. The composition algebra
# (`GridSampler` → `ApplyToChips` → `Stitch`) doesn't care which substrate
# it's running on — the operator just sees `patch.data` in whatever shape the
# backend produces.

# %%
from geotoolz.core import Lambda


def _patch_data_shape(d) -> str:
    """Walk a few common substrate-specific attrs to find a shape-like value."""
    if hasattr(d, "shape"):
        return repr(tuple(d.shape))
    for attr in ("da", "gdf", "ds"):
        sub = getattr(d, attr, None)
        if sub is not None and hasattr(sub, "shape"):
            return f"<{attr}> {tuple(sub.shape)}"
        if sub is not None and hasattr(sub, "sizes"):
            return f"<{attr}> dims={dict(sub.sizes)}"
    return f"<no shape: {type(d).__name__}>"


def _summarise(p: Patch) -> dict:
    return {
        "anchor": repr(p.anchor)[:40],
        "data_type": type(p.data).__name__,
        "data_shape": _patch_data_shape(p.data),
    }


_label = Lambda(_summarise, name="summary")

for name, patches in [
    ("RasterField", raster_patches),
    ("XarrayField (grid)", grid_patches),
    ("GeoPandasField (vector)", vector_patches),
    ("GeoPandasField (points)", point_patches),
    ("XvecField", xvec_patches),
]:
    summary = _label(patches[0])
    print(f"{name:>26s}: {summary}")

# %% [markdown]
# ## What this proves
#
# - The four-axis Patcher composition is substrate-agnostic. Same
#   `Sampler` / `Window` / `Aggregation` triples work across every Field;
#   only the `Geometry` dispatches on the `Domain` type.
# - Adding a new substrate is a single ~30-line `Field` adapter — implement
#   `domain`, `select(indexer)`, `with_data(array)`, gate the optional
#   dependency with a friendly import error.
# - Choice of backend is a deployment concern, not a modelling one. The same
#   pipeline that runs against an in-memory `GeoTensor` in a notebook runs
#   against a lazy `RasterioReader` in production with no code change beyond
#   the field wrapper.
