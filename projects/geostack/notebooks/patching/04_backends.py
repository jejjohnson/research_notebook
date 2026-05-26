# ---
# jupyter:
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
# # Backends — one Patcher, five `Field` adapters on real heterogeneous data
#
# `geopatcher` keeps the locality algebra (`Geometry`, `Sampler`,
# `Window`, `Aggregation`) decoupled from the substrate via the `Field`
# and `Domain` Protocols. The same `SpatialPatcher` therefore drives:
#
# | Field | Domain | Backend |
# |---|---|---|
# | `RasterField` | `RasterDomain` (`GeoDataBase`) | `RasterioReader`, `GeoTensor`, `AsyncGeoTIFFReader` |
# | `RioXarrayField` | `RasterDomain` | rioxarray `DataArray` |
# | `XarrayField` | `GridDomain` | `xarray.DataArray` (non-raster) |
# | `GeoPandasField` | `VectorDomain` or `PointDomain` | `geopandas.GeoDataFrame` |
# | `XvecField` | `PointDomain` | `xvec.Dataset` |
#
# Every section below uses a **real, heterogeneous data source** so the
# substrate diversity that motivates this module shows up in the
# substrates themselves:
#
# - **Raster (`RasterField` + `RioXarrayField`)** — a Sentinel-2 L2A
#   NIR chip over Lake Tahoe.
# - **Grid (`XarrayField`)** — the Copernicus DEM (GLO-30) over the
#   same AOI — a real geophysical grid with native coordinates.
# - **Vector polygons (`GeoPandasField`)** — Natural Earth admin-1
#   polygons for the US Pacific states.
# - **Vector points (`GeoPandasField(as_points=True)`)** — GBIF
#   occurrence points for California live oak.
# - **xvec data cube (`XvecField`)** — the GBIF points lifted into an
#   `xarray.Dataset` with a `xvec` geometry index plus a synthetic
#   "temperature" + "elevation" multivariate measurement per station.

# %%
import html
from io import BytesIO

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import requests
import rioxarray
import shapely
import xarray as xr
import xvec  # noqa: F401  — registers the .xvec accessor
from geopatcher import (
    GeoPandasField,
    Patch,
    RasterField,
    RioXarrayField,
    SpatialBoxcar,
    SpatialByIndex,
    SpatialExplicit,
    SpatialKNNGraph,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRadiusGraph,
    SpatialRectangular,
    SpatialRegularStride,
    XarrayField,
    XvecField,
)
from georeader.geotensor import GeoTensor
from geotoolz import Lambda
from IPython.display import HTML

from geostack import (
    LAKE_TAHOE_BBOX,
    load_gbif_points,
    load_natural_earth_admin1,
    load_s2_chip,
)


def collapsed(obj, summary: str = "click to inspect") -> HTML:
    """Wrap a rich repr in a collapsible <details> block."""
    body = (
        obj._repr_html_()
        if hasattr(obj, "_repr_html_")
        else "<pre style='margin:0'>" + html.escape(repr(obj)) + "</pre>"
    )
    return HTML(
        f"<details><summary><b>{html.escape(summary)}</b></summary>{body}</details>"
    )


xr.set_options(
    display_expand_data=False,
    display_expand_attrs=False,
    display_expand_coords=False,
    display_expand_data_vars=False,
    display_expand_indexes=False,
)

# %% [markdown]
# ## 1. `RasterField` — `GeoTensor` / `RasterioReader`
#
# The canonical raster backend. Wraps anything satisfying georeader's
# `GeoData` Protocol — `GeoTensor` (in-memory), `RasterioReader` (lazy
# file-backed), `AsyncGeoTIFFReader` (lazy async). The Patcher consumes
# them identically.

# %%
gt_bgrn = load_s2_chip(bbox=LAKE_TAHOE_BBOX)
nir = np.asarray(gt_bgrn)[3].astype("float32") * 1e-4  # NIR reflectance
# Take a 512×512 corner crop so the patcher's grid is easy to plot.
nir_crop = nir[600:1112, 200:712]
crop_transform = gt_bgrn.transform * rasterio.Affine.translation(200, 600)
gt = GeoTensor(
    values=nir_crop,
    transform=crop_transform,
    crs=gt_bgrn.crs,
    fill_value_default=0.0,
)
print(f"raster GeoTensor.shape: {gt.shape}  crs: {gt.crs}")

# %%
collapsed(gt, summary="GeoTensor (S2 L2A B08 over Lake Tahoe)")

# %%
raster_field = RasterField(gt)
print(f"raster_field.domain.shape: {raster_field.domain.shape}")
print(f"raster_field.domain.crs:   {raster_field.domain.crs}")

raster_patcher = SpatialPatcher(
    geometry=SpatialRectangular(size=(128, 128)),
    sampler=SpatialRegularStride(step=128),
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
# Same raster domain, accessed through the `xarray` surface (chunked
# Dask reads, unified xarray pipelines). The Patcher sees the affine +
# shape via the rioxarray accessor and treats it identically to a
# `RasterioReader`. We open the same Sentinel-2 B04 asset directly via
# `rioxarray` so the source is identical.

# %%
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
item = next(
    iter(
        catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=LAKE_TAHOE_BBOX,
            datetime="2024-06-01/2024-07-15",
            query={"eo:cloud_cover": {"lt": 5}, "s2:mgrs_tile": {"eq": "10SGJ"}},
        ).items()
    )
)
xr_raster = (
    rioxarray.open_rasterio(item.assets["B04"].href, masked=False)
    .rio.clip_box(*LAKE_TAHOE_BBOX, crs="EPSG:4326")
    .isel(x=slice(200, 712), y=slice(600, 1112))  # match the corner crop above
)
print(f"xr_raster shape: {xr_raster.shape}  crs: {xr_raster.rio.crs}")

# %%
collapsed(xr_raster, summary="xarray.DataArray (rioxarray-flavoured)")

# %%
rio_xr_field = RioXarrayField(xr_raster)
print(f"rio_xr_field.domain.shape: {rio_xr_field.domain.shape}")
print(f"rio_xr_field.domain.transform: {rio_xr_field.domain.transform}")

# %% [markdown]
# ## 3. `XarrayField` — non-raster N-D grid
#
# For dense, labeled cubes that aren't necessarily raster — climate
# reanalyses, model output, DEM stacks. We open the Copernicus
# GLO-30 DEM tile over the same AOI as a `(y, x)` `xarray.DataArray`
# with native lat/lon coordinates.

# %%
dem_items = list(
    catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=LAKE_TAHOE_BBOX,
    ).items()
)
dem = (
    rioxarray.open_rasterio(dem_items[0].assets["data"].href, masked=False)
    .squeeze("band", drop=True)
    .rio.clip_box(*LAKE_TAHOE_BBOX, crs="EPSG:4326")
)
print(f"DEM shape: {dem.shape}  crs: {dem.rio.crs}")

# Convert from rioxarray-flavoured (with rio accessor) to a plain
# xarray cube treated as a non-raster grid — drop the rio metadata so
# the GridDomain path is exercised rather than RasterDomain.
dem_cube = xr.DataArray(
    dem.values,
    dims=("lat", "lon"),
    coords={"lat": dem.y.values, "lon": dem.x.values},
    name="elevation_m",
)

# %%
collapsed(dem_cube, summary="xarray.DataArray (Copernicus DEM GLO-30)")

# %%
grid_field = XarrayField(dem_cube)
print(f"grid_field.domain.shape: {grid_field.domain.shape}")
print(f"grid_field.domain.coords keys: {list(grid_field.domain.coords)}")

# Patch every 100 lat × 100 lon block — coarse enough for a fast demo.
grid_patcher = SpatialPatcher(
    geometry=SpatialRectangular(size=(100, 100)),
    sampler=SpatialRegularStride(step=(100, 100)),
    window=SpatialBoxcar(),
    aggregation=SpatialByIndex(),  # ragged-friendly
)
grid_patches = list(grid_patcher.split(grid_field))
print(f"grid: {len(grid_patches)} patches")
print(f"  first.anchor: {grid_patches[0].anchor}")
print(f"  first.data.da.shape: {grid_patches[0].data.da.shape}")

# %% [markdown]
# ## 4. `GeoPandasField` — vector polygons
#
# Real vector geometries: **Natural Earth admin-1** polygons for the
# Pacific states. The `SpatialRadiusGraph` geometry dispatches on
# the polygon centroid to find all admin units within R degrees of an
# anchor point.

# %%
ne = load_natural_earth_admin1()
pacific = ne[ne["name"].isin(
    ["California", "Oregon", "Washington", "Nevada", "Idaho", "Arizona", "Utah"]
)].copy()
pacific = pacific.reset_index(drop=True)[["name", "geometry"]]
# Reproject to UTM 10N so the radius below is in metres.
pacific = pacific.to_crs("EPSG:32610")
print(f"pacific.shape: {pacific.shape}")

# %%
collapsed(pacific.head(8), summary="GeoDataFrame — Pacific admin-1 polygons (UTM 10N)")

# %%
vector_field = GeoPandasField(pacific)
print(f"vector_field.domain.crs: {vector_field.domain.crs}")
print(f"vector_field.domain.bounds: {vector_field.domain.bounds}")

# Anchor near Lake Tahoe, find every polygon within 600 km.
anchor_xy = shapely.Point(744000.0, 4325000.0)
vector_patcher = SpatialPatcher(
    geometry=SpatialRadiusGraph(radius=600_000),  # 600 km
    sampler=SpatialExplicit(anchors_=[anchor_xy]),
    window=SpatialBoxcar(),
    aggregation=SpatialByIndex(),
)
vector_patches = list(vector_patcher.split(vector_field))
print(f"vector: {len(vector_patches)} patches")
print(f"  first.indices length: {len(vector_patches[0].indices)}")
print(f"  matching names: {vector_patches[0].data.gdf['name'].tolist()}")

# %% [markdown]
# ## 5. `GeoPandasField(as_points=True)` — point cloud via geopandas
#
# Real GBIF occurrence points (California live oak) lifted into a
# `PointDomain` with a `cKDTree` so `KNNGraph` / `RadiusGraph`
# queries are cheap.

# %%
oak = load_gbif_points(species_key=5285750, limit=400)
oak = oak.to_crs("EPSG:32610")
print(f"oak shape: {oak.shape}")

# %%
collapsed(oak.head(6), summary="GeoDataFrame — Quercus agrifolia GBIF points")

# %%
point_field = GeoPandasField(oak, as_points=True)
print(f"point_field.domain.coords.shape: {point_field.domain.coords.shape}")
print(f"point_field.domain has kdtree: {point_field.domain.kdtree is not None}")

point_patcher = SpatialPatcher(
    geometry=SpatialKNNGraph(k=8),
    sampler=SpatialExplicit(anchors_=[np.array([744000.0, 4325000.0])]),
    window=SpatialBoxcar(),
    aggregation=SpatialByIndex(),
)
point_patches = list(point_patcher.split(point_field))
print(f"point: {len(point_patches)} patches")
print(f"  k-NN neighbour indices: {point_patches[0].indices.tolist()}")

# %% [markdown]
# ## 6. `XvecField` — xvec data cubes for in-situ multivariate data
#
# `xvec` puts a `shapely.Point` geometry coordinate on an
# `xarray.Dataset`, which is the modern pattern for stations / floats /
# swath samples with multiple variables and times. We lift the same
# GBIF points into a synthetic temperature + elevation cube — picture
# 24 hourly readings per occurrence record.

# %%
n_stations = len(oak)
n_hours = 24
rng = np.random.default_rng(0)
station_xy = np.column_stack([oak.geometry.x.values, oak.geometry.y.values])
station_geoms = gpd.points_from_xy(station_xy[:, 0], station_xy[:, 1])
xvec_ds = xr.Dataset(
    {
        "temperature": (
            ("geometry", "time"),
            15 + 5 * rng.standard_normal((n_stations, n_hours)),
        ),
        "elevation": (
            ("geometry",),
            100 + 800 * rng.random(n_stations),
        ),
    },
    coords={
        "geometry": station_geoms,
        "time": np.arange(n_hours),
    },
).xvec.set_geom_indexes("geometry", crs="EPSG:32610")

# %%
collapsed(xvec_ds, summary="xarray.Dataset (xvec, stations × time)")

# %%
xvec_field = XvecField(xvec_ds)
print(f"xvec_field.domain.coords.shape: {xvec_field.domain.coords.shape}")

xvec_patcher = SpatialPatcher(
    geometry=SpatialKNNGraph(k=4),
    sampler=SpatialExplicit(anchors_=[np.array([744000.0, 4325000.0])]),
    window=SpatialBoxcar(),
    aggregation=SpatialByIndex(),
)
xvec_patches = list(xvec_patcher.split(xvec_field))
print(f"xvec: {len(xvec_patches)} patches")
print(f"  k-NN neighbour indices: {xvec_patches[0].indices.tolist()}")
print(f"  data.ds.dims: {dict(xvec_patches[0].data.ds.sizes)}")

# %% [markdown]
# ## 7. A common operator across substrates
#
# Same `SpatialPatcher` shape, different backends. The composition
# algebra (`GridSampler` → `ApplyToChips` → `Stitch`) doesn't care
# which substrate it is running on — the operator just sees
# `patch.data` in whatever shape the backend produces.


# %%
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
        "anchor": repr(p.anchor)[:50],
        "data_type": type(p.data).__name__,
        "data_shape": _patch_data_shape(p.data),
    }


_label = Lambda(_summarise, name="summary")
for name, patches in [
    ("RasterField (S2 NIR)", raster_patches),
    ("XarrayField (DEM)", grid_patches),
    ("GeoPandasField (polygons)", vector_patches),
    ("GeoPandasField (GBIF points)", point_patches),
    ("XvecField (oak + temp/elev)", xvec_patches),
]:
    summary = _label(patches[0])
    print(f"{name:>32s}: {summary}")

# %% [markdown]
# ## What this proves
#
# - The four-axis Patcher composition is **substrate-agnostic**. Same
#   `Sampler` / `Window` / `Aggregation` triples work across every
#   Field; only the `Geometry` dispatches on the `Domain` type.
# - Adding a new substrate is a single ~30-line `Field` adapter —
#   implement `domain`, `select(indexer)`, `with_data(array)`, gate
#   the optional dependency with a friendly import error.
# - Choice of backend is a **deployment concern, not a modelling
#   one**. The same pipeline that runs against an in-memory
#   `GeoTensor` in a notebook runs against a lazy `RioXarrayField`
#   over a remote zarr archive in production with no code change
#   beyond the field wrapper.
