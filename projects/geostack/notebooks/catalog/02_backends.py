# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "Catalog — raster / xarray / vector backends"
# ---
#
# # Catalog backends — raster, xarray, vector on real data
#
# Three builders share one `GeoCatalog` shape. This notebook builds a
# catalog for each backend against **real public data**, then prints
# the underlying GeoDataFrame so the substrate diversity is concrete:
#
# | Backend | Real source |
# |---|---|
# | `build_raster_catalog` | Eight Sentinel-2 L2A scenes over Lake Tahoe (MPC) |
# | `build_xarray_catalog` | Copernicus DEM GLO-30 tile over the same AOI, saved as a NetCDF |
# | `build_vector_catalog` | Natural Earth admin-1 polygons (Pacific states) |

# %%
import tempfile
from pathlib import Path

import geocatalog as gc
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rioxarray
import xarray as xr

from geostack import (
    LAKE_TAHOE_BBOX,
    LAKE_TAHOE_TILE,
    load_natural_earth_admin1,
    load_stac_items,
)

tmp = Path(tempfile.mkdtemp(prefix="geocatalog_backends_"))
print(f"workdir: {tmp}")

# %% [markdown]
# ## 1. Raster backend — real Sentinel-2 L2A scenes
#
# Eight cloud-free scenes over Lake Tahoe (June–July 2024). The
# catalog stores the **signed asset URLs**; pixel data is read lazily
# on `load_raster`.

# %%
items = load_stac_items(
    "sentinel-2-l2a",
    LAKE_TAHOE_BBOX,
    "2024-06-01/2024-07-31",
    tile=LAKE_TAHOE_TILE,
    max_cloud_cover=15,
)
raster_cat = gc.build_raster_catalog(
    [it.assets["B04"].href for it in items],
    filename_regex=r".*_(?P<date>\d{8})T\d{6}_.*\.tif",
    target_crs="EPSG:32610",
)
print(raster_cat)
print(f"backend: {raster_cat.backend}")

# %% [markdown]
# The catalog is literally a `GeoDataFrame` — show it.

# %%
raster_cat.gdf[["filepath", "start_time", "end_time", "geometry"]].head()

# %% [markdown]
# ## 2. Xarray backend — Copernicus DEM
#
# A real DEM tile saved to disk as NetCDF; bounds come from the coord
# min/max, time from a configurable coordinate (default `"time"`).
# We add a singleton-time dim so the xarray builder's time-aware path
# is exercised even though a DEM is technically time-invariant.

# %%
mpc = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
dem_items = list(
    mpc.search(
        collections=["cop-dem-glo-30"],
        bbox=LAKE_TAHOE_BBOX,
    ).items()
)
dem = (
    rioxarray.open_rasterio(dem_items[0].assets["data"].href, masked=False)
    .squeeze("band", drop=True)
    .rio.clip_box(*LAKE_TAHOE_BBOX, crs="EPSG:4326")
)
nc_path = tmp / "cop_dem_lake_tahoe.nc"
ds = xr.Dataset(
    {"elevation": (("time", "y", "x"), dem.values[None, ...])},
    coords={
        "time": pd.date_range("2024-01-01", periods=1, freq="D"),
        "y": dem.y.values,
        "x": dem.x.values,
    },
)
ds.to_netcdf(nc_path)
print("Original xarray.Dataset:")
print(ds)

# %%
xa_cat = gc.build_xarray_catalog(
    [nc_path], target_crs="EPSG:4326", data_vars=["elevation"], time_var="time"
)
print(xa_cat)
print(f"backend: {xa_cat.backend}")
print(f"n_timesteps: {int(xa_cat.gdf['n_timesteps'].iloc[0])}")
xa_cat.gdf

# %% [markdown]
# ## 3. Vector backend — Natural Earth admin-1 polygons
#
# Polygon footprints come from each file's `total_bounds` in the
# target CRS. We persist one GeoPackage per state so the catalog has
# multiple rows.

# %%
ne = load_natural_earth_admin1()
pacific_states = ["California", "Oregon", "Nevada"]
vec_paths = []
for state in pacific_states:
    sub = ne[ne["name"] == state][["name", "geometry"]].copy()
    sub["class_id"] = pacific_states.index(state) + 1
    # Date-stamped filename so the regex picks up a date.
    p = tmp / f"admin1_{state.lower()}_20240101.gpkg"
    sub.to_file(p, driver="GPKG")
    vec_paths.append(p)
print(f"wrote {len(vec_paths)} admin GeoPackages to {tmp}")

# %%
vec_cat = gc.build_vector_catalog(
    vec_paths,
    filename_regex=r"admin1_[a-z]+_(?P<date>\d{8})\.gpkg",
)
print(vec_cat)
print(f"backend: {vec_cat.backend}")
vec_cat.gdf

# %% [markdown]
# ## 4. Rasterising the labels over the Lake Tahoe AOI
#
# `load_vector(..., task="semantic_segmentation")` turns the polygon
# overlay into a label `GeoTensor` that lines up with any imagery
# you load over the same AOI — the foundation for a vision-model
# training pipeline.

# %%
aoi = gc.GeoSlice(
    bounds=LAKE_TAHOE_BBOX,
    interval=pd.Interval(
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), closed="both"
    ),
    resolution=(0.001, 0.001),  # ~100 m at this latitude
    crs="EPSG:4326",
)
label_tensor = gc.load_vector(
    vec_cat, aoi, task="semantic_segmentation", label_field="class_id"
)
print(f"label_tensor.values.shape: {label_tensor.values.shape}")
print(f"unique class IDs: {sorted(np.unique(label_tensor.values).tolist())}")

fig, ax = plt.subplots(figsize=(7, 8))
im = ax.imshow(label_tensor.values[0], cmap="tab10", vmin=0, vmax=4)
ax.set_title("Pacific-states admin labels rasterised to Lake Tahoe AOI")
ax.axis("off")
cbar = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(["background", "California", "Oregon", "Nevada"])
plt.show()

# %% [markdown]
# ## 5. GeoParquet roundtrip
#
# Any catalog can be persisted as a GeoParquet artifact. The DuckDB
# backend (see [04_duckdb](04_duckdb.ipynb)) reads the same format
# directly without materialising the dataframe.

# %%
parquet_path = tmp / "raster_cat.parquet"
gc.to_geoparquet(raster_cat, parquet_path)
print(f"wrote {parquet_path.stat().st_size:,} bytes to {parquet_path}")

recovered = gc.from_geoparquet(parquet_path)
print(f"recovered: {recovered}")
print(f"len matches: {len(recovered) == len(raster_cat)}")

# %% [markdown]
# ## Recap
#
# Same `GeoCatalog` API, three substrates:
#
# | Builder | Reads | Stores per row |
# |---|---|---|
# | `build_raster_catalog` | GeoTIFFs / `.vsicurl/` URLs | filepath, bounds (footprint polygon), time interval |
# | `build_xarray_catalog` | NetCDF / zarr | filepath, bounds (from coord extent), time interval (from `time` coord), `n_timesteps` |
# | `build_vector_catalog` | GeoPackage / Shapefile / Parquet | filepath, bounds (`total_bounds`), time interval |
#
# All three round-trip through GeoParquet. All three implement the
# same `GeoCatalog` Protocol — `query`, `intersect`, `union`,
# `where`, `load_*` — so the [set algebra](03_set_algebra.ipynb) and
# [patching bridge](05_patching_bridge.ipynb) demos work uniformly
# regardless of which builder you used.
