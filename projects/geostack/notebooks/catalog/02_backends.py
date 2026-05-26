# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "Catalog — raster / xarray / vector backends"
# ---
#
# # Catalog backends: raster, xarray, vector
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/geocatalog/blob/main/docs/notebooks/catalog_backends.ipynb)
#
# Three builders share one `GeoCatalog` shape. This notebook builds a
# small catalog for each and prints the underlying data structure so
# you can see what each backend records.

# %%
import subprocess
import sys


try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "geocatalog @ git+https://github.com/jejjohnson/geocatalog@main",
        ],
        check=True,
    )

# %%
import tempfile
from pathlib import Path

import geocatalog as gc
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry
import xarray as xr
from rasterio.transform import from_bounds


tmp = Path(tempfile.mkdtemp(prefix="geocatalog_backends_"))
print(f"workdir: {tmp}")

# %% [markdown]
# ## 1. Raster backend
#
# The canonical case: a directory of GeoTIFFs, indexed by filename date.


# %%
def write_tif(name: str, value: int, bounds: tuple, crs: str = "EPSG:32629") -> Path:
    path = tmp / name
    xmin, ymin, xmax, ymax = bounds
    transform = from_bounds(xmin, ymin, xmax, ymax, 32, 32)
    data = np.full((3, 32, 32), value, dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=3,
        dtype="uint16",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


raster_files = [
    write_tif(
        "S2_T29SND_20240115_r0.tif", 10, (500_000, 4_000_000, 500_320, 4_000_320)
    ),
    write_tif(
        "S2_T29SND_20240116_r0.tif", 20, (500_320, 4_000_000, 500_640, 4_000_320)
    ),
]
raster_cat = gc.build_raster_catalog(
    raster_files,
    filename_regex=r"S2_T29SND_(?P<date>\d{8}).*\.tif",
    target_crs="EPSG:32629",
)
print(raster_cat)
print(f"backend: {raster_cat.backend}")

# %% [markdown]
# The catalog is literally a `GeoDataFrame` — show it.

# %%
raster_cat.gdf

# %% [markdown]
# ## 2. Xarray backend (extras: `xarray-raster`)
#
# A small NetCDF on disk; bounds come from the coord min/max, time from
# a configurable coordinate (default `"time"`).

# %%
nc_path = tmp / "modis_2024.nc"
ds = xr.Dataset(
    {
        "ndvi": (
            ("time", "y", "x"),
            np.linspace(0, 1, 5 * 16 * 16, dtype=np.float32).reshape(5, 16, 16),
        )
    },
    coords={
        "time": pd.date_range("2024-01-01", periods=5, freq="D"),
        "y": np.linspace(40.5, 40.0, 16),
        "x": np.linspace(-3.5, -3.0, 16),
    },
)
ds.to_netcdf(nc_path)
print("Original xarray.Dataset:")
print(ds)

# %%
xa_cat = gc.build_xarray_catalog(
    [nc_path], target_crs="EPSG:4326", data_vars=["ndvi"], time_var="time"
)
print(xa_cat)
print(f"backend: {xa_cat.backend}")
print(f"n_timesteps: {int(xa_cat.gdf['n_timesteps'].iloc[0])}")
xa_cat.gdf

# %% [markdown]
# ## 3. Vector backend
#
# Polygon footprints come from each file's `total_bounds` in the target
# CRS. Loaders rasterise into a label `GeoTensor` for ML targets.

# %%
vec_gdf = gpd.GeoDataFrame(
    {
        "class_id": [1, 2],
        "geometry": [
            shapely.geometry.box(500_000, 4_000_000, 500_160, 4_000_160),
            shapely.geometry.box(500_160, 4_000_160, 500_320, 4_000_320),
        ],
    },
    crs="EPSG:32629",
)
vec_path = tmp / "labels_20240115.gpkg"
vec_gdf.to_file(vec_path, driver="GPKG")

print("Original GeoDataFrame on disk:")
print(vec_gdf)

# %%
vec_cat = gc.build_vector_catalog(
    [vec_path], filename_regex=r"labels_(?P<date>\d{8})\.gpkg"
)
print(vec_cat)
print(f"backend: {vec_cat.backend}")
vec_cat.gdf

# %% [markdown]
# ## Rasterising the labels for an AOI

# %%
aoi = gc.GeoSlice(
    bounds=(500_000, 4_000_000, 500_320, 4_000_320),
    interval=pd.Interval(
        pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16"), closed="both"
    ),
    resolution=(10.0, 10.0),
    crs="EPSG:32629",
)
label_tensor = gc.load_vector(
    vec_cat, aoi, task="semantic_segmentation", label_field="class_id"
)
print(f"label_tensor.values.shape: {label_tensor.values.shape}   # (1, 32, 32)")
print(f"unique class IDs: {sorted(np.unique(label_tensor.values).tolist())}")

plt.imshow(label_tensor.values[0], cmap="tab10", vmin=0, vmax=4)
plt.title("Rasterised semantic-segmentation labels")
plt.colorbar(shrink=0.7)
plt.show()

# %% [markdown]
# ## GeoParquet roundtrip
#
# Any catalog can be persisted as a GeoParquet artifact. The Phase 2
# DuckDB backend reads the same format.

# %%
parquet_path = tmp / "raster_cat.parquet"
gc.to_geoparquet(raster_cat, parquet_path)
print(f"wrote {parquet_path.stat().st_size} bytes to {parquet_path}")

recovered = gc.from_geoparquet(parquet_path)
print(f"recovered: {recovered}")
print(f"len matches: {len(recovered) == len(raster_cat)}")
