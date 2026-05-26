# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# title: "Catalog — DuckDB-backed catalogs at scale"
# ---
#
# # DuckDB-backed catalogs at scale
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/geocatalog/blob/main/docs/notebooks/catalog_duckdb.ipynb)
#
# `DuckDBGeoCatalog` is the Phase-2 backend: a lazy SQL relation over a
# GeoParquet artifact. Same `GeoCatalog` Protocol as
# `InMemoryGeoCatalog`, but the rows live on disk (or in S3 / GCS /
# HuggingFace) and queries push down to the Parquet reader so you read
# only the row groups your AOI touches.
#
# This notebook builds a small catalog, persists it as GeoParquet,
# reopens it through DuckDB, and walks the Protocol surface:
# `query`, `intersect`, `union`, `iter_rows`, `materialize`.

# %%
import subprocess
import sys


try:
    import google.colab  # noqa: F401

    on_colab = True
except ImportError:
    on_colab = False

if on_colab:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "geocatalog[duckdb]"]
    )

# %%
import geocatalog as gc
import geopandas as gpd
import pandas as pd
import shapely.geometry


# %% [markdown]
# ## A small in-memory catalog
#
# We start from a hand-rolled `InMemoryGeoCatalog` of two tiles in
# UTM zone 29N — small enough to fit in RAM, but the same surface
# scales to 10⁶ rows once we route through DuckDB.

# %%
gdf = gpd.GeoDataFrame(
    {
        "geometry": [
            shapely.geometry.box(0, 0, 100, 100),
            shapely.geometry.box(200, 0, 300, 100),
        ],
        "start_time": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        "end_time": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
        "filepath": ["A.tif", "B.tif"],
    },
    geometry="geometry",
    crs="EPSG:32629",
)
mem = gc.InMemoryGeoCatalog(gdf, backend="raster")
mem

# %% [markdown]
# ## Persist as GeoParquet, reopen via DuckDB
#
# `to_geoparquet` writes the catalog with the GeoParquet 1.1 bbox
# covering struct, which DuckDB uses for predicate pushdown.
# `open_catalog(path)` is the factory — it prefers the DuckDB backend
# when `[duckdb]` is installed and silently falls back to the in-memory
# backend otherwise.

# %%
import pathlib
import tempfile


tmp = pathlib.Path(tempfile.mkdtemp())
gc.to_geoparquet(mem, tmp / "cat.parquet")

duck = gc.open_catalog(tmp / "cat.parquet")
duck

# %% [markdown]
# `len`, `total_bounds`, `temporal_extent` work just like the
# in-memory backend — but each is one SQL aggregate, not a Python
# loop.

# %%
print("rows           :", len(duck))
print("total_bounds   :", duck.total_bounds)
print("temporal_extent:", duck.temporal_extent)
print("config         :", duck.get_config())

# %% [markdown]
# ## Spatial + temporal queries push down
#
# A `GeoSlice` carries bounds + interval + CRS together. Passing one to
# `query` translates to a SQL `WHERE` clause that DuckDB can push down
# to the Parquet reader.

# %%
sl = gc.GeoSlice(
    bounds=(0, 0, 50, 50),
    interval=pd.Interval(
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        closed="both",
    ),
    resolution=(1.0, 1.0),
    crs="EPSG:32629",
)
hits = duck.query(sl)
print("matched", len(hits), "row(s):")
hits.materialize().gdf

# %% [markdown]
# ### Cross-CRS queries reproject internally
#
# An AOI in EPSG:4326 against a UTM-zone-29N catalog used to silently
# return zero rows in earlier homebrew catalogs (the §10.1 footgun in
# the design plan). The DuckDB backend reprojects the AOI before the
# SQL is built, so the right rows come back.

# %%
# UTM 29N (50, 50) ≈ (-13.488°, 0.00045°) in 4326.
duck.query(
    bounds=(-13.4885, 0.0001, -13.4880, 0.0008), crs="EPSG:4326"
).materialize().gdf

# %% [markdown]
# ## Set algebra: intersect + union as SQL joins
#
# `intersect` is a SQL spatial join clipped to `ST_Intersection`;
# `union` is `UNION ALL`. Both return new lazy relations.

# %%
labels = gc.InMemoryGeoCatalog(
    gpd.GeoDataFrame(
        {
            "geometry": [shapely.geometry.box(50, 50, 250, 150)],
            "start_time": [pd.Timestamp("2024-01-01")],
            "end_time": [pd.Timestamp("2024-01-04")],
            "filepath": ["labels.gpkg"],
        },
        geometry="geometry",
        crs="EPSG:32629",
    ),
    backend="vector",
)

joint = duck.intersect(labels)
joint.materialize().gdf

# %%
merged = duck.union(labels)
print("rows after union:", len(merged))

# %% [markdown]
# ## `iter_rows` — the streaming surface
#
# Loaders and the patcher bridge consume `CatalogRow`
# instances. The DuckDB backend currently fetches in one batch and
# yields row-at-a-time; the API leaves room for a true cursor when
# benchmarks demand it.

# %%
for row in duck.iter_rows():
    print(row.filepath, "—", row.geometry.bounds, "—", row.interval)

# %% [markdown]
# ## `materialize` — back to a GeoDataFrame when you need one
#
# When the rest of your pipeline expects a `GeoDataFrame`, pull the
# relation eagerly. Useful at the boundary between the catalog layer
# and a pandas-flavoured analytics step.

# %%
mat = duck.materialize()
mat.gdf

# %% [markdown]
# ## When to use DuckDB
#
# - Catalog scale past ~10⁵ rows (RAM ceiling for the gdf backend).
# - The catalog needs to be portable — a colleague queries it without
#   rebuilding.
# - You want cloud-hosted catalogs (`s3://bucket/cat.parquet`);
#   DuckDB's `httpfs` reads only the row groups your query touches.
# - You want full SQL escape-hatch power (`.sql("…")`).
#
# Stick with `InMemoryGeoCatalog` for prototyping, small training-set
# construction, or when you don't want the `[duckdb]` dependency.

# %% [markdown]
# ## Streaming build — `backend="duckdb"`
#
# The default builders (`build_raster_catalog`, `build_vector_catalog`,
# `build_xarray_catalog`) collect every row in RAM before returning an
# `InMemoryGeoCatalog`. Past ~10⁵ files the build step itself becomes
# the bottleneck.
#
# Pass `backend="duckdb"` to stream rows directly to a GeoParquet
# artifact in bounded memory (peak ≈ `batch_size × row_size`, not
# `O(n_rows)`). The result is a `DuckDBGeoCatalog` opened on the
# freshly written file.

# %%
import numpy as np
import rasterio
from rasterio.transform import from_bounds


# A handful of tiny GeoTIFFs in EPSG:32629 (UTM zone 29N).
scratch = pathlib.Path(tempfile.mkdtemp())
paths = []
for i, date in enumerate(["20240115", "20240116", "20240117"]):
    xmin = 500_000 + i * 160
    ymin = 4_000_000
    path = scratch / f"S2_T29SND_{date}_{xmin}_{ymin}.tif"
    transform = from_bounds(xmin, ymin, xmin + 160, ymin + 160, 32, 32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=1,
        dtype="uint16",
        crs="EPSG:32629",
        transform=transform,
    ) as dst:
        dst.write(np.full((1, 32, 32), 1, dtype=np.uint16))
    paths.append(path)

# %% [markdown]
# Build the catalog with the streaming branch. The `out_path` is required;
# the result is canonicalised to EPSG:4326 (the design's prescribed wire
# format) and Hilbert-sorted for row-group pruning at query time.

# %%
streamed = gc.build_raster_catalog(
    paths,
    filename_regex=r"S2_T29SND_(?P<date>\d{8})_\d+_\d+\.tif",
    backend="duckdb",
    out_path=scratch / "stream_cat.parquet",
    n_workers=1,  # bump to 4–8 on real workloads
    sort_by=("start_time", "geometry_hilbert"),
)
streamed

# %% [markdown]
# The artifact is a normal GeoParquet 1.1 file — readable by geopandas,
# DuckDB, GDAL, pandas. Reopening it via `open_catalog` gives back a
# `DuckDBGeoCatalog` with the same Protocol surface.

# %%
reopened = gc.open_catalog(scratch / "stream_cat.parquet", engine="duckdb")
print("rows           :", len(reopened))
print("CRS            :", reopened.crs)
print("total_bounds   :", reopened.total_bounds)

# %% [markdown]
# All the kwargs in one place:
#
# - `out_path`: required for `backend="duckdb"`.
# - `write_bbox=True`: emit the GeoParquet 1.1 covering bbox struct.
# - `sort_by=("start_time", "geometry_hilbert")`: post-write DuckDB
#   rewrite. `"geometry_hilbert"` is a literal token that expands to
#   `ST_Hilbert(ST_Centroid(geometry))`. `None` skips the rewrite.
# - `batch_size=10_000`: Arrow record-batch size; peak RAM ≈ `batch_size
#   × row_size`.
# - `n_workers=1`: `>1` spawns a process pool for per-file extraction.
# - `target_crs=None`: upgraded to `"EPSG:4326"` automatically in the
#   duckdb branch.
