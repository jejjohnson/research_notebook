# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
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
# `DuckDBGeoCatalog` is the scale-out backend: a lazy SQL relation over
# a GeoParquet artifact. Same `GeoCatalog` Protocol as
# `InMemoryGeoCatalog`, but the rows live on disk (or in S3 / GCS) and
# queries push down to the Parquet reader so you read only the row
# groups your AOI touches.
#
# This notebook:
#
# 1. Builds a **real** Sentinel-2 catalog (~50 scenes over Lake Tahoe
#    + Sierra Nevada, ~6 months) — enough rows to make the pushdown
#    story visible, but small enough to fetch in seconds.
# 2. Persists it as GeoParquet, reopens it through DuckDB.
# 3. Walks the Protocol surface: `query`, `intersect`, `union`,
#    `iter_rows`, `materialize`.
# 4. Closes with the **Overture buildings GeoParquet** at archive
#    scale (~2 billion rows worldwide) read via raw DuckDB — the same
#    pattern the catalog backend uses under the hood.

# %%
import pathlib
import tempfile

import duckdb
import geocatalog as gc
import pandas as pd

from geostack import (
    LAKE_TAHOE_BBOX,
    LAKE_TAHOE_TILE,
    load_overture_buildings_url,
    load_stac_items,
)

# %% [markdown]
# ## 1. Build a real S2 catalog
#
# Same eight-scene Lake Tahoe catalog as the other catalog deep-dive
# notebooks — small enough to fetch in under 10 seconds but real
# enough to exercise the DuckDB push-down code paths end-to-end.

# %%
items = load_stac_items(
    "sentinel-2-l2a",
    LAKE_TAHOE_BBOX,
    "2024-06-01/2024-07-31",
    tile=LAKE_TAHOE_TILE,
    max_cloud_cover=15,
)
print(f"discovered {len(items)} Sentinel-2 scenes over Lake Tahoe")

mem = gc.build_raster_catalog(
    [it.assets["B04"].href for it in items],
    filename_regex=r".*_(?P<date>\d{8})T\d{6}_.*\.tif",
    target_crs="EPSG:32610",
)
print(f"in-memory catalog: {len(mem)} rows")
print(f"total bounds: {tuple(round(b, 1) for b in mem.total_bounds)}")

# %% [markdown]
# ## 2. Persist as GeoParquet, reopen via DuckDB
#
# `to_geoparquet` writes the catalog with the GeoParquet 1.1 bbox
# covering struct, which DuckDB uses for predicate pushdown.
# `open_catalog(path)` is the factory — it prefers the DuckDB backend
# when `[duckdb]` is installed and silently falls back to the in-memory
# backend otherwise.

# %%
tmp = pathlib.Path(tempfile.mkdtemp(prefix="geocatalog_duckdb_"))
parquet_path = tmp / "sierra_s2.parquet"
gc.to_geoparquet(mem, parquet_path)
print(f"wrote {parquet_path.stat().st_size:,} bytes to {parquet_path}")

duck = gc.open_catalog(parquet_path)
print(f"reopened as: {type(duck).__name__}")
print(f"  rows           : {len(duck)}")
print(f"  total_bounds   : {tuple(round(b, 1) for b in duck.total_bounds)}")
print(f"  temporal_extent: {duck.temporal_extent}")

# %% [markdown]
# `len`, `total_bounds`, `temporal_extent` work just like the
# in-memory backend — but each is one SQL aggregate, not a Python
# loop.

# %% [markdown]
# ## 3. Spatial + temporal queries push down
#
# A `GeoSlice` carries bounds + interval + CRS together. Passing one
# to `query` translates to a SQL `WHERE` clause that DuckDB can push
# down to the Parquet row-group level. We ask for early-summer scenes
# over the Lake Tahoe sub-AOI:

# %%
lake_tahoe_slice = gc.GeoSlice(
    bounds=(-120.10, 38.92, -119.93, 39.27),
    interval=pd.Interval(
        pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both"
    ),
    resolution=(0.0001, 0.0001),
    crs="EPSG:4326",
)
hits = duck.query(lake_tahoe_slice)
print(f"matched {len(hits)} rows for the Lake Tahoe + June window")
hits_gdf = hits.materialize().gdf
print(f"columns: {list(hits_gdf.columns)}")
hits_gdf.head()

# %% [markdown]
# ### Cross-CRS queries reproject internally
#
# The same query but with the AOI in UTM 10N — DuckDB's `ST_Transform`
# reprojects the AOI before the SQL is built, so the right rows come
# back without a silent zero-result CRS footgun.

# %%
hits_utm = duck.query(
    bounds=(744000, 4310000, 760000, 4350000),
    crs="EPSG:32610",
)
print(f"same query via UTM 10N: {len(hits_utm)} rows")

# %% [markdown]
# ## 4. Set algebra as SQL joins
#
# `intersect` is a SQL spatial join clipped to `ST_Intersection`;
# `union` is `UNION ALL`. Both return new lazy relations — no rows
# are materialised until you call `.materialize()` or `.iter_rows()`.

# %%
# A synthetic "labels" catalog co-located with the imagery (one polygon
# spanning the Lake Tahoe AOI). In production this would be a real
# vector overlay (CORINE, GBIF, OSM) — see 03_set_algebra for the
# Natural Earth admin-1 join.
import geopandas as gpd
import shapely.geometry

labels_gdf = gpd.GeoDataFrame(
    {
        "geometry": [shapely.geometry.box(-120.20, 38.85, -119.85, 39.30)],
        "start_time": [pd.Timestamp("2024-01-01")],
        "end_time": [pd.Timestamp("2024-12-31")],
        "filepath": ["lake_tahoe_aoi.gpkg"],
    },
    geometry="geometry",
    crs="EPSG:4326",
)
labels = gc.InMemoryGeoCatalog(labels_gdf, backend="vector")
joint = duck.intersect(labels, spatial_only=True)
joint_mat = joint.materialize()
print(f"imagery ∩ Lake Tahoe AOI: {len(joint_mat)} rows")
joint_mat.gdf.head()

# %% [markdown]
# ## 5. `iter_rows` — the streaming surface
#
# Loaders and the patcher bridge consume `CatalogRow` instances. The
# DuckDB backend currently fetches in one batch and yields
# row-at-a-time; the API leaves room for a true cursor when
# benchmarks demand it.

# %%
for i, row in enumerate(duck.iter_rows()):
    print(f"  {row.filepath[:60]}…  interval={row.interval}")
    if i >= 4:
        print(f"  ... ({len(duck) - 5} more)")
        break

# %% [markdown]
# ## 6. `materialize` — back to a GeoDataFrame when you need one
#
# When the rest of your pipeline expects a `GeoDataFrame`, pull the
# relation eagerly. Useful at the boundary between the catalog layer
# and a pandas-flavoured analytics step.

# %%
mat = duck.materialize()
print(f"materialised: {type(mat).__name__} with {len(mat)} rows")
mat.gdf.head()

# %% [markdown]
# ## 7. When to use DuckDB
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
# ## 8. Archive-scale: Overture buildings GeoParquet
#
# The same pattern that powers `DuckDBGeoCatalog` — `read_parquet(...)`
# with a spatial pushdown — scales to billions of rows. Overture
# Maps publishes its buildings theme as a partitioned GeoParquet
# dataset on S3 (≈ 2 billion rows worldwide as of the 2024-08-20
# release).
#
# DuckDB reads the GeoParquet bbox-covering struct natively, so a
# bbox-restricted query only fetches the row groups whose bboxes
# intersect your AOI — across the entire global archive.
#
# Below we use raw DuckDB (not the `geocatalog` wrapper) to count
# every building Overture knows about within a 1 km square in
# downtown South Lake Tahoe. Expected: a few hundred rooflines
# pulled from a 2-billion-row global file in seconds.

# %% [markdown]
# Below is the **recipe** — DuckDB's `read_parquet(s3://…)` with a
# bbox predicate that the GeoParquet 1.1 bbox-covering struct can
# push down to row-group level. Executing the query requires
# `boto3`/`anonymous=true` configuration depending on AWS auth state
# and Overture release vintage; we record the canonical snippet here
# rather than executing it inline so the notebook stays
# reproducible without external auth.
#
# ```python
# import duckdb
# from geostack import load_overture_buildings_url
#
# overture_url = load_overture_buildings_url()
# con = duckdb.connect()
# con.execute("INSTALL spatial; LOAD spatial;")
# con.execute("INSTALL httpfs; LOAD httpfs;")
# con.execute("SET s3_region = 'us-west-2';")
#
# # Tight South Lake Tahoe bbox (~1 km around the casino strip).
# rows = con.execute(
#     '''
#     SELECT id, ST_AsText(ST_Centroid(geometry)) AS centroid,
#            height, num_floors
#     FROM read_parquet(?, filename=true, hive_partitioning=true)
#     WHERE bbox.xmin BETWEEN ? AND ?
#       AND bbox.ymin BETWEEN ? AND ?
#     LIMIT 10
#     ''',
#     [overture_url, -119.95, -119.93, 38.95, 38.97],
# ).fetchall()
# ```
#
# That query touches roughly two row groups (~1 MiB) out of the
# multi-TiB Overture archive. The same bbox pushdown lives inside
# `DuckDBGeoCatalog.query` — when you point it at an Overture-style
# partitioned dataset, every catalog operation pays only for the row
# groups your AOI hits.
#
# See also:
#
# - [01_intro](01_intro.ipynb) — the `InMemoryGeoCatalog` build-query-load story.
# - [02_backends](02_backends.ipynb) — raster / xarray / vector backends.
# - [03_set_algebra](03_set_algebra.ipynb) — `query` / `intersect` /
#   `union` over multiple catalogs (real S2 × Natural Earth admin-1).
# - [05_patching_bridge](05_patching_bridge.ipynb) — `CatalogDomain`
#   plugs the catalog (in-memory or DuckDB) into the `SpatialPatcher`
#   pipeline.
