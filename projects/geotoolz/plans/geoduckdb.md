# Dataset-builder snippets — Phase 2 design report (DuckDB backend)

> **Scope:** adding a DuckDB-backed catalog backend, GeoParquet-as-artifact, and SQL-native cross-catalog operations to the [`jejjohnson/georeader`](https://github.com/jejjohnson/georeader) `catalog` module proposed in Phase 1.
>
> **Status:** design proposal, no code changed yet. Assumes Phase 1 (the in-memory `GeoCatalog`, builders, loaders, samplers, and `to_geoparquet`/`from_geoparquet` round-trip) has shipped.

---

## 0. What's in place from Phase 1

The Phase 1 report proposed and (assume) shipped:

| Component | Provides |
| --- | --- |
| `GeoCatalog` protocol | one shape across raster / xarray / vector |
| `InMemoryGeoCatalog` | Phase 1 implementation: `gpd.GeoDataFrame` with `IntervalIndex` + `geometry` |
| `build_{raster,xarray,vector}_catalog` | per-backend builders |
| `load_{raster,xarray,vector}` | loaders that return `GeoTensor` / `xr.Dataset` |
| `random_sampler`, `grid_sampler`, `stitch` | ML glue |
| `GeoSlice` | unit of work passed between samplers and loaders |
| `query`, `intersect`, `union` | set algebra over catalogs |
| `to_geoparquet` / `from_geoparquet` | round-trip the catalog as a portable artifact |

What Phase 1 cannot do well:

- Catalogs much beyond ~10⁵ rows: `gpd.overlay` becomes painful, the gdf eats RAM.
- Sharing without re-running the build step: a pickled gdf is fragile across versions.
- Querying a catalog without loading the whole thing into Python.
- Catalog analytics ("monthly file count by tile, weighted by useful area inside this AOI").
- Streaming a catalog *during* construction so 10M files don't sit in RAM.

Phase 2 fills these gaps by adding a second backend behind the same `GeoCatalog` protocol.

---

## 1. Inventory: what DuckDB brings

DuckDB is an in-process columnar OLAP engine with a `spatial` extension that gives us everything we need at the index layer.

### 1.1 Core engine

- Single-binary, in-process (no server). ~30 MB.
- Native Parquet read/write with **predicate and projection pushdown** driven by row-group min/max statistics.
- Reads transparently from local filesystem, S3, GCS, Azure, HTTPS, HuggingFace via the `httpfs` extension.
- Arrow-native: zero-copy to `pyarrow`, then to `pandas` / `polars` / `geopandas`.
- Multi-threaded execution; vectorized columnar engine.
- A persistent file format (`.duckdb`) for storing tables + indexes when needed.

### 1.2 Spatial extension

GEOS-backed, GeoParquet 1.x compatible. The functions Phase 2 leans on:

| Category | Functions used |
| --- | --- |
| Constructors | `ST_MakeEnvelope`, `ST_GeomFromText`, `ST_GeomFromWKB`, `ST_AsWKB` |
| Predicates | `ST_Intersects`, `ST_Contains`, `ST_Within`, `ST_Overlaps` |
| Set ops | `ST_Intersection`, `ST_Union`, `ST_Difference` |
| Measures | `ST_Area`, `ST_Centroid`, `ST_Distance` |
| CRS | `ST_Transform` (requires the `spatial` extension's bundled PROJ data) |
| Indexing | `CREATE INDEX ... USING RTREE` (on `.duckdb` table files only) |

Spatial joins build an in-memory R-tree on the smaller side per query. Persistent on-disk R-trees are limited to materialized DuckDB tables, not Parquet — see [§7](#7-sharp-edges).

### 1.3 GeoParquet support

- Reads any GeoParquet 1.0/1.1 file produced by `geopandas.to_parquet` (or by DuckDB itself).
- Writes via `COPY ... TO 'file.parquet' (FORMAT 'parquet', COMPRESSION 'zstd')` — schema includes WKB geometry + GeoParquet-style column metadata when the spatial extension is loaded.
- GeoParquet **1.1** introduces a per-row `bbox` covering struct that lets the engine push spatial filters down to the row-group level *without parsing WKB*. For 10⁷-row catalogs this is roughly an order-of-magnitude speedup on selective queries.

> ⚠️ The `spatial` extension's GeoParquet *write* path is more recent than the read path. Pin a known-good DuckDB version (≥ 1.1) and add a smoke test that round-trips a small catalog.

---

## 2. User story

**Persona shift.** Phase 1's user has hundreds to thousands of files in one project. Phase 2's user has *too many to load eagerly*, or *wants to publish* a catalog so a collaborator (or a CI job, or a notebook on a different machine) can query it without rebuilding.

**Goal arc:**

1. *"I built a 1M-row catalog yesterday. Reload it instantly."* → `open_catalog("cat.parquet")` (no `build_*_catalog` needed).
2. *"My collaborator wants to query my catalog from her laptop."* → I push to `s3://bucket/cat.parquet`; she runs `open_catalog("s3://bucket/cat.parquet")`. Only the row groups her query touches get downloaded.
3. *"Intersect 200k imagery rows with 500k label rows."* → SQL spatial join, parallel, minutes instead of an OOM.
4. *"How does my coverage break down by month, by tile, by cloud %?"* → one SQL query against the catalog, no loop.
5. *"Stream candidate slices into the random sampler without materializing the join."* → DuckDB cursor → `iter_rows()` → sampler.
6. *"Append today's new acquisitions to yesterday's catalog without rebuilding."* → drop a new shard into a directory; `open_catalog("cat_dir/")` reads them as one virtual table.
7. *"Build the catalog directly into DuckDB during ingest so I never hold 10M rows in RAM."* → `build_raster_catalog(..., backend="duckdb", out_path="cat.parquet")` streams rows to disk.

The framing: **DuckDB doesn't replace Phase 1, it extends the working envelope by 2–3 orders of magnitude**, and turns the catalog into a first-class shareable artifact.

---

## 3. Motivation

### 3.1 The walls Phase 1 hits

| Dimension | Phase 1 ceiling | Why |
| --- | --- | --- |
| Catalog size in RAM | ~10⁵–10⁶ rows | each gdf row is ~1 KB; a 10M-row gdf is ~10 GB |
| `intersect` (overlay) | ~10⁴ × 10⁴ comfortable, 10⁵ × 10⁵ painful | `gpd.overlay` is `O(n × m)` worst case, single-threaded |
| Build cost | re-run every Python session | nothing is cached; rasterio metadata extraction is 1–10 ms/file |
| Sharing | pickled gdfs are fragile across versions | no canonical interchange format |
| Analytics | Python loops over the gdf | aggregations at scale need a query engine |
| Remote | "rsync the whole archive" | no lazy/partial access |

### 3.2 What DuckDB unlocks

| Need | DuckDB mechanism |
| --- | --- |
| Scale past memory | columnar parquet + row-group pushdown; nothing materializes until you ask |
| Persistence | GeoParquet is the artifact; the catalog is now a file you can version, hash, sign |
| Sharing | `httpfs` extension reads S3/GCS/HTTPS GeoParquet; predicate pushdown means low egress |
| Fast joins | spatial-join operator with parallel R-tree probe; band-join on temporal intervals |
| Analytics | full SQL surface |
| Streaming build | `INSERT ... SELECT` from a Python iterator into a DuckDB-managed Parquet writer |

### 3.3 Compared with alternatives

- **STAC + pgSTAC / stac-fastapi.** Heavier: needs Postgres, a schema, an HTTP service. Wins on standardization and federated discovery. Loses on dev velocity for single-user / single-team workflows.
- **PostGIS + GeoAlchemy.** Real database. More features (transactions, advanced indexes, triggers). Far more setup. Best when you already run Postgres for other reasons.
- **xstac + intake-stac.** Library-only, no server. But STAC items are JSON-per-file, so a 1M-item catalog is 1M JSON files or one giant blob — both worse than one Parquet file for the queries we care about.
- **Plain `gpd.read_parquet` + Python.** What Phase 1 already gives via `from_geoparquet`. Fine until you need joins or analytics at scale.

The unique slot DuckDB occupies: **zero-server, single-file, parallel, SQL, with native Parquet + spatial**. Nothing else hits all five.

---

## 4. Mathematics

### 4.1 Predicate pushdown via Parquet zone maps

A Parquet file is a sequence of *row groups*; each row group has, per column, min/max statistics. GeoParquet adds a per-row-group geometry bbox (and, in 1.1, an optional per-row bbox column).

For a query

```sql
WHERE tmax >= $T_lo AND tmin <= $T_hi
  AND ST_Intersects(geometry, ST_MakeEnvelope($x0,$y0,$x1,$y1))
```

a row group is pruned without reading any of its data when **any** of these holds:

$$
\text{group}.t_{\max\text{-max}} < T_{\text{lo}}, \quad
\text{group}.t_{\min\text{-min}} > T_{\text{hi}}, \quad
\text{group.bbox} \cap \text{query.bbox} = \emptyset.
$$

If the catalog is sorted by `tmin` (or, even better, written via a Hilbert/R-curve sort on geometry centroids), the fraction of row groups read for a narrow query approaches the fraction of rows that match. For a 10⁷-row catalog with 10k rows per group and a 0.1%-selective query, expect to read ~10 row groups, not 1000.

Practical recipe: sort the catalog by `(tmin, ST_Hilbert(centroid))` before writing.

### 4.2 Spatial-join algorithm

DuckDB's spatial join builds an R-tree on the smaller relation and probes with the larger one:

$$
\text{cost} \;\approx\; \underbrace{n \log n}_{\text{R-tree build on A}} \;+\; \underbrace{m \log n}_{\text{probes from B}} \;+\; \underbrace{k}_{\text{candidate refinement}}
$$

where `k` is the candidate match count after bbox filtering, refined by the exact `ST_Intersects` predicate. Compare with `gpd.overlay`'s effective `O(n + m + n × m')` (R-tree filter then per-pair intersection check), which dominates for dense overlap.

Empirically, on the same machine: a 200 k × 500 k spatial join takes minutes in DuckDB, hours in geopandas, and OOMs at 1M × 1M in geopandas while completing in DuckDB.

### 4.3 Temporal band-join

The interval-overlap predicate

```sql
a.tmax >= b.tmin AND a.tmin <= b.tmax
```

is recognized by DuckDB's optimizer as a **range join** (a.k.a. band join). The algorithm sorts both sides by `tmin`, then sweeps:

```
sort A by tmin, B by tmin
maintain a sliding window W of B-rows whose tmax >= current A.tmin
for each a in A:
    advance W: drop b where b.tmax < a.tmin; add b where b.tmin <= a.tmax
    emit (a, b) for b in W
```

Cost: $O(n \log n + m \log m + |\text{matches}|)$. No nested-loop blow-up.

### 4.4 Combined spatiotemporal join cost

For a query that joins on both space and time, DuckDB's planner picks the more selective predicate first based on column statistics. Typical plan:

1. Spatial join via R-tree (filters out 99%+ of pairs in most workloads).
2. Apply temporal predicate as a post-filter.
3. Compute `ST_Intersection` only for surviving pairs.

You can verify with `EXPLAIN`. The win versus Phase 1 is **2–3 orders of magnitude on real-world catalog sizes**.

### 4.5 GeoParquet bbox column (1.1)

GeoParquet 1.1 introduces an optional per-row covering struct:

```text
bbox: STRUCT<xmin DOUBLE, ymin DOUBLE, xmax DOUBLE, ymax DOUBLE>
```

For a spatial filter, DuckDB can evaluate

```sql
WHERE bbox.xmax >= $x0 AND bbox.xmin <= $x1
  AND bbox.ymax >= $y0 AND bbox.ymin <= $y1
  AND ST_Intersects(geometry, ST_MakeEnvelope($x0,$y0,$x1,$y1))
```

The first four conjuncts are pure double-comparison — no WKB parsing, fully vectorized. The exact `ST_Intersects` runs only on rows that pass the bbox filter. For very selective queries this is roughly 10× faster than calling `ST_Intersects` directly on every row.

The catalog writer should always emit the bbox column.

### 4.6 Streaming build cost model

Phase 1's `build_raster_catalog` materializes the whole gdf. For 10M files at ~1 KB/row, that's 10 GB plus Python overhead.

The streaming variant batches `B` rows into Arrow record batches and appends to a Parquet writer:

$$
\text{peak memory} \;\approx\; B \cdot \text{row\_size} \;+\; \text{open-file-handles} \cdot \text{rasterio overhead}
$$

For `B = 10 000` and ~1 KB/row, peak is ~10 MB plus rasterio's per-file overhead (tens of MB). Build time is dominated by `rasterio.open` calls, which can be parallelized with a `ThreadPoolExecutor` (GIL-friendly because rasterio releases the GIL during I/O).

---

## 5. Coupling with Phase 1

The whole point of the Phase 1 protocol design was to make Phase 2 transparent. Here's the seam:

| Operation | `InMemoryGeoCatalog` (Phase 1) | `DuckDBGeoCatalog` (Phase 2) |
| --- | --- | --- |
| `query(slice)` | `gdf.cx[...]` + `IntervalIndex.overlaps` | SQL `WHERE` with predicate pushdown |
| `intersect(other)` | `gpd.overlay` | SQL `JOIN ... ON ST_Intersects(...) AND <interval-overlap>` |
| `union(other)` | `pd.concat` | SQL `UNION ALL` |
| `iter_rows()` | `gdf.itertuples()` | Arrow batch iterator over the relation |
| `to_geoparquet(path)` | `gdf.to_parquet(...)` | `COPY ... TO ... (FORMAT 'parquet')` |
| `materialize()` | identity | execute the relation, return `InMemoryGeoCatalog` |
| `__len__` | `len(gdf)` | `SELECT COUNT(*)` (cached) |
| Backend-specific escape hatch | `.gdf` (full geopandas surface) | `.relation` (full SQL surface) |

**Loaders unchanged.** `load_raster(catalog, slice, ...)` only needs `catalog.iter_rows()` (or a single `query` + `iter_rows`) — it never touches the underlying gdf or relation.

**Samplers** need the smallest tweak: replace `catalog.gdf.iloc[i]` patterns with `catalog.iter_rows()` or `catalog.query(slice).iter_rows()`. Once that's done, samplers work identically against either backend.

**`stitch` unchanged.** It only consumes `GeoSlice` + `np.ndarray`, no catalog.

The catalog is the only seam. Get the protocol right and Phase 2 is purely additive.

---

## 6. Proposed API surface

### 6.1 Module layout

```text
georeader/
├── catalog/
│   ├── __init__.py
│   ├── base.py           # GeoCatalog protocol, CatalogRow, GeoSlice (re-export)
│   ├── memory.py         # InMemoryGeoCatalog (Phase 1)
│   ├── duckdb_backend.py # DuckDBGeoCatalog (Phase 2)        [extra: duckdb]
│   ├── parquet.py        # GeoParquet read/write helpers
│   ├── streaming.py      # StreamingParquetWriter for direct-to-disk builds
│   └── ops.py            # query / intersect / union dispatch
└── cli/
    └── catalog.py        # `georeader catalog ...` subcommands
```

### 6.2 Protocol additions

The Phase 1 `GeoCatalog` protocol stays mostly intact. Phase 2 adds:

```python
class GeoCatalog(Protocol):
    # ... Phase 1 surface ...

    # NEW: explicit construction from a parquet artifact (works for both backends)
    @classmethod
    def open(
        cls,
        source: str | Path | list[str | Path],
        *,
        backend: Literal["memory", "duckdb"] = "duckdb",
        table: str = "files",
    ) -> "GeoCatalog": ...

    # NEW: lazy / eager bridge (no-op on InMemory; executes on DuckDB)
    def materialize(self) -> "InMemoryGeoCatalog": ...

    # NEW: structured row iteration that both backends implement
    def iter_rows(self, *, batch_size: int = 1024) -> Iterator["CatalogRow"]: ...
```

Note `open` is a *factory*: it returns whichever backend the caller asked for, defaulting to DuckDB once Phase 2 ships because the lazy backend is almost always the right choice for an artifact-on-disk.

### 6.3 `CatalogRow`

```python
@dataclass(frozen=True)
class CatalogRow:
    filepath: str
    geometry: shapely.geometry.base.BaseGeometry   # decoded from WKB
    interval: pd.Interval                          # built from (tmin, tmax)
    crs: pyproj.CRS
    extras: dict[str, Any]                         # backend-specific extras (data_vars, layer, ...)
```

Loaders consume `CatalogRow`. Whether the row came from a gdf or a DuckDB cursor is invisible.

### 6.4 `DuckDBGeoCatalog`

```python
class DuckDBGeoCatalog(GeoCatalog):
    """Lazy, SQL-backed catalog. Operations return new relations; nothing
    runs until iter_rows() / materialize() / aggregate() is called."""
    relation: duckdb.DuckDBPyRelation
    con: duckdb.DuckDBPyConnection
    backend: Literal["raster", "xarray", "vector"]
    crs: pyproj.CRS

    @classmethod
    def open(cls, source: str | Path | list[str | Path],
             *, table: str = "files") -> "DuckDBGeoCatalog":
        """Open a parquet file, a list of parquet files, a directory of
        parquet shards, or a .duckdb file. Local or remote URIs."""

    @classmethod
    def from_memory(cls, mem: InMemoryGeoCatalog) -> "DuckDBGeoCatalog":
        """Register an in-memory catalog as a DuckDB view."""

    # Set algebra — all return DuckDBGeoCatalog (still lazy)
    def query(self, slice: GeoSlice, *, t_step: int | None = None) -> "DuckDBGeoCatalog": ...
    def intersect(self, other: "GeoCatalog", *, spatial_only: bool = False) -> "DuckDBGeoCatalog": ...
    def union(self, other: "GeoCatalog") -> "DuckDBGeoCatalog": ...

    # Persistence
    def to_geoparquet(self, path: str | Path, *,
                      partition_by: list[str] | None = None,
                      sort_by: list[str] | None = None,
                      compression: str = "zstd") -> None: ...
    def to_duckdb_file(self, path: str | Path, *,
                       with_rtree: bool = True) -> None: ...

    # Analytics escape hatch — return a DataFrame (eager)
    def aggregate(self, sql: str, **params) -> pd.DataFrame: ...
    def explain(self) -> str: ...   # debug

    # Underlying surface
    def sql(self, sql: str, **params) -> "DuckDBGeoCatalog": ...   # arbitrary SQL filter
```

### 6.5 Builders gain a `backend=` parameter

```python
def build_raster_catalog(
    filepaths: Sequence[str | Path],
    *,
    filename_regex: str,
    date_format: str = "%Y%m%d",
    target_crs: CRS | str | None = None,
    target_resolution: tuple[float, float] | None = None,

    # NEW
    backend: Literal["memory", "duckdb"] = "memory",
    out_path: str | Path | None = None,    # required if backend="duckdb"
    write_bbox: bool = True,               # GeoParquet 1.1 bbox column
    sort_by: tuple[str, ...] = ("tmin", "geometry_hilbert"),
    n_workers: int = 1,
) -> GeoCatalog: ...
```

When `backend="duckdb"`, the builder streams metadata into an Arrow record-batch writer and lands directly in `out_path` as GeoParquet. The returned object is a `DuckDBGeoCatalog` opened on that file. Memory stays bounded regardless of file count.

Same surface for `build_xarray_catalog` and `build_vector_catalog`.

### 6.6 GeoParquet schema (canonical)

```text
filepath        VARCHAR     NOT NULL
tmin            TIMESTAMP   NULL              -- NULL means "no date known"
tmax            TIMESTAMP   NULL
geometry        BLOB (WKB)  NOT NULL          -- GeoParquet column metadata = CRS + encoding
bbox            STRUCT      NOT NULL          -- xmin, ymin, xmax, ymax  (GeoParquet 1.1)
crs             VARCHAR     NOT NULL          -- usually one value per file; per-row allowed
backend         VARCHAR     NOT NULL          -- 'raster' | 'xarray' | 'vector'

-- raster-specific (NULL for other backends)
n_bands         INTEGER     NULL
dtype           VARCHAR     NULL
nodata          DOUBLE      NULL

-- xarray-specific
data_vars       VARCHAR[]   NULL
time_var        VARCHAR     NULL
n_timesteps     INTEGER     NULL

-- vector-specific
layer           VARCHAR     NULL

-- optional analytics columns (recommended)
area_m2         DOUBLE      NULL
cloud_pct       DOUBLE      NULL
sensor          VARCHAR     NULL
processing_ver  VARCHAR     NULL
```

Per-file extras land in additional columns (typed when the schema knows them, JSON-encoded otherwise).

### 6.7 CLI

```bash
$ georeader catalog inspect    cat.parquet
$ georeader catalog stats      cat.parquet --by month --by sensor
$ georeader catalog query      s3://bucket/cat.parquet \
                                --aoi AOI.geojson --tmin 2023-06-01 --tmax 2023-06-30 \
                                --out subset.parquet
$ georeader catalog intersect  imagery.parquet labels.parquet --out joint.parquet
$ georeader catalog union      l7.parquet l8.parquet         --out landsat.parquet
$ georeader catalog optimize   cat.parquet --sort tmin,hilbert --bbox --rewrite
```

The CLI is a thin wrapper around `DuckDBGeoCatalog`; everything it does is callable from Python too.

### 6.8 What I'd *not* add in Phase 2

- A query DSL that abstracts over pandas vs SQL. Just expose `.gdf` and `.relation` as escape hatches.
- A persistent server (HTTP API). That's STAC's job.
- Auto-conversion between CRSs at query time. Catalogs should be written in one CRS; `ST_Transform` is available but slow and a footgun.
- Materialized R-tree indexes on Parquet. They don't exist; if you need persistent indexes, write to `.duckdb` (see [§7](#7-sharp-edges)).

---

## 7. Sharp edges

1. **GeoParquet 1.0 vs 1.1.** 1.1 adds the per-row bbox column — order-of-magnitude speedup for spatial filters at scale. Default to writing 1.1; provide a `--geoparquet-version=1.0` opt-out for compatibility with older readers.
2. **DuckDB spatial extension version pinning.** APIs are stable but evolving. Pin a minimum DuckDB version (e.g. ≥ 1.1) and document the install (`INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;`).
3. **No persistent R-tree on Parquet.** Persistent R-tree indexes only exist in `.duckdb` table files. If a catalog will be queried thousands of times with the same predicate shapes, materialize to `.duckdb`:
   ```sql
   CREATE TABLE files AS SELECT * FROM 'cat.parquet';
   CREATE INDEX idx_geom ON files USING RTREE (geometry);
   ```
   Otherwise, rely on bbox + zone-map pushdown — usually fast enough.
4. **Multi-CRS catalogs.** Same problem as Phase 1, more visible because SQL doesn't auto-reproject. Strong recommendation: **write all GeoParquet in EPSG:4326**, store the native CRS in a column, reproject on load. `ST_Transform` works but requires PROJ data and is slow per call.
5. **NULL temporal sentinels.** Don't write `pd.Timestamp.min/.max` — they overflow `TIMESTAMP`. Write `NULL` and require explicit `OR tmin IS NULL` for queries that should include date-less files.
6. **Cursor lifecycle on `iter_rows()`.** The DuckDB connection must outlive the iterator. Use a context manager:
   ```python
   with catalog.cursor() as rows:
       for row in rows:
           ...
   ```
   Document that storing the iterator beyond the `with` block is unsupported.
7. **WKB → shapely deserialization cost.** Per-row `shapely.from_wkb` is fine at 10⁴ rows/s, painful at 10⁶. For tight inner loops, batch with `shapely.from_wkb(arr)` (vectorized, uses `pygeos` / shapely 2.x).
8. **Sampler weighting on lazy catalogs.** Phase 1's area-weighted random sampler ([§4.5 of Phase 1 report](#45-random-sampler-weighting)) needs all candidate areas. With DuckDB, either (a) maintain a precomputed `area_m2` column at build time (recommended), (b) use `USING SAMPLE 10%` to estimate, or (c) stream the area column once and cache.
9. **Parquet schema evolution.** Parquet is immutable. Add a column → rewrite the file. Mitigate by storing the catalog as a directory of dated shards with consistent schema; DuckDB reads them as one virtual table via `read_parquet('shards/*.parquet')`. Document the canonical schema and bump a version column.
10. **Authentication for remote.** `httpfs` picks up env vars, `~/.aws/credentials`, GCS service accounts automatically. Document this and provide one-line recipes per provider. Surface clear errors when auth fails (DuckDB's default messages are cryptic).
11. **Streaming builder + multiprocessing.** When `n_workers > 1`, the rasterio metadata extraction parallelizes well, but Arrow record-batch writes must serialize. Use a single writer thread fed by a queue. Don't try to parallelize the Parquet writer itself.
12. **Connection-per-thread.** DuckDB connections are *not* thread-safe; use `con.cursor()` in worker threads or open a fresh connection per process. The `DuckDBGeoCatalog` should expose a `.cursor()` method, not pass `con` around.

---

## 8. End-to-end examples

A varied gallery, organized by what's *new* in Phase 2 — persistence, analytics, scale, federation, and lazy ML pipelines. All examples assume the Phase 1 surface exists.

| § | Example | Pattern |
| --- | --- | --- |
| 8.1 | A. Save and reload a catalog locally | round-trip via GeoParquet |
| 8.1 | B. Publish a catalog to S3, query remotely | `httpfs` + predicate pushdown |
| 8.2 | C. Coverage histogram by month | `aggregate()` SQL |
| 8.2 | D. Per-tile statistics | grouped SQL |
| 8.2 | E. Custom SQL filter (low cloud, recent) | `.sql()` escape hatch |
| 8.3 | F. Million-row spatial join | DuckDB `intersect` at scale |
| 8.3 | G. Streaming build directly into GeoParquet | `backend="duckdb"` builder |
| 8.4 | H. Federated query across monthly shards | one virtual table from many files |
| 8.4 | I. Mix raster + vector catalogs in one SQL view | union via SQL |
| 8.4 | J. Cross-CRS catalog join with `ST_Transform` | when harmonization isn't possible |
| 8.5 | K. Lazy iterator from filtered catalog → sampler | DuckDB cursor → `random_sampler` |
| 8.5 | L. Stratified sampling via SQL window functions | balanced training set |
| 8.5 | M. Active learning with persistent uncertainty column | round-trip uncertainty back into the catalog |

### 8.1 Persistence and sharing

#### A. Save and reload a catalog locally

```python
from georeader.catalog import build_raster_catalog, open_catalog

catalog = build_raster_catalog(
    filepaths=glob("/data/s2/T29SND_*.tif"),
    filename_regex=r"T29SND_(?P<date>\d{8})_.*\.tif",
    target_crs="EPSG:4326",
    target_resolution=(0.0001, 0.0001),
)
catalog.to_geoparquet("/cat/s2_T29SND.parquet", sort_by=["tmin", "geometry_hilbert"])

# Tomorrow, in a fresh process:
catalog = open_catalog("/cat/s2_T29SND.parquet")    # DuckDBGeoCatalog (lazy)
print(len(catalog), catalog.total_bounds, catalog.temporal_extent)
```

#### B. Publish a catalog to S3, query remotely

```python
from georeader.catalog import open_catalog, GeoSlice

# Producer
catalog.to_geoparquet("s3://my-bucket/catalogs/s2_eu_2023.parquet")

# Consumer, on a different machine:
remote = open_catalog("s3://my-bucket/catalogs/s2_eu_2023.parquet")

aoi = GeoSlice(
    bounds=(-3.8, 40.3, -3.6, 40.5),
    interval=pd.Interval(pd.Timestamp("2023-06-01"), pd.Timestamp("2023-06-30"), closed="both"),
    resolution=(0.0001, 0.0001),
    crs="EPSG:4326",
)
sub = remote.query(aoi).materialize()              # only relevant row groups downloaded
print(f"Matched {len(sub)} files; downloaded ~{sub.bytes_read_estimate} bytes from S3.")
```

### 8.2 Catalog analytics

#### C. Coverage histogram by month

```python
from georeader.catalog import open_catalog

catalog = open_catalog("/cat/s2_T29SND.parquet")

df = catalog.aggregate("""
    SELECT
        date_trunc('month', tmin)              AS month,
        COUNT(*)                                AS n_files,
        AVG(cloud_pct)                          AS mean_cloud,
        SUM(area_m2) / 1e6                      AS total_area_km2
    FROM files
    GROUP BY 1
    ORDER BY 1
""")
df.plot.bar(x="month", y="n_files")
```

#### D. Per-tile statistics, restricted to an AOI

```python
df = catalog.aggregate("""
    SELECT
        substr(filepath, 12, 6)                 AS tile_id,
        COUNT(*)                                AS n_acquisitions,
        MIN(tmin)                               AS first_seen,
        MAX(tmax)                               AS last_seen,
        AVG(cloud_pct)                          AS mean_cloud
    FROM files
    WHERE ST_Intersects(geometry, ST_GeomFromText($aoi_wkt))
    GROUP BY 1
    ORDER BY n_acquisitions DESC
""", aoi_wkt=aoi_polygon.wkt)
```

#### E. Custom SQL filter (recent, low-cloud, large enough)

```python
fresh = catalog.sql("""
    SELECT * FROM files
    WHERE tmin >= '2023-01-01'
      AND cloud_pct < 10
      AND area_m2  > 1e8
""")     # returns DuckDBGeoCatalog (still lazy)

print(len(fresh))
fresh.to_geoparquet("/cat/s2_T29SND_clean_2023.parquet")
```

### 8.3 Scale operations

#### F. Million-row spatial join (intersect at scale)

```python
imagery = open_catalog("s3://my-bucket/imagery_eu.parquet")          # 1.2M rows
labels  = open_catalog("s3://my-bucket/labels_eu.parquet")           # 0.5M rows

joint = imagery.intersect(labels)                                     # lazy, no execution yet
joint.to_geoparquet("/cat/eu_imagery_x_labels.parquet",
                    partition_by=["sensor"], sort_by=["tmin"])

# DuckDB plan (printed for transparency):
print(joint.explain())
# │  ► HASH_JOIN  ON ST_Intersects(...) AND a.tmax >= b.tmin AND a.tmin <= b.tmax
# │  ├── PARQUET_SCAN  imagery_eu.parquet  [bbox + tmin/tmax pushdown]
# │  └── PARQUET_SCAN  labels_eu.parquet   [bbox + tmin/tmax pushdown]
```

#### G. Streaming build directly into GeoParquet

```python
from georeader.catalog import build_raster_catalog

# 8.5M files; would never fit in a gdf.
catalog = build_raster_catalog(
    filepaths=iter_files_from("/very_large_archive/"),     # any iterable
    filename_regex=FILENAME_REGEX,
    target_crs="EPSG:4326",
    target_resolution=(0.0001, 0.0001),
    backend="duckdb",
    out_path="/cat/global_archive.parquet",
    write_bbox=True,
    sort_by=("tmin", "geometry_hilbert"),
    n_workers=8,                                           # parallel rasterio metadata extraction
)
print(len(catalog))            # 8_532_104, peak RAM stayed under 1 GB
```

### 8.4 Federation and heterogeneity

#### H. Federated query across monthly shards

```python
# /cat/s2_2023/ contains 12 monthly parquet shards with the same schema
catalog = open_catalog("/cat/s2_2023/")           # DuckDB reads all shards as one table

aoi = GeoSlice(bounds=AOI, interval=pd.Interval("2023-04-01", "2023-09-30", closed="both"),
               resolution=(0.0001, 0.0001), crs="EPSG:4326")
sub = catalog.query(aoi)                           # touches only 6 shards (Apr–Sep)
```

Same pattern works for `s3://bucket/s2_2023/*.parquet` or HuggingFace datasets.

#### I. Mix raster and vector catalogs in one SQL view

```python
imagery_cat = open_catalog("/cat/s2.parquet")
labels_cat  = open_catalog("/cat/labels.parquet")

con = imagery_cat.con
con.register("img", imagery_cat.relation)
con.register("lab", labels_cat.relation)

paired = con.sql("""
    SELECT  img.filepath  AS img_fp,
            lab.filepath  AS lab_fp,
            ST_Intersection(img.geometry, lab.geometry) AS geometry,
            GREATEST(img.tmin, lab.tmin) AS tmin,
            LEAST   (img.tmax, lab.tmax) AS tmax
    FROM img
    JOIN lab
      ON ST_Intersects(img.geometry, lab.geometry)
     AND img.tmax >= lab.tmin AND img.tmin <= lab.tmax
    WHERE img.cloud_pct < 5
""")

paired_cat = DuckDBGeoCatalog.from_relation(paired, backend="raster", crs="EPSG:4326")
```

#### J. Cross-CRS catalog join (last resort)

If you really can't harmonize CRSs at write time, push `ST_Transform` into the join:

```python
joint = catalog.sql("""
    SELECT a.*, b.filepath AS dem_path
    FROM files a
    JOIN 'dem_utm.parquet' b
      ON ST_Intersects(
           a.geometry,
           ST_Transform(b.geometry, 'EPSG:32629', 'EPSG:4326')
         )
""")
```

Slower and PROJ-dependent. Document the cost.

### 8.5 Sampler / inference patterns at scale

#### K. Lazy iterator from a filtered catalog into the random sampler

The sampler doesn't care that the catalog is DuckDB-backed; it just calls `iter_rows()`.

```python
from georeader.catalog import open_catalog
from georeader.samplers import random_sampler

catalog = open_catalog("s3://bucket/imagery_eu.parquet")
clean   = catalog.sql("WHERE cloud_pct < 10 AND area_m2 > 1e8")     # still lazy

# random_sampler internally does clean.iter_rows() + reservoir sampling
for sl in random_sampler(clean, chip_size=(256, 256), length=100_000, seed=42):
    x = load_raster(catalog, sl, band_indexes=[2, 3, 4, 8]).values
    train_step(x)
```

DuckDB streams candidate rows; the sampler reservoir-samples without ever materializing the full filtered set.

#### L. Stratified sampling via SQL window functions

Balance a training set across, say, agro-ecological zones:

```python
catalog = open_catalog("/cat/s2_with_zones.parquet")    # has a `zone` column

stratified = catalog.sql("""
    SELECT * FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY zone ORDER BY random()) AS rn
        FROM files
        WHERE cloud_pct < 5
    )
    WHERE rn <= 1000      -- 1000 chips per zone
""")

for sl in random_sampler(stratified, chip_size=(256, 256)):
    ...
```

#### M. Active learning with a persistent uncertainty column

Round-trip the model's uncertainty back into the catalog to drive the next labeling round.

```python
catalog = open_catalog("/cat/al_round_3.parquet")

# Score current candidates
scores = []
for row in catalog.iter_rows():
    sl = GeoSlice(bounds=row.geometry.bounds, interval=row.interval,
                  resolution=(10.0, 10.0), crs=row.crs)
    x = load_raster(catalog, sl).values
    _, sigma = model_with_uncertainty(x)
    scores.append((row.filepath, float(sigma.mean())))

# Persist back as a derived catalog
con = catalog.con
con.execute("""
    CREATE OR REPLACE TABLE al_round_4 AS
    SELECT files.*, scores.uncertainty
    FROM files
    JOIN (SELECT * FROM (VALUES ($values)) AS t(filepath, uncertainty)) scores
      USING (filepath)
""", values=scores)
con.execute("COPY al_round_4 TO '/cat/al_round_4.parquet' (FORMAT 'parquet', COMPRESSION 'zstd')")

# The next round: pick the top-N most uncertain
next_round = open_catalog("/cat/al_round_4.parquet").sql(
    "SELECT * FROM files ORDER BY uncertainty DESC LIMIT 500"
)
```

### 8.6 Phase 2 API extensions implied by these examples

The new examples lean on a small set of additions that aren't strictly in §6:

- `DuckDBGeoCatalog.from_relation(...)` — wrap an arbitrary DuckDB relation as a catalog (used in I).
- `GeoCatalog.bytes_read_estimate` — telemetry surface from DuckDB's stats (used in B).
- `geometry_hilbert` virtual sort key — computed at write time from the geometry centroid (used in A, G).
- `catalog.cursor()` context manager around `iter_rows()` (mentioned in §7).

None require new dependencies; all are thin wrappers over DuckDB or geopandas.

---

## 9. Verdict

Phase 2 is a **clean, additive upgrade** that costs almost nothing for users on the in-memory path and unlocks 2–3 orders of magnitude more catalog scale, plus a real sharing story, for users who need it. The Phase 1 protocol design pays off here: loaders, samplers, and stitchers are unchanged.

What needs to happen to ship Phase 2:

- [ ] Implement `DuckDBGeoCatalog` behind a `[duckdb]` extra, satisfying the `GeoCatalog` protocol from Phase 1.
- [ ] Canonicalize the GeoParquet schema (§6.6) — types, NULL conventions, bbox column, sort order. Bump a `schema_version` column so future migrations are tractable.
- [ ] Add the streaming-write builders (`backend="duckdb"`) so multi-million-file ingest doesn't materialize a gdf.
- [ ] Round-trip tests on every backend at three scales: 10², 10⁴, 10⁶ rows.
- [ ] Cross-CRS smoke test (mixed-UTM-zone catalog harmonized to EPSG:4326 at write time).
- [ ] A `georeader catalog` CLI surface (§6.7) with the half-dozen subcommands above.
- [ ] One worked example per category in the docs, mirroring §8.

What I'd defer to Phase 3:

- A federated discovery layer (STAC bridge: produce a STAC catalog from a `GeoCatalog`, or open a STAC endpoint as a `GeoCatalog`).
- Persistent on-disk R-trees with automatic materialization based on query patterns.
- Distributed query (DuckDB has nothing here; this is the Iceberg / Delta Lake territory).
- A web UI for catalog inspection.

If we get §6 + §7 right, Phase 2 turns the catalog from "a Python object you build at the start of every script" into "a file you build once, version, share, and query lazily from anywhere." That's the upgrade.
