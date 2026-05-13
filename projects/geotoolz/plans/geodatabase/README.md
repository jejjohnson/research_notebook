---
title: Geodatabase
subject: geodatabase design
subtitle: Catalog Protocol with in-memory and DuckDB backends
short_title: Geodatabase
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, geodatabase, catalog, geoparquet
---

> **Status:** design proposal — split into two phases (Phase 1 + Phase 2 below).
> **Shipping shape:** incubated as `geotoolz.catalog` inside the `geotoolz` library — *not* a standalone `geocatalog` package at v0.1. Graduation is future work, gated on API stability and a real external user that wants the catalog without the operator algebra. See [`geopatcher/README.md`](../geopatcher/README.md) for the same incubation pattern; both submodules graduate together or independently when their APIs settle.
> **Scope:** a single `GeoCatalog` Protocol with two backends — an in-memory GeoDataFrame (Phase 1) and a DuckDB-backed GeoParquet store (Phase 2) — that share the same query API and the same `GeoSlice` unit of work.
> **Audience:** anyone touching the `geotoolz.catalog` submodule, or building downstream pipelines (`geotoolz.catalog_ops.CatalogPipeline`, ML training set builders) that consume catalogs.

---

## Summary

A *geocatalog* in this design is a **spatiotemporal index over geospatial files** — a queryable mapping from `(geometry, time interval, metadata)` to a backend-specific path or asset.
Today, every project that wants one rolls its own (the `jej_vc_snippets` repo has three: one for rasters, one for xarray datasets, one for vector files).
This design promotes those snippets into the `geotoolz.catalog` submodule as a single unified API, then extends it to scale. (Earlier drafts placed the catalog inside `georeader`; that was a layering inversion — `georeader` is the substrate library and shouldn't ship a file-index layer on top of itself.)

**Phase 1** ships an in-memory `GeoCatalog` Protocol with one default implementation (`InMemoryGeoCatalog`) that wraps a `gpd.GeoDataFrame` with an `IntervalIndex` for time and an R-tree for space.
Three builder entry points (`build_raster_catalog`, `build_xarray_catalog`, `build_vector_catalog`) cover the three backend types.
Set algebra (`query`, `intersect`, `union`) operates over catalogs.
A `GeoSlice` dataclass is the unit of work passed to samplers and loaders.

**Phase 2** adds a `DuckDBGeoCatalog` backend that swaps the in-memory GeoDataFrame for DuckDB's `spatial` extension over a GeoParquet artifact.
Same Protocol, same `GeoSlice`, same `query` / `intersect` / `union` shape — but now scaling to 10⁶+ rows, queryable via SQL, persistent as a portable GeoParquet file, and analyzable without loading into Python.

The work splits into two reviewable phases.
Phase 1 must land first; Phase 2 builds on it.

---

## Motivation

Three pressures make a unified catalog layer worth doing now:

1. **Every project rolls its own catalog.** The `jej_vc_snippets/remote_sensing/` directory contains three near-identical catalog implementations (raster, xarray, vector), each ~1500–1900 LOC, with subtly different APIs.
   Promoting them into `georeader` as one Protocol with three builders eliminates the duplication and gives downstream consumers a stable surface.

2. **Catalog scale has outgrown the GeoDataFrame.** A single Sentinel-2 archive across a continent is ~10⁵–10⁶ scenes; a 10-year time series across multiple sensors easily exceeds 10⁶. Phase 1's `gpd.GeoDataFrame` works fine up to ~10⁵ rows, then `gpd.overlay`-style operations become painful and RAM eats the gdf.
   A SQL-native backend handles the same operations on a different cost model.

3. **Catalogs need to be portable artifacts.** A pickled GeoDataFrame is fragile across versions.
   A serialised GeoParquet file is the emerging standard for portable spatial tables, queryable directly by DuckDB / GDAL / pandas / geopandas without ceremony.
   Making the Phase 2 backend GeoParquet-native turns the catalog itself into a shareable artifact.

The status quo can absorb each of these one at a time, but the three concerns are coupled — making the catalog SQL-native and making it portable are the same change.
A two-phase plan (in-memory first, scale-aware second) lets the design land incrementally without forcing a SQL dependency on small-scale users.

---

## Primer for newcomers

> **ELI5.** A geocatalog is like a **library card catalog for satellite files**: each card lists where the file is, what area it covers, and when it was taken.
> Searching for *"all files over Madagascar in June 2023"* becomes flipping through index cards, not opening every book.

### What's a "geocatalog"?

**What it is.** A *queryable index over geospatial files* — a table where each row describes one file and records its bbox, time interval, CRS, and path.
Given a query like "files overlapping AOI X between dates Y and Z," the catalog returns the matching paths fast without opening any of the files.

**How it works.** Each row carries a `geometry` column (a polygon — typically each file's bbox in some CRS) and time columns (start/end).
A spatial index over `geometry` and a time index over the dates lets queries skip non-matching rows without scanning the table.
The result is "STAC, but local and Pythonic" — same idea (a file index), different shape (in-process Python instead of a JSON API).

**What this means for us.** Today, every project that wants to scale beyond "open one file at a time" rolls its own catalog.
This design promotes the pattern into `georeader` as a single shape: one Protocol, two backends (in-memory for ≤10⁵ rows, DuckDB for larger / persistent), three builders (raster / xarray / vector).
Downstream code (`geotoolz.catalog_ops.CatalogPipeline`) consumes the Protocol, indifferent to which backend is in use.

```{mermaid}
flowchart LR
    Files[10k files in S3] --> Build[build_raster_catalog]
    Build --> Cat[GeoCatalog]
    Cat --> Q[query bbox + time]
    Q --> Hits[matching paths]
    Hits --> Loader[loader.read_geoslice]
    Loader --> GT[GeoTensor]
```

### R-tree spatial index

**What it is.** An R-tree is a tree-shaped data structure that organises geometric objects (boxes, polygons) so that "find all objects intersecting this query box" runs in `O(log n + k)` instead of `O(n)`.
Standard data structure for spatial queries.

**How it works.** The tree's leaves are the actual geometries; each internal node holds a bounding box that contains all of its children.
To answer "what intersects this query box," you descend only into branches whose bounding boxes overlap — pruning whole subtrees in one comparison.
`geopandas` builds an R-tree lazily on first spatial query; you don't construct it manually.

**What this means for us.** Phase 1's `InMemoryGeoCatalog` uses geopandas's `gdf.cx[xmin:xmax, ymin:ymax]` accessor, which uses the R-tree under the hood.
For a catalog of 10k tiles, queries are sub-millisecond.
R-tree size is `O(n)`; the in-memory backend hits a wall around 10⁵–10⁶ rows because the gdf itself becomes painful, not because the index does.

### IntervalIndex (temporal)

**What it is.** A pandas `IntervalIndex` is the time-axis equivalent of an R-tree — given a query interval, it returns all stored intervals that overlap, fast.

**How it works.** Each row in the catalog has a `pd.Interval(start, end, closed='both')`.
Putting them into an `IntervalIndex` lets you call `.overlaps(query_interval)` and get a boolean mask in `O(log n + k)` time.
Combined with the R-tree spatial filter, a `query(bbox, time)` is two index probes intersected.

**What this means for us.** Time-aware catalog queries don't have to scan the full row list — even with a year of daily files (~365 rows per tile × N tiles) the temporal filter takes microseconds.
The same `IntervalIndex` shape carries through Phase 2's DuckDB backend (as `bbox` + `start_time` + `end_time` columns with predicate pushdown).

### GeoParquet + DuckDB

**What it is.** **GeoParquet** is the emerging standard for storing spatial tables as Parquet files (binary columnar format) with geometry columns encoded in WKB. **DuckDB** is an embedded SQL database (think SQLite, but column-oriented and analytics-tuned) with a `spatial` extension that reads GeoParquet natively.

**How it works.** Build a catalog as a GeoParquet file once (`gdf.to_parquet("catalog.parquet")`).
Open it as a DuckDB table, write SQL queries against it: `SELECT path FROM catalog WHERE ST_Intersects(geometry, AOI) AND date BETWEEN ...`.
DuckDB pushes the spatial predicate down to the Parquet reader so most of the file is never read.

**What this means for us.** Phase 2 unlocks 10⁶+ row catalogs that don't fit in RAM, plus the catalog itself becomes a portable artifact you can share or query from another tool (pandas, geopandas, GDAL all read GeoParquet).
The Protocol surface is unchanged from Phase 1; only the backend changes.

```{mermaid}
classDiagram
    class GeoCatalog {
        <<Protocol>>
        query()
        intersect()
        union()
        iter_slices()
    }
    class InMemoryGeoCatalog {
        gpd.GeoDataFrame
        IntervalIndex + R-tree
        ~10⁵ rows in-RAM
    }
    class DuckDBGeoCatalog {
        DuckDB + GeoParquet
        spatial extension
        10⁶+ rows on-disk
    }
    GeoCatalog <|.. InMemoryGeoCatalog : Phase 1
    GeoCatalog <|.. DuckDBGeoCatalog : Phase 2
```

---

## Goals

- **Single `GeoCatalog` Protocol** that all backends honour.
  Same `query` / `intersect` / `union` / `iter_slices` / `to_geoparquet` / `from_geoparquet` shape regardless of which backend is in use.
- **Three storage backends covered:** raster files, xarray-shaped datasets (NetCDF/Zarr/HDF), vector files (Shapefile/GeoPackage/GeoJSON).
  One catalog API, three builder entry points.
- **`GeoSlice` as the unit of work** (specified separately in [`types/geoslice.md`](../types/geoslice.md)) — a dataclass carrying `(bounds, interval, resolution, crs)` that flows from samplers to loaders without either side knowing about the catalog backend.
- **Phase 1: in-memory backend** that's good for prototyping, ML training set construction, and catalogs up to ~10⁵ rows.
  Sub-second query times via R-tree + IntervalIndex.
- **Phase 2: DuckDB backend** that scales to 10⁶+ rows, queries via SQL, persists as GeoParquet, and supports streaming construction for catalogs that don't fit in RAM during the build step.
- **Round-tripping** between backends — load a Phase 2 catalog into Phase 1 for in-process operations, persist a Phase 1 catalog as GeoParquet for sharing.

---

## Non-goals

- **Replacing STAC.** STAC is an external interchange format; `GeoCatalog` is an in-package query API. A bridge to/from STAC is a separate, lighter design.
- **Reimplementing geopandas or DuckDB.** The catalog is a thin layer over the right tool for each backend.
- **Dataset-level transformations.** Catalogs index files; they don't apply operators.
  That's `geotoolz`'s job ([`catalog_ops.CatalogPipeline`](../geotoolz/geotoolz.md)).
- **Cross-CRS unification at index time.** Catalogs store geometries in their native CRS; cross-CRS queries reproject at query time.
  Forcing a global CRS would lose information.
- **Phase 1 backend retirement.** `InMemoryGeoCatalog` stays useful for small catalogs even after Phase 2 ships — no SQL dependency, no on-disk artifact required.

---

## Constraints

- **Builds on `georeader.GeoTensor`** — the loaders return `GeoTensor` (raster), `xr.Dataset` (xarray), or `gpd.GeoDataFrame` (vector).
  The `GeoTensor` design itself is documented in [Ch. 1 of the tutorial](../../georeader_tutorial/01_geotensor.md).
- **Builds on the reader Protocol from [Reader reconciliation](../georeader/README.md)** — loaders accept any `GeoData` (sync) or `AsyncGeoData` (async) reader.
  Catalog rows store paths/URIs; loaders open them via the configured reader class.
- **`GeoSlice` is shared with multiple designs** — [`types/geoslice.md`](../types/geoslice.md) is the source of truth for the dataclass, samplers, and stitch. Changes to that contract ripple through this design and through [`geotoolz.md`](../geotoolz/geotoolz.md).
- **Phase 2 introduces a DuckDB dependency** — opt-in via an extra (`georeader-spaceml[duckdb]`), not a hard dep.

---

## High-level shape

```
                ┌──────────────────────────────────┐
                │   GeoCatalog (Protocol)          │
                │   query / intersect / union      │
                │   iter_slices / to_geoparquet    │
                └──────────────────────────────────┘
                          ▲                ▲
                          │                │
              ┌───────────┘                └────────────┐
              │                                         │
   ┌────────────────────────┐          ┌───────────────────────────┐
   │ InMemoryGeoCatalog     │          │ DuckDBGeoCatalog          │
   │ (Phase 1)              │          │ (Phase 2)                 │
   │                        │          │                           │
   │ gpd.GeoDataFrame       │          │ DuckDB + GeoParquet       │
   │ + IntervalIndex (time) │          │ + spatial extension       │
   │ + R-tree (space)       │          │ + bbox column predicate   │
   │                        │          │   pushdown                │
   │ ~10⁵ rows max          │          │ 10⁶–10⁷ rows              │
   │ in-RAM                 │          │ disk-resident, lazy       │
   └────────────────────────┘          └───────────────────────────┘
              ▲                                         ▲
              │                                         │
              │   build_raster_catalog(paths, ...)      │
              │   build_xarray_catalog(paths, ...)      │
              │   build_vector_catalog(paths, ...)      │
              │                                         │
              └─────────────────────────────────────────┘
                       (same builder entry points,
                        backend selected via kwarg)
```

```python
# Same API across backends — only the constructor changes:
catalog = georeader.catalog.build_raster_catalog(
    paths=["scene1.tif", "scene2.tif", ...],
    backend="memory",                      # Phase 1 default
    # backend="duckdb",                     # Phase 2: persisted to GeoParquet
)

# Same query shape:
hits = catalog.query(
    bounds=(-122.5, 37.0, -122.0, 37.5),
    crs="EPSG:4326",
    time=("2024-01-01", "2024-12-31"),
)

# Same set algebra:
common = georeader.catalog.intersect(catalog_s2, catalog_landsat)
all_data = georeader.catalog.union(catalog_2023, catalog_2024)

# Same GeoSlice flow:
for slice_ in georeader.sampler.grid_geo_sampler(catalog, chip_size=(256, 256)):
    gt = catalog.load_geoslice(slice_)        # → GeoTensor
    yield op(gt)
```

The test of whether the design is right: `geotoolz.catalog_ops.CatalogPipeline(catalog, op)` should not branch on which backend the catalog uses.

---

## Sub-designs

The work splits into two phases:

| # | Sub-design | Owns |
|---|---|---|
| 1 | [`geocatalog.md`](geocatalog.md) | `GeoCatalog` Protocol; `InMemoryGeoCatalog` implementation; `build_{raster,xarray,vector}_catalog` builders; `GeoSlice` dataclass; `random_geo_sampler` / `grid_geo_sampler` / `stitch_predictions`; `query` / `intersect` / `union` set algebra; `to_geoparquet` / `from_geoparquet` round-trip. |
| 2 | [`geoduckdb.md`](geoduckdb.md) | `DuckDBGeoCatalog` backend; GeoParquet 1.1 with bbox-column predicate pushdown; SQL-native spatial/temporal joins; streaming builders to avoid RAM spike during construction; `backend="duckdb"` option on the existing builders. |

Phase 1 must merge before Phase 2 starts — the Protocol it locks down is what Phase 2 implements.

---

## Sequencing

```
Phase 1 (InMemoryGeoCatalog + Protocol + GeoSlice)
   │
   ▼
Phase 2 (DuckDBGeoCatalog backend)
```

- **Phase 1 lands first.** Locks the `GeoCatalog` Protocol and the `GeoSlice` dataclass.
- **Phase 2 implements the same Protocol over DuckDB.** No new public surface; the `backend="duckdb"` kwarg on the existing builders is the only API addition.
- **Both backends coexist long-term.** Phase 1 stays useful for small catalogs and for cases where a SQL dependency isn't wanted; Phase 2 takes over when scale or persistence matter.

---

## Open questions

These are unresolved and should be discussed before each phase starts.

### 1. Index-time CRS policy

Catalogs store geometries in each file's native CRS. Cross-CRS queries reproject at query time.
Alternatives:

- **Native-CRS storage (current proposal):** preserves precision, defers reprojection to query.
- **Force everything to EPSG:4326 at index time:** simpler queries, loses precision near poles and at high resolutions.
- **Per-row CRS column:** allows mixed catalogs but forces every query to materialise reprojections.

**Tentative pick: native-CRS storage**, but the cross-CRS query implementation (especially in DuckDB where reprojection isn't trivially SQL-able) is what would change this.

### 2. `GeoSlice` semantics

Open questions about the dataclass — `resolution` representation, antimeridian policy, time semantics in `stitch` — live in [`types/geoslice.md`](../types/geoslice.md) since they're cross-cutting.
The resolution decision affects both this design (loaders consume it) and the geotoolz operator layer.

### 3. Streaming construction in Phase 2

A 10⁷-row catalog can't fit in RAM during build.
Options:

- **Append-to-Parquet streaming:** builders write rows to GeoParquet incrementally; final file is sorted/optimised at the end.
- **Build small, then concatenate:** builders run in shards, output multiple GeoParquet files, merged at the end.
- **DuckDB INSERT INTO:** builders insert into a DuckDB table directly; export to GeoParquet at the end.

**Tentative pick: append-to-Parquet** — most portable, keeps the artifact format stable throughout.

---

## Alternatives considered

- **Don't unify; let each project keep its own catalog.** Rejected: the duplication is real, the APIs drift, and downstream consumers (`geotoolz`) need a stable surface.
- **Use STAC as the in-package catalog format.** Rejected: STAC is an interchange format, not a query API. A `GeoCatalog` over STAC items is fine as a *bridge*, but forcing every catalog to be STAC-shaped (with its full metadata schema) is too heavy for the in-package use case.
- **Skip Phase 1, go straight to DuckDB.** Rejected: forces a SQL dependency on small-scale users; loses the "in-process, no extra deps" path that's the right tool below ~10⁵ rows.
- **Two separate libraries (one for in-memory, one for DuckDB).** Rejected: identical API surface across the two means they should share a Protocol; splitting into separate packages just creates a coordination tax.

---

## Connection to other designs

- **[`geotoolz.md`](../geotoolz/geotoolz.md)** — `geotoolz.catalog_ops.CatalogPipeline(catalog, op)` consumes a `GeoCatalog`.
  The pipeline iterates the catalog, applies the operator per row, writes outputs.
- **[Reader reconciliation](../georeader/README.md)** — catalog loaders open files via the configured reader class.
  The `reader_class=...` strategy injection from that design plugs into the loader at the point where bytes are first read.
- **[`geostack.md`](../geostack.md)** — situates the catalog layer in the broader ecosystem (where DuckDB sits relative to titiler / lonboard / obstore).

The catalog layer sits between the reader layer (substrate) and the operator layer (composition).
Each side talks to the other through `GeoSlice` — the catalog produces them, the reader+operator consume them.

---

## Open questions, gotchas, and warnings

The unifying GeoCatalog design is sound; cross-cutting concerns to track:

- **Cross-CRS query footgun.** Catalogs are canonicalised to one CRS (Phase 2: EPSG:4326).
  User AOIs in another CRS silently return no rows.
  Mitigation lives in [`geocatalog.md` §10.1](geocatalog.md#101-cross-crs-query-footgun) — provide a CRS-aware query helper as the canonical path.
- **GeoParquet 1.1 writer adoption is uneven** across `geopandas` and `duckdb-spatial`.
  Pin known-good versions; add a round-trip CI test.
  See [`geocatalog.md` §10.2](geocatalog.md#102-geoparquet-11-writer-adoption-is-uneven) and [`geoduckdb.md` Open questions](geoduckdb.md#geoparquet-11-writer-support).
- **Phase 2 risk of bit-rot** if no real user emerges past 10⁶ rows.
  Gate Phase 2 release behind one real user or a multi-million-row CI fixture.
  See [`geocatalog.md` §10.3](geocatalog.md#103-phase-2-risk-of-bit-rot).
- **`schema_version` column** reserved from v0.1; bump on first substantive schema change.
- **Adapter scope honesty.** v0.1 ships `to_geoparquet` / `from_geoparquet` round-trip; STAC and torch adapters are v0.2+. Don't promise adapters that aren't built.
- **Concurrent writes** are safe under append-only patterns, not under in-place mutation.
  Document the recommended write pattern when multi-writer use cases emerge.
- **`ST_Transform` in WHERE clauses is slow.** The §4-style "store all geometries in 4326, native CRS as a column" convention exists exactly to avoid this.
  Loud documentation, not a code-level fix.

The full lists live per-doc: [`geocatalog.md` §10](geocatalog.md#10-open-questions-gotchas-and-warnings) for Phase 1; [`geoduckdb.md` Open questions](geoduckdb.md#open-questions-gotchas-and-warnings) for Phase 2.
