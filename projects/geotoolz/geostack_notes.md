---
title: The Modern GeoStack
subject: The Modern GeoStack
subtitle: "Motivation and ecosystem map for `geotoolz`, `xrtoolz`, and `GeoCatalog`"
short_title: GeoStack notes
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: geospatial, ecosystem, motivation, geotoolz, xrtoolz, geocatalog
---

> **Status:** living motivation document — companion to the per-design specs in [`plans/`](plans/).
> **Scope:** the modern cloud-native geospatial ecosystem and the gaps that motivate `geotoolz`, `xrtoolz`, and `GeoCatalog`.
> **Audience:** anyone trying to understand *why* these libraries exist and *where* they sit in the broader stack.

---

## Primer for newcomers

> **ELI5.** Imagine a planetary **shipping network**. **Warehouses** all over the world hold the cargo. **Trucks and ships** move it between ports. **Dock workers** crack open containers and stack the contents on **standard pallets**, which feed **factories** that build finished products. Customers find what they want through a **shipping registry**, and they see the result in **storefronts**.
>
> Modern geospatial software is exactly this network — but the cargo is *data* (satellite pixels, climate cubes, building footprints) instead of widgets, and the warehouses span continents and petabytes. The libraries I'm building are the missing pieces in this network: better **factories** for raster and cube data, and a better **registry** to tie everything together.

### The five-layer view

A flattened picture of the whole stack, before the detailed walk-through:

:::{include} diagrams/01_five_layer.html
:::

Read it from the bottom: bytes sit in **warehouses**, **forklifts** haul them out only when asked, the **registry** tells the forklift *which* warehouse to go to, **factories** turn raw cargo into finished products, and the **storefront** shows it to the human. Every layer above is useless without the one below — but the *registry* is the layer most people skip and then regret.

### Walking down the stack

The stack is eight conceptual layers — same anatomy whether the cargo is a satellite scene, a climate cube, or a building footprint. For each: what it does, how we used to do it, what changed, and what's still awkward.

**1. The warehouses (Storage).** Where the data physically lives. Today these are **object stores** in the cloud — buckets full of files, stacked in racks the size of city blocks, billed by the byte. *Old world:* tape archives, FTP servers, lab NAS appliances. Data lived where the people who collected it happened to work, and you got a copy by emailing someone or `scp`-ing it overnight. *What changed:* one address — `s3://…`, `gs://…` — for everyone, with effectively infinite capacity, and fault-tolerance baked in. *Still awkward:* the bytes alone are dumb; nothing useful happens until something fetches them, and "where exactly is my pallet inside this warehouse?" is a real problem (see layer 7).

**2. The forklifts (Transport).** Getting bytes out of the warehouse and onto your laptop. Modern transport speaks **HTTP** with **range requests** — "give me bytes 4096–8191 of this 10 GB file" — and dispatches many in parallel. *Old world:* full-file FTP downloads, mounted network shares, vendor-specific transfer protocols. If you wanted one tile out of a satellite scene, you downloaded the whole scene. *What changed:* concurrency hides latency; you only pay for what you read; a single laptop can saturate a 10 Gbps link. *Still awkward:* range reads only help if the **file format** lets you compute *which* bytes you want without reading the whole thing first — most pre-2015 formats don't.

**3. The retrofit team (Virtualisation).** A clever hack for legacy cargo. The world has petabytes of data sitting in monolithic file formats (HDF4, HDF5, NetCDF-3) that *aren't* range-readable. Rather than rewrite all of it, the retrofit team scans each old crate once, writes a tiny "where everything is inside" index, and exposes the index so modern forklifts can pretend the old crate is a new one. *Old world:* migrate everything — years of effort, double storage costs while in flight, and political fights over who owns the canonical copy. *What changed:* zero-migration access to legacy archives — the old data didn't move, but it now behaves cloud-native. *Still awkward:* one more layer of indirection to maintain, and the indices go stale when the underlying files change.

**4. The dock workers (Readers).** The code that actually opens a file and turns its bytes into something you can compute on. Modern readers are **asynchronous and concurrent** — many small range reads dispatched in parallel — and *lazy*, deferring work until you ask for the result. *Old world:* synchronous, one-file-at-a-time, blocking; or worse, click-through GUI dialogs in a vendor application (ArcToolbox, ENVI menus). *What changed:* a single Python process can drive thousands of concurrent reads. *Still awkward:* async code is harder to reason about, and each data geometry (raster vs cube vs vector) needs its own specialised reader — there's no one universal "open this geo-thing" function.

**5. The standard pallet (Substrate).** Once cargo is unloaded, what *shape* does it take in memory? The modern answer is a **typed carrier** — an array (or table) that remembers its coordinates: which axis is latitude, which row is "Paris", which column is "geometry", what the units are, what projection the world is in. *Old world:* bare arrays plus sidecar files (`.prj`, `.tfw`, `.hdr`, `.aux.xml`) you had to read by hand and keep in sync. Lose the sidecar, lose the meaning. *What changed:* coordinate-aware operations don't need manual bookkeeping; downstream code stays geographically honest. *Still awkward:* the three data geometries (raster / cube / vector) each have a different idea of "the right pallet" — there isn't yet one universal carrier.

**6. The factories (Compute / engines).** Where the science actually happens — pixel math, regridding, climatologies, spatial joins, ensemble statistics. Modern factories are **composable**: small typed operators chained into pipelines, like Lego blocks for analysis. *Old world:* monolithic GIS suites (ArcGIS, ENVI, GRASS, IDL) where every workflow was a vendor-specific click-trail, plus a long tail of one-off scripts copy-pasted between projects. *What changed:* reusable, testable units that work the same in a notebook, a script, and a server. *Still awkward:* the factory ecosystem is **fragmented** — each data geometry has its own factory, and operators don't always compose cleanly across them. (This is one of the gaps the libraries below try to close.)

**7. The shipping registry (Discovery / Catalog).** The single most underrated layer. A registry knows *what* exists, *where* each pallet is stored, and *what's inside* each container — without having to physically open them. You query the registry first, get back a short list of warehouse addresses, then send the forklifts. *Old world:* directory-of-files plus a README, hand-maintained CSV indices, FGDC metadata XML no one updated, or full-blown enterprise spatial databases for the lucky few. At petabyte scale there are heavyweight standardised registries (you'll meet them later); at lab scale most teams still wing it. *What changed:* the same columnar file format used for tabular data can *hold* the metadata index, queryable with the same SQL engine you already use for analytics — the registry becomes "just another file". *Still awkward:* no universal agreement on *what* metadata to record, and registries decay if no one keeps the crawls running.

**8. The storefront (Viz / UX).** Where humans actually see the result. Modern storefronts are **GPU-accelerated** and **in-browser**, reading straight from cloud storage — interactive maps that fetch tiles on demand and render millions of features at 60 fps. *Old world:* render to PNG, save to disk, open in a desktop GIS (QGIS, ArcMap), or maintain a heavyweight tile-server farm pre-rendering images for the web. *What changed:* the same notebook that ran the analysis can also render the map; the same cloud bytes feed both. *Still awkward:* GPU memory is finite, browsers have limits, and "ship a million polygons to a tablet" is still a real engineering problem.

### Two organising ideas

Borrowing the networking metaphor: the modern stack splits cleanly into a **control plane** and a **data plane**.

- **Control plane (the brain).** Knows *what* exists, *where* it lives, and *how* to find it. This is the **registry** (layer 7).
- **Data plane (the muscle).** Moves bytes, decodes pixels, runs math. This is layers 1–6 plus 8.

You need both. A registry with no readers is a phone book with no phones; readers with no registry means every workflow re-implements path-globbing and metadata parsing.

### The three stacks

Geospatial data has three fundamentally different geometries, and each gets its own optimised stack — **same eight layers, different specialisations** in each row:

| Stack | Geometry | Cargo (Storage) | Pallet (Substrate) |
|---|---|---|---|
| **Imagery** | 2D / 3D pixel grids | tiled-and-overviewed raster files | typed georeferenced array |
| **Dense** | N-D coordinate cubes | chunked array stores | labelled-coordinate dataset |
| **Vector** | Discrete features | columnar tabular files | typed spatial table |

The registry sits *above* all three and uses the **vector stack** to govern the other two — i.e., it stores raster and cube metadata as rows in a columnar tabular file and queries them with SQL.

> The rest of this document names the actual tools, walks each of the three stacks layer by layer, shows the byte-by-byte lifecycle of a query in each, and maps the gaps onto the libraries I'm building.

---

## Why this exists

The cloud-native geospatial ecosystem is **mature at the edges** (storage formats, byte transport, viz) and **fragmented in the middle** (slicing, indexing, composition). Three concrete gaps motivate this work:

**1. No unified compute layer for imagery.** `rio-tiler` does web tiles, `rasterio` does GDAL-backed reads, `eo-learn` does workflow DAGs — but nothing composes pixel-level operators (band math, masking, regridding, sensor calibration) over a typed `GeoTensor` carrier with the same ergonomics as PyTorch's `nn.Sequential`. → **`geotoolz`**.

**2. No first-class operator algebra for dense cubes.** `xarray` is a brilliant *substrate* (labelled arrays, coordinates, chunking) but operator composition still happens as ad-hoc function chains. Climatologies, regridding, EOF decompositions, and ensemble statistics all want a shared shape. → **`xrtoolz`**.

**3. No lightweight catalog for personal/lab-scale work.** STAC is the gold standard at petabyte scale, but spinning up a STAC API for a few hundred TB of project data is overkill. A GeoParquet-backed catalog queryable with DuckDB closes the gap between "directory of files" and "full STAC deployment". → **`GeoCatalog`** (see [`plans/geodatabase/`](plans/geodatabase/)).

---

## The unified picture

### The Grand Unified GeoStack

The full stack, with `GeoCatalog` as the discovery layer that ties the three data-plane stacks together:

:::{include} diagrams/02_unified_stack.html
:::

### The catalog is the glue

The key move across all three stacks: the **vector stack** (GeoParquet metadata, DuckDB query) is what makes the **imagery** and **dense** stacks navigable. Without a catalog, every workflow re-implements `glob()` + STAC parsing and the boilerplate compounds across projects. Each stack below ends with a **lifecycle of a query** sequence diagram showing exactly how the catalog hand-off works for that data geometry.

---

## The three stacks

Each stack is a **specialised silo** optimised for its data geometry, but they share a common transport foundation (`obstore`) and a common discovery layer (`GeoCatalog`).

### Shared anatomy

The same six layers repeat across all three stacks; only the row contents change.

| Layer | Imagery | Dense | Vector |
|---|---|---|---|
| **Storage** | COG | Zarr · HDF5 · NetCDF | GeoParquet · PostgreSQL · FlatGeobuf |
| **Transport** | `obstore` · GDAL/VSI | `obstore` · `kerchunk` | `obstore` · GDAL/VSI |
| **Readers** | `RasterioReader` · `async-geotiff` · `rio-tiler` | `zarr-python` · `icechunk` · `kerchunk` · `lazycogs` | `pyogrio` · `PostGIS` · `SedonaDB` |
| **Substrate** | `GeoTensor` · `GeoSlice` (`georeader`) | `DataArray` · `Dataset` · chunks (`xarray`) | `GeoArrow` · `GeoDataFrame` |
| **Compute** | **`geotoolz`** | **`xrtoolz`** | `DuckDB` · `PostGIS` · `SedonaDB` |
| **Viz** | `lonboard` · `titiler` · `terracotta` | `lonboard` · Holoviz · Datashader | `lonboard` · Felt |

### Stack A — Imagery (Rasterio territory)

**Focus.** 2D/3D satellite and aerial imagery.
**Philosophy.** Window-based processing — crop spatial slices from massive COGs via HTTP range requests.

:::{include} diagrams/03_stack_imagery.html
:::

#### Layers

**Storage — [COG (Cloud Optimized GeoTIFF)](https://www.cogeo.org/).** A standard GeoTIFF with two structural choices: pixels are organised into internal tiles (typically 256×256 or 512×512), and decimated overviews (½, ¼, ⅛ resolution) are appended to the same file. The IFD (Image File Directory) at the start of the file lists the byte offset and length of every tile and every overview. Because clients can read the IFD with a single small HEAD/GET, then fetch only the tiles overlapping their AOI via HTTP range requests, a 10 GB Sentinel-2 scene becomes a few-MB read for a small region. *Advantage: arbitrary spatial subsetting over HTTP without downloading the whole file.*

**Transport — [`obstore`](https://github.com/developmentseed/obstore) and [GDAL/VSI](https://gdal.org/user/virtual_file_systems.html).** Two parallel paths into cloud storage. **`obstore`** is a Python wrapper around the Rust [`object_store`](https://docs.rs/object_store/) crate — it speaks S3, GCS, Azure Blob, and HTTP natively, dispatches concurrent range requests, and benchmarks 2–5× faster than `fsspec`-based alternatives for byte-range workloads. **GDAL/VSI** is the C++ "Virtual File System" baked into GDAL (`vsis3://`, `vsigs://`, `vsicurl://` etc.), which is what `rasterio` uses by default — mature, ubiquitous, but synchronous, and configured through environment variables (`AWS_*`, `GDAL_HTTP_*`). *Advantage: `obstore` for new async pipelines, GDAL/VSI for the long tail of formats GDAL already understands.*

**Readers — `RasterioReader`, [`async-tiff`](https://github.com/developmentseed/async-tiff), [`rio-tiler`](https://github.com/cogeotiff/rio-tiler).** This is where bytes become arrays.
- **`RasterioReader`** — `rasterio`/GDAL-backed, synchronous, the workhorse. Supports every format GDAL does, not just COG. When you just want to open a file and read pixels, this is it.
- **`async-tiff`** — pure-Rust async COG reader. Parses IFDs and dispatches concurrent tile fetches through `object_store`. No GDAL dependency. Optimised for many-small-tiles workloads (web tile servers, parallel scene processing).
- **`rio-tiler`** — handles web-tile coordinate math: given an XYZ tile coordinate, compute the COG pixel window, read it, resample to 256×256, encode as PNG/JPEG.

*Advantage: pick the reader that matches your concurrency model — sync for scripts, async for high-concurrency servers and batch fan-out.*

**Substrate — [`georeader`](https://github.com/spaceml-org/georeader).** A small object model that lets every reader hand back the same shape of object, so downstream code doesn't care which reader produced it.
- **`GeoTensor`** — an N-D array (numpy `ndarray` subclass) carrying its CRS and affine transform, so coordinate-aware ops never lose georeferencing.
- **`GeoSlice`** — a declarative crop specification (bounds + CRS + optional time window). Decouples *what region* from *which file*.
- **`GeoData`** — protocol-style container any reader can satisfy.
- **`GeoCatalog`** — index over `GeoData` instances, queryable by bounds/time/attrs.

*Advantage: a typed carrier means coordinate metadata propagates automatically through pipelines, no `crs=` arguments threaded through every call.*

**Compute — `geotoolz`.** Operator algebra over `GeoTensor`. A single `Operator` is a typed function `GeoTensor → GeoTensor`; `Sequential` composes a linear chain (e.g. `mask → calibrate → NDVI`); `Graph` handles multi-input DAGs (e.g. fusing radar + optical inputs). Mirrors the ergonomics of PyTorch's `nn.Sequential`/`nn.Module` but for georeferenced rasters. *Advantage: composable, testable units with carrier-aware semantics.* See [`plans/geotoolz/`](plans/geotoolz/).

**Viz / UX — [`titiler`](https://github.com/developmentseed/titiler), [`terracotta`](https://github.com/DHI-GRAS/terracotta), [`lonboard`](https://github.com/developmentseed/lonboard).**
- **`titiler`** — FastAPI-based dynamic tile server. Reads COGs directly from cloud storage and returns XYZ tiles on demand. The standard for "give me a URL, get a slippy map".
- **`terracotta`** — lighter-weight, SQLite-indexed, pre-processed-tile server. Trades flexibility for raw serving speed.
- **`lonboard`** — GPU-accelerated Jupyter renderer using deck.gl. Pushes raw arrays straight to the GPU; ideal for inline notebook exploration of millions of pixels.

*Advantage: dynamic vs pre-baked tile serving for production; in-notebook GPU rendering for exploration.*

#### Lifecycle of a query

A user asks for a low-cloud Sentinel-2 mosaic of Paris in 2024:

:::{include} diagrams/04_lifecycle_imagery.html
:::

**The trick.** The reader makes *two* round trips — first a cheap header read to learn tile layout, then a fan-out of concurrent range reads for just the tiles that overlap the AOI. That's how you turn a 10 GB scene into a 3 MB transfer.

**What's specific.** GDAL/VSI is the *legacy* transport for synchronous readers; `obstore` is the *modern* transport for async readers. `geotoolz` carries `GeoTensor` (CRS + affine + array) end-to-end so operators stay coordinate-aware.

### Stack B — Dense (Xarray territory)

**Focus.** Multi-dimensional scientific grids — climate, weather, oceanography.
**Philosophy.** Coordinate-based DataCubes. Time, altitude, and variable are physical dimensions of a single labelled array.

:::{include} diagrams/05_stack_dense.html
:::

#### Layers

**Storage — [Zarr](https://zarr.dev/), [HDF5](https://www.hdfgroup.org/solutions/hdf5/), [NetCDF](https://www.unidata.ucar.edu/software/netcdf/).** All three store N-D arrays plus metadata, but with very different physical layouts.
- **Zarr** is *cloud-native by design*: the array is split into chunks, each chunk is a separate file (e.g. `var/0.0.5`), and metadata lives in small JSON files (`.zarray`, `.zattrs`). Readers fetch only the chunks they need.
- **HDF5 / NetCDF-4** are *legacy* monolithic binary formats — single big files with internal B-tree indices. Brilliant on local disk; painful in object storage because reading any chunk often requires walking the tree, which means many small range reads.

*Advantage of Zarr: parallel reads/writes scale linearly with object-store concurrency. Advantage of HDF5/NetCDF: ubiquitous in climate science and decades of tooling.*

**Transport — [`obstore`](https://github.com/developmentseed/obstore).** Same Rust-backed mover as Stack A. Zarr's "many small files" workload is exactly what `object_store` is tuned for — concurrent GETs without the per-request overhead of `s3fs`/`gcsfs`. *Advantage: makes a Zarr that lives in S3 feel like a Zarr that lives on a fast local disk.*

**Virtualisation — [`kerchunk`](https://github.com/fsspec/kerchunk) / [VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr).** A clever trick for the legacy formats: scan an HDF5/NetCDF file once, record the byte offsets of every chunk into a small JSON or Parquet "reference file", then expose that reference as a virtual Zarr store. Readers see a Zarr; under the hood `obstore` is doing range reads into the original HDF5. *Advantage: no rewriting of petabytes of legacy data — you get cloud-native access patterns for free.*

**Readers — [`zarr-python`](https://github.com/zarr-developers/zarr-python), [`Icechunk`](https://icechunk.io/), [`stackstac`](https://stackstac.readthedocs.io/) / [`odc-stac`](https://odc-stac.readthedocs.io/), [`lazycogs`](https://github.com/developmentseed/lazycogs).**
- **`zarr-python`** — the canonical Zarr reader. Recently (Zarr v3) gained a Rust-backed obstore-aware backend.
- **`Icechunk`** — Earthmover's transactional, version-controlled storage engine on top of Zarr. Adds Git-like snapshots, atomic commits, and time-travel to climate datasets — critical when you're updating a global temperature cube nightly and need rollback semantics.
- **`stackstac` / `odc-stac`** — the established pattern for "take a STAC item collection, expose it as a spatially-aligned `xarray.DataArray`". GDAL/rasterio under the hood.
- **`lazycogs`** — DevSeed's Rust-native answer to the same pattern. Same surface (a lazy `(band, time, y, x)` `xarray.DataArray` from a STAC-geoparquet collection of COGs), but the I/O stack is `async-geotiff` + `obstore` + `rustac` (DuckDB on stac-geoparquet) — no GDAL. **Independently validates the GeoCatalog Phase 2 design** by using the same `stac-geoparquet` + DuckDB pattern for catalog discovery.

*Advantage: `zarr-python` for vanilla cubes, `Icechunk` when you need ACID-like guarantees on a writable cube, `stackstac` / `odc-stac` for STAC-driven cube assembly via GDAL, `lazycogs` for the same pattern with a Rust-native I/O stack and STAC-geoparquet predicate pushdown.*

**Substrate — [`xarray`](https://docs.xarray.dev/).** The de-facto Python model for labelled N-D arrays.
- **`DataArray`** — a single N-D array with named dimensions and labelled coordinates (e.g. *temperature* at specific *lat*, *lon*, *time*).
- **`Dataset`** — a dict-like container of `DataArray`s sharing coordinates (e.g. *temperature* + *humidity* on the same grid).
- **Chunks** — internal subdivisions allowing Dask (or other engines) to process the cube in parallel.

*Advantage: coordinate-aware indexing (`ds.sel(time="2024-06", lat=slice(45, 50))`) means you write the science in the units of the science, not array indices.*

**Compute — `xrtoolz`.** Operator algebra over `xarray.Dataset`/`DataArray`, mirroring `geotoolz` but for cubes. Targets the recurring patterns: climatologies (rolling means, seasonal anomalies), regridding (between map projections), EOF / spectral decompositions, and ensemble statistics. *Advantage: composable units that respect xarray's labelled-coordinate semantics, instead of ad-hoc `groupby().mean()` chains scattered across notebooks.*

**Viz / UX — [`lonboard`](https://github.com/developmentseed/lonboard), [HoloViz](https://holoviz.org/) ([Datashader](https://datashader.org/), [hvPlot](https://hvplot.holoviz.org/)).**
- **`lonboard`** — pushes raw grid values to the GPU for real-time interaction with scientific data.
- **HoloViz / Datashader** — server-side aggregation of massive grids into browser-readable rasters. Essential when the data exceeds millions of points and naive rendering would melt the browser.

*Advantage: GPU rendering for interactive exploration; server-side aggregation for browser-friendly delivery of huge cubes.*

#### Lifecycle of a query

A climate scientist asks for monthly mean SST over the North Atlantic for 1990–2024:

:::{include} diagrams/06_lifecycle_dense.html
:::

**The trick.** Coordinate selection (`.sel`) is translated into chunk indices *before* any bytes are read; only the chunks intersecting the requested coordinate hyperrectangle are fetched. Lazy evaluation through Dask means downstream reductions (`.mean()`) collapse the chunk-fetch + compute graph into a single optimised pass.

**What's specific.** A *virtualisation* layer (`kerchunk`) sits between readers and transport — it makes legacy HDF5/NetCDF behave like Zarr without rewriting the underlying files. `xrtoolz` provides operator composition over `Dataset`/`DataArray` (climatologies, regridding, EOFs, ensemble statistics).

### Stack C — Vector (GeoPandas territory)

**Focus.** Discrete features — points, lines, polygons, and their attributes.
**Philosophy.** Tabular and relational. Geography is just another column in a high-performance database.

:::{include} diagrams/07_stack_vector.html
:::

#### Layers

**Storage — [GeoParquet](https://geoparquet.org/), [PostgreSQL/PostGIS](https://postgis.net/), [FlatGeobuf](https://flatgeobuf.org/).**
- **GeoParquet** — the modern standard. Apache Parquet (columnar, compressed, partitioned into row groups) with a spec extension for storing geometries (WKB or GeoArrow encoding) and per-row-group bbox statistics. Cloud-native, range-readable, predicate-pushdown friendly.
- **PostgreSQL + PostGIS** — the relational gold standard for stateful, multi-user vector data. Stores geometries as binary blobs with R-tree (GiST) indexes; handles concurrent writes and complex spatial joins.
- **FlatGeobuf** — a streamable flatbuffer format with a packed Hilbert R-tree at the start of the file. Allows fast spatial filtering over plain HTTP without any SQL engine.

*Advantage: GeoParquet for analytical workloads on cloud storage; PostGIS for transactional / multi-user; FlatGeobuf for lightweight HTTP-only access patterns.*

**Transport — [`obstore`](https://github.com/developmentseed/obstore), [GDAL/VSI](https://gdal.org/user/virtual_file_systems.html).** `obstore` pulls Parquet row groups from cloud storage; GDAL/VSI is what `pyogrio` uses to interface with the long tail of vector formats (GPKG, SHP, GeoJSON, KML, …). *Advantage: same dual-path story as Stack A — modern async path for Parquet, mature GDAL path for legacy formats.*

**Readers — [`pyogrio`](https://github.com/geopandas/pyogrio), [PostGIS](https://postgis.net/), [Sedona](https://sedona.apache.org/).**
- **`pyogrio`** — vectorised interface to GDAL/OGR vector drivers. Reads/writes via Arrow buffers, an order of magnitude faster than the legacy `fiona`-based path.
- **PostGIS** — the actual reader for rows out of a PostgreSQL spatial table; uses GiST indexes for spatial filtering.
- **Apache Sedona** — distributed engine for petabyte-scale vector data on Spark/Flink clusters. The right tool when "global road network" or "every building footprint on Earth" doesn't fit on one machine.

*Advantage: `pyogrio` for single-node analytics, PostGIS for transactional, Sedona for distributed.*

**Substrate — [GeoArrow](https://geoarrow.org/), [GeoPandas](https://geopandas.org/).**
- **GeoArrow** — a standardised Arrow memory layout for spatial data. Geometries are stored as native Arrow extension types (interleaved or struct coordinates), so DuckDB can hand a query result to `lonboard` *without re-serialising* — both speak Arrow buffers.
- **GeoPandas** — the Python `DataFrame`-with-a-geometry-column standard. Familiar pandas-like API; in-memory, single-node.

*Advantage: GeoArrow's zero-copy hand-off eliminates the "serialise to GeoJSON, parse, re-serialise" tax that used to dominate vector pipelines.*

**Compute / Engines — [DuckDB](https://duckdb.org/) (with [`spatial`](https://duckdb.org/docs/extensions/spatial/overview)), [PostGIS](https://postgis.net/), [Sedona](https://sedona.apache.org/).**
- **DuckDB + `spatial`** — in-process SQL engine that reads GeoParquet directly from S3, with predicate pushdown on the bbox column. The closest thing the geospatial world has to "SQLite for analytics".
- **PostgreSQL + PostGIS** — production multi-user spatial database. Concurrent writes, complex joins, mature query planner.
- **Sedona** — distributed spatial SQL on Spark.

*Advantage: DuckDB collapses a whole category of "load shapefile → filter in pandas" workflows into a single SQL statement against cloud storage. **This is the engine that powers `GeoCatalog`.***

**Viz / UX — [`lonboard`](https://github.com/developmentseed/lonboard), [Felt](https://felt.com/).**
- **`lonboard`** — pushes GeoArrow buffers directly to the GPU via deck.gl. Renders millions of polygons smoothly inside Jupyter.
- **Felt** — collaborative web mapping for sharing finished vector products with non-technical stakeholders.

*Advantage: GPU-rendered interactive exploration in notebooks; collaborative web sharing for handoff.*

#### Lifecycle of a query

An analyst asks for all building footprints in a flood zone:

:::{include} diagrams/08_lifecycle_vector.html
:::

**The trick.** Two stages of filtering: (1) **row-group pruning** using the bbox statistics in the Parquet footer skips entire row groups that can't possibly intersect, and only the surviving groups are fetched. (2) **column pruning** means non-needed attribute columns are never read off the wire. The result lands as Arrow buffers, so the hand-off to `lonboard` or GeoPandas is zero-copy.

**What's specific.** `GeoArrow` enables *zero-copy* hand-off — DuckDB can stream query results straight into `lonboard` without re-serialisation. This is also the stack that powers `GeoCatalog`: the catalog *is* a GeoParquet file queried with DuckDB.

---

## Where my libraries fit

Most of the modern stack is **already there**. Two of the three libraries below (`georeader` modernisation, `GeoCatalog`) are *reconciliation* work — taking pieces that already work and giving them a coherent surface. The third (`geotoolz` + `xrtoolz`) is the genuinely missing piece: the **factory layer** that doesn't yet have a good general-purpose shape.

:::{include} diagrams/09_libraries_fit.html
:::

Stars mark where my libraries plug in. Read it from the top: a user wants something → engines compose carriers → the registry tells them which files to open → readers translate cloud bytes into typed carriers → transport hauls only the bytes needed → cloud storage is the source of truth.

---

### 1. Modernising `georeader` *(reconciliation, not invention)*

`georeader` already gives us a clean object model: **`GeoTensor`** (georeferenced array), **`GeoSlice`** (declarative crop), and a small reader surface anchored on **`RasterioReader`**. What it doesn't yet have is a *protocol* — a shared shape every reader can satisfy — which means async / GDAL-free readers reinvent the substrate every time.

**What plugs in.**
- **[`async-geotiff`](https://github.com/developmentseed/async-geotiff)** — DevSeed's high-level async COG reader, Rust-backed via [`async-tiff`](https://github.com/developmentseed/async-tiff). Already does IFD walk, tile-fetch, decompression dispatch, request coalescing, and decoding off the event loop. We wrap it in a ~80-LOC `AsyncGeoTIFFReader` adapter.
- **[`obspec`](https://github.com/developmentseed/obspec)** — the upstream typed Protocol for object-store byte access (DevSeed). [`obstore`](https://github.com/developmentseed/obstore) is the reference implementation. We pass `obspec.AsyncStore` straight through to `async-geotiff`; we don't define our own `ByteStore` Protocol.

**What falls out for free.** Today's **`GeoData` / `GeoDataBase`** Protocols stay as the sync surface; we add a single new **`AsyncGeoData`** Protocol mirror (~30 LOC) for the async reader; cross-cutting types (`GeoSlice`, `Credential`) live in a shared home in [`plans/types/`](plans/types/); existing `RasterioReader` keeps doing what it does well; the new async reader is a thin adapter rather than a from-scratch COG reimplementation.

> **What about `lazycogs`?** [`developmentseed/lazycogs`](https://github.com/developmentseed/lazycogs) is excellent — but it returns `xarray.DataArray`, not `GeoTensor`, so it belongs in **Stack B (Dense / xarray)** above as a STAC-driven peer of `stackstac` / `odc-stac`. A sync GDAL-free `GeoTensor` reader for `geotoolz` was floated and deferred to v0.5+ (no clear customer that `RasterioReader` + `AsyncGeoTIFFReader` don't already cover); see [`plans/georeader/README.md` open question §3](plans/georeader/README.md).

**Sensor-specific readers built on the same Protocol.** Once the Reader Protocol stabilises, per-sensor readers slot in without re-inventing the substrate — each handles its own quirks (`+proj=geos` affines, bowtie distortion, irregular file formats) but emits the same `GeoTensor` carrier:

- **Geostationary** — `ABI_L1b` (GOES), `SEVIRI_Native` (Meteosat), with `MTG_FCI`, `Himawari_AHI` HSD, and `SEVIRI_HRIT` queued.
- **Polar-orbiting** — `MODIS_L1B` (Terra/Aqua), `VIIRS_L1B` (S-NPP/JPSS).
- **Public bucket helpers** — one-line URL constructors for the AWS/GCS open-data buckets each sensor lives in.

→ [`plans/readers/`](plans/readers/) for the per-sensor designs.

**Honest framing.** This isn't a new library. It's a refactor that turns several already-working tools into a coherent stack with a single Reader Protocol surface. *It's the prerequisite for everything else below* — `geotoolz` operators don't compose if every reader hands back a different shape of object.

→ Designs: [`plans/georeader/`](plans/georeader/), [`plans/types/`](plans/types/), [`plans/readers/`](plans/readers/).

---

### 2. `GeoCatalog` — *the registry as a file*

The pitch in one line: **store metadata as GeoParquet, query it with the tool that fits the scale — GeoPandas for most cases, DuckDB when the catalog gets big.**

**Why now.** [STAC](https://stacspec.org/) is the gold-standard registry at petabyte scale where you have a team running an API. It's overkill for the *lab-scale* problem most teams actually have: a few TB of project data on object storage that you want to slice by bbox + date + attribute without spinning up infrastructure. Meanwhile [GeoParquet](https://geoparquet.org/), [GeoPandas](https://geopandas.org/), and [DuckDB Spatial](https://duckdb.org/docs/extensions/spatial/overview) have matured to the point where the metadata index can be *just another file*.

**Two backends, one API.**

- **Phase 1 — GeoPandas in memory.** The catalog is *literally* a `geopandas.GeoDataFrame` with a `pd.IntervalIndex` for time. R-tree spatial indexing and interval-tree temporal lookup come for free from the existing libraries — no new spatial data-structure code. This handles the **90% case**: ≤10⁵ rows, fits in RAM, runs without a query engine. Round-trip via `to_geoparquet(...)` / `from_geoparquet(...)` for portability.
- **Phase 2 — DuckDB on GeoParquet.** When the catalog grows past ~10⁶ rows or has to live in cloud storage, the same API switches to a `DuckDBGeoCatalog`. SQL queries with predicate pushdown via the GeoParquet 1.1 bbox column; no daemon, no server. Most projects never need this — but the upgrade path is one constructor swap.

**The point: GeoParquet is the artifact, GeoPandas is the default tool, DuckDB is the scale-up.** You can do 90% of analysis with just the first two — `gpd.read_parquet(...)` plus the geopandas API you already know — and the third joins in transparently when row counts demand it.

**What's in the API.**
- **Builders** — one per substrate (`build_raster_*`, `build_xarray_*`, `build_vector_*`). Crawl storage once, extract bounds + CRS + timestamps, emit a `GeoCatalog`. Sensor-aware presets handle Sentinel-1/2, MODIS, VIIRS, ABI, SEVIRI, etc.
- **Loaders** — return `GeoTensor` (raster) or `xr.Dataset` (xarray-backed) for each catalog row, ready to compose with operators downstream.
- **Samplers** — `random_geo_sampler`, `grid_geo_sampler` produce `GeoSlice` objects from the catalog; `stitch_predictions` reassembles inference outputs back into geographic coordinates.
- **Set algebra** — `intersect`, `union`, `query` for cross-catalog operations (raster × label fusion, multi-sensor pairing, change-detection across epochs).
- **Adapters** — `GeoDataset` (torch `IterableDataset`); STAC read/write so a STAC catalogue can be lifted into the lighter format and back.

**Honest framing.** Nothing here is new technology. The contribution is the **convention** — the canonical schema, the IntervalIndex+R-tree pairing, the GeoParquet 1.1 bbox column convention — plus a thin Python API that hides the schema and connection plumbing. STAC-compatible read/write keeps the door open when you scale past lab size.

→ Designs: [`plans/geodatabase/geocatalog.md`](plans/geodatabase/geocatalog.md) (Phase 1, GeoPandas), [`plans/geodatabase/geoduckdb.md`](plans/geodatabase/geoduckdb.md) (Phase 2, DuckDB).

---

### 3. `geotoolz` + `xrtoolz` — *the factory layer that's missing*

This is the part of the stack where I keep hitting the same gap, regardless of which institution or project I'm working at. The storage / transport / discovery layers have gotten brilliant. The **research-notebook → operational-pipeline** leap is still mostly hand-rolled glue.

#### What shape is actually missing?

There *are* operator-flavoured libraries in the geospatial ecosystem. Each is excellent at what it does. None of them hits all three of *typed carrier* + *composable algebra* + *backend-agnostic execution*:

- **[`eo-learn`](https://github.com/sentinel-hub/eo-learn)** (Sentinel Hub) — workflow DAG over Earth-observation tasks. Wraps each step as an `EOTask` operating on a heavyweight `EOPatch` container, chained via `EOWorkflow`. **Different shape:** tightly coupled to `EOPatch` (not a general-purpose carrier); `EOWorkflow` is a *workflow runner*, not an operator algebra in the `nn.Module` sense; designed for batch processing, less so for interactive composition; doesn't address the dense-cube domain.
- **[`torchgeo`](https://github.com/microsoft/torchgeo)** — PyTorch-native datasets, samplers, and model architectures for ML on remote-sensing imagery. **Different shape:** ML-pipeline focused (`Dataset`/`DataLoader`/`LightningModule`), not a general analysis library; no `Sequential`-style composition for non-ML transforms; doesn't cover cubes.
- **[`xclim`](https://github.com/Ouranosinc/xclim)** (Ouranos) — climate indicators on `xarray` with rigorous CF-convention handling. **Different shape:** a domain-specific function library tied to climate-index semantics; no composition primitives; each indicator is a standalone callable.
- **[`xesmf`](https://github.com/pangeo-data/xESMF)** — regridding between map projections via ESMF. **Different shape:** a single-capability tool, brilliant at it; doesn't claim to be more.
- **[`xarray-spatial`](https://github.com/makepath/xarray-spatial)** — spatial-analytics primitives (slope, aspect, hillshade, viewshed, focal stats). **Different shape:** a pure-function library; no typed-operator surface, no composition machinery, no backend abstraction.
- **[`coordax`](https://github.com/neuralgcm/coordax)** (Google / NeuralGCM) — a JAX-native labelled-array library from the NeuralGCM team. Coordinate-aware arrays in the spirit of xarray but built for the differentiable / JAX ecosystem. **Closest thing to the missing carrier shape.** Different shape relative to the *operator-algebra* gap: `coordax` solves the carrier (and the JAX backend) but does not itself ship `Operator`/`Sequential`/`Graph` composition. **The honest take: `xrtoolz` should probably build *on top of* `coordax` rather than reinvent the carrier — and if `coordax` matures further, much of the cube-side work collapses to "just add the operator-algebra layer".**
- **[`geocube`](https://github.com/corteva/geocube)** — vector-to-raster rasterisation. Narrow scope.
- **[`rio-cogeo`](https://github.com/cogeotiff/rio-cogeo)**, **[`pyresample`](https://github.com/pytroll/pyresample)**, **[`rasterio.features`](https://rasterio.readthedocs.io/en/stable/topics/features.html)** — single-purpose utility libraries, not algebras.

What's missing is a **general-purpose operator algebra over typed georeferenced carriers** — the `nn.Module`/`Sequential`/`Graph` shape, but for `GeoTensor` (raster) and `Dataset`/`DataArray`/`coordax.Array` (cube). The libraries above each fill *one or two* of those properties; none combines all three. `coordax` gets us closest, and is treated below as a likely *foundation* for the cube side rather than competition.

#### The two-layer ladder *(Keras-style progression of complexity)*

One of the best lessons from Keras is that **complexity should be opt-in**. Beginners reach for `model = Sequential([Dense(10), Dense(1)])`; experts reach for the functional API or `Module` subclasses. The same library scales with the user's expertise. `geotoolz` and `xrtoolz` apply the same lesson:

- **Layer 0 — Primitives.** Pure `np.ndarray → np.ndarray` functions, jaxtyped at the signature (`Float[ndarray, "*batch bands H W"]`) so callers see expected dimensions in their IDE. `ndvi(arr)`, `mask(arr, cloud_mask)`, `lee_speckle(arr, window=7)`. Drop-in replacements for the one-liner you'd write today; nothing to learn beyond Python functions + jaxtyping shape annotations.
- **Layer 1 — Operators.** Composable typed objects on the carrier: `NDVI()`, `Mask()`, `LeeSpeckle()`. Take and return `GeoTensor`. Plug into `Sequential` chains and `Graph` DAGs. Pickle-able, version-pinnable, swappable, testable in isolation.

The same operation lives at both layers — `NDVI()(tensor)` and `ndvi(np.asarray(tensor))` produce the same result. The choice is ergonomic, not semantic. *This is the on-ramp that lets a researcher write a one-liner today, lift it into a pipeline tomorrow, and ship it as a serving endpoint next month — without rewriting.*

*(`geotoolz` deliberately ships **only** these two tiers, where its sibling `xrtoolz` ships three. The asymmetry is architectural: `GeoTensor` is an `np.ndarray` subclass with `__array_ufunc__`, so metadata round-trips through ufunc-pure primitives for free, and non-ufunc primitives get wrapped at the Operator boundary in a single line. `xarray.DataArray` is composition over `ndarray`, not a subclass — its middle "tensor function" tier earns its keep. Substrate dictates tier count.)*

#### The pattern

- **`Operator`** — a typed function `Carrier → Carrier`. Stateful when needed (config, learnable weights), stateless otherwise.
- **`Sequential`** — a linear chain of operators with type-checked composition.
- **`Graph`** — a multi-input DAG built from `Input` and `Node` primitives, for fusion workflows (e.g. radar + optical + DEM into a single classifier).
- **`ModelOp`** — a framework-agnostic operator that wraps any callable from `torch`, `jax`, or `sklearn`. Inference becomes just another node in the graph.
- **Sensor presets** — one-line constructors (`Sentinel2RGB()`, `S2NDVI()`, `EmitMethanePlume()`, …) that pre-configure entire pipelines for common sensors. Newcomers get a working pipeline in a single line; experts pop the hood and inspect the operator chain.
- **Hydra / hydra-zen friendly.** Every Operator is config-serialisable, so YAML-driven pipelines are first-class — the same graph runs from a notebook, a script, or a config file.
- **Carrier-aware.** CRS, transform, coordinates, units, fill values propagate automatically. The pipeline stays geographically honest without manual bookkeeping.
- **Backend-agnostic.** ***Carriers determine the surface, backends determine the scale.***

#### Why this matters — the use-case spectrum

The same operator graph supports:

- **Research notebooks** — composition of typed building blocks beats a 200-line one-shot script. Easier to read, easier to share.
- **Production pipelines** — the same graph runs in CI, batch jobs, scheduled tasks. No translation step.
- **Backend APIs / model serving** — lift the research graph straight into a FastAPI handler or a streaming worker. The operator graph *is* the request handler. No glue layer to maintain.
- **Reproducibility & governance** — operator graphs are serialisable, version-pinnable, and auditable. Run the same analysis on the same inputs years later; ship it as a regulatory deliverable; diff two analyses.
- **Composability across teams** — operators are installable units. A colleague's calibration operator drops into your pipeline like a Python package, because it *is* one. (`eo-learn`'s vision tied to `EOPatch`; this version is carrier-typed and lighter.)
- **Testability** — every operator is a typed function. Golden-file regression tests, property-based tests, and isolation testing become trivial.
- **Education / pedagogy** — composition of typed blocks is easier to teach than monolithic notebooks. Students learn the *shape* of an analysis instead of memorising a script.

The throughline: **the gap I keep running into, across affiliations and projects, is the research-to-production translation step.** The modern stack has gotten brilliant at storage and transport; the *factory layer* that lets a research idea graduate into operational infrastructure without a rewrite is still mostly hand-rolled. That's what these libraries aim at.

#### The scale spectrum

The same composition pattern carries across data sizes — *the carrier determines the surface, the backend determines the scale*:

| Regime | Data size | Backend | Typical workflow |
|---|---|---|---|
| **Small** | fits in RAM | numpy | interactive notebook, single-scene analysis — operators run as-is |
| **Medium** | fits on one box, doesn't fit in RAM | Dask, async batch, Ray | per-tile parallel processing, daily-scale pipelines — same operators dispatched per chunk/tile |
| **Big** | needs a cluster | distributed JAX (via `coordax`) | continental-scale ensemble analyses — same composition pattern, JAX-backed carriers |

The architectural commitment: operators don't bake in a backend. They run on whatever the carrier wraps.

**Honest scope.** "Same operator everywhere" holds for **single-machine numpy, Dask-orchestrated batch, and distributed JAX** — same composition pattern, sometimes with backend-specific implementations under one signature. Sedona / Spark and other distributed-SQL engines are *not* covered by the same code path; if you need them, the operator graph would have to emit Sedona SQL, which is a separate library and not currently planned. We'd rather deliver the 80% case cleanly than over-promise the 100%.

#### Honest research-to-prod scope

"Production" is a slippery word. The marquee pitch — *the same operator graph runs in a notebook and a serving endpoint* — holds for these targets:

- ✅ **Notebooks** (Jupyter, Marimo) — the native habitat.
- ✅ **Scripts** (`python pipeline.py`) — pickle-able Operator graphs.
- ✅ **Backend APIs** (FastAPI, Litestar, Flask) — Operator graph as the request handler.
- ✅ **Workflow orchestrators** (Airflow, Prefect, Dagster) — pickle-able graphs as task units.
- ✅ **Per-tile batch** (Dask, Ray, Modal, async-driven) — one graph per tile, parallelism at the orchestration layer.

What is *not* directly covered:

- ⚠️ **Distributed SQL engines** (Sedona, Spark, Snowflake) — different paradigm. Would require emitting SQL from operator graphs (out of scope, possible future work).
- ⚠️ **Streaming pipelines** (Flink, Beam, Kafka Streams) — operator graphs aren't streaming-aware. Wrapping each Operator as a Flink UDF works but loses composition guarantees.
- ⚠️ **Edge / mobile inference** (TFLite, ONNX runtime on mobile) — `ModelOp` wraps a callable; if the callable is exportable (ONNX), you ship the *model*, not the surrounding pre/post-processing operators.

The 80% case (notebook → API / pipeline / orchestrator / batch) is the deliverable target. The 20% (streaming, distributed SQL, edge) is acknowledged out-of-scope rather than promised and missed.

#### Specialisations

- **`geotoolz`** — carrier = **`GeoTensor`**. Domain = imagery (band math, cloud masking, atmospheric calibration, sensor presets, tile assembly, reprojection). Mirrors the "PyTorch for georeferenced rasters" shape. → [`plans/geotoolz/`](plans/geotoolz/).
- **`xrtoolz`** — carrier = **`xarray.Dataset` / `DataArray`**, with a JAX-native option via **[`coordax`](https://github.com/neuralgcm/coordax)** (NeuralGCM's labelled-array library). Domain = dense cubes (climatologies, regridding, EOFs, anomalies, ensemble statistics, spectral analysis). The carrier story is largely already solved — `xarray` for numpy/Dask, `coordax` for JAX — so `xrtoolz`'s contribution is the *operator-algebra layer* on top, mirroring `geotoolz`'s `Operator`/`Sequential`/`Graph` shape. Don't reinvent the array; build the algebra. → [github.com/jejjohnson/xr_toolz](https://github.com/jejjohnson/xr_toolz).

---

### 4. Supporting infrastructure

The cross-cutting types that flow *between* layers each get their own design — small, but load-bearing.

- **`GeoSlice`** — declarative crop specification (bbox + CRS + optional time window). Decouples *what region* from *which file*; produced by samplers, consumed by readers and operators. → [`plans/types/geoslice.md`](plans/types/geoslice.md).
- **`Credential`** — Protocol + per-cloud subclasses (Azure SAS / managed identity, AWS static / profile, GCS service account) + `from_config(...)` factory. Replaces the env-var-soup pattern every project currently re-implements. → [`plans/types/credentials.md`](plans/types/credentials.md).
- **Cloud byte transport — defer to [`obspec`](https://github.com/developmentseed/obspec).** We don't ship a Protocol of our own; `obspec.AsyncStore` is the upstream surface and `async-geotiff` already consumes it. We ship one tiny `geotoolz.io.open_store(url)` factory and nothing else. → [`plans/types/bytestore.md`](plans/types/bytestore.md).

---

### Future work — JAX bridge *(via `coordax`)*

`GeoTensor` is a `numpy.ndarray` subclass — fine for 99% of cases, but it walls off **differentiability**. The future-work direction is a thin bridge that translates `GeoTensor ↔ jax.Array` (most likely via `coordax`'s coordinate-aware array) while preserving CRS / transform metadata, opening the door to:

- **Differentiable operators** — calibration, atmospheric correction, geometric warps as learnable layers.
- **Surrogate models** — physics-informed networks where the simulator and the data sit in the same operator graph.
- **End-to-end training** — gradients propagating through the geospatial pipeline.

The bridge does **not** need to be invented from scratch — [`coordax`](https://github.com/neuralgcm/coordax) is JAX-native, coordinate-aware, and probably the right backbone. The real work is the `GeoTensor ↔ coordax.Array` adapter and the CRS-preserving operator implementations, not a new array library. (An earlier working name for this glue was `geoarrax`; if `coordax` keeps maturing, even the glue may shrink to a handful of converter functions.)

Out of scope for the initial library push; flagged here so the architecture leaves room for it — operators avoid `numpy`-only assumptions, carriers stay duck-typed where possible.

---

### Reference: library × layer × design

| Library / module | Layer | Status | Design doc |
|---|---|---|---|
| `georeader` (modernised) | Substrate / Readers | Reconciliation refactor | [`plans/georeader/`](plans/georeader/) |
| Sensor-specific readers (ABI, SEVIRI, MODIS, VIIRS, …) | Substrate / Readers | New, on shared Reader Protocol | [`plans/readers/`](plans/readers/) |
| `GeoCatalog` (Phase 1, GeoPandas) | Discovery | New (on existing GeoParquet + GeoPandas) | [`plans/geodatabase/geocatalog.md`](plans/geodatabase/geocatalog.md) |
| `DuckDBGeoCatalog` (Phase 2) | Discovery | New (on existing GeoParquet + DuckDB) | [`plans/geodatabase/geoduckdb.md`](plans/geodatabase/geoduckdb.md) |
| `geotoolz` (`Operator`, `Sequential`, `Graph`, `ModelOp`, sensor presets) | Logic (imagery) | New library | [`plans/geotoolz/`](plans/geotoolz/) |
| `xrtoolz` (operator algebra over xarray / `coordax`) | Logic (cubes) | External library | [github.com/jejjohnson/xr_toolz](https://github.com/jejjohnson/xr_toolz) |
| `GeoSlice` + samplers + `stitch_predictions` | Cross-cutting types | Extracted from existing designs | [`plans/types/geoslice.md`](plans/types/geoslice.md) |
| `Credential` + per-cloud subclasses | Cross-cutting types | New | [`plans/types/credentials.md`](plans/types/credentials.md) |
| `geotoolz.io.open_store(url)` (over upstream `obspec.AsyncStore`) | Cross-cutting types | New (~30 LOC factory only) | [`plans/types/bytestore.md`](plans/types/bytestore.md) |
| JAX bridge (via `coordax`) | Logic ↔ JAX | Future work | *TBD* |

---

## See also

- [`plans/geostack.md`](plans/geostack.md) — the engineering-focused ecosystem reference (strategy tables, bytes-paths triage, end-to-end flows). Eventually this motivation doc will merge in.
- [`plans/geotoolz/`](plans/geotoolz/) — `geotoolz` operator library design.
- [`plans/geodatabase/`](plans/geodatabase/) — `GeoCatalog` + DuckDB backend design.
- [`plans/georeader/`](plans/georeader/) — reader Protocols and concrete reader designs.
- [`plans/types/`](plans/types/) — cross-cutting type designs (`GeoSlice`, `Credential`; `bytestore.md` is a passthrough note for upstream `obspec`).
- [`plans/readers/`](plans/readers/) — per-sensor reader designs (geostationary, MODIS, …).

### The success-story anchor

The motivation in this doc is anchored on a concrete, projected end-to-end pipeline: **MARS-style methane attribution** rebuilt on the unified stack across three instruments (TROPOMI + GHGSat + EMIT) and three orchestration phases (research notebook → batch pipeline → FastAPI alert service). The full plan lives in:

- [`operational_attribution.md`](operational_attribution.md) — projected end-to-end pipeline + validation plan + migration story from current MARS-style work.

It's the single best demonstration of the *"same operator graph runs in research and production"* claim — and the place where the geotoolz / xrtoolz / GeoCatalog substrate, the per-sensor readers, and the `plumax` science modules all compose into one operational artifact.
