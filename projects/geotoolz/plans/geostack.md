---
title: RS cloud-native stack
subject: Ecosystem map
subtitle: obstore, georeader, geotoolz, titiler, lonboard — and how they fit
short_title: Stack
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: reference, ecosystem, stack
---

> **Scope:** a rundown of `obstore`, `RasterioReader`, `async-geotiff`, `georeader`, `geotoolz`, `titiler`, and `lonboard` — what each is for, how they depend on one another, and which combinations match common workflows.
> *(The `lazycogs` library ships an `xarray.DataArray` carrier and is part of the dense / cube stack; see [`../geostack_notes.md`](../geostack_notes.md) Stack B for that discussion.)*  **Status:** reference document, not a design proposal. Companion to the per-design documents in [`georeader/`](georeader/), [`geotoolz/`](geotoolz/), [`geodatabase/`](geodatabase/), [`readers/`](readers/), and [`types/`](types/) — read this first to orient on the ecosystem, then dive into a specific design when you want the full spec.

## Where each topic is fully specified

This file owns the visual ecosystem map (the layered diagram, the strategy-comparison tables, the bytes-paths triage diagram, the end-to-end flows).
For the deep design of any one tool, follow the pointer:

| Topic | Full design |
|---|---|
| Reader Protocol surface (`AsyncGeoData` added; `RasterioReader` widening) | [`georeader/reader_protocol.md`](georeader/reader_protocol.md) |
| Cloud byte transport — defer to `obspec` (no Protocol of our own) | [`types/bytestore.md`](types/bytestore.md) |
| `AsyncGeoTIFFReader` — thin adapter over `developmentseed/async-geotiff` | [`georeader/reader_async_geotiff.md`](georeader/reader_async_geotiff.md) |
| `geotoolz` operator library | [`geotoolz/geotoolz.md`](geotoolz/geotoolz.md) |
| `GeoCatalog` + builders + DuckDB backend | [`geodatabase/`](geodatabase/) |
| `GeoSlice` + samplers + stitch | [`types/geoslice.md`](types/geoslice.md) |
| Per-sensor readers (geostationary, MODIS, …) | [`readers/`](readers/) |

---

## The layered diagram

There's a clean six-layer stack across this set, much of which is shipped by DevSeed / Kyle Barron.

```text
                ┌──────────────────┐    ┌────────────────┐
                │    lonboard      │    │    titiler     │
   viz / UX  →  │  Jupyter/deck.gl │    │  HTTP tile API │
                └────────┬─────────┘    └────────┬───────┘
                         │                       │
                         │ GeoTensor / Arrow     │ XYZ tiles (PNG/JPEG)
                         │                       │
                ┌────────▼───────────────────────┴───────┐
   compute   →  │              geotoolz                  │
                │   Operator · Sequential · Graph        │
                └────────────────┬───────────────────────┘
                                 │ GeoTensor in / GeoTensor out
                                 │
                ┌────────────────▼───────────────────────┐
   substrate →  │     georeader (carriers + index)       │
                │   GeoTensor · GeoSlice · GeoData       │
                │   GeoCatalog                           │
                └─────┬───────────────────────┬──────────┘
                      │                       │
                      ▼                       ▼
              ┌─────────────┐         ┌──────────────────┐
   readers →  │ Rasterio-   │         │ async-geotiff    │
              │ Reader      │         │ (async, external,│
              │ (sync,      │         │  COG-only)       │
              │ in georead.)│         │                  │
              └──────┬──────┘         └────────┬─────────┘
                     │                         │
                     ▼                         ▼
              ┌──────────┐         ┌──────────────────────────┐
   transport →│  GDAL /  │         │   obstore  /  fsspec     │
              │  VSI     │         │  (Rust async / Py hybrid)│
              └─────┬────┘         └────────────┬─────────────┘
                    │                           │
                    └──────────────┬────────────┘
                                   ▼
                         ┌─────────────────┐
   storage   →           │  S3/GCS/Azure   │
                         └─────────────────┘
```

Bottom-up dependency direction.
Anything above can call anything below; nothing reaches up.
The substrate box (`georeader`) carries only the *types* — `GeoTensor`, `GeoSlice`, `GeoData`, `GeoCatalog`.
The reader implementations sit one layer down.
`RasterioReader` is in `georeader`'s own source tree but functionally peers with `async-geotiff`.
(georeader also ships sensor-specific readers — `S2_SAFE_reader`, EMIT, EnMAP, etc. — those live alongside `RasterioReader` and produce `GeoTensor` the same way.)

> **Note on `RasterioReader`'s bytes path.** The diagram shows `RasterioReader` going through GDAL VSI by default.
> That's the common path, but `RasterioReader` can also delegate to `obstore` or `fsspec` via rasterio's `opener=` parameter — so it isn't strictly bound to the GDAL/VSI lane.
> See [§ "What's actually inside `RasterioReader`"](#whats-actually-inside-rasterioreader) below.

---

## Per-tool rundown

### 1. `obstore` (transport)

**What it is.** Python bindings to the Rust `object_store` crate.
A unified async API for S3, GCS, Azure Blob, and local filesystems — plus HTTP. Made by DevSeed.

**Why it matters.** It's roughly 10× faster than `fsspec` + `aiobotocore` for parallel cloud reads, because the Rust runtime handles HTTP/2, connection pooling, and range coalescing properly.
Drop-in for anywhere you'd reach for `s3fs` or `fsspec`.

**Deps.** `pyo3` runtime; nothing in Python land.

**Use cases.**

- Bulk read of millions of small Parquet shards from S3.
- Range-request-driven COG reads (the layer above this one).
- Catalog hosting — when `georeader.GeoCatalog` opens a remote `.parquet`, `obstore` is the path your bytes travel.

**Talks to.** Everything above it that does cloud I/O. `async-geotiff` builds on it directly.
`georeader.RasterioReader` historically uses GDAL's VSI for cloud, but moving its remote path onto `obstore` is on the table.

### 2. `RasterioReader` (file reading — sync, in `georeader`)

**What it is.** The default sync reader in `georeader`.
Wraps `rasterio.open` with lazy-but-windowed semantics.
Universal driver coverage (every GDAL format), GDAL/VSI for cloud paths by default, fresh-per-read for process-safety.
The right answer ~90% of the time.

**See:** [`georeader/reader_protocol.md`](georeader/reader_protocol.md) for the full design (Protocol surface, `opener=`/`fs=` knobs, three-bytes-paths triage); [Tutorial Ch. 3](../georeader_tutorial/03_rasterio_reader.md) for the current implementation.

### 3. `async-geotiff` (file reading — async, external)

**What it is.** An async GeoTIFF / COG reader.
Tiles fetched via HTTP range requests through `obstore`; TIFF IFDs parsed concurrently; no GDAL. By DevSeed.
The basis for georeader's planned `AsyncGeoTIFFReader`.

**Why it matters.** Lights up workloads where many tile reads happen simultaneously (tile servers, hyper-parallel batch inference) — `asyncio.gather(*[read_window(w) for w in 1000_windows])` avoids thread-per-request overhead and exploits HTTP/2 multiplexing.

**See:** [`georeader/reader_async_geotiff.md`](georeader/reader_async_geotiff.md) for the full design.

### 4. `georeader` (substrate)

**What it is.** The Python library that owns the geospatial substrate types and I/O orchestration.

| Component | Role |
| --- | --- |
| `GeoTensor` | `np.ndarray` subclass carrying `transform`, `crs`, `dims`, `fill_value_default`. The numpy-shaped, geo-aware, ufunc-protocol-friendly substrate. |
| `GeoSlice` | A spatiotemporal descriptor — `bounds`, `interval`, `resolution`, `crs`. Unit of work passed between samplers and loaders. |
| `GeoData` | Higher-level container / abstract reader interface. Likely the base type for things like `S2_SAFE` scenes, EMIT scenes, and other multi-product readers. |
| `RasterioReader` | Sync, rasterio-backed reader. Lazy-but-windowed: open a file once, read sub-windows on demand. The default I/O path. |
| `GeoCatalog` | Catalog of files / scenes. Wraps a GeoDataFrame with `IntervalIndex` + geometry; query / intersect / union live here. |

**Why it matters.** This is the API surface RS users hold.
Everything above (`geotoolz`, `titiler` indirectly, `lonboard` indirectly) consumes `GeoTensor`.
Everything below (`async-geotiff`, `obstore`) is plumbing that produces them.

**Deps.** Hard: `numpy`, `rasterio`, `shapely`, `geopandas`, `pyproj`.
Optional: `obstore` / `async-geotiff` if/when remote-async paths are wired in.

**Use cases.** All RS workflows.
Loading a Sentinel-2 scene, reading a bbox from a Landsat archive, building a catalog over 1M files, sampling chips for ML, saving COGs.

**Talks to.** Below: rasterio (sync), `obstore` / `async-geotiff` (async paths).
Above: `geotoolz` for transformations, `titiler` and `lonboard` for serving / viz.

### 5. `geotoolz` (computation)

**What it is.** The composable Operator library on top of `GeoTensor`.
`Operator`, `Sequential`, `Graph`, plus the curated RS modules (`indices`, `radiometry`, `cloud`, `compositing`, `pansharpen`, `sar`, `hyperspectral`, `sampling`, `inference`, `catalog_ops`, `presets`).

**Why it matters.** Without it, every RS pipeline is bespoke glue code.
With it, `Sequential([MaskClouds(...), TOAToBOA(...), NDVI(...)])` — declared in YAML or Python.

**See:** [`geotoolz/README.md`](geotoolz/README.md) for the navigation entry; [`geotoolz/geotoolz.md`](geotoolz/geotoolz.md) for the full design report (architecture, 12-module surface, end-to-end examples, `xr_toolz` coexistence).

### 6. `titiler` (serving)

**What it is.** A dynamic tile server built on FastAPI + `rio-tiler`.
Serves XYZ / WMTS / OGC tiles from COGs, STAC items, or MosaicJSON. Comes with a viewer UI and OGC-compliant endpoints.
By DevSeed.

**Why it matters.** Once you've produced a COG (NDVI, classification map, segmentation result), you want to look at it on a web map.
`titiler` does on-the-fly tiling: a request for tile `z/x/y` triggers a `rio-tiler` read of just that COG window, color-mapped, returned as PNG/JPEG. No pre-rendering, no tile cache management.

**Deps.** `fastapi`, `uvicorn`, `rio-tiler`, `morecantile`.
Doesn't depend on `georeader` or `geotoolz` directly — it consumes COGs (or STAC), which the lower stack produced.

**Use cases.**

- Production tile API serving raster outputs from `geotoolz` pipelines.
- A research server that lets collaborators inspect intermediate results without downloading.
- Backend for `lonboard` raster layers (lonboard can pull tiles from titiler URLs).

**Talks to.** Below: reads COGs from object storage (potentially via `obstore` / `async-geotiff` if you wire `rio-tiler` that way; default is GDAL/VSI).
Above: HTTP clients, `lonboard`, web maps, leafmap.

### 7. `lonboard` (visualization)

**What it is.** Geospatial visualization in Jupyter using deck.gl.
Renders huge vector data (millions of features) and raster tiles efficiently by binary-streaming GeoArrow to the browser.
Recently added raster (XYZ tile-layer) support.
Also by DevSeed.

**Why it matters.** Folium / ipyleaflet choke on 100k+ features.
`lonboard` ships GeoArrow over the kernel-frontend boundary as a typed buffer and lets deck.gl's WebGL render it; the result is hundreds of millions of points / lines / polygons interactive in a notebook.

**Deps.** `pyarrow`, `geopandas`, `anywidget`, `deck.gl-py-bindings`.
Doesn't depend on `georeader` or `geotoolz` directly — accepts geopandas / GeoArrow / image arrays.

**Use cases.**

- Inspect a vector catalog you just queried (e.g. the GeoDataFrame backing `GeoCatalog`).
- Overlay a `geotoolz`-generated raster on a basemap during analysis.
- Drop a million-point ML dataset on a map without crashing the kernel.

**Talks to.** Below: takes geopandas / arrays directly, or pulls raster tiles from a `titiler` URL.

---

## The two readers compared

The choice of reader is usually the first decision in any pipeline.

| Property | `RasterioReader` | `async-geotiff` |
| --- | --- | --- |
| **Lives in** | `georeader` | external (DevSeed) |
| **Sync / async** | sync | async |
| **Transport** | GDAL / VSI | `obstore` (Rust async) |
| **Driver support** | every GDAL driver (TIFF, JP2, NetCDF, HDF5, GRIB, ENVI, …) | TIFF / COG only |
| **Format-spec coverage** | full, including non-tiled rasters | COG-shaped (tiled) |
| **CRS / warping** | GDAL warping, full PROJ stack | minimal — bytes → numpy |
| **Open cost** | low (header + metadata) | low |
| **Read cost (small bbox)** | one VSI range request × N tiles, sequential | one async batch of tile reads, parallel |
| **Read cost (whole file)** | streaming sequential read | parallel tiles |
| **Concurrency** | needs threadpool; GDAL not fully thread-safe | native asyncio |
| **Memory footprint** | bounded by window size | bounded by tile fan-out |
| **Best for** | single scenes, non-TIFF data, CRS-heavy work, batch jobs | tile servers, parallel batch reads of many COGs, random sampling across thousands of COGs |
| **Worst for** | cloud-heavy 1000-files-at-a-time | non-TIFF rasters, GDAL-only quirks |

A practical rule:

```text
        Is the file a COG in cloud storage,
        and do I need many concurrent reads?
                       │
                       ▼
                ┌─────────────┐
                │             │
                ▼             ▼
              YES            NO
                │             │
                ▼             ▼
        async-geotiff   RasterioReader.
                        (the safe default —
                         all formats, sync)
```

Both are interchangeable behind one Protocol surface — pipelines accept a `reader_class=` argument and the rest of the code is unchanged.
See [`georeader/README.md`](georeader/README.md) for the protocol surface that makes them swappable, and [`georeader/reader_protocol.md`](georeader/reader_protocol.md) for the refactor that locks it in.

> **What about `lazy-cogs`?** The [`developmentseed/lazycogs`](https://github.com/developmentseed/lazycogs) library is a STAC-driven lazy loader that returns `xarray.DataArray`, not `GeoTensor`.
> It's a peer of `stackstac` / `odc-stac` for the dense-cube / `xarray` stack rather than a `georeader` reader option.
> See [`../geostack_notes.md`](../geostack_notes.md) Stack B for that discussion.

---

## `obstore` vs `fsspec` compared

Once you're below the reader layer, you're choosing how the bytes themselves move.
The two real options are `obstore` and `fsspec`.
They overlap in scope but differ in shape, language backbone, and ecosystem fit.

| Property | `obstore` | `fsspec` |
| --- | --- | --- |
| **Language backbone** | Rust (`object_store` crate via `pyo3`) | Pure Python with per-backend extensions |
| **API style** | Object store (`get(key)`, `get_range(key, off, len)`, `put`, `list`) | Filesystem (`open(path)`, `seek`, `read`, `cat`, `glob`) |
| **Sync / async** | Async-native; sync helpers ride on top | Sync-native; async bolt-on (`asynchronous=True`) |
| **HTTP backend** | Rust `hyper` — HTTP/2, multiplexing, range coalescing | Per-backend lib (varies; often HTTP/1.1) |
| **Backends** | S3, GCS, Azure, HTTP, local, in-memory | All of the above + FTP, SFTP, HDFS, ADLS, OCI, GitHub, Dropbox, Google Drive, … |
| **Throughput on 1k parallel ranges** | ~10× over `s3fs`+`aiobotocore` | Baseline |
| **Ecosystem integration** | New (zarr 3, `async-geotiff`, `lazycogs`, `obstore-rs` consumers) | Wide and mature: pandas, xarray, zarr ≤ 2, dask, geopandas, parquet readers, anything that wraps `fs.open()` |
| **Auth** | Native credential chains (AWS / GCS / Azure SDKs) compiled in | Per-backend; quality varies |
| **Install footprint** | One Rust binary | Tiny core; per-backend extras (`s3fs`, `gcsfs`, `adlfs`) |
| **Maturity** | New (2024+); fast-moving | Mature (2017+); stable, ubiquitous |

A practical rule:

```text
        Is the workload "read many byte ranges from
        S3/GCS/Azure as fast as possible"?
                       │
                       ▼
                ┌─────────────┐
                │             │
                ▼             ▼
              YES            NO
                │             │
                │             ├─► Niche backend (FTP, SFTP, GitHub, …)?
                │             │     └─► fsspec (only option)
                │             │
                │             └─► Need to plug into pandas/xarray/zarr/dask?
                │                   └─► fsspec (the universal adapter)
                │
                ▼
              obstore.
              (5–10× faster on parallel COG reads)
```

The two coexist comfortably.
New code paths in `async-geotiff` and `lazycogs` default to `obstore`; older code paths in `geopandas`, `pandas`, `xarray`, and `zarr ≤ 2` go through `fsspec`.
`georeader.GeoCatalog` uses `obstore` for its parquet round-trip when reading remote catalogs because that's the hot path; but it can fall back to `fsspec` for niche storage.

For our async path: `AsyncGeoTIFFReader` accepts any [`obspec.AsyncStore`](https://github.com/developmentseed/obspec) (`obstore.S3Store` / `GCSStore` / `AzureStore`, etc.) via `store=`.
We don't ship a `ByteStore` Protocol of our own — `obspec` is the upstream Protocol and `obstore` is the reference implementation.
Sync reads with niche backends still go through `RasterioReader(fs=fsspec_fs)`.
See [`types/bytestore.md`](types/bytestore.md) for the (one-page) rationale and the small `geotoolz.io.open_store(url)` factory.

---

## What's actually inside `RasterioReader`

The main diagram shows `RasterioReader` going through `GDAL / VSI`.
That's the *default*, but it isn't the only option.
`rasterio.open(...)` accepts an `opener=` callable (added in rasterio 1.4) that GDAL uses for byte-level reads, which means `RasterioReader` can route bytes through `fsspec` or `obstore` instead of GDAL's built-in HTTP client.
Three paths, all sync, all sitting under the same Python class:

```text
              ┌──────────────────────┐
              │   RasterioReader     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  rasterio.open(...)  │
              └──────────┬───────────┘
                         │
              ┌──────────┼──────────────────┐
              ▼          ▼                  ▼
       ┌──────────┐ ┌──────────────┐ ┌──────────────────┐
       │ GDAL VSI │ │ opener=fs.open│ │ opener=<custom>  │
       │ (libcurl)│ │  (fsspec)     │ │ (obstore-aware)  │
       └────┬─────┘ └──────┬────────┘ └────────┬─────────┘
            │              │                   │
            └──────────────┴───────────────────┘
                           │
                           ▼
                  S3 / GCS / Azure / …
```

| Path | Who fetches the bytes | When you'd use it |
| --- | --- | --- |
| **GDAL VSI** (default) | libcurl inside the GDAL binary — `/vsis3/`, `/vsigs/`, `/vsiaz/`, `/vsicurl/`. Pure C; no Python in the byte-fetching loop. | Anything S3/GCS/Azure/HTTPS. Just works; the fastest non-async option for general use. |
| **`opener=fs.open`** (fsspec) | Python file-like via fsspec; GDAL calls back into Python for each byte range. Slower than VSI because of the Python ↔ C trip per range. | Niche backends GDAL doesn't natively support (FTP, SFTP, GitHub, custom auth flows), or when the rest of the pipeline already holds an fsspec filesystem and re-using it simplifies the auth story. |
| **`opener=<custom obstore callback>`** | Python adapter wrapping `obstore.ObjectStore.get_range`, given to GDAL via `opener=`. | Possible in principle. In practice you'd bypass `RasterioReader` entirely and use `async-geotiff` directly — already that path, without GDAL in between. |

Two takeaways from the diagram:

- **`RasterioReader` can fetch cloud bytes without fsspec or obstore.** GDAL's vsicurl (libcurl in C) is the default and is faster than the Python-ish alternatives for general use.
  Use the `opener=` escape hatch only for niche backends or shared-auth scenarios.
- **For thousands of parallel reads or millions of small chip fetches, a different reader is the right answer** — that's where `async-geotiff` shines, skipping GDAL entirely.

For the `opener=` / `fs=` constructor knobs and the refactor that wires them in, see [`georeader/reader_protocol.md`](georeader/reader_protocol.md) §"`RasterioReader` refactor".

---

## Three concrete combined flows

### Flow A — single-scene inference, all sync, simplest

```python
# Read → process → save
reader = georeader.RasterioReader("s3://bucket/scene.tif")
gt = reader.load()                                     # GeoTensor
ndvi = geotoolz.indices.NDVI(red_idx=2, nir_idx=3)(gt) # GeoTensor
georeader.save_cog(ndvi, "s3://out/ndvi.tif")          # COG written

# Then serve it:
#   titiler --src s3://out/ndvi.tif
#
# Then look at it:
import lonboard
lonboard.Map(layers=[lonboard.BitmapTileLayer(
    data="http://localhost:8000/cog/tiles/{z}/{x}/{y}.png?url=s3://out/ndvi.tif"
)])
```

Stack used: `rasterio` → `georeader.RasterioReader` → `georeader.GeoTensor` → `geotoolz` → COG → `titiler` → `lonboard`.
No async needed, no catalog.
**`RasterioReader` is the right reader here** — single scene, GDAL-friendly, no concurrency need.

### Flow B — catalog-driven async batch processing

```python
# Build / open a catalog of 50k COGs in S3
catalog = georeader.catalog.open_catalog("s3://bucket/s2_eu.parquet")     # uses obstore

# Define a per-tile pipeline — same as Flow A but as an Operator
per_tile = geotoolz.Sequential([
    geotoolz.cloud.MaskClouds(qa_band_idx=-1, bits=[10, 11]),
    geotoolz.indices.NDVI(red_idx=2, nir_idx=3),
    geotoolz.catalog_ops.WriteCOG(path_template="s3://out/{tile_id}.tif"),
])

# Run across the catalog
geotoolz.catalog_ops.CatalogPipeline(
    catalog,
    per_tile,
    reader_class=AsyncGeoTIFFReader,                   # async-geotiff under obstore
    n_concurrent=64,
).run()
```

Stack used: `obstore` → `async-geotiff` → `georeader.GeoTensor` → `geotoolz.Sequential` → `georeader.save_cog` → S3. The async path lights up because the workload is I/O-bound across thousands of small reads.
**`RasterioReader` would also work here** — just sequentially over a threadpool — but `async-geotiff` will be 5–10× faster for cloud-COG-heavy workloads.

### Flow C — ML dataloader with async COG fan-out + chip sampler

```python
catalog = georeader.catalog.open_catalog("s3://bucket/s2_eu.parquet")
sampler = geotoolz.sampling.RandomSampler(catalog, chip_size=(256, 256), length=100_000)

# Each chip is one async COG window read — no full file fetched.
# An AsyncGeoChipDataset wraps AsyncGeoTIFFReader.read_window calls inside
# the dataloader's worker loop (sync facade or per-worker event loop).
loader = torch.utils.data.DataLoader(
    AsyncGeoChipDataset(sampler, reader_class=AsyncGeoTIFFReader),
    batch_size=32, num_workers=8,
)

for batch in loader:
    preds = model(batch)
    # ...
```

Stack used: `obstore` → `async-geotiff` → `georeader.GeoSlice` → `geotoolz.sampling` → torch. The win is that 100k random chips across 50k COGs costs only the bytes in 100k tiny range requests, not 50k full files.
**`RasterioReader` would technically work** but you'd pay GDAL's VSI overhead on every chip — `async-geotiff`'s pure-Rust HTTP path is a meaningful speedup at this fan-out.

---

## How they overlap, where the seams are

A few places where two tools could do the same job and you have to pick:

- **`RasterioReader` vs `async-geotiff`.** See the comparison table above.
  Short version: `RasterioReader` is the safe default and the only choice for non-TIFF data; `async-geotiff` wins on parallel cloud-COG throughput and on "many files, small slice each."
- **`titiler` vs direct `lonboard` raster.** `titiler` is the right choice when you want a real HTTP API for many viewers.
  `lonboard`'s direct array input is the right choice when you're in one notebook and just want to look.
  Same picture, different audiences.
- **`obstore` vs `fsspec` / `s3fs`.** `obstore` is faster and async-native; `fsspec` has wider integration (zarr, parquet readers, etc. all speak fsspec).
  In practice `obstore` is the cloud transport for the new tools, and `fsspec` is what older libraries (including parts of geopandas/rasterio) still use.
  Coexist for now; new code prefers `obstore`.
- **`georeader.RasterioReader` vs `async-geotiff`.** Sync vs async; full GDAL coverage vs TIFF-only.
  `RasterioReader` is fine for batch jobs, notebooks, and any non-TIFF source; `async-geotiff` shines when you're either serving tiles or running thousands of parallel reads.

---

## In one sentence

`obstore` moves bytes; `RasterioReader` (sync, GDAL-backed, in georeader) and `async-geotiff` (async, cloud-COG-heavy) turn those bytes into numpy slices via two complementary strategies; `georeader` wraps those slices as `GeoTensor` and indexes them as `GeoCatalog`; `geotoolz` composes operators over `GeoTensor`; `titiler` serves the resulting COGs as web tiles; `lonboard` shows them in Jupyter.
