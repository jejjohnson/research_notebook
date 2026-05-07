# RS / cloud-native geospatial stack — tools, layers, and how they fit together

> **Scope:** a rundown of `obstore`, `RasterioReader`, `async-geotiff`, `lazy-cogs`, `georeader`, `geotoolz`, `titiler`, and `lonboard` — what each is for, how they depend on one another, and which combinations match common workflows. Assumes `geotoolz` is built per its design report (composable Operator library on `GeoTensor`).
>
> **Status:** reference document, not a design proposal.

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
                └─────┬─────────────┬───────────────┬────┘
                      │             │               │
                      ▼             ▼               ▼
              ┌─────────────┐ ┌──────────┐  ┌──────────────┐
   readers →  │ Rasterio-   │ │ async-   │  │  lazy-cogs   │
              │ Reader      │ │ geotiff  │  │              │
              │ (sync,      │ │ (async,  │  │ (lazy,       │
              │ in georead.)│ │ external)│  │  external)   │
              └──────┬──────┘ └────┬─────┘  └──────┬───────┘
                     │             │               │
                     ▼             └───────┬───────┘
              ┌──────────┐                 │
   transport →│  GDAL /  │                 ▼
              │  VSI     │      ┌──────────────────────────┐
              └─────┬────┘      │   obstore  /  fsspec     │
                    │           │  (Rust async / Py hybrid)│
                    │           └────────────┬─────────────┘
                    │                        │
                    └──────────┬─────────────┘
                               ▼
                     ┌─────────────────┐
   storage   →       │  S3/GCS/Azure   │
                     └─────────────────┘
```

Bottom-up dependency direction. Anything above can call anything below; nothing reaches up. The substrate box (`georeader`) carries only the *types* — `GeoTensor`, `GeoSlice`, `GeoData`, `GeoCatalog`. The reader implementations sit one layer down. `RasterioReader` is in `georeader`'s own source tree but functionally peers with `async-geotiff` and `lazy-cogs`. (georeader also ships sensor-specific readers — `S2_SAFE_reader`, EMIT, EnMAP, etc. — those live alongside `RasterioReader` and produce `GeoTensor` the same way.)

> **Note on `RasterioReader`'s bytes path.** The diagram shows `RasterioReader` going through GDAL VSI by default. That's the common path, but `RasterioReader` can also delegate to `obstore` or `fsspec` via rasterio's `opener=` parameter — so it isn't strictly bound to the GDAL/VSI lane. See [§ "What's actually inside `RasterioReader`"](#whats-actually-inside-rasterioreader) below.

---

## Per-tool rundown

### 1. `obstore` (transport)

**What it is.** Python bindings to the Rust `object_store` crate. A unified async API for S3, GCS, Azure Blob, and local filesystems — plus HTTP. Made by DevSeed.

**Why it matters.** It's roughly 10× faster than `fsspec` + `aiobotocore` for parallel cloud reads, because the Rust runtime handles HTTP/2, connection pooling, and range coalescing properly. Drop-in for anywhere you'd reach for `s3fs` or `fsspec`.

**Deps.** `pyo3` runtime; nothing in Python land.

**Use cases.**

- Bulk read of millions of small Parquet shards from S3.
- Range-request-driven COG reads (the layer above this one).
- Catalog hosting — when `georeader.GeoCatalog` opens a remote `.parquet`, `obstore` is the path your bytes travel.

**Talks to.** Everything above it that does cloud I/O. `async-geotiff` and `lazy-cogs` both build on it directly. `georeader.RasterioReader` historically uses GDAL's VSI for cloud, but moving its remote path onto `obstore` is on the table.

### 2. `RasterioReader` (file reading — sync, in `georeader`)

**What it is.** The default sync reader in `georeader`. Wraps `rasterio.open` with lazy-but-windowed semantics: opening the file is cheap (header + metadata only); calling `.load()` or reading a window via `read.read_from_bounds(...)` triggers the actual pixel fetch. Lives in `georeader/rasterio_reader.py`.

**Why it matters.** This is where 90% of users land by default, because (a) GDAL is the universal raster reader (handles every driver, every CRS quirk, every weird sensor product), (b) it works with both local files and cloud URLs out of the box via GDAL's VSI virtual file system (`/vsis3/`, `/vsigs/`, `/vsicurl/`), and (c) it's what every existing tutorial, notebook, and downstream tool expects.

**Key surface.**

- `RasterioReader(filepath, indexes=None, ...)` — opens lazily.
- `.load()` — read all configured bands into a `GeoTensor`.
- `.read_from_window(window)` — read a sub-window.
- Composed with `georeader.read.read_from_bounds(reader, bounds, ...)` for the bbox-driven path.
- Opens the file fresh on every `read()` call (process-safe; pickleable for parallel workers).

**Deps.** `rasterio`, `GDAL`, `numpy`, `shapely`, `pyproj`. No async runtime.

**Use cases.**

- Single-scene workflows (RGB visualisation, NDVI, save COG).
- Anything with non-COG drivers: `RasterioReader` reads ENVI HDR, NetCDF subdatasets, JPEG2000 (Sentinel-2 SAFE), HDF5 (EnMAP, EMIT), GRIB. async-geotiff and lazy-cogs are TIFF-only.
- Local-disk batch jobs where async I/O wouldn't help.
- Workflows where rasterio's CRS handling, masked-array support, or per-driver creation options matter (warping, COG profile generation, etc.).

**Talks to.** Below: `rasterio` + `GDAL`, which by default uses GDAL's own libcurl-based VSI handlers (`/vsis3/`, `/vsigs/`, `/vsiaz/`, `/vsicurl/`) for cloud paths — **no Python in the byte-fetching loop**. Optionally delegates to `fsspec` or `obstore` via rasterio's `opener=` parameter when the user passes a custom file opener (see [§ "What's actually inside `RasterioReader`"](#whats-actually-inside-rasterioreader)). Above: produces `GeoTensor`s consumed by `geotoolz`, `titiler` (via `rio-tiler` which is also rasterio-based), and any user code holding a `georeader` object.

### 3. `async-geotiff` (file reading — async, external)

**What it is.** An async GeoTIFF / COG reader. Reads tiles via HTTP range requests, parses TIFF IFDs concurrently, returns numpy arrays. By DevSeed.

**Why it matters.** GDAL is sync and process-bound. For workloads that need to fan out 1000 simultaneous tile reads (e.g. tile servers, hyper-parallel batch inference), an async reader avoids the thread-per-request overhead and exploits HTTP/2 multiplexing through `obstore`.

**Deps.** `obstore` for transport. No GDAL.

**Use cases.**

- Tile-server backends fetching arbitrary COG tiles fast.
- Batch processing where the bottleneck is "how many tiles per second can I pull from S3."
- ML dataloaders that want async over thread-pooled rasterio.

**Talks to.** `obstore` below. `georeader` above — `georeader.RasterioReader` is the sync reader, but a sibling `AsyncGeoTIFFReader` (or just direct `async-geotiff` calls) lands data into `GeoTensor` for the rest of the stack.

### 4. `lazy-cogs` (file reading — lazy, external)

**What it is.** Lazy COG access — open a COG without reading data; tiles are fetched only when sliced. Conceptually similar to what `rioxarray.open_rasterio(chunks=...)` does with dask, but lighter.

**Why it matters.** For workflows that touch many COGs but only read a small spatial subset each (catalog-driven processing, sampler-driven ML), eager reads are wasteful. Lazy access defers I/O until a slice is requested, then makes one HTTP range request for exactly the tiles the slice covers.

**Deps.** Typically `obstore` (or `fsspec`) for transport; pure-Python TIFF parsing for the header.

**Use cases.**

- "I have 50k COGs in S3, extract a 256×256 chip from each at coords X." Without laziness this is impossible; with it, it's seconds.
- Stacking timeseries where you fetch only the bbox you care about across all timesteps.

**Talks to.** Same shape as `async-geotiff`. The two overlap conceptually — you can think of `async-geotiff` as the async-first reader and `lazy-cogs` as the lazy-first abstraction; in practice they often fold into each other in user code.

### 5. `georeader` (substrate)

**What it is.** The Python library that owns the geospatial substrate types and I/O orchestration.

| Component | Role |
| --- | --- |
| `GeoTensor` | `np.ndarray` subclass carrying `transform`, `crs`, `dims`, `fill_value_default`. The numpy-shaped, geo-aware, ufunc-protocol-friendly substrate. |
| `GeoSlice` | A spatiotemporal descriptor — `bounds`, `interval`, `resolution`, `crs`. Unit of work passed between samplers and loaders. |
| `GeoData` | Higher-level container / abstract reader interface. Likely the base type for things like `S2_SAFE` scenes, EMIT scenes, and other multi-product readers. |
| `RasterioReader` | Sync, rasterio-backed reader. Lazy-but-windowed: open a file once, read sub-windows on demand. The default I/O path. |
| `GeoCatalog` | Catalog of files / scenes. Wraps a GeoDataFrame with `IntervalIndex` + geometry; query / intersect / union live here. |

**Why it matters.** This is the API surface RS users hold. Everything above (`geotoolz`, `titiler` indirectly, `lonboard` indirectly) consumes `GeoTensor`. Everything below (`async-geotiff`, `lazy-cogs`, `obstore`) is plumbing that produces them.

**Deps.** Hard: `numpy`, `rasterio`, `shapely`, `geopandas`, `pyproj`. Optional: `obstore` / `async-geotiff` if/when remote-async paths are wired in.

**Use cases.** All RS workflows. Loading a Sentinel-2 scene, reading a bbox from a Landsat archive, building a catalog over 1M files, sampling chips for ML, saving COGs.

**Talks to.** Below: rasterio (sync), `obstore` / `async-geotiff` / `lazy-cogs` (async / lazy paths). Above: `geotoolz` for transformations, `titiler` and `lonboard` for serving / viz.

### 6. `geotoolz` (computation)

**What it is.** The composable Operator library on top of `GeoTensor`. `Operator`, `Sequential`, `Graph`, plus the curated RS modules (`indices`, `radiometry`, `cloud`, `compositing`, `pansharpen`, `sar`, `hyperspectral`, `sampling`, `inference`, `catalog_ops`, `presets`).

**Why it matters.** Without it, every RS pipeline is bespoke glue code. With it, `Sequential([MaskClouds(...), TOAToBOA(...), NDVI(...)])` — declared in YAML or Python.

**Deps.** Hard: `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `georeader`. Optional: `torch` / `jax` (via `ModelOp`), `hydra-zen`, `xrpatcher`.

**Use cases.** Single-tile pipelines, tiled inference (`ApplyToChips`), catalog-driven processing (`CatalogPipeline`), Hydra-config workflows, sensor-specific presets.

**Talks to.** Below: consumes `GeoTensor` from `georeader`; reaches into `georeader.catalog` for the `CatalogPipeline` operator. Above: produces `GeoTensor` outputs that get saved as COGs (which `titiler` / `lonboard` then read).

### 7. `titiler` (serving)

**What it is.** A dynamic tile server built on FastAPI + `rio-tiler`. Serves XYZ / WMTS / OGC tiles from COGs, STAC items, or MosaicJSON. Comes with a viewer UI and OGC-compliant endpoints. By DevSeed.

**Why it matters.** Once you've produced a COG (NDVI, classification map, segmentation result), you want to look at it on a web map. `titiler` does on-the-fly tiling: a request for tile `z/x/y` triggers a `rio-tiler` read of just that COG window, color-mapped, returned as PNG/JPEG. No pre-rendering, no tile cache management.

**Deps.** `fastapi`, `uvicorn`, `rio-tiler`, `morecantile`. Doesn't depend on `georeader` or `geotoolz` directly — it consumes COGs (or STAC), which the lower stack produced.

**Use cases.**

- Production tile API serving raster outputs from `geotoolz` pipelines.
- A research server that lets collaborators inspect intermediate results without downloading.
- Backend for `lonboard` raster layers (lonboard can pull tiles from titiler URLs).

**Talks to.** Below: reads COGs from object storage (potentially via `obstore` / `async-geotiff` if you wire `rio-tiler` that way; default is GDAL/VSI). Above: HTTP clients, `lonboard`, web maps, leafmap.

### 8. `lonboard` (visualization)

**What it is.** Geospatial visualization in Jupyter using deck.gl. Renders huge vector data (millions of features) and raster tiles efficiently by binary-streaming GeoArrow to the browser. Recently added raster (XYZ tile-layer) support. Also by DevSeed.

**Why it matters.** Folium / ipyleaflet choke on 100k+ features. `lonboard` ships GeoArrow over the kernel-frontend boundary as a typed buffer and lets deck.gl's WebGL render it; the result is hundreds of millions of points / lines / polygons interactive in a notebook.

**Deps.** `pyarrow`, `geopandas`, `anywidget`, `deck.gl-py-bindings`. Doesn't depend on `georeader` or `geotoolz` directly — accepts geopandas / GeoArrow / image arrays.

**Use cases.**

- Inspect a vector catalog you just queried (e.g. the GeoDataFrame backing `GeoCatalog`).
- Overlay a `geotoolz`-generated raster on a basemap during analysis.
- Drop a million-point ML dataset on a map without crashing the kernel.

**Talks to.** Below: takes geopandas / arrays directly, or pulls raster tiles from a `titiler` URL.

---

## The three readers compared

The choice of reader is usually the first decision in any pipeline.

| Property | `RasterioReader` | `async-geotiff` | `lazy-cogs` |
| --- | --- | --- | --- |
| **Lives in** | `georeader` | external (DevSeed) | external |
| **Sync / async** | sync | async | sync API, lazy semantics |
| **Transport** | GDAL / VSI | `obstore` (Rust async) | `obstore` or `fsspec` |
| **Driver support** | every GDAL driver (TIFF, JP2, NetCDF, HDF5, GRIB, ENVI, …) | TIFF / COG only | TIFF / COG only |
| **Format-spec coverage** | full, including non-tiled rasters | COG-shaped (tiled) | COG-shaped (tiled) |
| **CRS / warping** | GDAL warping, full PROJ stack | minimal — bytes → numpy | minimal |
| **Open cost** | low (header + metadata) | low | very low (header only) |
| **Read cost (small bbox)** | one VSI range request × N tiles, sequential | one async batch of tile reads, parallel | one range request per slice access |
| **Read cost (whole file)** | streaming sequential read | parallel tiles | wasteful — was designed for slicing |
| **Concurrency** | needs threadpool; GDAL not fully thread-safe | native asyncio | sync, easy to wrap in threadpool |
| **Memory footprint** | bounded by window size | bounded by tile fan-out | bounded by what's been sliced |
| **Best for** | single scenes, non-TIFF data, CRS-heavy work, batch jobs | tile servers, parallel batch reads of many COGs | random sampling across thousands of COGs |
| **Worst for** | cloud-heavy 1000-files-at-a-time | non-TIFF rasters, GDAL-only quirks | reading the whole file |

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
   Will I touch         RasterioReader.
   many files,          (the safe default)
   each with a
   tiny bbox?
        │
        ▼
     ┌────┐    ┌────┐
     YES        NO
      │          │
      ▼          ▼
  lazy-cogs  async-geotiff
```

In `geotoolz`, this maps to a `reader_class=` argument on `ApplyToChips` / `CatalogPipeline` — the pipeline definition stays the same regardless of which reader actually fetches bytes:

```python
# Default — sync rasterio, fine for local or small jobs
geotoolz.catalog_ops.CatalogPipeline(catalog, op)

# Many cloud COGs in parallel
geotoolz.catalog_ops.CatalogPipeline(
    catalog, op,
    reader_class=georeader.AsyncGeoTIFFReader,
    n_concurrent=64,
)

# Random chips across many cloud COGs
geotoolz.sampling.RandomSampler(
    catalog, chip_size=(256, 256), length=100_000,
    reader_class=georeader.LazyCOGReader,
)
```

The reader is a strategy; the rest of the pipeline doesn't care.

---

## `obstore` vs `fsspec` compared

Once you're below the reader layer, you're choosing how the bytes themselves move. The two real options are `obstore` and `fsspec`. They overlap in scope but differ in shape, language backbone, and ecosystem fit.

| Property | `obstore` | `fsspec` |
| --- | --- | --- |
| **Language backbone** | Rust (`object_store` crate via `pyo3`) | Pure Python with per-backend extensions |
| **API style** | Object store (`get(key)`, `get_range(key, off, len)`, `put`, `list`) | Filesystem (`open(path)`, `seek`, `read`, `cat`, `glob`) |
| **Sync / async** | Async-native; sync helpers ride on top | Sync-native; async bolt-on (`asynchronous=True`) |
| **HTTP backend** | Rust `hyper` — HTTP/2, multiplexing, range coalescing | Per-backend lib (varies; often HTTP/1.1) |
| **Backends** | S3, GCS, Azure, HTTP, local, in-memory | All of the above + FTP, SFTP, HDFS, ADLS, OCI, GitHub, Dropbox, Google Drive, … |
| **Throughput on 1k parallel ranges** | ~10× over `s3fs`+`aiobotocore` | Baseline |
| **Ecosystem integration** | New (zarr 3, `async-geotiff`, `lazy-cogs`, `obstore-rs` consumers) | Wide and mature: pandas, xarray, zarr ≤ 2, dask, geopandas, parquet readers, anything that wraps `fs.open()` |
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

The two coexist comfortably. New code paths in `async-geotiff` and `lazy-cogs` default to `obstore`; older code paths in `geopandas`, `pandas`, `xarray`, and `zarr ≤ 2` go through `fsspec`. `georeader.GeoCatalog` uses `obstore` for its parquet round-trip when reading remote catalogs because that's the hot path; but it can fall back to `fsspec` for niche storage.

In `geotoolz` / `georeader`, this maps to a `store=` argument that any reader accepts:

```python
# Default — auto-pick obstore for s3://, gs://, az://, http(s)://; fsspec otherwise
reader = georeader.LazyCOGReader("s3://bucket/scene.tif")

# Explicit obstore
from obstore.store import S3Store
reader = georeader.LazyCOGReader(
    "scene.tif",
    store=ObstoreByteStore(S3Store("bucket")),
)

# Explicit fsspec — when you need a niche backend
import fsspec
fs = fsspec.filesystem("github", org="foo", repo="bar")
reader = georeader.LazyCOGReader(
    "path/to/scene.tif",
    store=FsspecByteStore(fs),
)
```

The reader doesn't care which transport actually fetches the bytes — both satisfy the same `ByteStore` Protocol (see the API reconciliation file).

---

## What's actually inside `RasterioReader`

The main diagram shows `RasterioReader` going through `GDAL / VSI`. That's the *default*, but it isn't the only option. `rasterio.open(...)` accepts an `opener=` callable (added in rasterio 1.4) that GDAL uses for byte-level reads, which means `RasterioReader` can route bytes through `fsspec` or `obstore` instead of GDAL's built-in HTTP client. Three paths, all sync, all sitting under the same Python class:

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
| **`opener=<custom obstore callback>`** | Python adapter wrapping `obstore.ObjectStore.get_range`, given to GDAL via `opener=`. | Possible in principle. In practice you'd bypass `RasterioReader` entirely and use `lazy-cogs` or `async-geotiff` directly — they're already that path, without GDAL in between. |

### What this really means

- **`RasterioReader` *can* fetch cloud bytes without fsspec or obstore.** Its default path is GDAL's native HTTP client (libcurl in C). That's been true since well before fsspec or obstore existed.
- **GDAL VSI does real range requests.** When `RasterioReader.read_window(...)` is called on a cloud COG, GDAL issues `GET .../scene.tif Range: bytes=N-M` for exactly the tiles needed. This is why a single-file workflow on a 1 GB COG only downloads the few KB you actually read.
- **The `opener=` escape hatch is for niche cases.** If GDAL's vsicurl works, use it — it's faster than fsspec-via-opener. If you need an unusual backend (FTP, GitHub, custom HTTP auth), `opener=fs.open` lets you reuse fsspec's coverage without leaving rasterio.
- **For "thousands of parallel reads" or "millions of small chip fetches", you want a different reader, not a fancy opener.** That's where `async-geotiff` and `lazy-cogs` exist — they skip GDAL entirely.

### How `georeader` exposes this

`RasterioReader` accepts the opener as a kwarg, typically through `rio_open_kwargs` or a top-level convenience like `fs=`:

```python
# Default — GDAL VSI handles s3://
reader = georeader.RasterioReader("s3://bucket/scene.tif")

# Force fsspec routing (e.g. for an S3-compatible endpoint with custom auth)
import fsspec
fs = fsspec.filesystem(
    "s3",
    endpoint_url="https://my-minio:9000",
    key=...,
    secret=...,
)
reader = georeader.RasterioReader(
    "s3://bucket/scene.tif",
    rio_open_kwargs={"opener": fs.open},
)

# Or via a georeader-level shortcut, if surfaced:
reader = georeader.RasterioReader("s3://bucket/scene.tif", fs=fs)
```

The reader's surface is identical regardless — same `.load()`, `.read_from_window(...)`, same `GeoTensor` output. Only the bytes path underneath changes.

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

Stack used: `rasterio` → `georeader.RasterioReader` → `georeader.GeoTensor` → `geotoolz` → COG → `titiler` → `lonboard`. No async needed, no catalog. **`RasterioReader` is the right reader here** — single scene, GDAL-friendly, no concurrency need.

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

Stack used: `obstore` → `async-geotiff` → `georeader.GeoTensor` → `geotoolz.Sequential` → `georeader.save_cog` → S3. The async path lights up because the workload is I/O-bound across thousands of small reads. **`RasterioReader` would also work here** — just sequentially over a threadpool — but `async-geotiff` will be 5–10× faster for cloud-COG-heavy workloads.

### Flow C — ML dataloader with lazy COGs + chip sampler

```python
catalog = georeader.catalog.open_catalog("s3://bucket/s2_eu.parquet")
sampler = geotoolz.sampling.RandomSampler(catalog, chip_size=(256, 256), length=100_000)

# Each chip is one lazy COG window read — no full file fetched
loader = torch.utils.data.DataLoader(
    GeoChipDataset(sampler, reader_class=LazyCOGReader),
    batch_size=32, num_workers=8,
)

for batch in loader:
    preds = model(batch)
    # ...
```

Stack used: `obstore` → `lazy-cogs` → `georeader.GeoSlice` → `geotoolz.sampling` → torch. The win is that 100k random chips across 50k COGs costs only the bytes in 100k tiny range requests, not 50k full files. **`RasterioReader` would technically work** but you'd pay GDAL's VSI overhead on every chip — `lazy-cogs` is a meaningful speedup at this fan-out.

---

## How they overlap, where the seams are

A few places where two tools could do the same job and you have to pick:

- **`RasterioReader` vs `async-geotiff` vs `lazy-cogs`.** See the comparison table above. Short version: `RasterioReader` is the safe default and the only choice for non-TIFF data; `async-geotiff` wins on parallel cloud-COG throughput; `lazy-cogs` wins on "many files, small slice each."
- **`async-geotiff` vs `lazy-cogs`.** Both read COGs without GDAL. If you need *concurrent* reads, `async-geotiff`. If you need *deferred* reads with a numpy-like slicing surface, `lazy-cogs`. They co-exist; `geotoolz`'s `ApplyToChips` could call either through a `reader_class` argument.
- **`titiler` vs direct `lonboard` raster.** `titiler` is the right choice when you want a real HTTP API for many viewers. `lonboard`'s direct array input is the right choice when you're in one notebook and just want to look. Same picture, different audiences.
- **`obstore` vs `fsspec` / `s3fs`.** `obstore` is faster and async-native; `fsspec` has wider integration (zarr, parquet readers, etc. all speak fsspec). In practice `obstore` is the cloud transport for the new tools, and `fsspec` is what older libraries (including parts of geopandas/rasterio) still use. Coexist for now; new code prefers `obstore`.
- **`georeader.RasterioReader` vs `async-geotiff`.** Sync vs async; full GDAL coverage vs TIFF-only. `RasterioReader` is fine for batch jobs, notebooks, and any non-TIFF source; `async-geotiff` shines when you're either serving tiles or running thousands of parallel reads.

---

## In one sentence

`obstore` moves bytes; `RasterioReader` (sync, GDAL-backed, in georeader), `async-geotiff` (async, cloud-COG-heavy), and `lazy-cogs` (lazy, slice-on-access) turn those bytes into numpy slices via three different strategies; `georeader` wraps those slices as `GeoTensor` and indexes them as `GeoCatalog`; `geotoolz` composes operators over `GeoTensor`; `titiler` serves the resulting COGs as web tiles; `lonboard` shows them in Jupyter.
