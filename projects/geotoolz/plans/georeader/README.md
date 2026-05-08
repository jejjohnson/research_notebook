---
title: Reader reconciliation
subject: georeader design
subtitle: One metadata surface, two read interfaces, three readers
short_title: Reader recon
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, georeader, reader, protocol
---

> **Status:** design proposal — split into three sub-designs (Issues 1–3 below).
> **Scope:** the long-term shape of the reader layer in `georeader`. Locks down a single metadata surface and two read interfaces; refactors the existing reader to conform; adds two new readers built on different transports.
> **Audience:** anyone touching `georeader/abstract_reader.py`, `georeader/rasterio_reader.py`, or building downstream pipelines that need to swap readers without rewriting call sites.

---

## Summary

Today, `georeader` ships one reader (`RasterioReader`) with a sync, GDAL-backed interface that's worked well for years. As the package's audience grows into cloud-native and async-first workloads, it needs to grow alongside — without breaking the call sites that already use it.

This design proposes **one shared metadata surface and two read interfaces** (sync and async) that all current and future readers honour. The existing `RasterioReader` is refactored to conform; two new readers (`LazyCOGReader`, `AsyncGeoTIFFReader`) are added; a `ByteStore` abstraction unifies cloud byte access across `obstore` and `fsspec`. Downstream code branches only on sync-vs-async, never on which concrete reader class is in use.

The work splits into three issues that can be reviewed independently.

---

## Motivation

Three pressures make this worth doing now:

1. **Cloud is the default substrate, not an exotic one.** New RS workflows assume reads from S3 / GCS / Azure; today's `RasterioReader` routes through GDAL VSI, which is excellent for the common case but offers no way to opt into competing transports — `obstore` (Rust core, HTTP/2, native parallel ranges) for hot-path throughput, or `fsspec` for niche backends and custom auth. The existing reader lacks the seam to plug them in.

2. **Async I/O is now first-class.** Tile servers, web maps, ML inference services, and any code that fans out reads concurrently are increasingly written async-first. `RasterioReader` is sync-only. Users wanting an async reader either roll their own or pull in an external library with a different API shape — there is no shared interface to compose against.

3. **COG-only readers can be substantially faster than full GDAL.** A pure-Python COG reader can skip per-call GDAL state and PROJ initialisation, batch parallel range requests directly via `obstore`, and coalesce close-by ranges. For tile-server fan-out across thousands of small windows the overhead difference is meaningful. A reader specialised to COG (the dominant cloud-native format) deserves a place alongside the general-purpose `RasterioReader`, not as a separate ecosystem with an incompatible API.

The status quo can absorb each of these one at a time, but the shapes start to drift apart and downstream code accumulates branches. A reconciliation pass — one metadata surface, two read interfaces, three readers — pays for itself the first time a user wants to swap GDAL VSI for obstore in a hot loop.

---

## Primer for newcomers

A handful of advanced concepts run through this design. Quick primers below; deeper specs in the per-issue sub-designs.

> **ELI5.** Reading a satellite image from the cloud is like ordering one slice of pizza from a giant pie that lives in another city. You don't want the whole pie shipped — just your slice. This design is about *how to ask for slices*, *who actually fetches them*, and *how to wait efficiently when you want a thousand at once*.

### What "reader" means in this package

**What it is.** A *reader* is a Python class that turns a file path or URL (local disk, S3, GCS, Azure, HTTP) into a `GeoTensor` — a numpy array with georeferencing attached. Today's package has one (`RasterioReader`); this design adds two more.

**How it works.** A reader has two phases. **Open** (cheap) reads only the file's header — enough to know the CRS, transform, shape, dtype. **Read** (expensive) actually fetches pixel bytes for a window and decodes them. The split lets you pass readers around as cheap handles and only pay I/O when you ask for data.

**What this means for us.** Code that takes a "reader" as input doesn't need the bytes — just the metadata. That's why this design defines two Protocols (`_ReaderMeta` for metadata-only, `SyncReader` for read-capable). Many georeader functions (window math, bounds queries, catalog construction) only need metadata and run instantly even on cloud-hosted files.

### Sync vs async I/O

**What it is.** *Sync* code blocks the calling thread until I/O completes (the standard Python flow). *Async* code uses `async def` / `await` so the thread can do other work while waiting. Two different control-flow models for the same fundamental operation.

**How it works.** Sync I/O is what you've used your whole life: `open(path).read()`. Async I/O uses `asyncio` (or `trio`); the runtime juggles many in-flight reads concurrently on one thread, which is dramatically more efficient for workloads where you'd otherwise spawn a thread-per-request (tile servers, 1000-window batch reads).

**What this means for us.** `RasterioReader` is sync — fine for batch jobs, scripts, notebooks. `AsyncGeoTIFFReader` is async — needed when you want to fan out 1000 reads concurrently from one process. The Protocol surface (`SyncReader` / `AsyncReader`) isolates the difference so user code only branches on `await` vs not, never on which concrete reader class is in use.

```{mermaid}
gantt
    title Three reads — sync (sequential) vs async (parallel)
    dateFormat X
    axisFormat %s
    section Sync
    read 1 :s1, 0, 3
    read 2 :s2, after s1, 3
    read 3 :s3, after s2, 3
    section Async
    read 1 :a1, 0, 3
    read 2 :a2, 0, 3
    read 3 :a3, 0, 3
```

### The "bytes path"

**What it is.** When a reader fetches data from cloud storage (S3, GCS, Azure), *something* has to translate "give me bytes 0–4096 of `s3://bucket/scene.tif`" into actual HTTP traffic. The library that does this is the **bytes path**.

**How it works.** Three options ship today: **GDAL VSI** (libcurl in C, default for `RasterioReader`), **obstore** (Rust core, fast for parallel ranges), and **fsspec** (Python, broadest backend coverage). They differ in throughput, async support, and which clouds they speak.

**What this means for us.** A single reader class can run on different bytes paths. `RasterioReader` defaults to VSI but the refactor in [Issue 1](reader_protocol.md) lets you swap to fsspec via `fs=` or to obstore via `opener=`. The new readers (`LazyCOGReader`, `AsyncGeoTIFFReader`) skip GDAL entirely and use obstore directly. Your call which trade-off matches the workload — see [`geostack.md` §"`obstore` vs `fsspec` compared"](../geostack.md#obstore-vs-fsspec-compared) for the comparison.

```{mermaid}
flowchart TD
    Need[Need to read raster bytes] --> Q1{Format?}
    Q1 -->|JP2 / NetCDF / HDF5 / GRIB| RR[RasterioReader<br/>full GDAL coverage]
    Q1 -->|TIFF / COG| Q2{Cloud-heavy fan-out?}
    Q2 -->|No, single scenes| RR
    Q2 -->|Yes, sync code| LC[LazyCOGReader<br/>obstore lazy reads]
    Q2 -->|Yes, async / tile server| AG[AsyncGeoTIFFReader<br/>asyncio.gather]
```

### Python Protocols

**What it is.** A `typing.Protocol` is a "structural type" — a class declaration that says *what methods/attributes a type must have* without requiring inheritance. Like duck typing with type-checker support.

**How it works.** Define a `Protocol` with the surface you want; any class that has the right attributes satisfies it automatically (no `class MyReader(SyncReader)` needed). With `@runtime_checkable`, `isinstance(x, Protocol)` works at runtime too.

**What this means for us.** The reader Protocols (`_ReaderMeta`, `SyncReader`, `AsyncReader`) let `RasterioReader`, `LazyCOGReader`, and `AsyncGeoTIFFReader` all be passed to the same function with no shared base class — they just satisfy the Protocol structurally. Same shape; three implementations; no inheritance hierarchy.

---

## Goals

- **Define a single metadata surface.** Every reader (current and future) exposes the same `crs` / `transform` / `bounds` / `shape` / `count` / `width` / `height` / `dtype` / `nodata` / `res` properties via a `_ReaderMeta` Protocol.
- **Define two read interfaces.** `SyncReader` and `AsyncReader` both build on `_ReaderMeta`; the only divergence is whether read methods return a `GeoTensor` or a `Coroutine[GeoTensor]`.
- **Refactor `RasterioReader` and `GeoData`** to conform to the new Protocols without breaking existing callers.
- **Add `LazyCOGReader`** — sync, COG-only, no GDAL — for fast cloud reads with reduced per-call overhead.
- **Add `AsyncGeoTIFFReader`** — async, COG-only, no GDAL — for high-concurrency fan-out.
- **Share a `ByteStore` abstraction** so `LazyCOGReader` and `AsyncGeoTIFFReader` stay agnostic between `obstore` and `fsspec`.

---

## Non-goals

- **Replacing GDAL.** `RasterioReader` stays the default. The new readers are specialisations, not replacements.
- **Reimplementing reprojection in pure Python.** `LazyCOGReader.read_bounds(target_crs=...)` falls back to scipy/skimage warping; it doesn't replicate GDAL's CRS-fix long tail.
- **Async-by-default for the existing reader.** `RasterioReader` stays sync; users wanting async use `AsyncGeoTIFFReader`.
- **Universal format support in the new readers.** `LazyCOGReader` and `AsyncGeoTIFFReader` are TIFF/COG-only. JP2, NetCDF, HDF5, GRIB, ENVI continue to route through `RasterioReader`.
- **Replacing `obstore` or `fsspec`.** The `ByteStore` Protocol is a thin compatibility shim, not a new transport library.

---

## Constraints

- **Backward compatibility.** Existing `RasterioReader` callers — and the `GeoData` / `GeoDataBase` Protocols in `abstract_reader.py` — must keep working. The current `read_from_window(window, boundless=True)` and `load(boundless=True)` methods stay; new methods are added alongside, not in place of.
- **`GeoTensor` already morally satisfies `_ReaderMeta`.** It exposes `crs`, `transform`, `bounds`, `shape`, `dtype`, `nodata` (as `fill_value_default`), `res`. Aligning it formally is a typing-only change; no runtime behaviour change.
- **Integer-pixel rounding behaviour can't change** silently. The `PIXEL_PRECISION = 3` tolerance in `window_utils` must be preserved across all readers.
- **The `GeoTensor` class lives in `georeader/geotensor.py` on the `feature/geotensor_npapi` branch** — see [Ch. 1 of the tutorial](../../georeader_tutorial/01_geotensor.md). The protocol definitions assume that branch is merged.

---

## High-level shape

Three readers, one shared metadata surface, two read interfaces:

| Reader | Lives in | Sync / async | Transport | Driver coverage |
|---|---|---|---|---|
| `RasterioReader` | `georeader` | sync | GDAL / VSI | every GDAL driver |
| `LazyCOGReader` | `georeader` | sync API, lazy semantics | `obstore` / `fsspec` | TIFF / COG only |
| `AsyncGeoTIFFReader` | `georeader` | async | `obstore` / `fsspec` | TIFF / COG only |

The metadata properties and the `read_window` / `read_bounds` / `read_geoslice` / `load` method names are identical across all three. The only divergence is whether reads are sync or async.

```python
# Sync path — RasterioReader or LazyCOGReader
def apply_to_chip(reader: SyncReader, slice_: GeoSlice, op: Operator) -> GeoTensor:
    with reader as r:
        gt = r.read_geoslice(slice_)
        return op(gt)

# Async path — AsyncGeoTIFFReader
async def apply_to_chip_async(reader: AsyncReader, slice_: GeoSlice, op: Operator) -> GeoTensor:
    async with reader as r:
        gt = await r.read_geoslice(slice_)
        return op(gt)                                   # op itself stays sync


# In geotoolz, the pipeline picks which world it lives in:
geotoolz.catalog_ops.CatalogPipeline(
    catalog,
    op,
    reader_class=georeader.RasterioReader,         # sync default
    # reader_class=georeader.LazyCOGReader,         # sync, lazy-on-open
    # reader_class=georeader.AsyncGeoTIFFReader,    # async, fan-out
)
```

Same metadata surface, same `read_*` method names, three different bytes paths underneath. The only tax on swapping is `await` — which is unavoidable as long as the cloud HTTP world is fundamentally async. For the side-by-side strategy comparison (open cost, read cost, concurrency, driver coverage), see the [stack-level overview in `geostack.md`](../geostack.md#the-three-readers-compared).

---

## Sub-designs

The work splits into three independently reviewable issues:

| # | Sub-design | Owns |
|---|---|---|
| 1 | [`reader_protocol.md`](reader_protocol.md) | `_ReaderMeta` + `SyncReader` Protocols; `RasterioReader` refactor with `opener=`/`fs=` knobs and the three-bytes-paths triage; `GeoData` / `GeoDataBase` alignment; `GeoTensor` Protocol conformance; tutorial chapter updates (02, 03). |
| 2 | [`reader_lazy_cog.md`](reader_lazy_cog.md) | `LazyCOGReader` class; IFD parsing; tile-fetch math; compression dispatch. |
| 3 | [`reader_async_geotiff.md`](reader_async_geotiff.md) | `AsyncReader` Protocol; `AsyncGeoTIFFReader` class; async `open(...)` classmethod; `asyncio.gather`-based parallel tile fetch; `max_concurrent_tiles` semaphore. |

The transport layer (`ByteStore` Protocol + `ObstoreByteStore` / `FsspecByteStore` adapters + `open_store(url)` factory) is specified separately in [`types/bytestore.md`](../types/bytestore.md) since it's consumed by both Issue 2 and Issue 3 and conceivably by future raw-byte-shaped readers.

Each sub-design is sized to be a single PR with a focused review.

---

## Sequencing

```
Issue 1 (protocols + RasterioReader refactor)
   │
   ▼
types/bytestore.md (ByteStore Protocol + adapters)
   │
   ▼
Issue 2 (LazyCOGReader)  ←──┬── independent (both consume ByteStore)
                             │
Issue 3 (AsyncGeoTIFFReader) ┘── shares COG-parsing helpers with Issue 2
```

- **Issue 1 lands first.** It locks the Protocol surface that 2 and 3 implement.
- **`types/bytestore.md` lands next** (or alongside Issue 1) — defines the transport seam both COG readers need.
- **Issues 2 and 3 can proceed in parallel** after the above. Issue 3 has a soft dependency on Issue 2 for the shared COG-parsing helpers (see open question #2 below).
- **No issue is blocking on the others' user-visible API.** The Protocol surface is locked in Issue 1; the transport surface is locked in `types/bytestore.md`; Issues 2 and 3 are pure implementations.

---

## Open questions

These are unresolved and should be discussed before Issue 1 starts.

### 1. `RasterioReader` file-handle caching

The current `RasterioReader` opens the file fresh on every `read()` call — see [Ch. 3 §1 of the tutorial](../../georeader_tutorial/03_rasterio_reader.md). That behaviour is **deliberate**: it makes the reader pickleable for `multiprocessing` / `joblib` / Dask workers, because a cached `rasterio.DatasetReader` cannot cross a process boundary.

The proposal in this design implies caching the open handle for the lifetime of the reader (with explicit `__enter__` / `__exit__` and `close()`). That's a behaviour change and the trade-off is real:

- **Cache the handle:** repeated reads in one process are faster (no per-call open cost). Pickling for multi-process work breaks; users would need to re-open in worker.
- **Open fresh per read (status quo):** pickleable across processes for free; pays a small per-call open cost.
- **Configurable:** add a `cache_handle: bool = False` kwarg. More API surface, but lets each call site pick.

**Decision needed before Issue 1.**

### 2. Where COG IFD parsing + tile math + decompression lives

Both `LazyCOGReader` and `AsyncGeoTIFFReader` need the same logic (IFD walk, `_tiles_for_window`, decompression dispatch). Three options:

- **Duplicate** — small (~150 LOC) but invites sync/async drift.
- **Shared internal `_cog_helpers.py` module** — one source of truth, both readers import.
- **`LazyCOGReader` exports the helpers** — `AsyncGeoTIFFReader` imports from it. Couples the two readers more tightly than necessary.

**Tentative pick: shared `_cog_helpers.py`** — cleaner than duplication, doesn't couple the two readers.

---

## Alternatives considered

- **Don't unify; let `lazy-cogs` and `async-geotiff` stay external libraries with different shapes.** Rejected: forces downstream code (`geotoolz`, ML pipelines) to special-case which library is in use, which is exactly the coordination tax the reconciliation removes.
- **Make the existing `RasterioReader` async-by-default with sync wrappers.** Rejected: too disruptive to existing callers, and the GDAL ecosystem isn't async-friendly underneath; the wrapper would be sync-pretending-to-be-async.
- **Use `rio-tiler` / `terracotta` as the COG reader.** Rejected: those are higher-level — they bake in tile-server assumptions and color/visualisation logic. The COG reader proposed here is a substrate, not a tile server.
- **Adopt `kerchunk` / `zarr`-shaped lazy access.** Rejected: incompatible with the rasterio-native `Window` and `Affine` API surface that the rest of `georeader` is built on. Could be added as a separate reader later.

---

## Tutorial alignment

Once these designs are implemented, the existing tutorial chapters need updates:

- [Ch. 2 — `abstract_reader`](../../georeader_tutorial/02_abstract_reader.md) — describe `_ReaderMeta` / `SyncReader` / `AsyncReader` Protocols alongside (or replacing) the current `GeoData` / `GeoDataBase` description.
- [Ch. 3 — `rasterio_reader`](../../georeader_tutorial/03_rasterio_reader.md) — describe the `opener=` / `fs=` constructor knobs and the three-bytes-paths triage.
- New chapters can be added for `LazyCOGReader` and `AsyncGeoTIFFReader` once those land — both natural successors to Ch. 3.

The tutorial today describes the **current** package state; updates land alongside each issue's implementation, not before.
