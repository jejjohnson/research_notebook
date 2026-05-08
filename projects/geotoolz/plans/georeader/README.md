# Reader reconciliation

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

### Strategy axis

| Reader | Open cost | Read cost (small bbox) | Concurrent reads | Driver coverage |
|---|---|---|---|---|
| `RasterioReader` | header + IFD via GDAL | one VSI call | sync (threadpool) | every GDAL driver |
| `LazyCOGReader` | one or two range requests | one tile-batch fetch | sync (threadpool) | TIFF/COG only |
| `AsyncGeoTIFFReader` | one or two async range requests | parallel tile fetch | native asyncio | TIFF/COG only |

Same metadata surface, same `read_*` method names, three different bytes paths underneath. The only tax on swapping is `await` — which is unavoidable as long as the cloud HTTP world is fundamentally async.

---

## Sub-designs

The work splits into three independently reviewable issues:

| # | Sub-design | Owns |
|---|---|---|
| 1 | [`reader_protocol.md`](reader_protocol.md) | `_ReaderMeta` + `SyncReader` Protocols; `RasterioReader` refactor with `opener=`/`fs=` knobs and the three-bytes-paths triage; `GeoData` / `GeoDataBase` alignment; `GeoTensor` Protocol conformance; tutorial chapter updates (02, 03). |
| 2 | [`reader_lazy_cog.md`](reader_lazy_cog.md) | `LazyCOGReader` class; IFD parsing; tile-fetch math; compression dispatch; `ByteStore` Protocol with `ObstoreByteStore` and `FsspecByteStore` adapters; `open_store(url)` factory. |
| 3 | [`reader_async_geotiff.md`](reader_async_geotiff.md) | `AsyncReader` Protocol; `AsyncGeoTIFFReader` class; async `open(...)` classmethod; `asyncio.gather`-based parallel tile fetch; `max_concurrent_tiles` semaphore. |

Each sub-design is sized to be a single PR with a focused review.

---

## Sequencing

```
Issue 1 (protocols + RasterioReader refactor)
   │
   ▼
Issue 2 (LazyCOGReader + ByteStore)  ←──┬── independent
                                         │
Issue 3 (AsyncGeoTIFFReader)  ───────────┘── reuses ByteStore from Issue 2
```

- **Issue 1 lands first.** It locks the Protocol surface that 2 and 3 implement.
- **Issues 2 and 3 can proceed in parallel** after Issue 1 merges. Issue 3 has a soft dependency on Issue 2 (it imports `ByteStore` and likely shares COG-parsing helpers); the order can be flipped if Issue 3 stubs the ByteStore import temporarily.
- **No issue is blocking on the others' user-visible API.** The Protocol surface is locked in Issue 1; Issues 2 and 3 are pure implementations of that surface.

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

### 2. Where `ByteStore` lives

`ByteStore` is needed first by `LazyCOGReader` (Issue 2) and reused by `AsyncGeoTIFFReader` (Issue 3). Three options:

- **Issue 2** — lands with its first consumer. Issue 3 imports.
- **Issue 1** — land all shared infrastructure (Protocols + ByteStore) up front. Issues 2 and 3 are then pure-implementation issues.
- **Separate Issue 0 / 1.5** — a small dedicated PR for ByteStore alone.

**Tentative pick: Issue 2** (matches "lands where first needed"), but Issue 1 is a defensible alternative.

### 3. Where COG IFD parsing + tile math + decompression lives

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
