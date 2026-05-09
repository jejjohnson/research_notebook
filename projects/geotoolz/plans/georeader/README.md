---
title: Reader reconciliation
subject: georeader design
subtitle: One metadata surface, two read interfaces, two readers
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

> **Status:** **revised 2026-05-09** — slimmed substantially. The earlier draft introduced a parallel `_ReaderMeta` / `SyncReader` / `AsyncReader` Protocol taxonomy, a custom `ByteStore` Protocol, and a from-scratch `_cog_helpers.py` async COG reader. On review, today's `GeoData` / `GeoDataBase` already covers the sync metadata + read surface; [`obspec`](https://github.com/developmentseed/obspec) already plays the role of `ByteStore`; and [`developmentseed/async-geotiff`](https://github.com/developmentseed/async-geotiff) already ships the async COG reader. So the design collapsed to: **add `AsyncGeoData` only, defer `ByteStore` to `obspec`, write `AsyncGeoTIFFReader` as a thin adapter over `async-geotiff`**.
> **Scope:** the long-term shape of the reader layer in `georeader`. Adds an `AsyncGeoData` Protocol alongside today's `GeoData` / `GeoDataBase`; adds one new reader (`AsyncGeoTIFFReader`) as a thin adapter over `async-geotiff`; documents an additive widening of `RasterioReader`'s bytes-path knobs.
> **Audience:** anyone touching `georeader/abstract_reader.py`, `georeader/rasterio_reader.py`, or building downstream pipelines that need to swap readers without rewriting call sites.

---

## Summary

Today, `georeader` ships one reader (`RasterioReader`) with a sync, GDAL-backed interface that's worked well for years. As the package's audience grows into cloud-native and async-first workloads, it needs to grow alongside — without breaking the call sites that already use it.

This design adds a single new Protocol (`AsyncGeoData`) alongside today's `GeoData` / `GeoDataBase` so async-shaped readers slot into the existing surface. One concrete async reader is added (`AsyncGeoTIFFReader`), implemented as a thin adapter over [`developmentseed/async-geotiff`](https://github.com/developmentseed/async-geotiff). Cloud byte access is delegated to [`obspec`](https://github.com/developmentseed/obspec) — the upstream Protocol that `async-geotiff` already consumes — rather than wrapped in a parallel `ByteStore` Protocol of our own. Downstream code branches only on sync-vs-async, never on which concrete reader class is in use.

The work splits into two small issues that can be reviewed independently.

---

## Motivation

Three pressures make this worth doing now:

1. **Cloud is the default substrate, not an exotic one.** New RS workflows assume reads from S3 / GCS / Azure; today's `RasterioReader` routes through GDAL VSI, which is excellent for the common case but offers no way to opt into competing transports — `obstore` (Rust core, HTTP/2, native parallel ranges) for hot-path throughput, or `fsspec` for niche backends and custom auth. The existing reader lacks the seam to plug them in.

2. **Async I/O is now first-class.** Tile servers, web maps, ML inference services, and any code that fans out reads concurrently are increasingly written async-first. `RasterioReader` is sync-only. Users wanting an async reader either roll their own or pull in an external library with a different API shape — there is no shared interface to compose against.

3. **COG-only readers can be substantially faster than full GDAL.** A pure-Rust COG reader (via `async-tiff`) can skip per-call GDAL state and PROJ initialisation, batch parallel range requests directly via `obstore`, and coalesce close-by ranges. For tile-server fan-out across thousands of small windows the overhead difference is meaningful. A reader specialised to COG (the dominant cloud-native format) deserves a place alongside the general-purpose `RasterioReader`, not as a separate ecosystem with an incompatible API. We don't have to *build* such a reader — `developmentseed/async-geotiff` exists, is actively maintained, and is the right thing to depend on. Our job is to expose it behind the same Protocol-shaped surface as `RasterioReader`.

The status quo can absorb each of these one at a time, but the shapes start to drift apart and downstream code accumulates branches. A reconciliation pass — `AsyncGeoData` Protocol + thin async-geotiff adapter — pays for itself the first time a user wants to swap GDAL VSI for `obstore` in a hot loop.

---

## Primer for newcomers

A handful of advanced concepts run through this design. Quick primers below; deeper specs in the per-issue sub-designs.

> **ELI5.** Reading a satellite image from the cloud is like ordering one slice of pizza from a giant pie that lives in another city. You don't want the whole pie shipped — just your slice. This design is about *how to ask for slices*, *who actually fetches them*, and *how to wait efficiently when you want a thousand at once*.

### What "reader" means in this package

**What it is.** A *reader* is a Python class that turns a file path or URL (local disk, S3, GCS, Azure, HTTP) into a `GeoTensor` — a numpy array with georeferencing attached. Today's package has one (`RasterioReader`); this design adds two more.

**How it works.** A reader has two phases. **Open** (cheap) reads only the file's header — enough to know the CRS, transform, shape, dtype. **Read** (expensive) actually fetches pixel bytes for a window and decodes them. The split lets you pass readers around as cheap handles and only pay I/O when you ask for data.

**What this means for us.** Code that takes a "reader" as input doesn't need the bytes — just the metadata. That's why georeader's existing Protocols split into two layers (`GeoDataBase` for metadata-only, `GeoData` for read-capable). Many georeader functions (window math, bounds queries, catalog construction) only need metadata and run instantly even on cloud-hosted files.

### Sync vs async I/O

**What it is.** *Sync* code blocks the calling thread until I/O completes (the standard Python flow). *Async* code uses `async def` / `await` so the thread can do other work while waiting. Two different control-flow models for the same fundamental operation.

**How it works.** Sync I/O is what you've used your whole life: `open(path).read()`. Async I/O uses `asyncio` (or `trio`); the runtime juggles many in-flight reads concurrently on one thread, which is dramatically more efficient for workloads where you'd otherwise spawn a thread-per-request (tile servers, 1000-window batch reads).

**What this means for us.** `RasterioReader` is sync — fine for batch jobs, scripts, notebooks. `AsyncGeoTIFFReader` is async — needed when you want to fan out 1000 reads concurrently from one process. The Protocol surface (`GeoData` / `AsyncGeoData`) isolates the difference so user code only branches on `await` vs not, never on which concrete reader class is in use.

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

**What this means for us.** A single reader class can run on different bytes paths. `RasterioReader` defaults to VSI but the optional widening in [Issue 1](reader_protocol.md) lets you swap to fsspec via `fs=` or to a custom callback via `opener=`. The new reader (`AsyncGeoTIFFReader`) skips GDAL entirely and accepts any [`obspec.AsyncStore`](https://github.com/developmentseed/obspec) (`obstore.S3Store` / `GCSStore` / `AzureStore` / etc.). Your call which trade-off matches the workload — see [`geostack.md` §"`obstore` vs `fsspec` compared"](../geostack.md#obstore-vs-fsspec-compared) for the comparison.

```{mermaid}
flowchart TD
    Need[Need to read raster bytes] --> Q1{Format?}
    Q1 -->|JP2 / NetCDF / HDF5 / GRIB| RR[RasterioReader<br/>full GDAL coverage]
    Q1 -->|TIFF / COG| Q2{Cloud-heavy fan-out?}
    Q2 -->|No, single scenes| RR
    Q2 -->|Yes, async / tile server| AG[AsyncGeoTIFFReader<br/>asyncio.gather + obstore]
```

### Python Protocols

**What it is.** A `typing.Protocol` is a "structural type" — a class declaration that says *what methods/attributes a type must have* without requiring inheritance. Like duck typing with type-checker support.

**How it works.** Define a `Protocol` with the surface you want; any class that has the right attributes satisfies it automatically (no `class MyReader(GeoData)` declaration required). With `@runtime_checkable`, `isinstance(x, Protocol)` works at runtime too.

**What this means for us.** The reader Protocols (`GeoDataBase`, `GeoData`, `AsyncGeoData`) let `RasterioReader` and `AsyncGeoTIFFReader` (and any future sensor-specific or raw-byte reader) be passed to the same function with no shared base class — they just satisfy the Protocol structurally. Same shape; independent implementations; no inheritance hierarchy.

---

## Goals

- **Reuse today's metadata surface.** Every reader (current and future) keeps using the existing `crs` / `transform` / `bounds` / `shape` / `width` / `height` / `dtype` / `fill_value_default` / `res` properties from `GeoDataBase` and `GeoData`. No parallel `_ReaderMeta` Protocol.
- **Add one new read interface.** `AsyncGeoData` mirrors `GeoData` with `async` read methods; user code typed `data: AsyncGeoData` accepts any conforming async reader.
- **Add `AsyncGeoTIFFReader`** as a thin adapter over `developmentseed/async-geotiff` — async, COG-only, no GDAL — for high-concurrency fan-out. ~80 LOC.
- **Defer cloud byte transport to `obspec`.** No custom `ByteStore` Protocol. We pass `obspec.AsyncStore` straight through to `async-geotiff`. We ship a small `geotoolz.io.open_store(url)` factory and nothing else.
- **(Optional, additive)** widen `RasterioReader` with `opener=` / `fs=` / `rio_open_kwargs=` keyword-only knobs so users can route bytes through GDAL VSI / fsspec / a custom callback explicitly. Pure addition, no breaking changes.

---

## Non-goals

- **Replacing GDAL.** `RasterioReader` stays the default. The new reader is a specialisation, not a replacement.
- **Reimplementing the COG reader.** `developmentseed/async-geotiff` already does IFD walk, tile-fetch math, decompression dispatch, range coalescing, and obspec transport. Our reader is a ~80-LOC adapter, not a peer reimplementation.
- **Reprojection / warping / resampling in the async path.** `async-geotiff` explicitly disclaims warp; we follow suit. `AsyncGeoTIFFReader.read_bounds(target_crs=...)` raises `NotImplementedError` and points users at `georeader.read.read_reproject_like` (post-step) or `RasterioReader` (WarpedVRT). See [open question §4](#4-async-warp-resample-overview-pick-deferred) for revisit.
- **Async-by-default for the existing reader.** `RasterioReader` stays sync; users wanting async use `AsyncGeoTIFFReader`.
- **Universal format support in the new reader.** `AsyncGeoTIFFReader` is TIFF/COG-only. JP2, NetCDF, HDF5, GRIB, ENVI continue to route through `RasterioReader`.
- **A sync GDAL-free GeoTensor reader for v0.1.** Speculative, no clear customer; `RasterioReader` covers sync, `AsyncGeoTIFFReader` covers GDAL-free. If a real workload emerges later we'll add a sync sibling (or a sync facade over `AsyncGeoTIFFReader`); see [open question §3](#3-a-sync-gdal-free-geotensor-reader-deferred).
- **A custom `ByteStore` Protocol.** [`obspec`](https://github.com/developmentseed/obspec) (DevSeed) already plays that role and is what `async-geotiff` consumes. We pass it through. See [`types/bytestore.md`](../types/bytestore.md) for the rationale.
- **A parallel `_ReaderMeta` / `SyncReader` taxonomy.** Today's `GeoDataBase` and `GeoData` already cover the metadata + sync-read surface. Adding a parallel layer would force every concept to have two names forever. See [Issue 1 §"Why the rewrite"](reader_protocol.md#why-the-rewrite).

---

## Constraints

- **Backward compatibility.** Existing `RasterioReader` callers — and the `GeoData` / `GeoDataBase` Protocols in `abstract_reader.py` — must keep working. The current `read_from_window(window, boundless=True)` and `load(boundless=True)` methods stay; new methods are added alongside, not in place of.
- **`GeoTensor` already morally satisfies `GeoData`.** It exposes `crs`, `transform`, `bounds`, `shape`, `dtype`, `fill_value_default`, `res`. Confirming it formally is a typing-only change; no runtime behaviour change.
- **Integer-pixel rounding behaviour can't change** silently. The `PIXEL_PRECISION = 3` tolerance in `window_utils` must be preserved across all readers.
- **The `GeoTensor` class lives in `georeader/geotensor.py` on the `feature/geotensor_npapi` branch** — see [Ch. 1 of the tutorial](../../georeader_tutorial/01_geotensor.md). The protocol definitions assume that branch is merged.

---

## High-level shape

Two readers, one shared metadata surface, two read interfaces:

| Reader | Lives in | Sync / async | Transport | Driver coverage |
|---|---|---|---|---|
| `RasterioReader` | `georeader` | sync | GDAL / VSI | every GDAL driver |
| `AsyncGeoTIFFReader` | `georeader` | async | `obstore` / `fsspec` | TIFF / COG only |

The metadata properties and the `read_window` / `read_bounds` / `read_geoslice` / `load` method names are identical across both. The only divergence is whether reads are sync or async.

```python
# Sync path — RasterioReader satisfies GeoData
def apply_to_chip(reader: GeoData, slice_: GeoSlice, op: Operator) -> GeoTensor:
    with reader as r:
        gt = r.read_geoslice(slice_)
        return op(gt)

# Async path — AsyncGeoTIFFReader satisfies AsyncGeoData
async def apply_to_chip_async(reader: AsyncGeoData, slice_: GeoSlice, op: Operator) -> GeoTensor:
    async with reader as r:
        gt = await r.read_geoslice(slice_)
        return op(gt)                                   # op itself stays sync


# In geotoolz, the pipeline picks which world it lives in:
geotoolz.catalog_ops.CatalogPipeline(
    catalog,
    op,
    reader_class=georeader.RasterioReader,         # sync default
    # reader_class=georeader.AsyncGeoTIFFReader,    # async, fan-out
)
```

Same metadata surface, same `read_*` method names, two different bytes paths underneath. The only tax on swapping is `await` — which is unavoidable as long as the cloud HTTP world is fundamentally async. For the side-by-side strategy comparison (open cost, read cost, concurrency, driver coverage), see the [stack-level overview in `geostack.md`](../geostack.md#the-two-readers-compared).

---

## Sub-designs

The work splits into two independently reviewable issues:

| # | Sub-design | Owns |
|---|---|---|
| 1 | [`reader_protocol.md`](reader_protocol.md) | `AsyncGeoData` Protocol (single new Protocol); `GeoTensor` Protocol-conformance check; tutorial chapter updates (02). Optional bundle: `RasterioReader` constructor widening with `opener=`/`fs=`/`rio_open_kwargs=` knobs + three-bytes-paths writeup in tutorial Ch. 3. |
| 2 | [`reader_async_geotiff.md`](reader_async_geotiff.md) | `AsyncGeoTIFFReader` class — thin (~80 LOC) adapter over `developmentseed/async-geotiff`; async `open(...)` classmethod; `Window`/`RasterArray` translators; passthrough of `obspec.AsyncStore` to `GeoTIFF.open`. |

Cloud byte transport is delegated to `obspec` (see [`types/bytestore.md`](../types/bytestore.md)); we ship a small `geotoolz.io.open_store(url)` factory and nothing else. There is no `ByteStore` Protocol of our own.

Each sub-design is sized to be a single PR with a focused review.

---

## Sequencing

```
Issue 1 (AsyncGeoData Protocol; optional RasterioReader widening)
   │
   ▼
types/bytestore.md (one-page obspec passthrough note + open_store helper)
   │
   ▼
Issue 2 (AsyncGeoTIFFReader thin adapter over async-geotiff)
```

- **Issue 1 lands first.** It defines `AsyncGeoData` so Issue 2 has a typed seam to satisfy.
- **`types/bytestore.md`** is documentation, not code — it picks `obspec` as the transport surface and specifies `geotoolz.io.open_store(url)` (~30 LOC). Can land alongside Issue 1.
- **Issue 2 is a single focused PR**: the ~80-LOC async adapter, two small `Window`/`RasterArray` translators, the `geotoolz.io.open_store` helper.
- **No `_cog_helpers.py`, no semaphore policy, no decompression dispatch in our code.** All of that is in `async-geotiff` (and its Rust dep `async-tiff`); we depend on it.

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

In **`developmentseed/async-geotiff`** (and its Rust dep `async-tiff`). We don't host these primitives ourselves — `AsyncGeoTIFFReader` is a thin adapter over `GeoTIFF.open` and `overview.read(window=...)`. The earlier draft of this plan proposed a private `_cog_helpers.py` module; that scope was removed when the review showed `async-geotiff` already covers IFD walk, tile-fetch math, decompression dispatch, request coalescing, and decoding off the event loop. See [Issue 2 §"Why the rewrite"](reader_async_geotiff.md#why-the-rewrite).

If a future reader needs the same primitives (sync facade, sensor-specific COG variant), the right path is to call `async-geotiff` from sync code via `asyncio.run(...)` — not to fork the helpers.

### 3. A sync GDAL-free GeoTensor reader (deferred)

Earlier drafts of this design proposed a `LazyCOGReader` — a sync, GDAL-free, COG-only `GeoTensor` reader. It was originally pitched as a wrapper around the [`developmentseed/lazycogs`](https://github.com/developmentseed/lazycogs) library, which turned out to return `xarray.DataArray` (not `GeoTensor`) and to be properly part of the `xrtoolz` / dense-cube stack — see the `geostack_notes.md` discussion for that re-routing.

The sync GDAL-free GeoTensor workload itself is plausible (notebooks, FastAPI sync handlers, batch scripts), but **doesn't yet have a clear customer** that `RasterioReader` (sync, GDAL) and `AsyncGeoTIFFReader` (async, GDAL-free) don't already cover between them. If a real workload emerges, the cheapest path is a sync facade that wraps `AsyncGeoTIFFReader` with `asyncio.run(...)` for one-call use cases; the more expensive path is a from-scratch sync IFD reader. **Decide if and when.**

### 4. Async warp / resample / overview-pick (deferred)

`async-geotiff` explicitly disclaims [warping, resampling, and automatic overview selection](https://github.com/developmentseed/async-geotiff#anti-features). Their guidance is "load with async-geotiff, then warp via `rasterio.MemoryFile` if needed". Our v1 plan adopts the same boundary: `AsyncGeoTIFFReader.read_bounds(target_crs=...)` raises `NotImplementedError` and points users at:

- **(a) Two-step pattern.** `gt = await reader.read_bounds(bounds)` (native CRS) → `gt = georeader.read.read_reproject_like(gt, target=...)` (sync warp post-step). User owns the post-step.
- **(b) Use `RasterioReader` instead.** It has WarpedVRT integration on the sync path.

This is fine for the workloads we know about (tile servers serving native-CRS overviews, fan-out batch reads in a single CRS). It will **not** fit a future tile server that needs Web-Mercator output from a UTM source without GDAL anywhere in the loop. When that customer materialises, the options are:

- (i) Inline post-warp via `rasterio.warp` in `loop.run_in_executor`. Adds GDAL back into the async dependency cone (defeats part of the point).
- (ii) `WarpedAsyncGeoTIFFReader` wrapper class that composes (i). Cleaner API, same dep cost.
- (iii) Pure-Python or pure-Rust warp (long-tail engineering; not on anyone's roadmap).

Same logic applies to overview auto-selection (`request_resolution`-style helper) and to in-CRS resampling. **Deferred for a later discussion** — flagging here so we don't accidentally bake a no-warp assumption deep into downstream code that would later be hard to lift.

---

## Alternatives considered

- **Don't unify; let `async-geotiff` stay an external library with a different shape.** Rejected: forces downstream code (`geotoolz`, ML pipelines) to special-case which library is in use, which is exactly the coordination tax the reconciliation removes. *(`lazycogs` was previously named here as a parallel external library; on closer inspection it's `xarray`-shaped, so it belongs in the `xrtoolz` discussion rather than this one — see [`geostack_notes.md`](../../geostack_notes.md).)*
- **Make the existing `RasterioReader` async-by-default with sync wrappers.** Rejected: too disruptive to existing callers, and the GDAL ecosystem isn't async-friendly underneath; the wrapper would be sync-pretending-to-be-async.
- **Use `rio-tiler` / `terracotta` as the COG reader.** Rejected: those are higher-level — they bake in tile-server assumptions and color/visualisation logic. The COG reader proposed here is a substrate, not a tile server.
- **Adopt `kerchunk` / `zarr`-shaped lazy access.** Rejected: incompatible with the rasterio-native `Window` and `Affine` API surface that the rest of `georeader` is built on. Could be added as a separate reader later.

---

## Tutorial alignment

Once these designs are implemented, the existing tutorial chapters need updates:

- [Ch. 2 — `abstract_reader`](../../georeader_tutorial/02_abstract_reader.md) — add a small section describing the new `AsyncGeoData` Protocol alongside the existing `GeoData` / `GeoDataBase` writeup.
- [Ch. 3 — `rasterio_reader`](../../georeader_tutorial/03_rasterio_reader.md) — describe the `opener=` / `fs=` constructor knobs and the three-bytes-paths triage.
- A new chapter can be added for `AsyncGeoTIFFReader` once it lands — a natural successor to Ch. 3.

The tutorial today describes the **current** package state; updates land alongside each issue's implementation, not before.

---

## Open questions, gotchas, and warnings

The reconciliation is mostly low-risk — pieces exist, the work is plumbing. A few things to manage actively:

- **`feature/geotensor_npapi` merge timing is critical-path** for `geotoolz`. The ndarray-subclass `GeoTensor` with `__array_ufunc__` underpins `geotoolz`'s two-tier model. If the branch stalls upstream, downstream blocks. Track the upstream merge as a v0.1 release blocker; contingency is to vendor `GeoTensor` in `geotoolz` until upstream catches up.
- **`__array_function__` (NEP-18) coverage.** `__array_ufunc__` covers ufuncs only; `np.fft.*`, `np.linalg.*`, `np.einsum`, `np.percentile` go through `__array_function__`. Verify `GeoTensor` implements both protocols, or document which numpy submodules strip the subclass. Add a CI test that round-trips metadata through every numpy submodule the readers and downstream operators touch.
- **ndarray subclass survival across third-party libraries.** `GeoTensor` survives numpy + scipy + skimage + matplotlib. It does **not** survive PyTorch (`torch.from_numpy` strips), JAX (`jnp.asarray` strips), or Dask without explicit `meta=` plumbing. Document this boundary in the user docs so consumers don't assume the subclass flows everywhere.
- **Async ↔ sync boundary.** `AsyncGeoData` returns awaitables; downstream sync code (Operators, batch loops) needs `asyncio.run()` per call, which costs an event loop per invocation. `geotoolz` will need to pick a strategy (`AsyncOperator` family, or restrict async to the `CatalogPipeline` boundary) — see [`geotoolz.md` §11.2](../geotoolz/geotoolz.md#112-implementation-gotchas-test-these-in-ci). Worth coordinating before v0.1.
- **Sensor-reader scope reduction for v0.1.** [`readers/`](../readers/) lists ABI, SEVIRI Native, MTG-FCI, Himawari-AHI HSD, SEVIRI HRIT, MODIS, VIIRS. Each "hard" sensor (irregular file formats, bowtie distortion) is 1–2 weeks *with full product spec access*. **Recommendation:** ship MODIS + ABI as v0.1 sensor proofs; defer SEVIRI / MTG / Himawari to v0.5+ unless an active user has a concrete need.
- **Credential per-reader isolation.** The `Credential` design (`apply()` returning a dict, no global env-var mutation) only works if every reader is updated to consume the per-call dict instead of reading from `os.environ`. Backwards-compat path: legacy `apply_to_os_environ()` is provided but discouraged. Audit each reader on the way through reconciliation.
- **`async-geotiff` API stability.** `async-geotiff` is at v0.1+ and pre-1.0; the API may shift between minor releases. Pin a minor range in `pyproject.toml` (`async-geotiff>=0.1,<0.2`) and bump deliberately. Document the bump policy in `geotoolz`'s release notes when we cut v0.1.
- **Per-sensor public bucket helpers** are user-friendly but couple `georeader` to specific cloud-bucket layouts that providers can change without notice (Sentinel-2 on AWS moved twice). Pin reader behaviour to a documented bucket convention; add a smoke test that fails loudly if a bucket layout changes.
