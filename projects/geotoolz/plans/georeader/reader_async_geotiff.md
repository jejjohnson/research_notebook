---
title: AsyncGeoTIFFReader
subject: georeader design
subtitle: Thin adapter over `developmentseed/async-geotiff`
short_title: AsyncGeoTIFF
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, georeader, async, cog, async-geotiff
---

> **Parent:** [README.md](README.md) **Depends on:** [Issue 1](reader_protocol.md) — adds an `AsyncGeoData` Protocol that mirrors today's `GeoData` with `async` read methods.
> **Status:** **revised 2026-05-09** — collapsed from a "build a COG reader from scratch" design (~400 LOC, private `_cog_helpers.py`, semaphore policy) into a thin adapter (~80 LOC) over [`developmentseed/async-geotiff`](https://github.com/developmentseed/async-geotiff).
> See §"Why the rewrite" below.
> **Scope:** an async, COG-only reader for high-concurrency fan-out workloads — tile servers, web maps, async ML inference services.
> Wraps `async_geotiff.GeoTIFF`; translates async-geotiff's `Window` / `RasterArray` into our `Window` / `GeoTensor`.

---

## Why this issue exists

Sync I/O is fine when you have one read at a time.
For workloads where many reads happen concurrently (a tile server fielding 1000 simultaneous tile requests, an async ML service that fans out across hundreds of windows from one process), async-native fetching is the right shape — `asyncio.gather(*[reader.read_window(w) for w in windows])` becomes a one-line concurrency primitive.

Today, `georeader` has no async story.
Users wanting async reads either roll their own or pull in an external library with a different API. This issue adds `AsyncGeoTIFFReader` — same metadata surface as `RasterioReader`, async read methods.

We do **not** rebuild any of the COG plumbing.
[`async-geotiff`](https://github.com/developmentseed/async-geotiff) (DevSeed, ~50★, last updated 2026-05-09) already ships:

- `await GeoTIFF.open(path, store=...)` — async constructor, fetches and parses the IFD chain.
- `geotiff.transform`, `geotiff.crs`, `geotiff.overviews[i]`, `overview.read(window=...)`.
- IFD walk, GeoKey parsing, GDAL metadata.
- Per-tile range fetch, decompression, assembly into a `(bands, height, width)` ndarray.
- **Request coalescing for adjacent tiles** (in the Rust `async-tiff` core).
- **CPU-bound decoding off the event loop** (Rust thread pool).
- Decompressors: Deflate, JPEG, JPEG2000, LERC, LERC_DEFLATE, LERC_ZSTD, LZMA, LZW, WebP, ZSTD.
- Cloud transport: accepts any [`obspec.AsyncStore`](https://github.com/developmentseed/obspec) — including `obstore.S3Store`, `GCSStore`, `AzureStore`, `HTTPStore`, `LocalStore`.

Our reader is therefore a **thin adapter**: instantiate `GeoTIFF.open`, expose its metadata behind our `GeoDataBase`-shaped property surface, translate windows on the way in and `RasterArray → GeoTensor` on the way out.
Roughly 80 LOC.

---

## Why the rewrite

The earlier draft of this doc proposed:

- A private `_cog_helpers.py` module with COG header parsing, `_tiles_for_window` math, decompression dispatch (~150–200 LOC).
- An `asyncio.Semaphore`-based `max_concurrent_tiles` policy.
- A custom `ByteStore` Protocol consumed via `await self._store.get_range_async(...)` inside `asyncio.gather(...)`.
- ~400 LOC total.

On review (2026-05-09), every one of those concerns is already solved upstream:

| Concern | Where it lives in async-geotiff |
|---|---|
| IFD walk, GeoKey parsing | `async_geotiff/_geotiff.py` + `_crs.py` + `_gdal_metadata.py` |
| Window → tile coords + tile fetch math | `async_geotiff/_read.py` |
| Per-tile range fetch + decompression | `async_geotiff/_fetch.py` (delegates to Rust `async-tiff`) |
| Tile assembly into ndarray | `async_geotiff/_read.py::assemble_tiles()` |
| Overviews | `geotiff.overviews[]` |
| Request coalescing | Rust `async-tiff` core, automatic |
| Concurrency / fan-out control | Rust thread pool + obspec backend's connection pool |
| Decompressor coverage | Deflate, JPEG, JPEG2000, LERC*, LZMA, LZW, WebP, ZSTD |

A pure-Python reimplementation would be slower, less correct (decompression dispatch is fiddly), and would drift from upstream.
The right call is to depend on `async-geotiff` and write the smallest possible adapter on top of it.

The custom `ByteStore` Protocol also evaporated in the same rewrite — see [`types/bytestore.md`](../types/bytestore.md).
We pass `obspec.AsyncStore` straight through.

---

## Primer for newcomers

> **ELI5.** Sync code is like **waiting in line at one cashier** — you can't do anything else until your turn finishes.
> Async code is like **leaving your order at 25 different counters** and collecting the results as they're ready.
> Same hardware; far more food per unit time when each order is mostly waiting.

### `async` / `await` basics

**What it is.** Python's `async def` defines a *coroutine* — a function that can pause itself with `await` and let other coroutines run on the same thread until the awaited operation completes.
It's not threading; it's cooperative multitasking inside one event loop.

**How it works.** When you call `result = await some_async_function()`, the runtime suspends the current coroutine until `some_async_function()` finishes, then resumes with the result.
While suspended, the event loop runs other ready coroutines.
Async only helps when the work is I/O-bound — waiting on network, disk, etc. — because that's when there's idle time to fill.

**What this means for us.** Cloud raster reads are dominated by network round-trips.
With `async`, one process can have hundreds of `read_window(...)` calls in flight concurrently, all sharing one OS thread.
The CPython GIL doesn't get in the way because nobody's computing — they're all waiting on HTTP. Same hardware; far more throughput; far simpler than thread pools.

### `asyncio.gather` and parallel awaits

**What it is.** `asyncio.gather(coro1, coro2, coro3, ...)` runs multiple coroutines concurrently and returns when all finish (or one raises).
It's the canonical way to issue many parallel I/O operations in one call.

**How it works.** Each argument is a coroutine that hasn't started yet.
`gather(...)` schedules them all on the event loop, lets them run interleaved, and collects their results into a list in input order.

**What this means for us.** A *single* `read_window(...)` is just one `await` on async-geotiff (the parallelism inside it is in the Rust core).
The interesting `gather(...)` is at the *outer* layer — `await asyncio.gather(*[reader.read_window(w) for w in 1000_windows])` for tile-server-shaped workloads.

```{mermaid}
gantt
    title 25 tile reads — sync vs async
    dateFormat X
    axisFormat %Lms
    section Sync
    tile 1 :s1, 0, 50
    tile 2 :s2, after s1, 50
    tile 3 :s3, after s2, 50
    ... :s4, after s3, 1100
    tile 25 :s25, after s4, 50
    section Async (asyncio.gather)
    tile 1 :a1, 0, 50
    tile 2 :a2, 0, 50
    tile 3 :a3, 0, 50
    ... :a4, 0, 50
    tile 25 :a25, 0, 50
```

### Async classmethod for construction

**What it is.** Python's `__init__` *can't* be an `async def`.
There's no `__ainit__` magic method.
So when initialisation requires async work (fetching the COG IFD over HTTP), the convention is a classmethod `open(...)` that's `async`.

**How it works.** `__init__` does cheap synchronous setup (store the URL, the credential, allocate state).
The user calls `await Cls.open(url)`, which constructs an instance via `__init__` and then runs the async setup before returning the fully-initialised reader.
Same pattern used by `aiohttp.ClientSession`, `asyncpg.connect`, and `async_geotiff.GeoTIFF` itself.

**What this means for us.** Users write `reader = await AsyncGeoTIFFReader.open("s3://bucket/scene.tif")`.
The reader's `crs` / `transform` / etc. don't work until `open()` has been awaited — accessing them earlier raises `RuntimeError("Reader not opened")`.
Slight cost in clarity; major win in not faking sync APIs over async work.

```{mermaid}
sequenceDiagram
    participant App
    participant Cls as AsyncGeoTIFFReader
    participant AGT as async_geotiff.GeoTIFF
    participant Store as obspec.AsyncStore

    App->>Cls: await open(url, store=...)
    Cls->>Cls: __init__ (cheap)
    Cls->>AGT: await GeoTIFF.open(url, store=store)
    AGT->>Store: range request — IFD bytes
    Store-->>AGT: bytes
    AGT-->>Cls: GeoTIFF instance
    Cls-->>App: ready AsyncGeoTIFFReader
    App->>Cls: reader.crs ✓
    App->>Cls: await reader.read_window(w)
    Cls->>AGT: await overview.read(window=...)
    AGT->>Store: parallel range requests (Rust core)
    Store-->>AGT: bytes
    AGT-->>Cls: RasterArray
    Cls-->>App: GeoTensor
```

---

## Deliverables

1. **`AsyncGeoData` Protocol** in `georeader/abstract_reader.py` — mirrors today's `GeoData` with async read methods.
   Defined in [Issue 1](reader_protocol.md).
2. **`AsyncGeoTIFFReader` class** in `georeader/async_geotiff_reader.py` (~80 LOC).
3. **`async open(...)` classmethod** — wraps `await async_geotiff.GeoTIFF.open(path, store=...)`.
4. **Async read methods** — `read_window`, `read_bounds`, `read_geoslice`, `load`.
   Each is one or two lines of adapter code over `async-geotiff`.
5. **Two small translators** — `Window ↔ async_geotiff.Window` and `RasterArray → GeoTensor`.
6. **Optional `geotoolz.io.open_store(url)`** — see [`types/bytestore.md`](../types/bytestore.md).
   Builds an `obstore` backend from the URL scheme; if the user passes their own `obspec.AsyncStore` we use it as-is.
7. **Async context-manager** — `__aenter__` / `__aexit__`.
   `aclose` is a no-op (obstore pools its own connections; async-geotiff has no resource to release).
8. **Tests** — open + read a real COG asynchronously; concurrent fan-out across N windows; numerical equivalence vs `RasterioReader.read_window` for the same file/window within rounding.

This issue does **not** ship: a custom byte-store Protocol, a `_cog_helpers.py` module, a semaphore-based concurrency cap, decompression dispatch, IFD parsing, or tile-fetch math.
All of that is in `async-geotiff`.

---

## `AsyncGeoTIFFReader` class — full sketch

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from async_geotiff import GeoTIFF, RasterArray
from async_geotiff import Window as AGTWindow
from rasterio.windows import Window

from georeader.abstract_reader import AsyncGeoData
from georeader.geotensor import GeoTensor
from geotoolz.io import open_store

if TYPE_CHECKING:
    import obspec
    import pyproj
    from rasterio import Affine


class AsyncGeoTIFFReader(AsyncGeoData):
    """Async COG reader. Thin adapter over async_geotiff.GeoTIFF.

    Use for high-concurrency fan-out (tile servers, async ML inference
    services). For one-off sync reads, use RasterioReader instead.

    Open is async because the IFD fetch is async. After open, every
    metadata property is sync; every read method is a coroutine that
    delegates to async-geotiff (which uses a Rust thread pool for
    decoding and obspec for transport).
    """

    def __init__(
        self,
        path_or_url: str,
        *,
        store: "obspec.AsyncStore | None" = None,
        overview_level: int = 0,
    ) -> None:
        """Cheap. Does NOT fetch the header — call .open() first."""
        self.path_or_url = path_or_url
        self._store = store or open_store(path_or_url)
        self._overview_level = overview_level
        self._geotiff: GeoTIFF | None = None

    @classmethod
    async def open(
        cls,
        path_or_url: str,
        *,
        store: "obspec.AsyncStore | None" = None,
        overview_level: int = 0,
    ) -> "AsyncGeoTIFFReader":
        """Async constructor. Most users call this rather than __init__."""
        self = cls(path_or_url, store=store, overview_level=overview_level)
        self._geotiff = await GeoTIFF.open(path_or_url, store=self._store)
        return self

    # ------------------------------------------------------------------ metadata
    def _require_open(self) -> GeoTIFF:
        if self._geotiff is None:
            raise RuntimeError(
                "Reader not opened — call `await AsyncGeoTIFFReader.open(...)`",
            )
        return self._geotiff

    @property
    def _overview(self):
        return self._require_open().overviews[self._overview_level]

    @property
    def crs(self) -> "pyproj.CRS":
        return self._require_open().crs

    @property
    def transform(self) -> "Affine":
        return self._overview.transform

    @property
    def shape(self) -> tuple[int, int, int]:
        ovr = self._overview
        # async-geotiff doesn't expose band count separately on the overview;
        # take it from the primary IFD samples-per-pixel.
        count = self._require_open().ifd.samples_per_pixel
        return (count, ovr.height, ovr.width)

    @property
    def dtype(self) -> np.dtype:
        # one cheap probe — async-geotiff exposes this on its IFD.
        return np.dtype(self._require_open().ifd.dtype)

    @property
    def fill_value_default(self):
        return self._require_open().nodata

    # ... bounds / res / footprint inherited from AsyncGeoData defaults

    # --------------------------------------------------------------------- reads
    async def read_window(self, window: Window) -> GeoTensor:
        arr: RasterArray = await self._overview.read(
            window=_to_agt_window(window),
        )
        return _rasterarray_to_geotensor(arr, fill_value=self.fill_value_default)

    async def read_bounds(
        self,
        bounds: tuple[float, float, float, float],
        *,
        target_resolution: tuple[float, float] | None = None,
        target_crs: "pyproj.CRS | str | None" = None,
    ) -> GeoTensor:
        if target_crs is not None or target_resolution is not None:
            raise NotImplementedError(
                "AsyncGeoTIFFReader.read_bounds does not warp or resample. "
                "Read in the native CRS, then call georeader.read.read_reproject_like "
                "(or wrap with RasterioReader for WarpedVRT-based on-the-fly warping). "
                "See plans/georeader/reader_async_geotiff.md open question §1.",
            )
        from georeader import window_utils
        win = window_utils.window_from_bounds(self, bounds)
        return await self.read_window(win)

    async def read_geoslice(self, slice_) -> GeoTensor:
        return await self.read_bounds(slice_.bounds)

    async def load(self) -> GeoTensor:
        ovr = self._overview
        return await self.read_window(
            Window(col_off=0, row_off=0, width=ovr.width, height=ovr.height),
        )

    # -------------------------------------------------------------- lifecycle
    async def aclose(self) -> None:
        # obstore pools its own connections; async-geotiff has no
        # explicit resource to release. No-op.
        pass

    async def __aenter__(self) -> "AsyncGeoTIFFReader":
        if self._geotiff is None:
            self._geotiff = await GeoTIFF.open(self.path_or_url, store=self._store)
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()


# ---- adapters ------------------------------------------------------------

def _to_agt_window(w: Window) -> AGTWindow:
    return AGTWindow(
        col_off=int(w.col_off),
        row_off=int(w.row_off),
        width=int(w.width),
        height=int(w.height),
    )


def _rasterarray_to_geotensor(
    arr: RasterArray,
    *,
    fill_value=None,
) -> GeoTensor:
    """Map async-geotiff's RasterArray onto our GeoTensor.

    RasterArray.data is (bands, height, width). If a mask is present,
    apply it: substitute fill_value where mask is False.
    """
    data = arr.data
    if arr.mask is not None and fill_value is not None:
        # mask=True means valid; broadcast across the band axis.
        invalid = np.broadcast_to(~arr.mask, data.shape)
        data = np.where(invalid, fill_value, data)
    return GeoTensor(
        values=data,
        transform=arr.transform,
        crs=arr._geotiff.crs,
        fill_value_default=fill_value,
    )
```

That's the whole reader.
Roughly 80 LOC counting blank lines and helpers.
The interesting work is two adapters; the rest is property passthrough.

---

## What we are *not* doing (anti-goals)

These are explicitly out of scope — pulling them in would either reinvent async-geotiff functionality or rebuild the GDAL warping layer in a context where rasterio already exists.

- **Reprojection / warping in `read_bounds(target_crs=...)`.** async-geotiff has the same non-goal. Users wanting cross-CRS reads either (a) read native + post-process via `georeader.read.read_reproject_like`, or (b) use `RasterioReader` (which has WarpedVRT integration).
  The async-reader path raises on `target_crs!=None` for v1; revisit per [open question §1](#1-async-warp-revisit-later).
- **Resampling in `read_bounds(target_resolution=...)`.** Same reasoning. async-geotiff's overview ladder is what we expose; downsampling between overview levels happens in a sync post-step via existing `georeader` functions.
- **Auto overview selection.** Caller picks `overview_level` (default 0 = full-res).
  `geotoolz` can layer "pick the smallest overview ≥ requested resolution" on top later — this is a one-screen helper, not reader plumbing.
- **A custom byte-store Protocol.** See [`types/bytestore.md`](../types/bytestore.md).
- **A semaphore for `max_concurrent_tiles`.** async-tiff's Rust core handles intra-call coalescing; obstore's connection pool bounds outer concurrency.
  Adding our own semaphore competes with both.
  If a benchmark shows us getting hammered, revisit.
- **Decompression dispatch.** async-tiff already covers Deflate, JPEG, JPEG2000, LERC, LERC_DEFLATE, LERC_ZSTD, LZMA, LZW, WebP, ZSTD. If a sensor we care about uses something exotic, that's a per-issue workaround, not a reason to fork.

---

## Module layout

| Component | Source |
|---|---|
| `obspec.AsyncStore` Protocol | external — [`developmentseed/obspec`](https://github.com/developmentseed/obspec) |
| `obstore.S3Store` / `GCSStore` / `AzureStore` / ... | external — [`developmentseed/obstore`](https://github.com/developmentseed/obstore) |
| `GeoTIFF.open`, `Overview.read`, `RasterArray` | external — [`developmentseed/async-geotiff`](https://github.com/developmentseed/async-geotiff) |
| `open_store(url)` factory | `geotoolz/io.py` (~30 LOC) |
| `AsyncGeoData` Protocol | `georeader/abstract_reader.py` (added in [Issue 1](reader_protocol.md), ~30 LOC) |
| `AsyncGeoTIFFReader` adapter | `georeader/async_geotiff_reader.py` (~80 LOC) |
| `_to_agt_window` / `_rasterarray_to_geotensor` | colocated with the reader (small private helpers) |

The original code in this issue is the ~80-LOC adapter and its two helpers.
Everything else is dependencies.

---

## Acceptance criteria

- `AsyncGeoTIFFReader` instances satisfy `AsyncGeoData` per static type-check.
- `await AsyncGeoTIFFReader.open("s3://...")` returns a fully initialised reader; metadata properties work after open and raise `RuntimeError` before.
- `await reader.read_window(window)` returns a `GeoTensor` numerically matching `RasterioReader.read_window(window)` for the same file and window (within rounding).
- Concurrent fan-out: `await asyncio.gather(*[reader.read_window(w) for w in windows])` for 100 windows from one reader instance completes without errors and faster than the sync equivalent.
- `read_bounds(target_crs=...)` / `read_bounds(target_resolution=...)` raise `NotImplementedError` with a message pointing at `read_reproject_like` or `RasterioReader`.
- `async with await AsyncGeoTIFFReader.open(...)` context-manager works.
- Reader passes a user-supplied `obspec.AsyncStore` straight through to `async_geotiff.GeoTIFF.open(...)` — no wrapping, no Protocol of our own.

---

## Issue-specific open questions

In addition to the [parent design's open questions](README.md#open-questions):

### 1. Async warp — revisit later

[`async-geotiff` explicitly says no warp / resample / overview-pick.](https://github.com/developmentseed/async-geotiff#anti-features) Their guidance is "load with async-geotiff, then warp via `rasterio.MemoryFile`".
For v1 we raise on `target_crs!=None`.
If a real customer materialises (e.g. an async tile server that needs Web-Mercator output from a UTM source), the options are:

- (a) post-process inside `read_bounds`: fetch native, then warp via `rasterio.warp` in a thread (`loop.run_in_executor`).
  Adds GDAL back into the async dependency cone, which partly defeats the point.
- (b) document the two-step pattern (`reader.read_bounds(...)` → `georeader.read.read_reproject_like(...)`) and let the user own the post-step.
- (c) build a `WarpedAsyncGeoTIFFReader` wrapper that composes (a).
  Keeps the surface clean but ships the GDAL dep regardless.

Tentative pick: stay at "raise" for v1, document (b) in the error message, revisit when a customer asks.
**Tagged for later discussion** (per the 2026-05-09 review).

### 2. Overview selection

`overview_level: int = 0` is a constructor arg today (full-res by default).
Adding a `request_resolution` shortcut that picks the smallest overview ≥ that resolution is a 10-line helper; could live on the reader or as a free function in `geotoolz`.
Not blocking.

### 3. Async iteration over many readers (one per file)

Tile-server workloads often have one COG per scene and N concurrent client requests.
The natural shape is `await asyncio.gather(*[AsyncGeoTIFFReader.open(p) for p in paths])` for setup, then per-request reads against a cached pool of opened readers.
Caching strategy (LRU? per-process? shared across workers?) is **not** in this issue — it's an application-level concern.
Just calling out so we don't accidentally bake in a single-reader-per-call assumption.

### 4. `path_or_url` typing on the Protocol

`AsyncGeoData` (defined in Issue 1) inherits from `GeoData`, which doesn't currently mandate `path_or_url`.
We only need it on the concrete reader for diagnostics / repr.
Don't lift it onto the Protocol.

### 5. Pinned vs floating `async-geotiff` version

`async-geotiff` is at v0.1+ and pre-1.0; the API may shift.
Pin a minor range in `pyproject.toml` (`async-geotiff>=0.1,<0.2`) and bump deliberately.
Document the bump policy in `geotoolz`'s release notes when we cut v0.1.
