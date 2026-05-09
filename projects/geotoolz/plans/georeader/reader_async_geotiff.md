---
title: AsyncGeoTIFFReader
subject: georeader design
subtitle: Async COG reader for high-concurrency fan-out
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
keywords: design, georeader, async, cog
---

> **Parent:** [README.md](README.md)
> **Depends on:** [Issue 1](reader_protocol.md) (Protocols) + [`types/bytestore.md`](../types/bytestore.md) (`ByteStore` Protocol).
> **Scope:** an async, COG-only reader for high-concurrency fan-out workloads — tile servers, web maps, async ML inference services. Owns the COG-parsing primitives (IFD walk, tile-fetch math, decompression dispatch) in a private `_cog_helpers.py` module so future raw-byte readers can reuse them.

---

## Why this issue exists

Sync I/O is fine when you have one read at a time. For workloads where many reads happen concurrently (a tile server fielding 1000 simultaneous tile requests, an async ML service that fans out across hundreds of windows from one process), async-native fetching is the right shape — `asyncio.gather(*[reader.read_window(w) for w in windows])` becomes a one-line concurrency primitive.

Today, `georeader` has no async story. Users wanting async reads either roll their own or pull in an external library with a different API. This issue adds `AsyncGeoTIFFReader` — same metadata surface as the sync readers, async read methods.

The design reuses the `ByteStore` Protocol from [`types/bytestore.md`](../types/bytestore.md) and ships its own COG-parsing primitives (header walking, `_tiles_for_window` math, decompression dispatch) in a private `_cog_helpers.py` module. The async-specific work is small:

- Header fetch happens in an async classmethod (`open(...)`).
- Read methods are coroutines that issue per-tile `await self._store.get_range_async(...)` calls inside `asyncio.gather(...)` for parallel fetch. (`get_ranges_async` for store-level coalescing is a possible alternative — see [Open question §5](#5-per-tile-get_range_async-vs-store-level-get_ranges_async).)
- A `max_concurrent_tiles` semaphore caps the fan-out per call.

---

## Primer for newcomers

> **ELI5.** Sync code is like **waiting in line at one cashier** — you can't do anything else until your turn finishes. Async code is like **leaving your order at 25 different counters** and collecting the results as they're ready. Same hardware; far more food per unit time when each order is mostly waiting.

### `async` / `await` basics

**What it is.** Python's `async def` defines a *coroutine* — a function that can pause itself with `await` and let other coroutines run on the same thread until the awaited operation completes. It's not threading; it's cooperative multitasking inside one event loop.

**How it works.** When you call `result = await some_async_function()`, the runtime suspends the current coroutine until `some_async_function()` finishes, then resumes with the result. While suspended, the event loop runs other ready coroutines. Async only helps when the work is I/O-bound — waiting on network, disk, etc. — because that's when there's idle time to fill.

**What this means for us.** Cloud raster reads are dominated by network round-trips. With `async`, one process can have hundreds of `read_window(...)` calls in flight concurrently, all sharing one OS thread. The CPython GIL doesn't get in the way because nobody's computing — they're all waiting on HTTP. Same hardware; far more throughput; far simpler than thread pools.

### `asyncio.gather` and parallel awaits

**What it is.** `asyncio.gather(coro1, coro2, coro3, ...)` runs multiple coroutines concurrently and returns when all finish (or one raises). It's the canonical way to issue many parallel I/O operations in one call.

**How it works.** Each argument is a coroutine that hasn't started yet (`some_async_func(arg)` without `await`). `gather(...)` schedules them all on the event loop, lets them run interleaved, and collects their results into a list in input order. If any raises, `gather` propagates the exception (or you can pass `return_exceptions=True` to collect them).

**What this means for us.** `AsyncGeoTIFFReader.read_window(...)` decomposes a window into N tiles, builds N coroutines (one per range request), and `await asyncio.gather(...)` fetches them in parallel. Tile-server workloads can additionally do `await asyncio.gather(*[reader.read_window(w) for w in 1000_windows])` for outer parallelism — concurrent reads across many readers.

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

### Semaphores (concurrency limits)

**What it is.** An `asyncio.Semaphore(N)` is a counter that lets at most N coroutines proceed at once; the rest wait. Standard concurrency primitive.

**How it works.** `async with semaphore:` blocks the coroutine until the counter has room, increments it, runs the block, decrements on exit. Used to cap concurrent operations regardless of how many you've launched.

**What this means for us.** Without a cap, `read_window(window)` for a window covering 100 tiles would issue 100 parallel HTTP requests — possibly exhausting connection pools or hitting rate limits. `max_concurrent_tiles=32` (default) wraps each fetch in `async with sem:` so at most 32 are in flight per call. Outer fan-out (across `read_window` calls from 100 concurrent users) is also bounded by the surrounding framework's concurrency (FastAPI / aiohttp request handlers).

### Async classmethod for construction

**What it is.** Python's `__init__` *can't* be an `async def`. There's no `__ainit__` magic method. So when initialisation requires async work (fetching the COG IFD over HTTP), the convention is a classmethod `open(...)` that's `async`.

**How it works.** `__init__` does cheap synchronous setup (store the URL, the credential, allocate state). The user calls `await Cls.open(url)`, which constructs an instance via `__init__` and then runs the async setup before returning the fully-initialised reader. Same pattern used by `aiohttp.ClientSession`, `asyncpg.connect`, etc.

**What this means for us.** Users write `reader = await AsyncGeoTIFFReader.open("s3://bucket/scene.tif")` instead of `reader = AsyncGeoTIFFReader(...)`. The reader's `crs` / `transform` / etc. don't work until `open()` has been awaited — accessing them earlier raises `RuntimeError("Reader not opened")`. Slight cost in clarity; major win in not faking sync APIs over async work.

```{mermaid}
sequenceDiagram
    participant App
    participant Cls as AsyncGeoTIFFReader
    participant Init as __init__
    participant Header as _fetch_header

    App->>Cls: await open(url)
    Cls->>Init: __init__(url)
    Init-->>Cls: instance (no header yet)
    Note over Cls: header is None — calling .crs raises
    Cls->>Header: await _fetch_header()
    Header-->>Cls: COGHeader cached
    Cls-->>App: ready instance
    App->>App: reader.crs ✓
    App->>App: await reader.read_window(w)
```

---

## Deliverables

1. **`AsyncReader` Protocol** in `georeader/abstract_reader.py` — extends `_ReaderMeta` with async read methods. (Could move to Issue 1; landing here for now to avoid scope creep there.)
2. **`AsyncGeoTIFFReader` class** in `georeader/async_geotiff_reader.py`.
3. **Async classmethod `open(...)`** — fetches the IFD chain via `await store.get_range_async(...)`.
4. **Async read methods** — `read_window`, `read_bounds`, `read_geoslice`, `load`.
5. **`max_concurrent_tiles` semaphore** — caps parallel range requests per call.
6. **Async context-manager** — `__aenter__` / `__aexit__`.
7. **Tests** — open + read a real COG asynchronously; concurrent fan-out across N windows.

This issue does **not** ship new transport infrastructure — it imports `ByteStore`, `ObstoreByteStore`, `FsspecByteStore`, `open_store` from [`types/bytestore.md`](../types/bytestore.md). The COG header parsing and tile-fetch math live in a private `_cog_helpers.py` module owned by this reader; future raw-byte-shaped readers can import from there if they need the same primitives.

---

## `AsyncReader` Protocol

```python
class AsyncReader(_ReaderMeta, Protocol):
    """Async read interface — AsyncGeoTIFFReader."""

    async def read_window(self, window: Window) -> GeoTensor: ...
    async def read_bounds(
        self,
        bounds: tuple[float, float, float, float],
        *,
        target_resolution: tuple[float, float] | None = None,
        target_crs: pyproj.CRS | str | None = None,
    ) -> GeoTensor: ...
    async def read_geoslice(self, slice_: GeoSlice) -> GeoTensor: ...
    async def load(self) -> GeoTensor: ...
    async def aclose(self) -> None: ...

    async def __aenter__(self) -> "AsyncReader": ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool | None: ...
```

Mirrors `SyncReader` (Issue 1) on every method name; only the `await` keyword diverges. Downstream code in `geotoolz` accepts `SyncReader` or `AsyncReader` and branches on which is in use — see [parent §High-level shape](README.md#high-level-shape).

---

## `AsyncGeoTIFFReader` class

```python
class AsyncGeoTIFFReader(AsyncReader):
    """Async COG reader (TIFF/COG only, no GDAL).

    Open is async because the IFD fetch is async. After open, every read
    method dispatches `len(tiles)` parallel range requests via obstore and
    awaits all of them before assembling the array. Designed for high
    concurrency — e.g. a tile server fetching 1000 windows concurrently.
    """
    path_or_url: str
    indexes: tuple[int, ...] | None
    _store: ByteStore
    _header: "COGHeader | None"                      # populated after .open()
    _max_concurrent_tiles: int

    def __init__(
        self,
        path_or_url: str,
        indexes: int | Sequence[int] | None = None,
        *,
        store: ByteStore | None = None,
        max_concurrent_tiles: int = 32,
    ):
        """Cheap. Does NOT fetch the header — call .open() first."""
        self.path_or_url = path_or_url
        self.indexes = _normalise_indexes(indexes)
        self._store = store or open_store(path_or_url, prefer="auto")
        self._header = None
        self._max_concurrent_tiles = max_concurrent_tiles

    @classmethod
    async def open(
        cls,
        path_or_url: str,
        indexes: int | Sequence[int] | None = None,
        *,
        store: ByteStore | None = None,
        max_concurrent_tiles: int = 32,
    ) -> "AsyncGeoTIFFReader":
        """Async constructor: build instance, fetch and parse the IFD chain.
        Most users call this rather than __init__."""
        self = cls(path_or_url, indexes, store=store,
                   max_concurrent_tiles=max_concurrent_tiles)
        await self._fetch_header()
        return self

    # internal
    async def _fetch_header(self) -> None:
        """One or two async range requests to pull and parse the TIFF IFDs.
        Uses the COG parser from the private _cog_helpers module."""
        ...
    def _tiles_for_window(self, window: Window) -> list["TileSpec"]:
        """Tile-fetch math from _cog_helpers."""
        ...
    def _decompress_and_assemble(
        self,
        tile_bytes: list[bytes],
        tiles: list["TileSpec"],
        window: Window,
    ) -> np.ndarray:
        """Per-tile decompression dispatch from _cog_helpers."""
        ...

    # metadata — sync after .open() has been awaited
    @property
    def crs(self) -> pyproj.CRS:
        """Raises RuntimeError if .open() hasn't been awaited yet."""
        if self._header is None:
            raise RuntimeError("Reader not opened — call AsyncGeoTIFFReader.open(...)")
        return self._header.crs
    # ... (the rest of _ReaderMeta surface)

    # reads — same shape as the sync readers, but coroutines
    async def read_window(self, window: Window) -> GeoTensor:
        tiles = self._tiles_for_window(window)
        # parallel fetch with concurrency cap
        sem = asyncio.Semaphore(self._max_concurrent_tiles)

        async def _fetch(t):
            async with sem:
                return await self._store.get_range_async(
                    self.path_or_url, t.offset, t.length,
                )

        bytes_list = await asyncio.gather(*[_fetch(t) for t in tiles])
        out = self._decompress_and_assemble(bytes_list, tiles, window)  # ndarray
        return GeoTensor(
            values=out,
            transform=self._header.window_transform(window),
            crs=self._header.crs,
            fill_value_default=self._header.nodata,
        )

    async def read_bounds(self, bounds, *, target_resolution=None, target_crs=None) -> GeoTensor: ...
    async def read_geoslice(self, slice_: GeoSlice) -> GeoTensor: ...
    async def load(self) -> GeoTensor:
        """Fetches every tile in parallel. Use sparingly."""
        ...

    async def aclose(self) -> None: ...               # no-op; obstore is pooled
    async def __aenter__(self) -> "AsyncGeoTIFFReader":
        if self._header is None:
            await self._fetch_header()
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool | None: ...
```

---

## Why an async classmethod for open

`AsyncGeoTIFFReader.__init__` is sync and cheap — it stores the URL, the store, and the indexes, but does **not** fetch any bytes. The IFD fetch happens in `await AsyncGeoTIFFReader.open(...)` because:

- IFD fetch requires an `await store.get_range_async(...)` call.
- `__init__` can't be async in Python — there's no `__ainit__`.
- Workarounds (event-loop hacks, lazy initialisation on first read) are worse than the explicit two-step pattern.

The accepted Python idiom for "async constructor" is `cls.open(...)` as a classmethod; this is what `aiohttp.ClientSession`, `asyncpg.connect`, etc. all use. Documented prominently so users don't trip on the "I instantiated it but `.crs` raises" failure mode.

---

## Concurrency control — `max_concurrent_tiles`

Without a cap, `read_window(window)` could issue dozens to hundreds of parallel range requests, exhausting connection pools or rate-limiting on the cloud side. The semaphore caps fan-out per call:

```python
sem = asyncio.Semaphore(self._max_concurrent_tiles)

async def _fetch(t):
    async with sem:
        return await self._store.get_range_async(self.path_or_url, t.offset, t.length)

bytes_list = await asyncio.gather(*[_fetch(t) for t in tiles])
```

Default `max_concurrent_tiles=32` is a sensible starting point — enough parallelism for the common case (`read_window` of a 1024×1024 area pulling ~25 tiles), conservative enough that fan-out across N concurrent `read_window` calls doesn't OOM.

For tile-server workloads where 1000 client requests fan out to 1000 reader calls each pulling 25 tiles, the *outer* concurrency is also bounded by `aiohttp` / FastAPI's request-handler pool — the inner reader semaphore prevents a single request from monopolising the connection pool.

---

## Module layout

This issue ships its own COG-parsing primitives in a private module so future raw-byte readers can reuse them:

| Component | Source | Used for |
|---|---|---|
| `ByteStore` Protocol + adapters | [`types/bytestore.md`](../types/bytestore.md) | byte fetching |
| `open_store(url)` factory | [`types/bytestore.md`](../types/bytestore.md) | auto-pick transport |
| COG header parser | private `_cog_helpers.py` (this issue) | IFD walk, `COGHeader` dataclass |
| `_tiles_for_window` math | private `_cog_helpers.py` (this issue) | window → tile-spec list |
| Decompression dispatch | private `_cog_helpers.py` (this issue) | per-tile decode |

The original code in this issue is:

- `AsyncReader` Protocol declaration.
- `_cog_helpers.py` — IFD walk, `COGHeader`, `_tiles_for_window`, decompression dispatch.
- `AsyncGeoTIFFReader.__init__` / `open` / async read methods.
- `asyncio.Semaphore` concurrency cap.
- Async context-manager.

Roughly ~400 LOC total (~150–200 of which is the COG-parsing helpers, the rest is the async reader proper).

---

## Acceptance criteria

- `AsyncGeoTIFFReader` instances satisfy `AsyncReader` per static type-check.
- `await AsyncGeoTIFFReader.open("s3://...")` returns a fully initialised reader; metadata properties work after open.
- `await reader.read_window(window)` returns a `GeoTensor` numerically matching `RasterioReader.read_window(window)` for the same file and window (within rounding).
- Concurrent fan-out: `await asyncio.gather(*[reader.read_window(w) for w in windows])` for 100 windows from one reader instance completes without errors and faster than the sync equivalent.
- `max_concurrent_tiles` semaphore is honoured — verifiable by mocking `store.get_range_async` to count concurrent invocations.
- `async with await AsyncGeoTIFFReader.open(...)` context-manager works.
- Reusing `ByteStore` from [`types/bytestore.md`](../types/bytestore.md) — no duplication of transport code.

---

## Issue-specific open questions

In addition to the [parent design's open questions](README.md#open-questions):

1. **`AsyncReader` Protocol location.** Currently planned for this issue, but could move to Issue 1 if there's any chance of an *async-conformant* `RasterioReader` shim landing later (e.g., `RasterioReader.read_window_async` that runs the sync read in a threadpool). See [parent open question](README.md#open-questions).
2. **Sync `__init__` + async `open(...)` vs always-async open.** Some libraries (e.g., `httpx.AsyncClient`) allow a sync `__init__` that defers expensive setup. The current proposal does this — `__init__` is cheap, `open(...)` does the IFD fetch. Alternative: always require `await open(...)` and make `__init__` private. The user-facing API is roughly the same; the internal state-machine is simpler with the always-async-open path but the existing pattern is more familiar from other libraries.
3. **`run_in_executor` shim for sync ByteStore adapters.** If `FsspecByteStore` is given a non-async-capable backend, its `get_range_async` can fall back to `loop.run_in_executor(None, ...)` to wrap the sync read. This is part of `FsspecByteStore`'s adapter spec in [`types/bytestore.md`](../types/bytestore.md); just calling out that `AsyncGeoTIFFReader` works against any `ByteStore` regardless of whether the backend is natively async.
4. **Trio / anyio support.** The proposal uses `asyncio.gather` and `asyncio.Semaphore` directly. Wrapping in `anyio` for trio compatibility is straightforward but adds a dependency. Recommendation: stay asyncio-only for v1; add anyio later if there's demand.

5. **Per-tile `get_range_async` vs store-level `get_ranges_async`.** The current proposal uses per-tile `get_range_async` calls inside `asyncio.gather(...)` so the `max_concurrent_tiles` semaphore can wrap each individual fetch. The alternative is a single `await self._store.get_ranges_async(url, [(o, l), ...])` call that lets the store (e.g., `obstore`) do its own coalescing of close-by ranges and parallelism control. **Tentative pick: per-tile + semaphore** for v1 (explicit, debuggable, and lets us cap concurrency precisely). Switch to store-level if benchmarking shows obstore's coalescing meaningfully outperforms manual gather on real workloads.
