# Issue 3 — `AsyncGeoTIFFReader`

> **Parent:** [README.md](README.md)
> **Depends on:** [Issue 1](reader_protocol.md) (Protocols) + [Issue 2](reader_lazy_cog.md) (`ByteStore`, COG-parsing helpers).
> **Scope:** an async, COG-only reader for high-concurrency fan-out workloads — tile servers, web maps, async ML inference services.

---

## Why this issue exists

Sync I/O is fine when you have one read at a time. For workloads where many reads happen concurrently (a tile server fielding 1000 simultaneous tile requests, an async ML service that fans out across hundreds of windows from one process), async-native fetching is the right shape — `asyncio.gather(*[reader.read_window(w) for w in windows])` becomes a one-line concurrency primitive.

Today, `georeader` has no async story. Users wanting async reads either roll their own or pull in an external library with a different API. This issue adds `AsyncGeoTIFFReader` — same metadata surface as the sync readers, async read methods.

The design reuses everything from [Issue 2](reader_lazy_cog.md) that's not sync-specific: COG header parsing, `_tiles_for_window` math, decompression dispatch, the `ByteStore` Protocol. The only async-specific work is:

- Header fetch happens in an async classmethod (`open(...)`).
- Read methods are coroutines that `await self._store.get_ranges_async(...)`.
- A `max_concurrent_tiles` semaphore caps the fan-out per call.

---

## Deliverables

1. **`AsyncReader` Protocol** in `georeader/abstract_reader.py` — extends `_ReaderMeta` with async read methods. (Could move to Issue 1; landing here for now to avoid scope creep there.)
2. **`AsyncGeoTIFFReader` class** in `georeader/async_geotiff_reader.py`.
3. **Async classmethod `open(...)`** — fetches the IFD chain via `await store.get_range_async(...)`.
4. **Async read methods** — `read_window`, `read_bounds`, `read_geoslice`, `load`.
5. **`max_concurrent_tiles` semaphore** — caps parallel range requests per call.
6. **Async context-manager** — `__aenter__` / `__aexit__`.
7. **Tests** — open + read a real COG asynchronously; concurrent fan-out across N windows.

This issue does **not** ship new transport infrastructure — it imports `ByteStore`, `ObstoreByteStore`, `FsspecByteStore`, `open_store` from [Issue 2](reader_lazy_cog.md). It also imports the COG header parsing and tile-fetch math from the shared module created there (see [parent open question #3](README.md#open-questions)).

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
    async def read_geoslice(self, slice: GeoSlice) -> GeoTensor: ...
    async def load(self) -> GeoTensor: ...
    async def aclose(self) -> None: ...

    async def __aenter__(self) -> "AsyncReader": ...
    async def __aexit__(self, *exc) -> None: ...
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
        Reuses the shared COG parser from Issue 2's _cog_helpers module."""
        ...
    def _tiles_for_window(self, window: Window) -> list["TileSpec"]:
        """Same math as LazyCOGReader; imported from shared _cog_helpers."""
        ...
    def _decompress_and_assemble(
        self,
        tile_bytes: list[bytes],
        tiles: list["TileSpec"],
        window: Window,
    ) -> np.ndarray:
        """Same dispatch as LazyCOGReader; imported from shared _cog_helpers."""
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
        return self._decompress_and_assemble(bytes_list, tiles, window)

    async def read_bounds(self, bounds, *, target_resolution=None, target_crs=None) -> GeoTensor: ...
    async def read_geoslice(self, slice: GeoSlice) -> GeoTensor: ...
    async def load(self) -> GeoTensor:
        """Fetches every tile in parallel. Use sparingly."""
        ...

    async def aclose(self) -> None: ...               # no-op; obstore is pooled
    async def __aenter__(self) -> "AsyncGeoTIFFReader":
        if self._header is None:
            await self._fetch_header()
        return self
    async def __aexit__(self, *exc) -> None: ...
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

## Reuse of `LazyCOGReader` infrastructure

This issue is small because most of the work is shared:

| Component | Source | Used for |
|---|---|---|
| `ByteStore` Protocol + adapters | [Issue 2](reader_lazy_cog.md) | byte fetching |
| `open_store(url)` factory | [Issue 2](reader_lazy_cog.md) | auto-pick transport |
| COG header parser | shared `_cog_helpers.py` (Issue 2) | IFD walk, `COGHeader` dataclass |
| `_tiles_for_window` math | shared `_cog_helpers.py` (Issue 2) | window → tile-spec list |
| Decompression dispatch | shared `_cog_helpers.py` (Issue 2) | per-tile decode |

The only original code in this issue is:

- `AsyncReader` Protocol declaration.
- `AsyncGeoTIFFReader.__init__` / `open` / async read methods.
- `asyncio.Semaphore` concurrency cap.
- Async context-manager.

Roughly ~150–200 LOC of new code on top of Issue 2's helpers.

---

## Acceptance criteria

- `AsyncGeoTIFFReader` instances satisfy `AsyncReader` per static type-check.
- `await AsyncGeoTIFFReader.open("s3://...")` returns a fully initialised reader; metadata properties work after open.
- `await reader.read_window(window)` returns a `GeoTensor` matching `LazyCOGReader.read_window(window)` (same file, same window).
- Concurrent fan-out: `await asyncio.gather(*[reader.read_window(w) for w in windows])` for 100 windows from one reader instance completes without errors and faster than the sync equivalent.
- `max_concurrent_tiles` semaphore is honoured — verifiable by mocking `store.get_range_async` to count concurrent invocations.
- `async with await AsyncGeoTIFFReader.open(...)` context-manager works.
- Reusing `ByteStore` from Issue 2 — no duplication of transport code.

---

## Issue-specific open questions

In addition to the [parent design's open questions](README.md#open-questions):

1. **`AsyncReader` Protocol location.** Currently planned for this issue, but could move to Issue 1 if there's any chance of an *async-conformant* `RasterioReader` shim landing later (e.g., `RasterioReader.read_window_async` that runs the sync read in a threadpool). See [parent open question](README.md#open-questions).
2. **Sync `__init__` + async `open(...)` vs always-async open.** Some libraries (e.g., `httpx.AsyncClient`) allow a sync `__init__` that defers expensive setup. The current proposal does this — `__init__` is cheap, `open(...)` does the IFD fetch. Alternative: always require `await open(...)` and make `__init__` private. The user-facing API is roughly the same; the internal state-machine is simpler with the always-async-open path but the existing pattern is more familiar from other libraries.
3. **`run_in_executor` shim for sync ByteStore adapters.** If `FsspecByteStore` is given a non-async-capable backend, its `get_range_async` can fall back to `loop.run_in_executor(None, ...)` to wrap the sync read. This is implemented in Issue 2's `FsspecByteStore`; just calling out that `AsyncGeoTIFFReader` works against any `ByteStore` regardless of whether the backend is natively async.
4. **Trio / anyio support.** The proposal uses `asyncio.gather` and `asyncio.Semaphore` directly. Wrapping in `anyio` for trio compatibility is straightforward but adds a dependency. Recommendation: stay asyncio-only for v1; add anyio later if there's demand.
