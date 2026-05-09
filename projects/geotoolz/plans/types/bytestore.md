---
title: ByteStore Protocol
subject: Core types
subtitle: Unified transport for obstore and fsspec
short_title: ByteStore
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, types, transport, obstore
---

> **Parent:** [README.md](README.md) — Core types.
> **Status:** design proposal. Extracted as a cross-cutting Protocol because [`AsyncGeoTIFFReader`](../georeader/reader_async_geotiff.md) consumes it for COG tile reads, and any future raw-byte-shaped reader (sync or async) will too. Lives here rather than buried inside one reader's design doc.
> **Scope:** the `ByteStore` Protocol that abstracts cloud byte access, plus the `ObstoreByteStore` and `FsspecByteStore` adapters and the `open_store(url)` factory.

---

## Summary

A `ByteStore` is a small Protocol exposing object-store-shaped byte access — `get`, `get_range`, `get_ranges`, `put`, `list` — in both sync and async forms. Two adapters implement it: `ObstoreByteStore` wraps the Rust-backed `obstore.ObjectStore` (HTTP/2, native parallel ranges, ~10× throughput on cloud reads); `FsspecByteStore` wraps any `fsspec.AbstractFileSystem` (universal backend coverage including FTP, SFTP, GitHub, Dropbox).

Readers that fetch bytes directly (`AsyncGeoTIFFReader`, future raw-byte-shaped readers) accept a `store=` kwarg typed against this Protocol. Same reader code, two transports underneath; the only thing that differs is which compiled artefact handles `GET /bucket/key Range: bytes=offset-end`.

`open_store(url)` is the unified factory — auto-picks obstore for major clouds, falls back to fsspec for niche backends, with explicit `prefer="obstore"` / `prefer="fsspec"` overrides for the cases where the user wants to force one.

---

## Motivation

Three pressures make a typed transport surface worth doing:

1. **Two real cloud-byte libraries with overlapping scope but different shapes.** `obstore` is async-native, Rust-backed, fast, and covers the major clouds (S3 / GCS / Azure / HTTP). `fsspec` is sync-native, Python, slower per call, and covers *everything* (including FTP, SFTP, GitHub, Dropbox, and a long tail of niche backends). Code that needs to read bytes shouldn't care which is in use; it should care about the bytes.

2. **More than one reader will need the same abstraction.** [`AsyncGeoTIFFReader`](../georeader/reader_async_geotiff.md) fetches COG tiles via range requests; future raw-byte-shaped readers (sync facades, alternative format readers, sensor-specific COG variants) face the same transport choice. Without `ByteStore`, each reader either hardcodes one transport (loses flexibility) or reimplements the dispatch (duplicates code).

3. **The Credential × ByteStore boundary needs an explicit answer.** [`credentials.md`](credentials.md) is the typed credential surface for the GDAL-VSI path. The fsspec and obstore paths carry credentials inside their `fs` / `store` constructors. When users want to swap credentials for an `AsyncGeoTIFFReader`, they construct a new `ByteStore`. Making `ByteStore` a first-class Protocol surfaces this design choice rather than burying it in one reader's design doc.

The pattern mirrors the credentials extraction: a Protocol that abstracts a process-level concern (transport, auth) for the reader layer, with a small set of concrete implementations and a factory.

---

## Primer for newcomers

> **ELI5.** A `ByteStore` is a uniform **"fetch me bytes from the cloud"** helper. Underneath, two real fetchers exist: one **Rust-fast** (obstore), one **widely-compatible** (fsspec). Code that wants bytes doesn't pick — it just asks the helper, who picks the right fetcher for the job.

### HTTP range requests

**What it is.** An HTTP request can include a `Range: bytes=N-M` header, asking the server for *just* bytes N through M of the resource — not the whole thing. Every modern object store (S3, GCS, Azure, plain HTTPS-served TIFFs) honours this.

**How it works.** Client sends `GET /bucket/key.tif` with `Range: bytes=0-16383`. Server responds `206 Partial Content` and the requested bytes. Multiple range requests can be issued in parallel; HTTP/2 (used by obstore's Rust HTTP client) multiplexes them over one TCP connection. This is what makes "read 25 tiles from a 1 GB COG" cost ~6 MB and one round-trip's worth of latency, not 1 GB.

**What this means for us.** The whole point of `ByteStore.get_range(...)` and `ByteStore.get_ranges(...)` is to translate user-friendly "I want this chunk of this object" into HTTP range requests. Adapters (obstore, fsspec) do the translation; readers (`AsyncGeoTIFFReader`, plus any future raw-byte readers) issue the calls. Without range requests, none of this works — you'd be downloading whole files.

```{mermaid}
sequenceDiagram
    participant Reader
    participant Store as ByteStore
    participant HTTP as HTTPS server (S3 / GCS / Azure)

    Reader->>Store: get_range(url, offset=1024, length=8192)
    Store->>HTTP: GET url<br/>Range: bytes=1024-9215
    HTTP-->>Store: 206 Partial Content<br/>(8192 bytes)
    Store-->>Reader: bytes
```

### Object storage vs filesystem APIs

**What it is.** Two different shapes for accessing remote bytes.

**Object storage** (obstore's shape) treats S3/GCS/Azure as flat key-value stores: `store.get(key)`, `store.get_range(key, off, len)`, `store.put(key, data)`, `store.list(prefix)`. No directories; just keys.

**Filesystem** (fsspec's shape) treats the same backends as POSIX-like filesystems: `fs.open(path)`, `fs.cat(path)`, `fs.glob(pattern)`, `fs.ls(path)`. Familiar API; sometimes awkward over actually-flat backends.

**How it works.** Object storage is closer to how S3/GCS/Azure actually work under the hood. Filesystem APIs translate to/from the underlying store — `fs.open(path).seek(off).read(n)` ultimately becomes an HTTP range request, but with extra Python layers. obstore is faster because it skips the filesystem-emulation layer.

**What this means for us.** `ByteStore` is shaped after object storage (the simpler / more efficient pattern). The `FsspecByteStore` adapter translates filesystem calls to the object-storage shape on the way in. Reader code calls `ByteStore.get_ranges(...)` regardless of which transport is underneath; the adapter handles whichever weirdness the underlying library imposes.

### HTTP/2 multiplexing

**What it is.** HTTP/2 (the protocol replacing HTTP/1.1 since ~2015) lets multiple requests share one TCP connection, with responses interleaved. HTTP/1.1 needed one connection per concurrent request — opening 25 connections to fetch 25 tiles is expensive (TLS handshake per connection); HTTP/2 needs one.

**How it works.** Client opens a TCP connection + TLS handshake (slow, one-time). Then it sends 25 `GET ... Range: bytes=...` requests on that connection in parallel; the server responds with 25 interleaved bodies. Total latency is ~1 round-trip-time for *all* 25 ranges, not 25 round-trips.

**What this means for us.** obstore's Rust HTTP client speaks HTTP/2 natively. fsspec's per-backend libraries (`s3fs`, `gcsfs`, etc.) vary — some do HTTP/2, some don't. This is why the strategy advice is "obstore for hot paths over major clouds; fsspec for niche backends" — same reads, but obstore wraps them in HTTP/2 and fsspec often doesn't.

### Sync vs async transport

**What it is.** Sync `ByteStore` methods (`get_range`, `get_ranges`) block the calling thread until I/O completes. Async methods (`get_range_async`, `get_ranges_async`) are coroutines that suspend on `await` and let the event loop do other work.

**How it works.** `ObstoreByteStore`'s native shape is async — sync methods are thin wrappers that block on `obstore.get_range_async(...)` via a sync runtime. `FsspecByteStore`'s native shape is sync — async methods only run truly async on backends constructed with `asynchronous=True` (like `s3fs`, `gcsfs`, `adlfs`); on synchronous backends they fall back to running the sync call in an executor.

**What this means for us.** `AsyncGeoTIFFReader` calls `await store.get_ranges_async(...)`; a future sync raw-byte reader (or sync facade over `AsyncGeoTIFFReader`) would call `store.get_ranges(...)`. Same Protocol, two access patterns. The async path is the high-throughput path for tile-server-shaped workloads; the sync path is the "I just want one chip" path. Both paths exist on every adapter — you don't lose async support by picking fsspec, you just get a slower async (per-backend).

```{mermaid}
flowchart TD
    Url[open_store url, prefer=auto]
    Url --> Q{URL scheme?}
    Q -->|s3:// gs:// az://| Obs[ObstoreByteStore<br/>fast HTTP/2 parallel]
    Q -->|http:// https://<br/>file:// memory://| Obs
    Q -->|ftp:// sftp://<br/>github:// dropbox://| Fs[FsspecByteStore<br/>broad backend coverage]
    Q -->|other| Fs
    Obs --> Adapt[satisfies ByteStore Protocol]
    Fs --> Adapt
```

---

## Goals

- **A `ByteStore` Protocol** with sync + async method pairs covering whole-object reads, range reads, parallel range reads, writes, and listing.
- **Two adapter implementations** wrapping `obstore.ObjectStore` and `fsspec.AbstractFileSystem`.
- **A unified `open_store(url, prefer=...)` factory** that auto-picks the transport from URL scheme.
- **Reader integration** via a `store: ByteStore | None = None` constructor kwarg on COG-shaped readers — same Protocol, different concrete `ByteStore`.

---

## Non-goals

- **Replacing `obstore` or `fsspec`.** `ByteStore` is a thin compatibility shim, not a new transport library.
- **Owning the credential layer.** Credentials live in the `ByteStore`'s construction (e.g., `obstore.S3Store(access_key=..., secret_key=...)` or `fsspec.filesystem("s3", key=..., secret=...)`). The boundary between this and the typed [`Credential`](credentials.md) Protocol is documented in §"Connections" below.
- **Universal backend coverage from one adapter.** `obstore` covers the majors; `fsspec` covers the long tail. Users pick by URL scheme or explicit override.
- **Sync-async unification.** The Protocol exposes sync + async pairs because both are needed by different consumers; an adapter can leave one pair as best-effort (e.g., `FsspecByteStore`'s async path is only fast on async-capable backends).

---

## Constraints

- **`get_ranges` is the load-bearing method.** Fetch N byte ranges from one object in one parallel call. obstore implements this natively (with optional coalescing of close-by ranges); fsspec doesn't, so its adapter falls back to `asyncio.gather` over single-range fetches. Performance differences flow from this.
- **Sync methods always work.** Even when the underlying transport is async-native (obstore), sync methods are provided as thin wrappers blocking on the async path. Sync reader code (e.g. a future sync facade over `AsyncGeoTIFFReader`) doesn't have to know.
- **Async methods may be best-effort.** `FsspecByteStore.*_async` methods only run truly async on backends constructed with `asynchronous=True` (s3fs, gcsfs, adlfs). Other backends serialise inside `asyncio.gather`. Documented at the adapter level.
- **`open_store(url)` makes a sensible default.** Users who don't want to think about which transport is in use should be able to write `AsyncGeoTIFFReader("s3://bucket/scene.tif")` and get the right thing.

---

## The `ByteStore` Protocol

```python
from typing import AsyncIterator, Iterator, Protocol


class ByteStore(Protocol):
    """Unified byte-store API. Both obstore.ObjectStore and fsspec
    AbstractFileSystem can satisfy this via thin adapters.

    Every method comes in sync + async pairs. Adapter implementations
    can leave one pair as best-effort (e.g. fsspec's async path is
    only fast on async-capable backends like s3fs / gcsfs / adlfs)."""

    # whole-object reads
    def get(self, key: str) -> bytes: ...
    async def get_async(self, key: str) -> bytes: ...

    # range reads — the hot path for COG tile fetches
    def get_range(self, key: str, offset: int, length: int) -> bytes: ...
    async def get_range_async(self, key: str, offset: int, length: int) -> bytes: ...

    # parallel range reads — the BIG win for tile fan-out
    def get_ranges(
        self, key: str, ranges: list[tuple[int, int]],
    ) -> list[bytes]: ...
    async def get_ranges_async(
        self, key: str, ranges: list[tuple[int, int]],
    ) -> list[bytes]: ...

    # writes
    def put(self, key: str, data: bytes) -> None: ...
    async def put_async(self, key: str, data: bytes) -> None: ...

    # listing
    def list(self, prefix: str = "") -> Iterator[str]: ...
    async def list_async(self, prefix: str = "") -> AsyncIterator[str]: ...
```

`get_ranges` is the load-bearing method — fetch N byte ranges from one object in one parallel call. obstore implements this natively (with optional coalescing of close-by ranges); fsspec doesn't, so its adapter falls back to `asyncio.gather` over single-range fetches.

---

## `ObstoreByteStore` — wraps `obstore.ObjectStore`

```python
class ObstoreByteStore(ByteStore):
    """Wrap an obstore.ObjectStore as a ByteStore.

    Async path is the native one — sync methods are thin wrappers that
    block on the async path via the obstore sync runtime.
    """
    _store: "obstore.ObjectStore"

    def __init__(self, store: "obstore.ObjectStore"):
        self._store = store

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "ObstoreByteStore":
        """Auto-pick the right obstore backend from URL scheme:
        s3:// → S3Store, gs:// → GCSStore, az:// → AzureStore,
        http(s):// → HTTPStore, file:// → LocalStore, memory:// → MemoryStore."""
        ...

    # whole object
    def get(self, key: str) -> bytes:
        return self._store.get(key).bytes()
    async def get_async(self, key: str) -> bytes:
        result = await self._store.get_async(key)
        return await result.bytes_async()

    # single range
    def get_range(self, key: str, offset: int, length: int) -> bytes:
        return self._store.get_range(key, offset, length).bytes()
    async def get_range_async(self, key: str, offset: int, length: int) -> bytes:
        result = await self._store.get_range_async(key, offset, length)
        return await result.bytes_async()

    # parallel ranges — obstore's get_ranges is native; coalesces close ranges
    def get_ranges(self, key, ranges):
        return [b.bytes() for b in self._store.get_ranges(key, ranges)]
    async def get_ranges_async(self, key, ranges):
        results = await self._store.get_ranges_async(key, ranges)
        return [await r.bytes_async() for r in results]

    # writes
    def put(self, key, data): self._store.put(key, data)
    async def put_async(self, key, data): await self._store.put_async(key, data)

    # listing
    def list(self, prefix=""):
        for entry in self._store.list(prefix=prefix):
            yield entry.path
    async def list_async(self, prefix=""):
        async for entry in self._store.list_async(prefix=prefix):
            yield entry.path
```

---

## `FsspecByteStore` — wraps `fsspec.AbstractFileSystem`

```python
class FsspecByteStore(ByteStore):
    """Wrap an fsspec filesystem as a ByteStore.

    Sync methods always work. Async methods require an async-capable
    filesystem (constructed with asynchronous=True or a backend that
    supports it like s3fs / adlfs / gcsfs). Parallel ranges have no
    native fsspec equivalent — the adapter falls back to asyncio.gather
    over single-range fetches, which is throughput-limited by the
    backend."""
    _fs: "fsspec.AbstractFileSystem"
    _root: str                                       # bucket / container prefix

    def __init__(self, fs: "fsspec.AbstractFileSystem", root: str = ""):
        self._fs = fs
        self._root = root.rstrip("/")

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "FsspecByteStore":
        """Build via fsspec.url_to_fs(url)."""
        ...

    def _path(self, key: str) -> str:
        return f"{self._root}/{key}" if self._root else key

    # whole object
    def get(self, key: str) -> bytes:
        return self._fs.cat_file(self._path(key))
    async def get_async(self, key: str) -> bytes:
        return await self._fs._cat_file(self._path(key))

    # single range
    def get_range(self, key: str, offset: int, length: int) -> bytes:
        with self._fs.open(self._path(key), "rb") as f:
            f.seek(offset)
            return f.read(length)
    async def get_range_async(self, key: str, offset: int, length: int) -> bytes:
        return await self._fs._cat_file(
            self._path(key), start=offset, end=offset + length,
        )

    # parallel ranges — no native parallel; serial fallback for sync,
    # asyncio.gather for async (effective only on async-capable backends)
    def get_ranges(self, key, ranges):
        with self._fs.open(self._path(key), "rb") as f:
            out = []
            for offset, length in ranges:
                f.seek(offset)
                out.append(f.read(length))
            return out
    async def get_ranges_async(self, key, ranges):
        return await asyncio.gather(*[
            self.get_range_async(key, o, l) for (o, l) in ranges
        ])

    # writes
    def put(self, key, data):
        with self._fs.open(self._path(key), "wb") as f:
            f.write(data)
    async def put_async(self, key, data):
        await self._fs._pipe_file(self._path(key), data)

    # listing
    def list(self, prefix=""):
        return iter(self._fs.ls(self._path(prefix)))
    async def list_async(self, prefix=""):
        for p in await self._fs._ls(self._path(prefix)):
            yield p
```

---

## Unified factory

```python
def open_store(
    url: str,
    *,
    prefer: Literal["obstore", "fsspec", "auto"] = "auto",
    **backend_kwargs,
) -> ByteStore:
    """Build a ByteStore for the given URL.

    Selection:
      "auto"   — obstore for s3:// / gs:// / az:// / http(s):// / file:// / memory://;
                 fsspec for any other scheme (ftp://, sftp://, github://, …).
      "obstore" — force obstore; raise if backend not supported.
      "fsspec" — force fsspec; useful when the rest of the pipeline expects an
                 fsspec-shaped object (zarr 2, geopandas, etc.).
    """
    ...
```

---

## How readers consume it

Every reader that fetches bytes directly takes an optional `store: ByteStore | None = None`. If `None`, the reader calls `open_store(url)` internally:

```python
class AsyncGeoTIFFReader(AsyncReader):
    def __init__(
        self,
        path_or_url: str,
        indexes: int | Sequence[int] | None = None,
        *,
        store: ByteStore | None = None,
        max_concurrent_tiles: int = 32,
    ):
        self.path_or_url = path_or_url
        self._store = store or open_store(path_or_url, prefer="auto")
        # ...

    async def read_window(self, window):
        tiles = self._tiles_for_window(window)
        # one call into the unified protocol — adapter handles obstore vs fsspec
        tile_bytes = await self._store.get_ranges_async(
            self.path_or_url,
            [(t.offset, t.length) for t in tiles],
        )
        return self._decompress_and_assemble(tile_bytes, tiles, window)
```

A future sync raw-byte reader (or sync facade over `AsyncGeoTIFFReader`) follows the same shape, calling `self._store.get_ranges(...)` instead of `await self._store.get_ranges_async(...)` — the Protocol carries both surfaces.

The reader code knows nothing about which transport is active. It calls `self._store.get_ranges(...)` and the adapter does the right thing.

---

## Connections to other designs

| Design | How it touches `ByteStore` |
|---|---|
| [`georeader/reader_async_geotiff.md`](../georeader/reader_async_geotiff.md) | `AsyncGeoTIFFReader` accepts `store: ByteStore | None`. Async path; uses `store.get_range_async` / `store.get_ranges_async` under `asyncio.gather`. Defaults to `open_store(url, prefer="auto")`. |
| [`georeader/reader_protocol.md`](../georeader/reader_protocol.md) | `RasterioReader`'s opener-path triage (`fs=`, `opener=`) is a related but distinct seam — it routes bytes through GDAL's callback API rather than through a `ByteStore`. The two coexist; `RasterioReader` doesn't currently accept a `ByteStore`. |
| [`credentials.md`](credentials.md) | Credential × ByteStore overlap — see §"Open question 1" below. |
| [`geostack.md`](../geostack.md) §"`obstore` vs `fsspec` compared" | The ecosystem-level comparison table that frames *why* both transports coexist. Read first to orient on the trade-off; this design is the typed surface that abstracts it. |

---

## Open questions

### 1. `Credential` × `ByteStore` overlap

Both Protocols touch authentication. The split today:

- `Credential` ([`credentials.md`](credentials.md)) is for the **GDAL-VSI path** in `RasterioReader`. It applies env vars to a `rasterio.Env(...)` context. GDAL's libcurl handles HTTP.
- `ByteStore` (this doc) is for the **`AsyncGeoTIFFReader` path** (and any future raw-byte reader). Credentials live inside the `ObstoreByteStore`'s `obstore.S3Store(access_key=..., secret_key=...)` or the `FsspecByteStore`'s `fsspec.filesystem("s3", key=..., secret=...)`.

When a user wants to swap credentials for an `AsyncGeoTIFFReader`, they construct a new `ByteStore`. When for a `RasterioReader`, they pass `credential=`. The two surfaces don't currently interoperate — and probably shouldn't, since the underlying transports are different.

But there's a cleaner shape: a `Credential` could *produce* a `ByteStore` (e.g., `aws_credential.to_obstore_s3_store(bucket="...")`) so users have one credential surface that flows through both reader families. **Tentative pick:** add a small set of `to_*_store()` helpers on `Credential` subclasses; document that the GDAL-VSI path uses `apply()` and the byte-fetch path uses `to_*_store()`. Don't unify the two Protocols.

### 2. Should `ByteStore` be a `runtime_checkable` Protocol?

Currently typed as `Protocol` (compile-time only). Making it `@runtime_checkable` would let user code `isinstance(x, ByteStore)` for the duck-typed case. Same question we resolved for `Credential` (yes, runtime-checkable). Probably yes here too for symmetry.

### 3. Should `ByteStore` carry a default key?

Many `AsyncGeoTIFFReader` use cases construct one reader per file; the URL is stored on the reader and passed to every `store.get_ranges_async(self.path_or_url, ...)`. Could `ByteStore` carry an optional default `key=` so the reader doesn't repeat itself? Probably overkill — explicit is fine; the duplication is minor.

### 4. `put` and `list` — needed for read-only readers?

`AsyncGeoTIFFReader` only calls `get_range_async` / `get_ranges_async`. The Protocol includes `put` and `list` because the abstraction is general, but read-only consumers won't use them. Leaving them in keeps the Protocol useful for catalog-style writers. Adapters that don't support `put` can raise; the Protocol shape doesn't have to change.

### 5. URL scheme detection in `open_store(prefer="auto")`

The auto-pick logic needs an explicit list of "obstore-supported" schemes. As obstore grows backends, the list grows. Either hardcode the current list (and update on each obstore release) or detect dynamically (try obstore first, fall back to fsspec on `NotImplementedError`). **Tentative pick:** hardcoded list for predictability; document where it lives so it's easy to update.

---

## Alternatives considered

- **Extend `obstore.ObjectStore` directly** with fsspec compatibility shims. Rejected: forces obstore as a hard dep on every install. The Protocol approach lets users with only fsspec installed still work.
- **Use `fsspec.AbstractFileSystem` as the universal interface.** Rejected: fsspec is sync-native, and the async story is awkward (per-backend, not all backends async-capable). The Protocol design respects both libraries' shapes instead of forcing one onto the other.
- **Drop one transport.** Rejected: obstore wins on throughput for major clouds; fsspec wins on backend breadth. Both have legitimate use cases.
