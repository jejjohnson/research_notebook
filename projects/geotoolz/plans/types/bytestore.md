---
title: ByteStore — defer to `obspec`
subject: Core types
subtitle: We don't ship a Protocol; we pass `obspec.AsyncStore` straight through
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
keywords: design, types, transport, obspec
---

> **Parent:** [README.md](README.md) — Core types.
> **Status:** **revised 2026-05-09** — collapsed from a full Protocol design into a one-page passthrough note.
> See §"Why the rewrite" below.
> **Scope:** how readers in `geotoolz` get bytes from cloud object stores.
> Short answer: we use [`obspec`](https://github.com/developmentseed/obspec) and ship a tiny `open_store(url)` factory.
> We do *not* define our own `ByteStore` Protocol.

---

## Summary

[`obspec`](https://github.com/developmentseed/obspec) is the upstream typed Protocol for object-store byte access — `Store` (sync) and `AsyncStore` (async) with `get`, `get_range`, `get_ranges`, `put`, `list`.
It's by DevSeed, designed around the `obstore` API, and is the same Protocol that [`async-geotiff`](https://github.com/developmentseed/async-geotiff) consumes for its `GeoTIFF.open(path, store=...)` call.

We don't need a parallel Protocol of our own.
Concretely:

- [`AsyncGeoTIFFReader`](../georeader/reader_async_geotiff.md) accepts a `store: obspec.AsyncStore | None` kwarg and forwards it straight to `async_geotiff.GeoTIFF.open(path, store=store)`.
  The Rust core handles range requests, coalescing, decoding off the event loop.
- [`RasterioReader`](../georeader/reader_rasterio.md) is unchanged on the bytes side — it routes through GDAL VSI / fsspec / a user-supplied `opener=` callback.
  There is no `store=` kwarg.
- The [`Credential`](credentials.md) Protocol exposes small `to_obstore_*_store()` helpers that build a real `obstore.S3Store` / `GCSStore` / `AzureStore`.
  Those satisfy `obspec.AsyncStore` because `obstore` is the reference implementation of the `obspec` Protocol.

We ship one helper of our own: a `geotoolz.io.open_store(url, *, prefer="auto")` factory that picks an `obstore` backend from the URL scheme. ~30 LOC; not a Protocol.

---

## Why the rewrite

The earlier draft of this doc proposed a full `ByteStore` Protocol with sync + async method pairs and two adapter classes (`ObstoreByteStore`, `FsspecByteStore`).
On review (2026-05-09), it was clear that:

1. **`obspec` is the upstream Protocol.** Defining our own duplicates the surface and forces consumers to learn two names for the same concept.
   `async-geotiff` already consumes `obspec` natively.
2. **`async-geotiff` does the work, not us.** Earlier drafts of [`reader_async_geotiff.md`](../georeader/reader_async_geotiff.md) planned a private `_cog_helpers.py` module with IFD walk, tile-fetch math, and decompression dispatch. All of that is already in `async-geotiff` (and its Rust dep `async-tiff`).
   Our reader is a thin adapter over `GeoTIFF.open` and `overview.read(window=...)` — there is no place where we need to call `store.get_ranges_async(...)` ourselves.
3. **`fsspec` doesn't need a `ByteStore` adapter from us.** Sync `RasterioReader` already takes an `fs=` shortcut for fsspec backends; that path is untouched.
   We're not asking `RasterioReader` to use `obspec`.
4. **No second async raw-byte reader is on the roadmap.** The original "`AsyncGeoTIFFReader` plus future raw-byte readers will share `ByteStore`" framing was speculative; collapsing the speculative `LazyCOGReader` (see [`georeader/README.md` open question §3](../georeader/README.md#3-deferred-a-sync-gdal-free-geotensor-reader)) removed the only second consumer.

Net: a 419-line design doc became a one-page passthrough note.

---

## What we ship

### `open_store(url, prefer="auto")` — the only helper

```python
# geotoolz/io.py
from typing import Literal

import obspec


def open_store(
    url: str,
    *,
    prefer: Literal["obstore", "auto"] = "auto",
    **backend_kwargs,
) -> obspec.AsyncStore:
    """Build an obspec-conformant store for ``url``.

    Selection:
      "auto"     — obstore for s3:// / gs:// / az:// / http(s):// /
                   file:// / memory://. Raises NotImplementedError on
                   schemes obstore doesn't speak (ftp://, sftp://, …);
                   route those through RasterioReader(fs=fsspec_fs) instead.
      "obstore"  — force obstore; same as auto with explicit intent.
    """
    ...
```

That's the entire `geotoolz` surface for cloud byte access.
`obspec.AsyncStore` is the type; `obstore` is the implementation we recommend; `open_store` is convenience.

### What we do **not** ship

- A `ByteStore` Protocol of our own.
- An `ObstoreByteStore` wrapper (`obstore.S3Store` already satisfies `obspec.AsyncStore` directly).
- An `FsspecByteStore` adapter. fsspec users go through `RasterioReader(fs=fs)` for sync reads.
  There is no async-from-fsspec path through `geotoolz`; if someone needs that later, the right answer is probably `obspec-fsspec` (community shim) or pull `obspec.AsyncStore` shapes directly from fsspec async backends — neither lives in our package.
- A `prefer="fsspec"` mode.
  We don't have a path that consumes fsspec for async reads.

---

## How the readers use it

### `AsyncGeoTIFFReader` — passthrough to `async-geotiff`

```python
from async_geotiff import GeoTIFF
import obspec

from geotoolz.io import open_store


class AsyncGeoTIFFReader(AsyncGeoData):
    def __init__(
        self,
        path_or_url: str,
        *,
        store: obspec.AsyncStore | None = None,
        ...
    ):
        self._store = store or open_store(path_or_url)
        # ...

    @classmethod
    async def open(cls, path_or_url, *, store=None, **kw):
        self = cls(path_or_url, store=store, **kw)
        self._geotiff = await GeoTIFF.open(path_or_url, store=self._store)
        return self
```

No `get_range_async` calls in our code; no semaphore; no decompression dispatch. The Rust core in `async-tiff` handles range coalescing and concurrency.

### `RasterioReader` — unchanged

`RasterioReader` keeps its existing `opener=` / `fs=` / `rio_open_kwargs=` knobs (see [`reader_protocol.md`](../georeader/reader_protocol.md) §"The three bytes paths").
It does **not** consume `obspec`.
Mixing GDAL VSI's libcurl with an `obspec.Store` would mean wrapping the store in a Python file-like callback, which defeats the point — for fast cloud reads use `AsyncGeoTIFFReader`.

---

## Connections to other designs

| Design | How it touches obspec |
|---|---|
| [`georeader/reader_async_geotiff.md`](../georeader/reader_async_geotiff.md) | Accepts `store: obspec.AsyncStore | None`; forwards to `GeoTIFF.open`. Defaults to `open_store(url)`. |
| [`credentials.md`](credentials.md) | `Credential.to_obstore_*_store(...)` helpers build concrete `obstore.S3Store` / `GCSStore` / `AzureStore` — all of which satisfy `obspec.AsyncStore`. |
| [`georeader/reader_rasterio.md`](../georeader/reader_rasterio.md) | Untouched. Sync reader uses GDAL VSI / fsspec / opener; no obspec. |

---

## Open questions

### 1. Do we ever need a sync `obspec.Store` consumer?

`obspec` defines both `Store` (sync) and `AsyncStore` (async).
`AsyncGeoTIFFReader` only needs the async surface.
If a future need arises (e.g. a sync notebook helper that reads one COG chip at a time without entering an event loop), we could add a `sync_open_store(url)` companion.
Not on the roadmap.

### 2. Should `open_store` fall back to fsspec on unsupported schemes?

No (current pick).
If `prefer="auto"` hits `ftp://` it raises and the user is told to use `RasterioReader(fs=...)` instead.
Reason: there's no `obspec` path that wraps fsspec async backends in our package, so silently returning something that doesn't work with `AsyncGeoTIFFReader` is worse than a clear error.
Revisit if a community `obspec-fsspec` shim becomes mature.

### 3. Where does `open_store` live?

Tentative: `geotoolz/io.py`.
Could equally live in `geotoolz/_stores.py` or directly on `AsyncGeoTIFFReader` as a classmethod.
Pick during implementation.
