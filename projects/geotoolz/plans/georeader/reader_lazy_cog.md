# Issue 2 — `LazyCOGReader` + `ByteStore`

> **Parent:** [README.md](README.md)
> **Depends on:** [Issue 1](reader_protocol.md) — `SyncReader` Protocol.
> **Scope:** a sync, COG-only reader that skips GDAL; the `ByteStore` Protocol that makes it agnostic between `obstore` and `fsspec`.

---

## Why this issue exists

`RasterioReader` via GDAL VSI is excellent for general-purpose reads, but for COG-on-cloud hot paths it pays per-call GDAL state and PROJ initialisation that a pure-Python reader can skip. For tile-server fan-out across thousands of small windows, the difference matters.

This issue adds `LazyCOGReader` — same `SyncReader` interface as `RasterioReader`, different bytes path underneath:

- Open is one or two HTTP range requests for the TIFF IFD chain (the file is *not* downloaded).
- Reads issue exactly the range requests for the COG tiles overlapping the requested region; obstore can coalesce close-by ranges into a single HTTP/2 multiplexed call.
- No GDAL state, no PROJ init per call, no Python ↔ C trip per range.

`ByteStore` ships in this issue because it's first needed here. [Issue 3](reader_async_geotiff.md) reuses it for `AsyncGeoTIFFReader`.

---

## Deliverables

1. **`LazyCOGReader` class** in `georeader/lazy_cog_reader.py` (or similar) — implements `SyncReader`.
2. **COG header parsing** — `_fetch_header_sync`, parses the IFD chain into a `COGHeader` dataclass.
3. **Tile fetching** — `_tiles_for_window` math, `_decompress_and_assemble`.
4. **Compression dispatch** — DEFLATE / LZW / JPEG / Zstd / LZMA / none via `imagecodecs`.
5. **Best-effort reprojection** — `read_bounds(target_crs=...)` via `scipy.ndimage.map_coordinates` or `skimage.transform.warp` (no GDAL).
6. **`ByteStore` Protocol** in `georeader/bytestore.py` — sync + async pairs for `get` / `get_range` / `get_ranges` / `put` / `list`.
7. **`ObstoreByteStore` adapter** — wraps `obstore.ObjectStore`.
8. **`FsspecByteStore` adapter** — wraps `fsspec.AbstractFileSystem`.
9. **`open_store(url, prefer="auto")` factory** — auto-pick based on URL scheme.
10. **`__getitem__` numpy-style sugar** on `LazyCOGReader`.

---

## `LazyCOGReader` class

```python
class LazyCOGReader(SyncReader):
    """Lazy COG reader (TIFF/COG only, no GDAL).

    Open is one or two HTTP range requests for the TIFF IFD chain — the
    file is *not* downloaded. Reads are sync (the API matches RasterioReader)
    but every read_window / read_bounds call issues exactly the range
    requests needed for the COG tiles overlapping the requested region.

    No file handle to close; obstore manages connection pooling.
    """
    path_or_url: str
    indexes: tuple[int, ...] | None
    _store: ByteStore                         # constructed from URL if not passed
    _header: "COGHeader"                      # parsed IFD chain (tile offsets, sizes, layout)

    def __init__(
        self,
        path_or_url: str,
        indexes: int | Sequence[int] | None = None,
        *,
        store: ByteStore | None = None,
    ):
        """Sync constructor — fetches the IFD chain immediately."""
        self.path_or_url = path_or_url
        self.indexes = _normalise_indexes(indexes)
        self._store = store or open_store(path_or_url, prefer="auto")
        self._header = self._fetch_header_sync()

    # internal
    def _fetch_header_sync(self) -> "COGHeader":
        """One or two sync range requests to pull and parse the TIFF IFDs."""
        ...
    def _tiles_for_window(self, window: Window) -> list["TileSpec"]:
        """COG tiles overlapping the window: byte offset + length + (i, j) grid pos."""
        ...
    def _decompress_and_assemble(
        self,
        tile_bytes: list[bytes],
        tiles: list["TileSpec"],
        window: Window,
    ) -> np.ndarray:
        """Per-tile DEFLATE/LZW/JPEG decompress, paste into the output window."""
        ...

    # metadata — populated by __init__'s header fetch
    @property
    def crs(self) -> pyproj.CRS: ...
    # ... (the rest of _ReaderMeta surface)

    # reads — same shape as RasterioReader, different bytes path
    def read_window(self, window: Window) -> GeoTensor:
        """Identify overlapping COG tiles, fetch each via store.get_ranges,
        decompress, paste into the windowed array, wrap as GeoTensor."""
        ...
    def read_bounds(self, bounds, *, target_resolution=None, target_crs=None) -> GeoTensor:
        """Bounds → window via inverse transform; reproject via scipy/skimage
        if target_crs differs (no GDAL); read_window."""
        ...
    def read_geoslice(self, slice: GeoSlice) -> GeoTensor: ...
    def load(self) -> GeoTensor:
        """Read every tile. Discouraged — use RasterioReader if you want the whole file."""
        ...

    # numpy-style sugar — defers to read_window
    def __getitem__(self, key: tuple[slice, slice] | tuple[slice, slice, slice]) -> GeoTensor:
        """gt = reader[100:356, 200:456] → exactly the tiles for that bbox."""
        ...

    def close(self) -> None: ...                     # no-op; obstore is pooled
    def __enter__(self) -> "LazyCOGReader": ...
    def __exit__(self, *exc) -> None: ...
```

---

## Header parsing — what happens at construction time

```text
LazyCOGReader("s3://bucket/scene.tif")
        │
        ▼
1. store.get_range(url, 0, 16_384)              # one HTTP range request
        │
        ▼
2. Parse TIFF magic + first IFD pointer
        │
        ▼
3. Walk IFD chain (one or two more range requests if it spills past 16 KB)
        │
        ▼
4. Cache COGHeader: per-IFD tile size, tile_offsets[],
   tile_byte_counts[], compression, dtype, photometric,
   transform, crs (from GeoKeys), nodata
        │
        ▼
self._header = COGHeader(...)                   # ~few KB total fetched
```

A TIFF stores file-level metadata in an *IFD* (Image File Directory). A COG has one IFD per resolution level (full-res + overviews); each IFD lists tile byte offsets and lengths in its `TileOffsets` / `TileByteCounts` tags. Parsing all IFDs lets the reader decide, for any output window, exactly which tiles to fetch and from which IFD (overview level).

---

## Tile fetching — what happens on `read_window(window)`

For COG tile size `(T_x, T_y)` and a window `w = (col_off, row_off, width, height)`:

$$
\begin{aligned}
i \in \,&\bigl[\,\lfloor w_{\text{col\_off}} / T_x \rfloor,\; \lceil (w_{\text{col\_off}} + w_{\text{width}}) / T_x \rceil\,\bigr) \\
j \in \,&\bigl[\,\lfloor w_{\text{row\_off}} / T_y \rfloor,\; \lceil (w_{\text{row\_off}} + w_{\text{height}}) / T_y \rceil\,\bigr)
\end{aligned}
$$

Tile sizes are typically 512×512 or 256×256 px.

```python
def _tiles_for_window(self, window: Window) -> list["TileSpec"]:
    """Compute the (i, j) tile indices that cover the window, then look up
    each tile's byte offset/length from the cached IFD."""
    Tx, Ty = self._header.tile_size                          # px
    i0 = window.col_off // Tx
    i1 = ceil((window.col_off + window.width)  / Tx)
    j0 = window.row_off // Ty
    j1 = ceil((window.row_off + window.height) / Ty)
    n_cols = self._header.n_cols_in_tiles                    # ceil(W / Tx)

    tiles = []
    for j in range(j0, j1):
        for i in range(i0, i1):
            idx = j * n_cols + i
            tiles.append(TileSpec(
                grid=(i, j),
                offset=self._header.tile_offsets[idx],
                length=self._header.tile_byte_counts[idx],
                compression=self._header.compression,
            ))
    return tiles
```

Then the read itself batches every range request into one parallel call:

```python
def read_window(self, window: Window) -> GeoTensor:
    tiles = self._tiles_for_window(window)

    # Batch all tile range requests; obstore can coalesce close-by ranges.
    tile_bytes = self._store.get_ranges(
        self.path_or_url,
        [(t.offset, t.length) for t in tiles],
    )

    # Decompress each tile (DEFLATE / LZW / JPEG / Zstd / none) and paste
    # into the output buffer.
    out = np.empty(self._output_shape(window), dtype=self._header.dtype)
    for t, raw_bytes in zip(tiles, tile_bytes):
        tile_arr = decompress_tile(
            raw_bytes, t.compression, self._header.tile_size, self._header.dtype,
        )
        self._paste_tile_into_window(out, tile_arr, t.grid, window)

    return GeoTensor(
        values=out,
        transform=self._header.window_transform(window),
        crs=self._header.crs,
        fill_value_default=self._header.nodata,
    )
```

---

## Decompression dispatch

Per-tile decompression is dispatched on the `Compression` IFD tag:

| Compression code | Codec | Decoder |
|---|---|---|
| `1` | none | `np.frombuffer(...).reshape(...)` |
| `8` | DEFLATE | `zlib.decompress` |
| `5` | LZW | `imagecodecs.lzw_decode` |
| `7` | JPEG | `imagecodecs.jpeg_decode` |
| `34925` | LZMA | `lzma.decompress` |
| `50000` / `50001` | Zstd | `imagecodecs.zstd_decode` (or `zstandard`) |

`imagecodecs` is the canonical fast decoder for all of these and is usually a hard dep.

---

## Why this is fast

For a 1 GB COG with 256×256 px tiles, a `read_window` for a 1024×1024 px output area touches **at most 25 tiles** (a 5×5 grid). At ~256 KB compressed each, that's ~6 MB of bytes fetched in **one parallel `get_ranges` call**. The other 994 MB of the file is never read.

Compared to:

- **Downloading the whole file:** 1 GB.
- **GDAL VSI for the same window:** also fetches just the right tiles, *but* via 25 sequential range requests (libcurl reuses the connection but doesn't natively coalesce). With HTTP/2 multiplexing this gap narrows.
- **Naive `cat` then slice:** the worst path; never do this.

The win for `LazyCOGReader` over `RasterioReader`-via-VSI is mostly **per-tile Python overhead saved** (no GDAL state, no PROJ init per call) and **range coalescing** when obstore identifies close-by ranges.

---

## What this loses vs `RasterioReader`

- **No GDAL warping.** `read_bounds(target_crs=...)` has to be done in Python — typically `scipy.ndimage.map_coordinates` or `skimage.transform.warp`. Slower and less accurate than GDAL/PROJ for difficult projections.
- **TIFF/COG only.** No JP2, NetCDF, HDF5, GRIB, ENVI. Use `RasterioReader` for those.
- **Limited compression set.** Whatever `imagecodecs` ships. Old `JPEG2000` tiles in some legacy COGs may not decode.
- **No CRS-quirk handling.** GDAL has a long tail of CRS fixes (Web Mercator latitude clamp, ESRI WKT variants, datum-shift grids) that a pure-Python reader doesn't replicate.

The trade is intentional: skip GDAL for speed and concurrency-friendliness, accept narrower scope.

---

## `ByteStore` Protocol

```python
from typing import Iterator, Protocol


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
    async def list_async(self, prefix: str = "") -> Iterator[str]: ...
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

## Transport strategy axis

| Backend | Hot-path throughput | Niche backends | Sync API | Ecosystem fit |
|---|---|---|---|---|
| `ObstoreByteStore` | very high (HTTP/2, parallel ranges) | S3, GCS, Azure, HTTP, file, memory | sync helpers, async-native | new code (zarr 3, async-geotiff, lazy-cogs) |
| `FsspecByteStore` | moderate (per-backend) | everything (FTP, SFTP, GitHub, Dropbox, …) | sync-native, async on capable backends | older code (pandas, xarray, geopandas, zarr ≤ 2) |

Same `ByteStore` protocol, same reader code, two transports underneath. The split is **breadth vs throughput**: pick obstore for hot paths over major clouds, pick fsspec when you need a niche backend.

---

## Acceptance criteria

- `LazyCOGReader` instances satisfy `SyncReader` per static type-check.
- Open a real public COG from S3 (or a httpbin-served test COG); inspect `crs`, `transform`, `bounds`, `shape` — all match GDAL's reading of the same file to within `PIXEL_PRECISION`.
- `read_window(window)` returns a `GeoTensor` matching `RasterioReader.read_window(window)` for the same file (within float tolerance per resampling method).
- `read_bounds(bounds, target_crs=...)` works for the most common reprojection (UTM ↔ Web Mercator, UTM ↔ EPSG:4326).
- Tile-fetch latency: a 1024×1024 read from a 1 GB COG completes in seconds (target: < 5 s on a typical home connection, dominated by network).
- `ByteStore`, `ObstoreByteStore`, `FsspecByteStore`, `open_store` exported from `georeader.bytestore`.
- `LazyCOGReader("s3://...")` works via auto-picked obstore; `LazyCOGReader("ftp://...")` works via fsspec.

---

## Issue-specific open questions

In addition to the [parent design's open questions](README.md#open-questions):

1. **COG helpers location.** This issue ships `_tiles_for_window`, decompression dispatch, and IFD parsing. [Issue 3](reader_async_geotiff.md) reuses them. See parent open question #3 — the working assumption is shared `_cog_helpers.py` module, with `LazyCOGReader` and `AsyncGeoTIFFReader` both importing.
2. **Reprojection fallback choice.** scipy vs skimage vs custom. Both are in the dependency tree elsewhere in the package; either works.
3. **`load()` discouragement strength.** The docstring says "Discouraged — use RasterioReader if you want the whole file." Should this raise a `UserWarning`, or stay docstring-only?
4. **Index validation.** Today's `RasterioReader` uses 1-based indexes (rasterio convention). Should `LazyCOGReader` follow that, or use 0-based (numpy convention)? Following rasterio is less surprising for downstream `geotoolz` operators that already pass band indices.
