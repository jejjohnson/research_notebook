
# Reader reconciliation — `RasterioReader`, `LazyCOGReader`, `AsyncGeoTIFFReader`

> **Scope:** rough class-level signatures showing how the three readers share one metadata surface and split into sync vs async read interfaces. Same level of granularity as the `GeoSlice` / `GeoCatalog` design sketches — types and method shapes, not full implementations.
>
> **Status:** reference document, design-shaped. The exact field names, transports, and decoder choices may differ in the actual implementations.

---

## Overview

Three readers, one shared metadata surface, two read interfaces:

| Reader | Lives in | Sync / async | Transport | Driver |
| --- | --- | --- | --- | --- |
| `RasterioReader` | `georeader` | sync | GDAL / VSI | every GDAL driver |
| `LazyCOGReader` | external (`lazy-cogs`) | sync API, lazy semantics | `obstore` / `fsspec` | TIFF / COG only |
| `AsyncGeoTIFFReader` | external (`async-geotiff`) | async | `obstore` | TIFF / COG only |

The metadata properties (`crs`, `transform`, `bounds`, `shape`, `count`, `dtype`, `nodata`, `res`) and the read-method names (`read_window`, `read_bounds`, `read_geoslice`, `load`) are identical across all three. The only divergence is whether reads are sync or async.

---

## Shared protocol

```python
from typing import Protocol, Sequence
import numpy as np
import pyproj
from rasterio import Affine
from rasterio.windows import Window
from georeader.geotensor import GeoTensor
from georeader.geoslice import GeoSlice


class _ReaderMeta(Protocol):
    """Metadata surface shared by every georeader reader.

    All properties are cheap after construction (header-only). Subclasses
    decide *when* the header is fetched — eagerly in __init__ (sync readers)
    or via an async classmethod (async readers).
    """
    path_or_url: str
    indexes: tuple[int, ...] | None        # bands to read; None = all

    @property
    def crs(self) -> pyproj.CRS: ...
    @property
    def transform(self) -> Affine: ...
    @property
    def bounds(self) -> tuple[float, float, float, float]: ...   # (xmin, ymin, xmax, ymax)
    @property
    def shape(self) -> tuple[int, int, int]: ...                  # (count, height, width)
    @property
    def count(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def nodata(self) -> float | None: ...
    @property
    def res(self) -> tuple[float, float]: ...                     # (x_res, y_res)


class SyncReader(_ReaderMeta, Protocol):
    """Sync read interface — RasterioReader and LazyCOGReader."""

    def read_window(self, window: Window) -> GeoTensor: ...
    def read_bounds(
        self,
        bounds: tuple[float, float, float, float],
        *,
        target_resolution: tuple[float, float] | None = None,
        target_crs: pyproj.CRS | str | None = None,
    ) -> GeoTensor: ...
    def read_geoslice(self, slice: GeoSlice) -> GeoTensor: ...
    def load(self) -> GeoTensor: ...
    def close(self) -> None: ...

    def __enter__(self) -> "SyncReader": ...
    def __exit__(self, *exc) -> None: ...


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

Both `SyncReader` and `AsyncReader` share the entire metadata surface. The only thing that diverges is whether the read methods return a `GeoTensor` or a `Coroutine[GeoTensor]`.

---

## `RasterioReader` (sync, GDAL-backed) — lives in `georeader`

```python
class RasterioReader(SyncReader):
    """Sync, GDAL-backed reader. The default in georeader.

    Opens a rasterio dataset on construction (cheap — header + IFD). The
    file handle is cached for the lifetime of the reader so repeated reads
    don't pay the open cost. Closes on context-exit or explicit .close().

    Reads happen via rasterio.read(window=...). The bytes path *under*
    the rasterio call has three modes — see "Inside RasterioReader" below:

      1. opener=None and fs=None  → GDAL VSI (libcurl in C); the default.
                                     Cloud paths /vsis3/, /vsigs/, /vsiaz/, /vsicurl/.
      2. opener=callable          → GDAL calls the callable for each byte range.
      3. fs=fsspec_filesystem     → shortcut: equivalent to opener=fs.open.

    On-the-fly reprojection in read_bounds() is done via rasterio.warp.WarpedVRT.
    """
    path_or_url: str
    indexes: tuple[int, ...] | None
    _dataset: "rasterio.DatasetReader | None"        # cached open handle
    _opener: "Callable[[str, str], BinaryIO] | None"
    _fs: "fsspec.AbstractFileSystem | None"
    _rio_open_kwargs: dict

    def __init__(
        self,
        path_or_url: str,
        indexes: int | Sequence[int] | None = None,
        *,
        opener: "Callable[[str, str], BinaryIO] | None" = None,    # custom file opener
        fs: "fsspec.AbstractFileSystem | None" = None,              # shortcut: opener=fs.open
        rio_open_kwargs: dict | None = None,                        # other rasterio.open kwargs
    ): ...

    # internal
    def _ensure_open(self) -> "rasterio.DatasetReader":
        """Open the dataset if not already open; return the handle."""
        ...

    # metadata — straight passthrough to rasterio dataset attrs
    @property
    def crs(self) -> pyproj.CRS: ...
    # ... (the rest of _ReaderMeta surface)

    # reads
    def read_window(self, window: Window) -> GeoTensor:
        """ds.read(indexes=..., window=...) → GeoTensor with windowed transform."""
        ...
    def read_bounds(self, bounds, *, target_resolution=None, target_crs=None) -> GeoTensor:
        """Wrap in WarpedVRT if target_crs differs from native; window the
        VRT to the requested bounds; read."""
        ...
    def read_geoslice(self, slice: GeoSlice) -> GeoTensor:
        """Convenience: read_bounds(slice.bounds, target_resolution=slice.resolution,
                                    target_crs=slice.crs)."""
        ...
    def load(self) -> GeoTensor: ...
    def close(self) -> None: ...
    def __enter__(self) -> "RasterioReader": ...
    def __exit__(self, *exc) -> None: ...
```

### Inside `RasterioReader` — the three bytes paths

`RasterioReader` ultimately delegates to `rasterio.open(...).read(window=...)`. Under that delegation, GDAL's byte-fetching loop can run in three modes depending on what the constructor was given:

```text
              ┌──────────────────────┐
              │   RasterioReader     │
              │   __init__(...)      │
              └──────────┬───────────┘
                         │
              ┌──────────┼──────────────────┐
              ▼          ▼                  ▼
       opener=None  opener=fs.open   opener=<custom>
       fs=None      (or fs=…)        (e.g. obstore-aware)
            │              │                   │
            ▼              ▼                   ▼
       ┌──────────┐ ┌──────────────┐ ┌──────────────────┐
       │ GDAL VSI │ │ Python       │ │ Python adapter   │
       │ (libcurl)│ │ file-like    │ │ over obstore     │
       │  in C    │ │ via fsspec   │ │ get_range        │
       └────┬─────┘ └──────┬───────┘ └────────┬─────────┘
            │              │                  │
            └──────────────┴──────────────────┘
                           │
                           ▼
                  S3 / GCS / Azure / …
```

#### Path resolution in `__init__`

```python
def _resolve_open_kwargs(self) -> dict:
    """Translate the constructor's opener/fs knobs into rasterio.open kwargs."""
    kwargs = dict(self._rio_open_kwargs or {})
    if self._opener is not None:
        kwargs["opener"] = self._opener
    elif self._fs is not None:                       # fs= shortcut
        kwargs["opener"] = self._fs.open
    # else: no opener key → rasterio uses GDAL VSI for cloud paths
    return kwargs

def _ensure_open(self) -> "rasterio.DatasetReader":
    if self._dataset is None or self._dataset.closed:
        self._dataset = rasterio.open(
            self.path_or_url, **self._resolve_open_kwargs(),
        )
    return self._dataset
```

#### Per-path summary

| Path | Trigger | Bytes fetched by | Speed | Driver coverage |
| --- | --- | --- | --- | --- |
| **GDAL VSI** | `opener=None`, `fs=None` (default) | libcurl inside GDAL | fastest sync option | every GDAL driver |
| **fsspec** | `fs=fsspec_fs` or `opener=fs.open` | Python file-like via fsspec; GDAL calls back per range | slower (Python ↔ C trip per range) | every GDAL driver, Python boundary added |
| **obstore** (custom callback) | `opener=<callback wrapping obstore.get_range>` | Python adapter over `obstore.ObjectStore` | similar to fsspec — same Python ↔ C bottleneck | every GDAL driver, but only for what obstore can serve |

#### Usage

```python
# Default — GDAL VSI handles s3:// directly; fastest option
reader = RasterioReader("s3://bucket/scene.tif")

# fsspec shortcut — for niche backends or custom auth
import fsspec
fs = fsspec.filesystem(
    "s3", endpoint_url="https://my-minio:9000", key=..., secret=...,
)
reader = RasterioReader("s3://bucket/scene.tif", fs=fs)

# Equivalent: explicit opener
reader = RasterioReader(
    "s3://bucket/scene.tif",
    rio_open_kwargs={"opener": fs.open},
)

# obstore via custom callable — possible but rarely the right tool;
# at this point you'd use LazyCOGReader or AsyncGeoTIFFReader instead
def obstore_opener(path: str, mode: str) -> "BinaryIO":
    """Return a file-like wrapping obstore range fetches."""
    ...
reader = RasterioReader("s3://bucket/scene.tif", opener=obstore_opener)
```

The three options share the metadata surface and the public API — **only the bytes path differs underneath**, and `_resolve_open_kwargs` is the only Python code that knows which path is active.

---

## `LazyCOGReader` (sync API, lazy-on-open + on-slice fetch)

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
    _store: "obstore.ObjectStore"            # constructed from URL if not passed
    _header: "COGHeader"                      # parsed IFD chain (tile offsets, sizes, layout)

    def __init__(
        self,
        path_or_url: str,
        indexes: int | Sequence[int] | None = None,
        *,
        store: "obstore.ObjectStore | None" = None,
    ):
        """Sync constructor — fetches the IFD chain immediately."""
        ...

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
        """Identify overlapping COG tiles, fetch each via store.get_range_sync,
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

### Inside `LazyCOGReader` — IFD parsing and tile fetching

The pure-Python reader has two halves: **header parsing** (once, on `__init__`) and **tile fetching** (once per `read_*` call). Both are sync; the async sibling is `AsyncGeoTIFFReader` below.

#### Header parsing — what happens at construction time

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

#### Tile fetching — what happens on `read_window(window)`

Window-to-tiles mapping. For COG tile size `(T_x, T_y)` and a window `w = (col_off, row_off, width, height)`:

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

#### Decompression

Per-tile decompression is dispatched on the `Compression` IFD tag:

| Compression code | Codec | Decoder |
| --- | --- | --- |
| `1` | none | `np.frombuffer(...).reshape(...)` |
| `8` | DEFLATE | `zlib.decompress` |
| `5` | LZW | `imagecodecs.lzw_decode` |
| `7` | JPEG | `imagecodecs.jpeg_decode` |
| `34925` | LZMA | `lzma.decompress` |
| `50000` / `50001` | Zstd | `imagecodecs.zstd_decode` (or `zstandard`) |

`imagecodecs` is the canonical fast decoder for all of these and is usually a hard dep.

#### Why this is fast

For a 1 GB COG with 256×256 px tiles, a `read_window` for a 1024×1024 px output area touches **at most 25 tiles** (a 5×5 grid). At ~256 KB compressed each, that's ~6 MB of bytes fetched in **one parallel `get_ranges` call**. The other 994 MB of the file is never read.

Compare to:

- **Downloading the whole file:** 1 GB.
- **GDAL VSI for the same window:** also fetches just the right tiles, *but* via 25 sequential range requests (libcurl reuses the connection but doesn't natively coalesce). With HTTP/2 multiplexing this gap narrows.
- **Naive `cat` then slice:** the worst path; never do this.

The win for `LazyCOGReader` over `RasterioReader`-via-VSI is mostly **per-tile Python overhead saved** (no GDAL state, no PROJ init per call) and **range coalescing** when obstore identifies close-by ranges.

#### What this loses vs `RasterioReader`

- **No GDAL warping.** `read_bounds(target_crs=...)` has to be done in Python — typically `scipy.ndimage.map_coordinates` or `skimage.transform.warp`. Slower and less accurate than GDAL/PROJ for difficult projections.
- **TIFF/COG only.** No JP2, NetCDF, HDF5, GRIB, ENVI. Use `RasterioReader` for those.
- **Limited compression set.** Whatever `imagecodecs` ships. Old `JPEG2000` tiles in some legacy COGs may not decode.
- **No CRS-quirk handling.** GDAL has a long tail of CRS fixes (Web Mercator latitude clamp, ESRI WKT variants, datum-shift grids) that a pure-Python reader doesn't replicate.

The trade is intentional: skip GDAL for speed and concurrency-friendliness, accept narrower scope.

---

## `AsyncGeoTIFFReader` (async, obstore-backed)

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
    _store: "obstore.ObjectStore"
    _header: "COGHeader | None"                      # populated after .open()
    _max_concurrent_tiles: int

    def __init__(
        self,
        path_or_url: str,
        indexes: int | Sequence[int] | None = None,
        *,
        store: "obstore.ObjectStore | None" = None,
        max_concurrent_tiles: int = 32,
    ):
        """Cheap. Does NOT fetch the header — call .open() first."""
        ...

    @classmethod
    async def open(
        cls,
        path_or_url: str,
        indexes: int | Sequence[int] | None = None,
        *,
        store: "obstore.ObjectStore | None" = None,
        max_concurrent_tiles: int = 32,
    ) -> "AsyncGeoTIFFReader":
        """Async constructor: build instance, fetch and parse the IFD chain.
        Most users call this rather than __init__."""
        ...

    # internal
    async def _fetch_header(self) -> None:
        """One or two async range requests to pull and parse the TIFF IFDs."""
        ...
    def _tiles_for_window(self, window: Window) -> list["TileSpec"]: ...
    def _decompress_and_assemble(
        self,
        tile_bytes: list[bytes],
        tiles: list["TileSpec"],
        window: Window,
    ) -> np.ndarray: ...

    # metadata — sync after .open() has been awaited
    @property
    def crs(self) -> pyproj.CRS:
        """Raises RuntimeError if .open() hasn't been awaited yet."""
        ...
    # ... (the rest of _ReaderMeta surface)

    # reads — same shape as the sync readers, but coroutines
    async def read_window(self, window: Window) -> GeoTensor:
        """tiles = self._tiles_for_window(window)
           bytes_list = await asyncio.gather(*[
               self._store.get_range_async(self.path_or_url, t.offset, t.length)
               for t in tiles
           ])  # parallel
           return self._decompress_and_assemble(bytes_list, tiles, window) → GeoTensor."""
        ...
    async def read_bounds(self, bounds, *, target_resolution=None, target_crs=None) -> GeoTensor: ...
    async def read_geoslice(self, slice: GeoSlice) -> GeoTensor: ...
    async def load(self) -> GeoTensor:
        """Fetches every tile in parallel. Use sparingly."""
        ...

    async def aclose(self) -> None: ...               # no-op; obstore is pooled
    async def __aenter__(self) -> "AsyncGeoTIFFReader": ...
    async def __aexit__(self, *exc) -> None: ...
```

---

## How they're swappable in `geotoolz`

Because the metadata surface and read-method names are identical, downstream code only branches on sync vs async:

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

---

## Strategy axis at a glance

| Reader | Open cost | Read cost (small bbox) | Concurrent reads | Driver coverage |
| --- | --- | --- | --- | --- |
| `RasterioReader` | header + IFD via GDAL | one VSI call | sync (threadpool) | every GDAL driver |
| `LazyCOGReader` | one or two range requests | one tile-batch fetch | sync (threadpool) | TIFF/COG only |
| `AsyncGeoTIFFReader` | one or two async range requests | parallel tile fetch | native asyncio | TIFF/COG only |

Same metadata surface, same `read_*` method names, three different bytes paths underneath. The only tax on swapping is `await` — which is unavoidable as long as the cloud HTTP world is fundamentally async.

---

## Transport reconciliation — `obstore` vs `fsspec`

One layer below the readers, the bytes themselves are fetched by one of two transports. Both can serve the same use case (range reads from S3 / GCS / Azure) but they have very different shapes:

| Library | API style |
| --- | --- |
| `obstore` | Object store: `store.get(key)`, `store.get_range(key, off, len)`, `store.put(key, data)`, `store.list(prefix)`. Async-native. |
| `fsspec` | Filesystem: `fs.open(path, "rb").seek(off).read(n)`, `fs.cat(path)`, `fs.glob(pattern)`. Sync-native, async via `asynchronous=True`. |

For COG-tile-shaped reads (`get_range` / read at offset N for length L), both libraries can serve the same need. A unified `ByteStore` Protocol lets the readers stay agnostic.

### The shared protocol

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

### `ObstoreByteStore` — wraps `obstore.ObjectStore`

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

### `FsspecByteStore` — wraps an `fsspec.AbstractFileSystem`

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

### Unified factory

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

### How readers consume it

Every reader that needs cloud byte access takes an optional `store: ByteStore | None = None`. If `None`, the reader calls `open_store(url)` internally:

```python
class LazyCOGReader(SyncReader):
    def __init__(
        self,
        path_or_url: str,
        indexes: int | Sequence[int] | None = None,
        *,
        store: ByteStore | None = None,
    ):
        self.path_or_url = path_or_url
        self._store = store or open_store(path_or_url, prefer="auto")
        # ...

    def read_window(self, window):
        tiles = self._tiles_for_window(window)
        # one call into the unified protocol — adapter handles obstore vs fsspec
        tile_bytes = self._store.get_ranges(
            self.path_or_url,
            [(t.offset, t.length) for t in tiles],
        )
        return self._decompress_and_assemble(tile_bytes, tiles, window)


class AsyncGeoTIFFReader(AsyncReader):
    def __init__(
        self,
        path_or_url: str,
        indexes: int | Sequence[int] | None = None,
        *,
        store: ByteStore | None = None,
        max_concurrent_tiles: int = 32,
    ):
        # same as above, async path
        ...

    async def read_window(self, window):
        tiles = self._tiles_for_window(window)
        tile_bytes = await self._store.get_ranges_async(
            self.path_or_url,
            [(t.offset, t.length) for t in tiles],
        )
        return self._decompress_and_assemble(tile_bytes, tiles, window)
```

### The strategy axis

| Backend | Hot-path throughput | Niche backends | Sync API | Ecosystem fit |
| --- | --- | --- | --- | --- |
| `ObstoreByteStore` | very high (HTTP/2, parallel ranges) | S3, GCS, Azure, HTTP, file, memory | sync helpers, async-native | new code (zarr 3, async-geotiff, lazy-cogs) |
| `FsspecByteStore` | moderate (per-backend) | everything (FTP, SFTP, GitHub, Dropbox, …) | sync-native, async on capable backends | older code (pandas, xarray, geopandas, zarr ≤ 2) |

Same `ByteStore` protocol, same reader code, two transports underneath. The only thing that differs is which compiled artefact handles `GET /bucket/key Range: bytes=offset-end`.
