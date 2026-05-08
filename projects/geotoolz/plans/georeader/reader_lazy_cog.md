---
title: LazyCOGReader
subject: georeader design
subtitle: Sync, COG-only, GDAL-free reader
short_title: LazyCOG
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, georeader, cog, lazy
---

> **Parent:** [README.md](README.md)
> **Depends on:** [Issue 1](reader_protocol.md) — `SyncReader` Protocol; [`types/bytestore.md`](../types/bytestore.md) — `ByteStore` Protocol.
> **Scope:** a sync, COG-only reader that skips GDAL.

---

## Why this issue exists

`RasterioReader` via GDAL VSI is excellent for general-purpose reads, but for COG-on-cloud hot paths it pays per-call GDAL state and PROJ initialisation that a pure-Python reader can skip. For tile-server fan-out across thousands of small windows, the difference matters.

This issue adds `LazyCOGReader` — same `SyncReader` interface as `RasterioReader`, different bytes path underneath:

- Open is one or two HTTP range requests for the TIFF IFD chain (the file is *not* downloaded).
- Reads issue exactly the range requests for the COG tiles overlapping the requested region; obstore can coalesce close-by ranges into a single HTTP/2 multiplexed call.
- No GDAL state, no PROJ init per call, no Python ↔ C trip per range.

The `ByteStore` Protocol that abstracts obstore vs fsspec is specified separately in [`types/bytestore.md`](../types/bytestore.md). `LazyCOGReader` consumes it; [Issue 3 (`AsyncGeoTIFFReader`)](reader_async_geotiff.md) consumes the same Protocol.

---

## Primer for newcomers

> **ELI5.** A COG is a satellite image organised like a **paper atlas**: a table of contents at the front, then a grid of small page-tiles. To read your area of interest, you flip to the table of contents (a few KB), look up which pages cover it, and jump straight to those pages. You never flip through the rest of the atlas.

### What's a COG (Cloud Optimized GeoTIFF)?

**What it is.** A COG is a regular GeoTIFF organised so that an HTTP client can fetch *just the parts it needs* with byte-range requests. Same `.tif` extension; different on-disk layout.

**How it works.** A standard TIFF stores its metadata (the IFDs — Image File Directories — see below) at the *end* of the file, so to read anything you have to download the tail first. A COG flips this: header at the start, then the image data organised as small tiles (typically 256×256 or 512×512 pixels, each independently compressed). To read a 1024×1024 window from a 1 GB COG, you fetch the header (~few KB), look up which tiles overlap your window, and issue one HTTP range request per tile (or one batched parallel request). No need to download the whole file.

**What this means for us.** COGs are the dominant cloud-native raster format because they make "read a small bbox from a huge file in S3" tractable. `LazyCOGReader` exists to exploit this layout without going through GDAL's overhead. For files that aren't COGs (legacy GeoTIFFs, JP2, NetCDF) you fall back to `RasterioReader`.

### TIFF IFDs and tile offsets

**What it is.** An IFD (Image File Directory) is the TIFF format's metadata block — a list of tags describing one image (its width, height, dtype, compression, and *where the pixel data lives in the file*). A COG has one IFD per resolution level (full-res + each overview).

**How it works.** Each IFD has two arrays of interest: `TileOffsets` (the byte offset of each tile in the file) and `TileByteCounts` (the byte length of each tile). Together they tell you "tile (i, j) of resolution level k starts at byte N and is L bytes long." Given a window, the reader computes which (i, j) tiles overlap, looks up their offsets/lengths, and issues range reads for exactly those bytes.

**What this means for us.** Parsing the IFD chain on `__init__` is what makes the rest of the read flow possible. Tile-fetching code is small (~50 lines of math) once you have the IFD; the rest of `LazyCOGReader` is the IFD parser plus per-tile decompression.

```{mermaid}
sequenceDiagram
    participant App
    participant Reader as LazyCOGReader
    participant Store as ByteStore
    participant Cloud as S3

    Note over App,Cloud: __init__ — parse IFD (one-time, few KB)
    App->>Reader: LazyCOGReader(url)
    Reader->>Store: get_range(0, 16384)
    Store->>Cloud: GET Range bytes 0-16383
    Cloud-->>Store: header bytes
    Store-->>Reader: bytes
    Note over Reader: parse TIFF magic + IFD chain<br/>cache tile_offsets[], tile_byte_counts[]

    Note over App,Cloud: read_window — fetch overlapping tiles
    App->>Reader: read_window(w)
    Note over Reader: compute (i,j) tiles overlapping w
    Reader->>Store: get_ranges([(o1,l1), (o2,l2), ...])
    Store->>Cloud: parallel range requests (HTTP/2)
    Cloud-->>Store: tile bytes
    Store-->>Reader: list[bytes]
    Note over Reader: decompress + paste into output
    Reader-->>App: GeoTensor
```

### HTTP range requests

**What it is.** An HTTP request with a `Range: bytes=N-M` header asks the server for *just* bytes N through M of the resource, not the whole thing. Every modern object store (S3, GCS, Azure, plain HTTPS) honours this.

**How it works.** Standard HTTP feature, supported since HTTP/1.1. The server responds with `206 Partial Content` and just the requested bytes. With HTTP/2 (which obstore uses), many concurrent ranges can multiplex over one TCP connection — fetching 25 tiles in parallel costs not much more than fetching one.

**What this means for us.** A `LazyCOGReader.read_window(...)` for a 1024×1024 area on a 1 GB COG might fetch ~6 MB across 25 tiles in one parallel call. The other 994 MB is never read. This is what makes "100k random chips across 50k COGs" feasible in seconds rather than infeasible at all.

### Compression dispatch

**What it is.** COG tiles are individually compressed — typically with DEFLATE (zlib), LZW, JPEG, or Zstd. The reader has to decompress each tile after fetching its bytes.

**How it works.** Each IFD has a `Compression` tag (an integer code: `8` = DEFLATE, `5` = LZW, `7` = JPEG, `50000` = Zstd, etc.). The reader dispatches on this code to the right decoder — `zlib.decompress`, `imagecodecs.lzw_decode`, etc. After decompression, each tile is a small numpy array that gets pasted into the output window.

**What this means for us.** The compression code is fixed once per IFD (so the dispatch happens once per file at open, not per tile), and `imagecodecs` handles all the common codecs in C. Compression-dispatch isn't a performance bottleneck; the network is.

```{mermaid}
flowchart TD
    Start[tile bytes + Compression tag]
    Start --> Q{Compression code}
    Q -->|1| None[np.frombuffer + reshape]
    Q -->|8| Deflate[zlib.decompress]
    Q -->|5| LZW[imagecodecs.lzw_decode]
    Q -->|7| JPEG[imagecodecs.jpeg_decode]
    Q -->|34925| LZMA[lzma.decompress]
    Q -->|50000 / 50001| Zstd[imagecodecs.zstd_decode]
    None --> Out[tile ndarray]
    Deflate --> Out
    LZW --> Out
    JPEG --> Out
    LZMA --> Out
    Zstd --> Out
```

---

## Deliverables

1. **`LazyCOGReader` class** in `georeader/lazy_cog_reader.py` (or similar) — implements `SyncReader`.
2. **COG header parsing** — `_fetch_header_sync`, parses the IFD chain into a `COGHeader` dataclass.
3. **Tile fetching** — `_tiles_for_window` math, `_decompress_and_assemble`.
4. **Compression dispatch** — DEFLATE / LZW / JPEG / Zstd / LZMA / none via `imagecodecs`.
5. **Best-effort reprojection** — `read_bounds(target_crs=...)` via `scipy.ndimage.map_coordinates` or `skimage.transform.warp` (no GDAL).
6. **`__getitem__` numpy-style sugar** on `LazyCOGReader`.

The `ByteStore` Protocol + `ObstoreByteStore` / `FsspecByteStore` adapters + `open_store(url)` factory are specified in [`types/bytestore.md`](../types/bytestore.md) and ship as part of that work. This issue depends on them but doesn't own them.

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
    def read_geoslice(self, slice_: GeoSlice) -> GeoTensor: ...
    def load(self) -> GeoTensor:
        """Read every tile. Discouraged — use RasterioReader if you want the whole file."""
        ...

    # numpy-style sugar — defers to read_window
    def __getitem__(self, key: tuple[slice, slice] | tuple[slice, slice, slice]) -> GeoTensor:
        """gt = reader[100:356, 200:456] → exactly the tiles for that bbox."""
        ...

    def close(self) -> None: ...                     # no-op; obstore is pooled
    def __enter__(self) -> "LazyCOGReader": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None: ...
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

## Transport via `ByteStore`

`LazyCOGReader` accepts a `store: ByteStore | None = None` constructor kwarg. The `ByteStore` Protocol — its sync + async method pairs, the `ObstoreByteStore` and `FsspecByteStore` adapters, and the `open_store(url, prefer="auto")` factory — is specified separately in [`types/bytestore.md`](../types/bytestore.md) since it's also consumed by [`AsyncGeoTIFFReader`](reader_async_geotiff.md) and conceivably by any future raw-byte-shaped reader.

When `store=None`, `LazyCOGReader` calls `open_store(url, prefer="auto")` — obstore for `s3://` / `gs://` / `az://` / `http(s)://` / `file://` / `memory://`, fsspec for niche backends (`ftp://`, `sftp://`, `github://`, …). Override via `store=` to force a specific transport or to inject pre-configured credentials.

For the obstore-vs-fsspec comparison (HTTP backend, async story, install footprint, ecosystem fit) and the decision tree, see [`geostack.md` §"`obstore` vs `fsspec` compared"](../geostack.md#obstore-vs-fsspec-compared).

---

## Acceptance criteria

- `LazyCOGReader` instances satisfy `SyncReader` per static type-check.
- Open a real public COG from S3 (or a httpbin-served test COG); inspect `crs`, `transform`, `bounds`, `shape` — all match GDAL's reading of the same file to within `PIXEL_PRECISION`.
- `read_window(window)` returns a `GeoTensor` matching `RasterioReader.read_window(window)` for the same file (within float tolerance per resampling method).
- `read_bounds(bounds, target_crs=...)` works for the most common reprojection (UTM ↔ Web Mercator, UTM ↔ EPSG:4326).
- Tile-fetch latency: a 1024×1024 read from a 1 GB COG completes in seconds (target: < 5 s on a typical home connection, dominated by network).
- `LazyCOGReader("s3://...")` works via auto-picked obstore; `LazyCOGReader("ftp://...")` works via fsspec (both via `ByteStore`).

---

## Issue-specific open questions

In addition to the [parent design's open questions](README.md#open-questions):

1. **COG helpers location.** This issue ships `_tiles_for_window`, decompression dispatch, and IFD parsing. [Issue 3](reader_async_geotiff.md) reuses them. See parent open question #2 — the working assumption is shared `_cog_helpers.py` module, with `LazyCOGReader` and `AsyncGeoTIFFReader` both importing.
2. **Reprojection fallback choice.** scipy vs skimage vs custom. Both are in the dependency tree elsewhere in the package; either works.
3. **`load()` discouragement strength.** The docstring says "Discouraged — use RasterioReader if you want the whole file." Should this raise a `UserWarning`, or stay docstring-only?
4. **Index validation.** Today's `RasterioReader` uses 1-based indexes (rasterio convention). Should `LazyCOGReader` follow that, or use 0-based (numpy convention)? Following rasterio is less surprising for downstream `geotoolz` operators that already pass band indices.
