# Ch. 12 — `save`

> **Module:** `georeader/save.py` (586 LOC; the empty `save_cog.py` is a deprecated stub)
> **Role:** the export side. Take a `GeoTensor` (or anything `GeoData`-shaped), write it to disk as a tiled GeoTIFF or a Cloud-Optimized GeoTIFF (COG). Handles cloud-storage destinations (`gs://`, `s3://`, `az://`) transparently.

---

## 1. Why COGs

A traditional GeoTIFF stores data as a single big image, with the header at the *end* of the file. To read any part of it you need to first download enough bytes to find the header, then seek into the data. Over HTTP this is awful — you pay for the round-trip on every read.

A Cloud Optimized GeoTIFF (COG) reorganises the same byte stream so:

- The **header lives at the start**. A few KB of HTTP range-read tells you everything about the file.
- Data is **tiled** (default 256×256 blocks). To read one tile you fetch one tile's bytes — not the whole file.
- **Overviews** (decimated pyramid layers — 2×, 4×, 8× downsampled copies) are interleaved at the start. Want a thumbnail? Read overview 8 — a tiny number of bytes.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│              CLOUD OPTIMIZED GEOTIFF STRUCTURE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Traditional GeoTIFF            Cloud Optimized GeoTIFF (COG)           │
│  ────────────────────           ─────────────────────────────           │
│                                                                          │
│  ┌─────────────────┐            ┌─────────────────┐                     │
│  │ Header (end)    │            │ Header (start)  │ ← HTTP range 0-N   │
│  │                 │            │ ─────────────── │                     │
│  │                 │            │ Overview 8x     │ ← Pyramid layers   │
│  │    Full        │            │ Overview 4x     │   for fast zoom    │
│  │    Resolution  │            │ Overview 2x     │                     │
│  │    Data        │            │ ─────────────── │                     │
│  │                 │            │ ┌───┬───┬───┐  │ ← Tiled structure  │
│  │                 │            │ │256│256│256│  │   (default 256x256)│
│  │                 │            │ ├───┼───┼───┤  │                     │
│  │                 │            │ │256│256│256│  │                     │
│  └─────────────────┘            │ └───┴───┴───┘  │                     │
│                                 └─────────────────┘                     │
│                                                                          │
│  Benefits:                                                               │
│  • Read any region without downloading whole file                       │
│  • Fast preview via overviews (pyramid layers)                          │
│  • Efficient streaming for web mapping applications                     │
│  • Compatible with all GeoTIFF readers                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

The cost: COG files are slightly larger than minimal GeoTIFFs (overviews add ~33% on top of full resolution). The benefit: for any file you'll touch over HTTP, COG is transformative — `RasterioReader("https://...cog.tif")` becomes interactive.

The practical rule: **save everything as COG by default unless you have a reason not to**.

---

## 2. Compression options

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPRESSION RECOMMENDATIONS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Compression    Best For                    Speed    Size                │
│  ───────────    ────────                    ─────    ────                │
│  lzw           General purpose (DEFAULT)    Fast     Medium             │
│  deflate       Better compression ratio     Medium   Small              │
│  zstd          Modern, fast + small         Fast     Small              │
│  lzma          Maximum compression          Slow     Smallest           │
│  jpeg          RGB imagery (lossy)          Fast     Tiny               │
│  webp          RGB imagery (lossy/lossless) Fast     Tiny               │
│  none          Already compressed data      N/A      Original           │
│                                                                          │
│  For scientific data: Use lzw or zstd (lossless)                        │
│  For visualization: Consider jpeg or webp (lossy ok)                    │
│  For integer masks: Use deflate with predictor=2                        │
└─────────────────────────────────────────────────────────────────────────┘
```

The package default is **`lzw`** — a sensible "fast and lossless" choice that's universally supported by GeoTIFF readers (going back decades). Three other choices show up often:

- **`zstd`** — modern, faster than lzw, smaller files. Recommended whenever your downstream readers support it (rasterio + GDAL ≥ 3.0 do; older proprietary tools may not). Probably the best default for new projects.
- **`deflate` with `predictor=2`** — ideal for integer masks and class-label rasters. The predictor encodes pixel-to-pixel differences before deflate, giving 5–10× better ratios on data that varies smoothly.
- **`jpeg` / `webp`** — *lossy*. Use only for visualisation outputs where you don't care about exact pixel values. Never for ML training data.

Override via `profile_arg={"compress": "zstd"}`.

---

## 3. Data type mapping

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  NumPy dtype     GeoTIFF dtype    Notes                                 │
│  ────────────    ─────────────    ─────                                 │
│  uint8           Byte             8-bit unsigned (0-255)                │
│  uint16          UInt16           16-bit unsigned (0-65535)             │
│  int16           Int16            16-bit signed                         │
│  uint32          UInt32           32-bit unsigned                       │
│  int32           Int32            32-bit signed                         │
│  float32         Float32          32-bit floating point                 │
│  float64         Float64          64-bit floating point                 │
│  complex64       CFloat32         Complex 32-bit                        │
│  bool            Byte             Converted to 0/1                      │
└─────────────────────────────────────────────────────────────────────────┘
```

The `bool → Byte` conversion is the only sneaky one — booleans are written as `uint8` `{0, 1}` because GeoTIFF has no boolean type. Reading them back gives a `uint8` array, and you'd usually `> 0` to convert it back to bool. The `validmask`/`invalidmask` `GeoTensor` properties (Chapter 1) are aware of this.

`profile_from_dtype(dtype)` gives you the rasterio profile dict for any of these — used internally by `save_cog`.

---

## 4. The two save functions

### `save_cog(data_save, path_tiff_save, descriptions=None, tags=None, profile_arg=None, ...)`

The standard export. Steps:

1. Materialise `data_save` to an in-memory ndarray (calls `.load()` if it's a reader).
2. Build a profile from the dtype + the COG defaults (LZW, 256×256 blocks, BIGTIFF if needed).
3. Apply user-supplied `profile_arg` overrides.
4. Write a tiled GeoTIFF.
5. Build overviews via `rasterio.rio.overview` (powers of 2 down to ≤256 pixels).
6. **Reorganise** the file via `rasterio_shutil.copy(..., COG_DRIVER)` so the header and overviews land at the start.

The two-pass approach (write, then reorganise) is what GDAL's COG driver does internally — this module just orchestrates it via rasterio.

### `save_tiled_geotiff(data_save, path_tiff_save, ...)`

The "no overviews" variant. Same profile, same compression, same tiling, but skips the overview-building and reorganisation steps. Cheaper to write; you can't preview at low res over HTTP. Use when:

- Output is already a tile of a larger product and the consumer never zooms.
- You're producing intermediate files that get re-read locally.
- You'll add overviews later via `gdaladdo` once the file is finalised.

Source: [save.py:163 (`save_tiled_geotiff`)](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/save.py#L163), [save.py:327 (`save_cog`)](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/save.py#L327).

---

## 5. Cloud destinations

`save_cog("gs://bucket/key.tif", ...)` works. The implementation:

- Detects the prefix (`gs://`, `s3://`, `az://`, `abfs://`, `oss://`, `http://`, `https://`).
- Writes locally to a temp file.
- Uploads via fsspec (or a user-supplied `fs` filesystem object) once writing is complete.
- Cleans up the temp file.

The constant `REMOTE_FILE_EXTENSIONS` at the top of the module is the prefix list. Extending support to a new cloud (`oss://` etc.) is mostly a one-line addition there + ensuring the right fsspec backend is installed.

For HTTP destinations, the upload phase requires a destination that accepts PUTs (presigned S3 URLs work; static HTTP URLs do not). Most "I want to write to https://..." workflows mean "actually I want to write to s3://, then serve via https://" — pass the `s3://` URL.

The `fs` parameter is the escape hatch for custom auth: pass `fs=gcsfs.GCSFileSystem(token=...)` to use a non-default credential.

---

## 6. The signature

```python
save_cog(
    data_save,                          # GeoData (GeoTensor or RasterioReader)
    path_tiff_save,                     # str — local path or gs://, s3://, etc.
    descriptions=None,                  # list[str] — per-band names
    tags=None,                          # dict[str, str] — file-level metadata
    profile_arg=None,                   # dict — overrides the default profile
    fs=None,                            # fsspec filesystem for cloud writes
    nodata=None,                        # override fill_value_default
)
```

A few non-obvious points:

- **`descriptions`** maps to `rasterio`'s per-band `descriptions` field — visible in `gdalinfo`, used by some viewers as the band-picker label. Highly recommended (`["Red", "Green", "Blue"]` etc.) for any file you ship.
- **`tags`** is file-level GDAL metadata, written into the GeoTIFF's GDAL_METADATA tag. Use for provenance: `{"source": "Sentinel-2", "date": "2024-01-15", "processing": "ndvi"}`. Survives round-trip through `RasterioReader.tags()`.
- **`profile_arg={"compress": "zstd", "predictor": 2}`** is the override hatch. The full list of COG-relevant profile keys: `compress`, `blockxsize`, `blockysize`, `predictor`, `BIGTIFF`, `TILED`, `COPY_SRC_OVERVIEWS`.
- **`nodata=None`** means "use `data_save.fill_value_default`". Pass explicitly when your output should have a different nodata than the input (e.g., `nodata=255` for a uint8 mask).

---

## 7. Idiomatic usage

**Save an in-memory `GeoTensor` as a COG with band names and provenance:**

```python
save.save_cog(
    ndvi_gt,
    "output/ndvi_2024_06.tif",
    descriptions=["NDVI"],
    tags={"source": "S2_L2A", "tile": "T29SND", "date": "2024-06-15"},
    profile_arg={"compress": "zstd"},
)
```

**Stream a large lazy reader to cloud storage without ever materialising in full:**

```python
save.save_cog(
    rasterio_reader_for_huge_scene,
    "gs://bucket/output.tif",
)
```

The function does call `.load()` internally (step 1 of section 4), so this still allocates the full array briefly — the "without ever materialising" claim above is wrong as the module currently stands. For genuinely streaming I/O on truly huge files, fall back to manually iterating windows and calling `dataset.write(window=...)` per chunk.

---

## 8. Internal helpers worth knowing

- **`_add_overviews(rst_out, tile_size, verbose=False)`** — iterates over decimation factors (2, 4, 8, 16, ...) and stops once an overview would be ≤ `tile_size`. Public-internal: it's prefixed with `_` but stable.
- **`_save_cog(out_np, path_tiff_save, profile, ...)`** — the actual write step. Takes a numpy array (already materialised) and a complete rasterio profile. Used by `save_cog` after step 1.
- **`PROFILE_TILED_GEOTIFF_DEFAULT`** — module-level dict. The defaults: `lzw` compress, `BIGTIFF=IF_SAFER`, 256×256 blocks. Inspect or override via `profile_arg`.
- **`BLOCKSIZE_DEFAULT = 256`** — module-level constant. Change at module level if your downstream tools want different blocking; pass via `profile_arg` per call otherwise.

---

## 9. Sharp edges

- **`save_cog` materialises in memory.** Don't use it for files that won't fit. Use `save_tiled_geotiff` with windowed writes via rasterio directly for stream-write scenarios.
- **Nodata for floating-point.** Default nodata of 0 is wrong for float reflectance (0 is a valid dark pixel). Pass `nodata=np.nan` or use a per-band sentinel like `-9999`.
- **`bool` arrays write as `uint8`.** Round-trip gives uint8; cast with `.astype(bool)` after read if you need bool semantics.
- **Big-endian and signed-byte exotica aren't supported.** The dtype mapping only covers what numpy + GeoTIFF agree on. If you have weird dtypes, cast first.
- **`COPY_SRC_OVERVIEWS` doesn't apply here.** That GDAL flag is for *copying* an existing file's overviews; this module always builds them fresh.
- **Cloud writes are atomic at the destination.** The temp-file-then-upload pattern means partial writes don't appear at the destination URL. But a crash mid-upload leaves the temp file orphaned in `/tmp`.
- **The `save_cog.py` file in the package is empty.** It's a placeholder kept around for backwards-import-compatibility. The real code is in `save.py`.

---

## 10. Connection to `geotoolz`

Two `geotoolz` operators wrap this module:

- **`catalog_ops.WriteCOG(path_template="...")`** — a terminal Operator. Takes a `GeoTensor` from upstream, formats `path_template` with metadata from the catalog row, calls `save.save_cog(...)`. The `path_template="{tile_id}_{date}.tif"` form lets `Sequential` pipelines write per-tile outputs to deterministic filenames.
- **`presets.s2.S2_L2A_NDVI(...)`** — implicitly uses `save_cog` if the user passes a `path` argument; otherwise returns an in-memory `GeoTensor`.

The empty `save_cog.py` file alongside `save.py` is the back-compat stub from when `save_cog` lived in its own module. Don't import from it; use `from georeader.save import save_cog` (or the package-level re-export from `__init__.py`).

Next chapter: [13_misc_io.md](13_misc_io.md) — the smaller utility modules (`io.py`, `dataarray.py`, `plot.py`).
