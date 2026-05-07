# Chapter 3 — `rasterio_reader.py`: the lazy file-backed reader

> **Module:** `georeader/rasterio_reader.py` (1630 LOC)
> **Role:** the canonical `GeoData` implementation. Wraps `rasterio` to give you a `GeoTensor`-shaped interface over a file (local, S3, GCS, Azure, HTTP) **without** reading the bytes until you ask.

---

## 1. Why a lazy reader exists

Three concrete reasons rasterio alone isn't enough:

1. **Process-safety.** `RasterioReader` opens the file *fresh on every `read()` call*. That's the unlock for `multiprocessing` / `joblib` / Dask workers — you can pickle the reader, send it to workers, and each worker opens its own dataset handle. A cached `rasterio.DatasetReader` cannot be pickled safely.
2. **A `GeoTensor`-shaped surface without the bytes.** `reader.shape`, `reader.transform`, `reader.bounds`, `reader.dtype`, `reader.isel(...)` all work without reading data. Only `read()` / `load()` / `read_from_window().load()` materialise.
3. **Multi-file stacks as one object.** Pass a list of paths, get a `(T, C, H, W)` reader. `isel({"time": 0})` returns a single-time-step reader. The time dimension is a structural feature, not a wrapper.

This is the class your operators should accept whenever they don't strictly need the data in memory.

---

## 2. RasterioReader vs GeoTensor

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                 RASTERIOREADER vs GEOTENSOR                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RasterioReader (Lazy)              GeoTensor (In-Memory)               │
│  ─────────────────────              ────────────────────                │
│                                                                         │
│  • Data on disk/cloud               • Data in RAM                       │
│  • Read on demand                   • Instant access                    │
│  • Memory efficient                 • Full numpy API                    │
│  • Parallel-safe                    • Arithmetic operations             │
│  • Overview/pyramid support         • Broadcasting                      │
│                                                                         │
│  Use for:                           Use for:                            │
│  • Large files                      • Processing pipelines              │
│  • Cloud data                       • CNN inference                     │
│  • Tiled processing                 • Index calculations                │
│  • Quick previews                   • Visualizations                    │
│                                                                         │
│  Convert: reader.load() ────────────────────────────────► GeoTensor     │
└─────────────────────────────────────────────────────────────────────────┘
```

The mental model: `RasterioReader` is the **address book**, `GeoTensor` is the **delivered package**. `reader.load()` is the postman.

The arithmetic ops (`+`, `*`, etc.) only exist on `GeoTensor`. If you write `reader * 2` you get an error — and that's deliberate. Arithmetic on a lazy reader implies you've decided to materialise; the explicit `.load()` makes that decision visible at the call site.

---

## 3. Multi-file reading: time series as structure

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-FILE READING                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: List of paths                Output array shape                 │
│  ────────────────────                ──────────────────                 │
│                                                                         │
│  paths = [                                                              │
│    "2023-01.tif",   ─────┐                                              │
│    "2023-02.tif",   ─────┼──────► stack=True:  (T, C, H, W)             │
│    "2023-03.tif"    ─────┘                      (3, 4, 1000, 1000)      │
│  ]                                                                      │
│                                                                         │
│  Each file: (4, 1000, 1000)        stack=False: (T×C, H, W)             │
│  4 bands, 1000×1000 pixels                       (12, 1000, 1000)       │
│                                                                         │
│  Requirements for multi-file:                                           │
│  • Same CRS                                                             │
│  • Same transform (resolution, origin)                                  │
│  • Same shape (unless allow_different_shape=True)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

Two modes worth distinguishing:

- **`stack=True` (default, time-as-axis).** Result `dims = ("time", "band", "y", "x")`. This is what you almost always want for time-series analysis. `isel({"time": 0})` returns a (C, H, W) reader pointing at one file.
- **`stack=False` (concat along bands).** Result `(T*C, H, W)`. Useful when files are *band groups* of a single observation rather than time steps — e.g., S2 SAFE where each band is its own JP2 and there's no time meaning to the file list.

Validation is strict: `__init__` checks CRS / transform / shape match across files unless you opt out via `allow_different_shape=True`. The check only relaxes shape — CRS and transform mismatches always raise. (See [rasterio_reader.py:301](../../../../tmp/georeader_src/georeader/rasterio_reader.py#L301).)

---

## 4. Window focus — "view" semantics

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    WINDOW FOCUS CONCEPT                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Full raster (10000 × 10000)                                            │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                                                                │     │
│  │                                                                │     │
│  │        ┌─────────────────────┐                                 │     │
│  │        │    window_focus     │  ← reader.set_window(...)       │     │
│  │        │    (2000 × 2000)    │                                 │     │
│  │        │                     │  After set_window:              │     │
│  │        │  ┌───────────┐      │  • reader.shape → (C, 2000, 2000)│    │
│  │        │  │ read()    │      │  • reader.bounds → window bounds│     │
│  │        │  │ window    │      │  • read(window=...) is relative │     │
│  │        │  └───────────┘      │    to window_focus              │     │
│  │        └─────────────────────┘                                 │     │
│  │                                                                │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  Benefits: • Work with large files efficiently                          │
│            • Coordinates/bounds reflect the focused region              │
│            • Tiled processing with consistent interface                 │
└─────────────────────────────────────────────────────────────────────────┘
```

`set_window(window_focus)` is the single most useful trick in this class for tiled processing:

- After it's set, **the reader pretends the focused region is the whole raster.** `reader.shape` shrinks; `reader.bounds` reflects the window; `reader.transform` is rewritten with the new origin.
- All subsequent `read()` / `read_from_window()` / `isel()` calls are **relative to the focused window** — you can hand a focused reader to a function that doesn't know about windowing and it will Just Work on the sub-region.
- Overlapping window scans become a loop of `set_window` + `load()` without ever computing nested coordinate offsets in user code.

The non-mutating sister is `read_from_window(window, boundless=True)` ([rasterio_reader.py:654](../../../../tmp/georeader_src/georeader/rasterio_reader.py#L654)) which returns a *new reader* with the focus applied — preferred for parallel pipelines where mutating shared state is awkward.

---

## 5. Constructor parameters

```python
RasterioReader(
    paths,                          # str | list[str]
    allow_different_shape=False,
    window_focus=None,              # rasterio.windows.Window
    fill_value_default=None,        # falls back to file's nodata, then 0
    stack=True,                     # only matters for list paths
    indexes=None,                   # 1-based band indices (rasterio convention)
    overview_level=None,            # 0 = first overview, None = full res
    check=True,                     # validate CRS/transform/shape across paths
    rio_env_options=None,           # GDAL options dict (vsi creds etc.)
)
```

A few non-obvious ones:

- **`indexes`** is **1-based** because rasterio is. `indexes=[1, 2, 3]` on a 4-band S2 image gives you BGR (the first three bands). The reader stores the requested indices and lazily honours them on every read.
- **`overview_level=2`** lets you skip the full-res file entirely for previews. COGs ship overviews at 2×, 4×, 8× downsampled levels; reading from `overview_level=2` is roughly 64× cheaper than full-res reads.
- **`rio_env_options`** is the escape hatch for GDAL config — proxy creds, custom timeout, AWS_REGION, etc. Defaults to a sensible `RIO_ENV_OPTIONS_DEFAULT` defined near the top of the file.
- **`fill_value_default=None`** means "use the file's nodata value if it has one, else 0." Set it explicitly if you have a reason — the default is rarely wrong but rarely what you'd choose if asked.

---

## 6. The four read paths

Each method emphasises a different ergonomic. They all ultimately call into `rasterio.DatasetReader.read`.

| Method | Returns | Use for |
|---|---|---|
| `load(boundless=True)` | `GeoTensor` of full focus | "give me everything in memory now" |
| `read(**kwargs)` | `np.ndarray` (no metadata) | rasterio-compatible; passes kwargs through |
| `read_from_window(window, boundless=True)` | `RasterioReader` (focused) | tiled inference; chain with `.load()` to materialise |
| `read_from_tile(x, y, z, ...)` | `np.ndarray` | XYZ web-tile schema (used by `tileserver` reader) |

The `boundless=True` default on `read_from_window`/`load` matches `GeoTensor.read_from_window` — you get the requested shape no matter where the window lands, padded with `fill_value_default`. This is critical for batched CNN inference: every chip has the same shape, batches stack cleanly.

---

## 7. xarray-style `isel` over time + band

```python
reader.isel({"time": [0, 2], "band": [1, 2, 3]})
```

Same dim-name vocabulary as `GeoTensor.isel` (Chapter 1 §6) but with `"time"` admitted for stacked multi-file readers. Returns a *new reader* — still lazy. Spatial dims (`"x"`, `"y"`) accept slices and rewrite the focus window.

The internal mechanism: `isel` returns a copied reader with `indexes` adjusted (band selection), `paths` filtered (time selection), and `window_focus` updated (spatial slice). Nothing reads. Source: [rasterio_reader.py:739](../../../../tmp/georeader_src/georeader/rasterio_reader.py#L739).

---

## 8. Overviews — the "free preview" path

```python
reader = RasterioReader("s3://bucket/big.tif", overview_level=2)
preview = reader.load()  # ~64× cheaper than full res
```

Two methods to know:

- **`reader.overviews(index=1, time_index=0)`** — list available decimation factors for a band. Empty list means no overviews in the file (it's not a COG, or it's a single-resolution GeoTIFF).
- **`reader.reader_overview(overview_level)`** — return a *new reader* configured to read from that level. Useful when you've already opened a full-res reader and decide you want a thumbnail without re-instantiating from the path.

This is also the right tool for "should I load this whole scene to decide if it's cloudy?" — read overview level 3, run your cloud check on the small image, then go full-res only if it's worth it.

---

## 9. The cloud story — VSI paths and credentials

`rasterio` (via GDAL) handles cloud paths transparently if you pass them in the right form:

| URI form | Backend |
|---|---|
| `s3://bucket/key.tif` | GDAL VSI `/vsis3/` |
| `gs://bucket/key.tif` | `/vsigs/` |
| `https://...cog.tif` | `/vsicurl/` (range requests) |
| `az://account/key.tif` | `/vsiaz/` |

Internal helper `_get_rio_options_path(path)` (and the module-level `_vsi_path` in `geotensor.py`) translate user-friendly URIs to VSI form. Credentials come from `rio_env_options` or from environment (`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`, etc.) — same as plain rasterio.

The [`georeader.md` plan](../plans/georeader.md) is about widening this seam: `LazyCOGReader` and the async-obstore reader plug in here as alternative implementations of the same interface, swapping GDAL VSI for direct HTTP-range / obstore reads. That's the unification work.

---

## 10. Method reference (the public surface)

**Lifecycle**
- `__init__(paths, ...)` — opens *no* file yet; reads metadata once if `check=True`
- `__copy__()`, `copy()` — shallow copy with the same path, fresh state
- `__repr__()` — multi-line human-readable summary

**Inspection (no I/O after `__init__`)**
- `shape`, `crs`, `transform`, `dtype`, `count`, `width`, `height`, `bounds`, `res`, `nodata`, `fill_value_default`, `dims`, `attrs`
- `tags()` — file/band-level GDAL tags
- `descriptions()` — band names
- `same_extent(other, precision=1e-3)` — extent equality (Chapter 2 §6)
- `footprint(crs=None)` — outer polygon
- `meshgrid(dst_crs=None)` — pixel-centre coordinates

**Configuration (mutating)**
- `set_indexes(indexes, relative=True)` — change which bands to read
- `set_indexes_by_name(names)` — same, but by description
- `set_window(window_focus, ...)` — restrict to a sub-region

**Read**
- `load(boundless=True)` → `GeoTensor`
- `read(**kwargs)` → `np.ndarray`
- `read_from_window(window, boundless=True)` → `RasterioReader`
- `read_from_tile(x, y, z, ...)` → `np.ndarray` (XYZ tiles)
- `isel(sel, boundless=True)` → `RasterioReader`
- `values` (property) → eager array

**Overviews**
- `overviews(index=1, time_index=0)` → list of decimation factors
- `reader_overview(overview_level)` → `RasterioReader` at that level
- `block_windows(bidx=1, time_idx=0)` → list of `(block_id, Window)` for native tiling

---

## 11. Sharp edges

- **`indexes` is 1-based** (rasterio convention), but `block_windows` and overview-level integers are 0-based. Easy to flip without realising.
- **`set_window` mutates state.** If you're sharing a reader across functions or threads, prefer `read_from_window` / `isel` which return new readers. The mutating form exists because some legacy call sites needed it.
- **`stack=True` requires identical metadata across files.** When that's not true (e.g., bands at different resolutions), `mosaic` or `griddata` is the right tool, not `RasterioReader`.
- **Cloud reads are *not* free.** Each `read_from_window().load()` issues HTTP range requests. For tile-loops over the same scene, a `set_window` + `load()` of a coarse window into memory followed by `GeoTensor` slicing is often dramatically faster than N small cloud reads.
- **Files are opened on every read.** This is a feature (process-safety) but it costs ~ms per call. For tight inner loops over the same file in one process, batching reads matters.
- **Overviews don't always exist.** Plain GeoTIFFs without pyramids have empty `reader.overviews()`. The reader doesn't *create* overviews on the fly; that's `gdaladdo` territory.

---

## 12. Connection to the rest of the package

Where `RasterioReader` shows up downstream:

- **`georeader.read`** — every `read_from_*` function takes a `GeoData`, and a `RasterioReader` is the typical input. Reprojection ([Chapter 5](05_read.md)) is where lazy reading really pays off: you compute the source window from the destination grid first, then read only that window.
- **`georeader.mosaic`** — combine multiple `RasterioReader`s into one composite, reading only the windows that cover the target.
- **`georeader.save`** — `save_cog(reader, path)` streams a reader to disk without ever having the full array in memory.
- **`georeader.readers.S2_SAFE_reader`** — the Sentinel-2 reader is a `RasterioReader` underneath, with band-group logic on top.

For curvilinear sensors (PRISMA, EnMAP, MODIS) `RasterioReader` is the wrong tool — those use a custom reader pattern routed through `griddata` ([Chapter 7](07_griddata.md)).

Next chapter: [04_window_utils.md](04_window_utils.md) — the window/coordinate-system math that all of the above relies on.
