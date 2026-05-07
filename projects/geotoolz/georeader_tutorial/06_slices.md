# Ch. 6 — `slices`

> **Module:** `georeader/slices.py` (404 LOC)
> **Role:** divide a raster into tiles. Three diagrams cover the *what* (overlap vs not), the *vocabulary* (Python `slice` vs `rasterio.windows.Window`), and the *what-do-I-do-at-the-edge* problem. Three public functions: `create_slices`, `create_windows`, plus the dict↔window converters.

---

## 1. The job

When a raster doesn't fit in RAM (or a model has a fixed input size), you process it in tiles. The whole module exists to answer one question:

**Given a raster shape, a tile size, and an overlap, give me the list of windows that cover it.**

Everything else — overlap semantics, edge handling, dict-vs-tuple ergonomics — is a knob on that core operation.

---

## 2. Tiling strategies

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    TILING STRATEGIES                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Non-overlapping (overlap=0)          Overlapping (overlap>0)           │
│  ─────────────────────────            ──────────────────────            │
│                                                                          │
│  ┌────┬────┬────┬────┐              ┌────┬────┬────┬────┐               │
│  │ 1  │ 2  │ 3  │ 4  │              │ 1 ─┼─ 2 ┼─ 3 ┼─ 4 │               │
│  ├────┼────┼────┼────┤              ├───┬┼───┬┼───┬┼───┬┤               │
│  │ 5  │ 6  │ 7  │ 8  │              │ 5 │├ 6 │├ 7 │├ 8 ││               │
│  ├────┼────┼────┼────┤              ├───┼┼───┼┼───┼┼───┼┤               │
│  │ 9  │ 10 │ 11 │ 12 │              │ 9 ││10 ││11 ││12 ││               │
│  └────┴────┴────┴────┘              └───┴┴───┴┴───┴┴───┴┘               │
│                                          └─ overlap region              │
│  Best for:                          Best for:                           │
│  • Independent tiles                • Edge-sensitive algorithms         │
│  • Aggregation tasks                • Convolutions/filters              │
│  • Simple mosaicking                • Seamline blending                 │
└─────────────────────────────────────────────────────────────────────────┘
```

**Stride** = `tile_size - overlap`. That single equation drives the whole module:

- `overlap=0` → stride = tile_size → tiles abut perfectly. Best for per-pixel work that has no spatial neighbourhood (mean per tile, classification heads applied independently).
- `overlap>0` → stride < tile_size → tiles share rows/cols with neighbours. Necessary for any operation with a spatial kernel (convolutions, Lee-speckle filters, cubic resampling). The overlap should be at least the kernel half-width plus a margin — typical values 16/32/64 for U-Net-shaped models.

The overlapped form composes with `slice_save_for_pred` ([Chapter 4 §9](04_window_utils.md)) for the standard "predict on padded chip, save the centre" tile-and-stitch recipe.

---

## 3. Slices vs Windows

```text
┌─────────────────────────────────────────────────────────────────────────┐
│              SLICES (Python) vs WINDOWS (Rasterio)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  slice(start, stop)              Window(col_off, row_off, width, height)│
│  ──────────────────              ──────────────────────────────────────│
│                                                                          │
│  • Python array indexing         • Rasterio file reading                │
│  • 2D: (row_slice, col_slice)    • 2D: explicit offsets + sizes         │
│  • End-exclusive                 • Width/height inclusive               │
│                                                                          │
│  Conversion:                                                             │
│    slices_to_windows((row_slice, col_slice)) → Window                   │
│    window_to_slices(window) → (row_slice, col_slice)                    │
│                                                                          │
│  Example:                                                                │
│    slice(100, 356), slice(200, 456)  →  Window(200, 100, 256, 256)      │
│    (rows 100-355, cols 200-455)          (col=200, row=100, 256x256)    │
└─────────────────────────────────────────────────────────────────────────┘
```

Two vocabularies for the same rectangle. Pick by audience:

| Use slices when | Use windows when |
|---|---|
| Indexing a numpy / xarray array directly | Calling `rasterio.read(window=...)` |
| Composing with `xarray.isel` | Calling `read_from_window` |
| Working in numpy `(rows, cols)` order | Working in rasterio `(col, row)` order |

The conversion is purely arithmetic — there's no CRS or transform involved. The example: rows 100–355 inclusive (256 rows) and cols 200–455 inclusive (256 cols) is `Window(col_off=200, row_off=100, width=256, height=256)`. Note the order swap: numpy is rows-first, Window is col-first.

`create_slices` returns dicts (`{"y": slice(...), "x": slice(...)}`); `create_windows` returns `Window` objects. They're the same data wearing different hats.

---

## 4. Edge handling — three knobs

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    EDGE HANDLING OPTIONS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Raster: 500px, Tile: 128px, Overlap: 0                                 │
│  ──────────────────────────────────────                                 │
│                                                                          │
│  include_incomplete=True (default):                                     │
│  ┌────────┬────────┬────────┬────────┬────┐                             │
│  │  128   │  128   │  128   │  128   │ 12 │  ← 5 tiles, last is 12px   │
│  └────────┴────────┴────────┴────────┴────┘                             │
│                                                                          │
│  include_incomplete=False:                                              │
│  ┌────────┬────────┬────────┐                                           │
│  │  128   │  128   │  128   │  ← 3 tiles, drops edge                    │
│  └────────┴────────┴────────┘                                           │
│                                                                          │
│  trim_incomplete=True:                                                   │
│  Same as include_incomplete=True but last slice is trimmed              │
│  slice(384, 500) instead of slice(384, 512)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

500 isn't divisible by 128 — what to do with the remainder? Three flags settle it:

- **`include_incomplete=True` (default).** Keep the partial last tile. Result has 5 tiles for our example, the last covering only 12 pixels. Use this for visualization / aggregation where every pixel must be covered.
- **`include_incomplete=False`.** Drop the partial last tile. Result has 3 tiles covering 384 of the 500 pixels. Use this for ML training where a partial tile would have a different shape from the rest of the batch.
- **`trim_incomplete=True` (default).** Last slice is `slice(384, 500)` — trimmed to the raster bound. The alternative (`trim_incomplete=False`) is `slice(384, 512)` — extending past the raster, requiring `boundless=True` reads to pad with `fill_value_default`. The trimmed version is what you want when reading; the un-trimmed is what you want when *writing back* into a fixed-size grid that already has padding allocated.

There's also a fourth, narrower flag:

- **`start_negative_if_padding=True`.** Shifts the first tile by `-overlap//2`, so overlap is symmetric across the *whole* raster (including the first/last edges) rather than just between interior tiles. Required when tile-and-stitch with a CNN that introduces edge artefacts even at the global boundary; usually paired with `boundless=True`.

---

## 5. The three public functions

### `create_windows(geodata_shape, window_size, overlap=None, ...)`

The primary entry point. `geodata_shape=(height, width)` (numpy order); `window_size=(height, width)`; `overlap=(row_overlap, col_overlap)`. Returns `list[rasterio.windows.Window]`.

**Use this** for tiled reading from a raster.

### `create_slices(named_shape, dims, overlap=None, ...)`

Generalised N-dimensional version. `named_shape={"y": H, "x": W, ...}`, `dims={"y": tile_h, "x": tile_w, ...}`. Returns `list[dict[str, slice]]` — one dict per tile, with one slice per named dimension.

**Use this** for xarray-style indexing or for tiling along non-spatial dims (time chunks, band groups).

Internally, `create_windows` is a thin wrapper around `create_slices` with `dims={"y": ..., "x": ...}` and a final dict-to-window conversion.

### `slices_to_windows((row_slice, col_slice))` / `window_to_slices(window)`

Pair-wise converters. No tiling logic — just rectangle-shape translation.

There are also private helpers `_slices` (1D), `_slices_nd` (N-D tuple form), and `_slices_2d` (2D tuple form) — used internally and occasionally exposed in older code paths.

---

## 6. Composing with the rest of georeader

The three idiomatic patterns:

**Pattern A — sequential tiled read into memory:**

```python
windows = slices.create_windows(
    geodata_shape=reader.shape[-2:],
    window_size=(512, 512),
    overlap=(64, 64),
)
for w in windows:
    chip = reader.read_from_window(w, boundless=True).load()
    # chip is (C, 512, 512) GeoTensor regardless of position
```

**Pattern B — tile-and-stitch CNN inference:**

```python
windows = slices.create_windows(
    geodata_shape=reader.shape[-2:],
    window_size=(512, 512),
    overlap=(64, 64),
    start_negative_if_padding=True,    # symmetric edge handling
    trim_incomplete=False,             # let boundless padding handle the edge
)
for w_read in windows:
    chip = reader.read_from_window(w_read, boundless=True).load()
    pred = model(chip.values)
    # crop the prediction to remove the 32-px overlap on each side
    # save back to global output via slice_save_for_pred (Chapter 4 §9)
```

**Pattern C — N-D tiling for time-series chunks:**

```python
chunks = slices.create_slices(
    named_shape={"time": T, "band": C, "y": H, "x": W},
    dims={"time": 30, "y": 512, "x": 512},     # tile time + space; full bands
    overlap={"y": 64, "x": 64},                 # overlap only spatially
)
for sel in chunks:
    sub = data.isel(sel)                        # xarray-style
```

Pattern C is the one that lets you process a year of S2 in monthly time-chunks without ever materialising the whole stack.

---

## 7. Sharp edges

- **Numpy axis order vs Window order, again.** `geodata_shape=(H, W)`, `window_size=(H, W)`, `overlap=(row_overlap, col_overlap)` — all numpy `(y, x)`. The returned `Window` objects are `(col_off, row_off, width, height)` — rasterio `(x, y)`. The function does the swap; you have to pass numpy order in.
- **`include_incomplete=False` *drops* coverage.** The 12-pixel remainder of the worked example is unread. If you don't want to drop and don't want a small tile, pad the raster first or use `start_negative_if_padding`.
- **`trim_incomplete` interacts with `boundless`.** Trimmed tiles are smaller than `window_size` — chip batches won't stack. Untrimmed-and-boundless gives uniform shape but allocates padding. Pick consciously.
- **No CRS / transform involvement.** This module is pure pixel arithmetic. If you want geographic-coordinate tiles (e.g., "tiles aligned to UTM grid lines"), you compute a `figure_out_transform` first and use `bounds_to_windows` ([Chapter 4 §3](04_window_utils.md)) per tile-bbox.
- **`overlap` argument order matches the dims order.** In `create_windows` it's `(row_overlap, col_overlap)`. In `create_slices` it's `{"y": ..., "x": ...}` — the dict keys make this less ambiguous.

---

## 8. Why this module is small and stays small

`slices.py` is one of the cleanest files in georeader. 404 lines, three public functions, no dependencies beyond `rasterio.windows`. The strategic minimalism is deliberate: tiling is a generator, not a pipeline. The pipeline (read → predict → stitch) is the user's responsibility, composed from this module + `read.py` + `window_utils.py`.

For `geotoolz.sampling`:

- `geotoolz.sampling.GridSampler(chip_size, stride)` is `slices.create_windows(window_size=chip_size, overlap=chip_size-stride, ...)` plus a CRS-aware wrapper if the user wants geographic chip queries.
- `geotoolz.sampling.RandomSampler` is **not** in this module — random sampling is just `np.random.randint` over the geographic bbox; no tiling structure to share.
- `geotoolz.sampling.Stitch` lives in [Chapter 4 §9](04_window_utils.md) (`slice_save_for_pred`) — this module gives you the chips, that one tells you how to put predictions back.

Next chapter: [07_griddata.md](07_griddata.md) — irregular-grid interpolation and the geolocation-lookup-table (GLT) pattern. Where the package grows past "rectilinear rasters" into swath sensors like MODIS / PRISMA.
