---
title: GeoSlice
subject: Core types
subtitle: Unit of work between catalog, sampler, loader, operator
short_title: GeoSlice
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, types, sampler, geoslice
---

> **Parent:** [README.md](README.md) — Core types.
> **Status:** design proposal. Promoted from the [Geodatabase Phase 1](../geodatabase/geocatalog.md) writeup, where it was treated as a footnote despite being the unit of work flowing between three layers.
> **Scope:** the `GeoSlice` dataclass and the three primitives that produce or consume it (`random_sampler`, `grid_sampler`, `stitch`).

---

## Summary

A `GeoSlice` is a **bounded request for data** — a bbox, a time interval, a target resolution, and a CRS. It is the unit of work that catalogs produce, that loaders consume, and that operators (in `geotoolz`) compose against.
The three sampler/stitch primitives are the canonical producers and consumer:

- `random_sampler(catalog, chip_size, ...)` → iterator of `GeoSlice` for ML training.
- `grid_sampler(catalog, chip_size, ...)` → iterator of `GeoSlice` for tiled inference.
- `stitch(predictions, slices, ...)` → reverses the chip operation back into a single output raster.

Today, all four live in `jej_vc_snippets/sampler.py` (~1400 LOC).
**Ownership update:** the three primitives above belong in `geotoolz.patch` today (with [`geopatcher`](../geopatcher/README.md) as the plan/future extraction) — they are concrete instances of `Patcher(Rectangular × RegularStride × Boxcar × OverlapAdd)` (grid sampler), `Patcher(Rectangular × Random × Boxcar × …)` (random sampler), and `OverlapAdd.merge` (stitch). `georeader.samplers` is not the right home: `georeader` owns the substrate (Protocols, carriers, byte paths); the patching/sampling algebra belongs in `geotoolz.patch`. The dataclass itself stays here in `types/` — it's the cross-cutting wire format between the catalog (producer), `geotoolz.patch` (consumer), and `georeader`'s loaders (consumer). This document gives the dataclass its own attention so the invariants, sampler math, and stitch reductions are specified in one place rather than scattered across other designs.

---

## Motivation

Three reasons this deserves its own design doc rather than a section inside the catalog plan:

1. **It's the inter-layer contract.** Catalogs produce `GeoSlice`s.
   Loaders consume them.
   `geotoolz.sampling.GridSampler` wraps `grid_sampler`; `geotoolz.inference.ApplyToChips` consumes the iterator and `stitch`es predictions.
   Six designs reference `GeoSlice`; none was the right home to fully specify it.

2. **The math is non-trivial and easy to get wrong.** Random-sampler weighting has a documented bias when tile sizes are heterogeneous.
   Grid-sampler stride math has subtle edge-case behaviour at the trailing row/column.
   Stitch's four reduction modes (`average` / `max` / `first` / `last`) each have different ordering semantics.
   Putting it all in one place makes it auditable.

3. **`GeoSlice` overlaps with two other concepts** — `rasterio.windows.Window` (pixel-space rectangle) and `slices.create_windows` (chunking generator).
   Reconciling the three is a design decision in its own right, not a footnote in another doc.

---

## Primer for newcomers

> **ELI5.** A `GeoSlice` is a **delivery slip**: it says where (bbox), when (time), at what zoom (resolution), and in what coordinate system (CRS).
> The catalog writes the slip; the loader reads it; whoever's in between never has to ask "wait, which file did this come from?" — the slip has everything they need.

### Frozen dataclasses (immutability)

**What it is.** A `@dataclass(frozen=True)` is a Python class with auto-generated `__init__` / `__repr__` / `__eq__` *and* a guarantee that its attributes can't be mutated after construction.
Trying to assign a new value to a field raises `FrozenInstanceError`.

**How it works.** The `@dataclass` decorator inspects type annotations and synthesises the constructor.
`frozen=True` adds `__setattr__` and `__delattr__` overrides that raise.
The instance is also hashable by default (because nothing can change), which lets you use it as a dict key or set member.

**What this means for us.** `GeoSlice` is frozen so that once constructed, it can be passed across function boundaries, stored in caches, used as a dict key for "slices I've already processed," etc. Code that wants to *change* a slice creates a new one (`dataclasses.replace(slice_, bounds=new_bounds)`) — explicit by design.
Mutable units of work flowing between layers are a recipe for bugs.

### `pd.Interval` and `IntervalIndex`

**What it is.** A `pd.Interval(start, end, closed='both')` is pandas's typed representation of a time range.
Combined into an `IntervalIndex`, you get fast "find all rows whose interval overlaps this query interval" queries — the temporal counterpart to a spatial R-tree.

**How it works.** Each interval is an immutable object with a `closed` policy (`'both'`, `'left'`, `'right'`, `'neither'`).
Standard set ops (`overlaps`, `contains`) are vectorised at the IntervalIndex level.
Scales cleanly: 10k intervals → microseconds per query.
Same primitive used inside catalogs to filter by time.

**What this means for us.** `GeoSlice.interval` is a `pd.Interval`, not a `(tmin, tmax)` tuple, because the catalog stores intervals the same way and "does this slice's interval overlap any catalog row?" is a single function call (no manual min/max gymnastics).
Convention is `closed='both'` everywhere — both endpoints inclusive.

### `pyproj.CRS` (vs string EPSG codes)

**What it is.** `pyproj.CRS` is a typed CRS object that knows its EPSG code, WKT representation, axis order, datum shifts, and reprojection rules.
Compared to a bare `"EPSG:4326"` string, it carries semantic information.

**How it works.** Constructed from `"EPSG:32630"`, a WKT string, or a `pyproj.CRS.from_epsg(32630)` call.
Carries methods for axis-order checking, transformation parameter lookup, and "is this CRS equivalent to that one" comparisons.
Reprojection to/from another CRS goes through `pyproj.Transformer`.

**What this means for us.** `GeoSlice.crs` is a `pyproj.CRS`, not a string, so two slices in slightly different CRS representations (e.g., `EPSG:32630` vs full WKT for the same UTM zone) can be compared sensibly.
The package's `compare_crs(a, b)` helper does the right thing across the various forms.

### "Unit of work" between layers

**What it is.** A small, immutable, self-contained value that flows from a *producer* (catalog, sampler) to a *consumer* (loader, operator) across a function boundary, carrying everything the consumer needs to do its job — no further reference to the producer required.

**How it works.** `GeoSlice` has four fields: `bounds` (where in space), `interval` (when in time), `resolution` (at what pixel size), `crs` (in which coordinate system).
That's enough for a loader to fetch a chip without consulting the catalog or the sampler again.
The producer hands you a slice; you do whatever; you hand the result downstream.
No hidden state.

**What this means for us.** The pattern decouples layers cleanly.
A `random_sampler` doesn't know which loader will consume the slice; a `read_geoslice(slice)` method doesn't know whether the slice came from a sampler, a manual user query, or a JSON config. Same `GeoSlice` shape; many producers and consumers; no coupling beyond the type.

```{mermaid}
sequenceDiagram
    participant Cat as GeoCatalog
    participant Sampler as grid_sampler
    participant Reader
    participant Op as Operator
    participant Stitch

    Cat->>Sampler: iter_rows()
    loop per chip
        Sampler-->>Reader: GeoSlice<br/>(bounds, time, res, crs)
        Reader->>Reader: open file at slice.bounds
        Reader-->>Op: GeoTensor
        Op-->>Stitch: prediction
    end
    Stitch->>Stitch: combine slices + predictions
    Stitch-->>Cat: stitched GeoTensor
```

---

## Goals

- **Lock the `GeoSlice` dataclass shape** — fields, invariants, immutability semantics, derived properties.
- **Specify the three primitives** — signatures, math, edge cases for `random_sampler` / `grid_sampler` / `stitch`.
- **Reconcile with `rasterio.windows.Window` and `georeader.slices`** — what's the relationship, when does each apply, when do they convert.
- **Make the producers and consumers explicit** — which designs hand `GeoSlice` upstream, which receive it, what guarantees flow each way.

---

## Non-goals

- **Defining the `GeoCatalog` Protocol** — that's [Geodatabase](../geodatabase/README.md).
  This doc only describes how catalogs *produce* `GeoSlice` via the samplers.
- **Defining the loader contract** — loaders that take a `GeoSlice` and return a `GeoTensor` are part of the catalog/reader designs.
- **Owning the model-loop helper.** The old `run_inference_with_grid_sampler` is deliberately not promoted; `geotoolz.inference.ApplyToChips` is the sanctioned successor.
- **Async samplers.** All three primitives are sync.
  An async iterator variant is plausible but lands later if needed.

---

## Constraints

- **`GeoSlice.crs` is `pyproj.CRS`-shaped.** Matches the rest of `georeader`; no string-only or EPSG-int-only constraint.
- **Time intervals are `pd.Interval`-shaped.** Matches the catalog's `IntervalIndex`.
  `closed='both'` is the convention everywhere.
- **`GeoSlice` is immutable.** `frozen=True` dataclass — slices are passed across function boundaries and shouldn't be mutated in flight.
- **`stitch` produces a `GeoTensor`**, not a bare ndarray.
  Carrying CRS/transform through is non-negotiable; otherwise the result isn't georeferenced.
- **No torch in the core path.** `random_sampler` returns an `Iterator[GeoSlice]`, not a `DataLoader`.
  ML adapters can be built on top.

---

## The `GeoSlice` dataclass

```python
import pandas as pd
import pyproj
from dataclasses import dataclass

@dataclass(frozen=True)
class GeoSlice:
    """A bounded request for data — produced by samplers, consumed by loaders.

    Carries everything a loader needs to fetch a chip without consulting
    the catalog: bbox in CRS units, time interval, target resolution, CRS.
    """
    bounds: tuple[float, float, float, float]   # (xmin, ymin, xmax, ymax)
    interval: pd.Interval                       # closed='both'
    resolution: tuple[float, float]             # (x_res, y_res) in CRS units
    crs: pyproj.CRS

    @property
    def shape(self) -> tuple[int, int]:
        """Output grid shape (height, width) implied by bounds + resolution."""
        x_res, y_res = self.resolution
        xmin, ymin, xmax, ymax = self.bounds
        return (
            round((ymax - ymin) / y_res),
            round((xmax - xmin) / x_res),
        )

    @property
    def transform(self) -> "rasterio.Affine":
        """North-up affine from bounds + resolution."""
        ...

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
```

### Invariants

- `bounds` is `(xmin, ymin, xmax, ymax)` with `xmin < xmax` and `ymin < ymax`.
  Antimeridian-crossing bboxes are forbidden at the slice level — split before constructing.
- `interval.closed == "both"`.
  Half-open intervals are accepted by `pd.Interval` but rejected here for consistency with the catalog's `IntervalIndex`.
- `resolution` is positive in both axes.
  The sign of the y-resolution is **not** flipped — slices store positive resolutions and the `transform` property handles the y-axis sign.
- `(bounds, resolution)` is consistent: `(xmax - xmin) / x_res` and `(ymax - ymin) / y_res` should round to integers within `PIXEL_PRECISION = 3` decimal digits.
  Producers (samplers) guarantee this; loaders may assume it.

### Why `GeoSlice` and not `Window` or `slice`?

| | `rasterio.windows.Window` | `slice` (Python) | `GeoSlice` |
|---|---|---|---|
| Coordinate system | pixel | array index | CRS units |
| Carries CRS | no | no | yes |
| Carries time | no | no | yes |
| Carries resolution | no | no | yes |
| Use for | within-file extraction | array slicing | inter-layer contract |

`GeoSlice` exists because the inter-layer contract needs all four.
A `Window` is what a loader produces *from* a `GeoSlice` once it knows the source file's transform.
A Python `slice` is what `slices.create_windows` ([Tutorial Ch. 6](../../georeader_tutorial/06_slices.md)) emits inside a single file, with no geographic meaning.

### Conversion to/from `Window`

```python
def slice_to_window(slice_: GeoSlice, transform: Affine) -> Window:
    """Convert a CRS-unit GeoSlice to a pixel-space Window in the given file."""
    return rasterio.windows.from_bounds(*slice_.bounds, transform=transform)

def window_to_slice(window: Window, transform: Affine, crs: CRS,
                    interval: pd.Interval, resolution: tuple[float, float]) -> GeoSlice:
    """Inverse — usually only useful in tests / debugging."""
    bounds = rasterio.windows.bounds(window, transform)
    return GeoSlice(bounds=bounds, interval=interval, resolution=resolution, crs=crs)
```

The asymmetry is deliberate: a `GeoSlice` carries strictly more information than a `Window`, so going `slice → window` is lossy in the time/CRS axes (you have to drop them) and going `window → slice` requires extra arguments to fill them back in.

---

## `random_sampler`

### Signature

```python
def random_sampler(
    catalog: GeoCatalog,
    chip_size: tuple[int, int],
    *,
    length: int | None = None,
    roi: shapely.Polygon | None = None,
    toi: pd.Interval | None = None,
    units: Literal["pixels", "crs"] = "pixels",
    seed: int | None = None,
    weight: Literal["area", "uniform"] = "area",
) -> Iterator[GeoSlice]: ...
```

- `catalog` is consulted via `iter_rows()` (Phase 1) or a streaming cursor (Phase 2).
  The sampler doesn't materialise the full catalog.
- `length=None` means "infinite iterator"; downstream code uses `itertools.islice` to bound.
- `roi` / `toi` filter the catalog before sampling.
- `units="pixels"` interprets `chip_size` in pixels; `units="crs"` in CRS units (relevant for cross-resolution catalogs).
- `seed` makes a sampler deterministic.
- `weight` picks the tile-selection distribution (see below).

### Area-weighted sampling math

Default `weight="area"`.
Tiles big enough to fit a chip are kept; each surviving tile gets weight proportional to its area:

$$
w_i = \frac{(x_{\max,i} - x_{\min,i}) \cdot (y_{\max,i} - y_{\min,i})}{\sum_j A_j}
$$

A tile is drawn with `numpy.random.Generator.choice(p=w)`, then a chip is placed uniformly inside the tile:

```python
x_offset = uniform(0, tile_width  - chip_width)
y_offset = uniform(0, tile_height - chip_height)
```

The timestamp is sampled uniformly (in seconds) inside the tile's interval and assigned to both `tmin` and `tmax`.

### The weighting bias (open question)

Area-weighted sampling pushes samples *toward* large tiles.
For uniform-density imagery this is what you want — every pixel has equal probability of being sampled.
For tiled mosaics with one giant tile and many small ones, it over-samples the giant tile.

Three options:

- **`weight="area"` (default)** — what's described above.
  Right for uniform imagery.
- **`weight="uniform"`** — each tile has equal probability regardless of size.
  Right for tiled mosaics where each tile represents one "scene" and you want balanced training data.
- **`weight=callable`** — user-supplied function over the catalog row.
  Maximum flexibility; rarely needed.

The proposal is to expose `weight={"area", "uniform"}` and document the bias.
See [parent README §Open questions](README.md) for the unresolved decision on whether `callable` is worth adding in v1.

---

## `grid_sampler`

### Signature

```python
def grid_sampler(
    catalog: GeoCatalog,
    chip_size: tuple[int, int],
    *,
    stride: tuple[int, int] | None = None,
    roi: shapely.Polygon | None = None,
    toi: pd.Interval | None = None,
    units: Literal["pixels", "crs"] = "pixels",
) -> Iterator[GeoSlice]: ...
```

- `stride=None` defaults to `chip_size` (no overlap).
- `stride < chip_size` produces overlapping chips — required by inference pipelines that pair with `stitch` to reduce edge artefacts.
- `roi` / `toi` filter; `units` parallel to `random_sampler`.

### Stride math

For each tile of size `(W, H)` and chip `(w, h)` with stride `(s_x, s_y)`:

$$
n_{\text{cols}} = \left\lceil \frac{W - w}{s_x} \right\rceil + 1, \qquad
n_{\text{rows}} = \left\lceil \frac{H - h}{s_y} \right\rceil + 1
$$

Final-row/column chips are **shifted** (not truncated) so chip size stays exact across the grid.
Total chip count = sum over tiles.
Matches the standard sliding-window convention used by `xrpatcher` / `xbatcher`.

### Reconciliation with `slices.create_windows`

[`slices.create_windows`](../../georeader_tutorial/06_slices.md) does the same stride math but at the pixel level inside a single file.
The relationship:

| Layer | Function | Input | Output |
|---|---|---|---|
| Inter-file (catalog) | `grid_sampler` | catalog + chip size in CRS units | iterator of `GeoSlice` |
| Intra-file (one raster) | `slices.create_windows` | `(H, W)` shape + chip size in pixels | iterator of `Window` |

`grid_sampler` calls into the catalog to find tiles; for each tile it can either reimplement stride math itself or convert the tile's bounds into a `(H, W)` shape and delegate to `create_windows`, then convert each `Window` back to a `GeoSlice`.
The implementation detail is open; the two functions stay separately useful at their respective scopes.

---

## `stitch`

### Signature

```python
def stitch(
    predictions: Sequence[np.ndarray],
    slices: Sequence[GeoSlice],
    *,
    method: Literal["average", "max", "first", "last"] = "average",
    roi: shapely.Polygon | None = None,
) -> GeoTensor: ...
```

- `predictions[i]` is the model output for `slices[i]`.
  Same length, same order.
- `roi` is optional — clips the output to a polygon footprint.
- Returns a `GeoTensor` with transform/CRS derived from the union of the slices' bounds and the assumed-uniform resolution.

### Reduction methods

| Method | Update rule |
|---|---|
| `average` | `out += pred; counts += 1; out /= max(counts, 1)` |
| `max`     | `out = np.maximum(out, pred)` |
| `first`   | write only where `counts == 0`, then increment counts |
| `last`    | unconditional overwrite |

Pixel placement uses the output transform implicit in `(xmin, ymax, x_res, y_res)`:

```python
col = int((slice_.bounds[0] - xmin) / x_res)
row = int((ymax - slice_.bounds[3]) / y_res)   # y-flip: rows count from north downward
```

### Order-dependence

`first` and `last` are **order-dependent** by definition — they reduce based on the iteration order of `predictions`.
`average` and `max` are order-independent.

For deterministic pipelines, sort `slices` by `(slice_.bounds, slice_.interval.left)` before passing.
The samplers don't guarantee a stable order — `random_sampler` is intentionally random; `grid_sampler` iterates tiles in catalog order, which depends on the backend.

---

## Connections to other designs

| Design | How it touches `GeoSlice` |
|---|---|
| [Geodatabase / `geocatalog.md`](../geodatabase/geocatalog.md) | `GeoCatalog.query(slice_: GeoSlice)` consumes one. Catalog rows + `iter_rows()` is what the samplers iterate. |
| [Geodatabase / `geoduckdb.md`](../geodatabase/geoduckdb.md) | DuckDB cursor iteration into the samplers; catalog produces lazy row stream, samplers reservoir-sample. |
| [Reader reconciliation](../georeader/README.md) | `GeoData.read_geoslice(slice)` (sync) and `AsyncGeoData.read_geoslice(slice)` (async) are the canonical loader entry points. |
| [`geotoolz.md`](../geotoolz/geotoolz.md) | `geotoolz.sampling.GridSampler` wraps `grid_sampler`. `geotoolz.inference.ApplyToChips` consumes the iterator and uses `stitch` for the inverse step. The Stitch operator in geotoolz is a direct re-export. |

The *flow* of a `GeoSlice` through a typical inference pipeline:

```
GeoCatalog  ──grid_sampler──►  GeoSlice  ──reader.read_geoslice──►  GeoTensor
                                  │                                       │
                                  └──────────────────────┐              model
                                                          ▼                │
                                                   stitch(slices,         │
                                                          predictions) ◄──┘
                                                            │
                                                            ▼
                                                       GeoTensor (stitched)
```

Every arrow is sync in this proposal. The async equivalent (using `AsyncGeoData.read_geoslice` and `asyncio.gather`) lands as a follow-up if needed.

---

## Open questions

### 1. `weight=callable` in `random_sampler`

Section above proposes `weight={"area", "uniform"}` for v1. A callable variant (`weight=lambda row: ...`) is requested by some users for stratified sampling but adds API surface.
Defer to v2 or include now?
**Tentative pick: defer.**

### 2. `GeoSlice` antimeridian policy

The invariants require `xmin < xmax`.
Antimeridian-crossing AOIs are forbidden at the slice level.
Where does the split happen?

- **At the producer** (sampler refuses to emit antimeridian-crossing slices, splits internally).
- **At the consumer** (loader detects and splits the read into two).
- **At a higher level** (reject antimeridian-crossing inputs entirely with a clear error).

[`window_utils.bounds_to_windows`](../../georeader_tutorial/04_window_utils.md) currently handles this in the loader path by returning a list of windows.
Whether `GeoSlice` should mirror that or refuse outright is a judgement call.

### 3. `GeoSlice.dims` for higher-rank slices

Today's `shape` returns `(h, w)`.
For 4D `(t, b, h, w)` chips, do we want a `dims` property?
The number of channels can be inferred from the catalog row + loader, not from the slice itself.
**Tentative pick: keep slices 2D-shaped; let downstream code assemble multi-dim chunks.**

### 4. `stitch` default for `method`

`average` is the proposal default.
For probability-output models this is *wrong* — averaging probabilities pre-softmax gives the wrong posterior.
For most practical cases users want `last` (faster, no allocation of the counts array) or `average` (smoother).
Pick: keep `average` as default, document the gotcha.
**Tentative pick: keep `average`.**

### 5. Time semantics in `stitch`

`stitch` produces a single `GeoTensor` from many slices.
If those slices have different `interval` fields (likely, when stitching predictions across time), what does the output's `interval` become?

- **Union of all input intervals** — semantically defensible.
- **Drop the time field** — the output is spatial; time is no longer meaningful.
- **Require uniform `interval` across inputs** — rejects mixed-time stitching with an error.

**Unresolved.** The implementation in `jej_vc_snippets` is silent on this — the stitched output is just an ndarray with no time metadata.

---

## Alternatives considered

- **Use `BoundingBox` from torchgeo.** Rejected: brings a torch dependency into the inter-layer contract; doesn't carry resolution; carries time only optionally.
- **Reuse `rasterio.windows.Window`.** Rejected: `Window` is pixel-space and CRS-agnostic.
  Cross-CRS catalogs would need conversion at every layer boundary.
- **Use a STAC `Item` as the unit of work.** Rejected: too heavy.
  STAC items carry rich asset metadata that the inter-layer contract doesn't need; forcing every slice to be STAC-shaped pulls in a JSON-schema dep.
- **Skip the dataclass; pass `(bounds, interval, resolution, crs)` as a tuple.** Rejected: the sampler/loader/stitch surface is wide enough that a positional tuple becomes a bug magnet.
  Field names matter.
