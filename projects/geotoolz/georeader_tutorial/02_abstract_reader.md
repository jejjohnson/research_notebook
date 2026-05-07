# Chapter 2 — `abstract_reader.py`: the type protocols

> **Module:** `georeader/abstract_reader.py` (257 LOC)
> **Role:** the duck-typing contract that lets `GeoTensor` (in-memory) and `RasterioReader` (lazy on-disk) be passed interchangeably to every function in the package.

---

## 1. The one-line idea

Anything with `transform`, `crs`, `shape` is **`GeoDataBase`**. Anything that *additionally* knows how to materialise its data (`values`, `load()`, `read_from_window()`) is **`GeoData`** (alias `AbstractGeoData`). Most of `georeader.read`, `georeader.window_utils`, `georeader.mosaic` etc. type-annotate against these protocols, so the same function body works on either substrate.

This is the seam that the [`georeader.md` plan](../plans/georeader.md) wants to widen — make `RasterioReader`, `LazyCOGReader`, and `AsyncGeoTIFFReader` all honour the same protocol, then user code does `reader_class=...` strategy injection.

---

## 2. The type hierarchy

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    GEOREADER TYPE HIERARCHY                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  GeoDataBase (Protocol)           Minimal interface for geospatial data │
│  ├── transform: Affine            Pixel → coordinate mapping            │
│  ├── crs: Any                     Coordinate reference system           │
│  └── shape: Tuple                 (C, H, W) or (H, W) dimensions        │
│       │                                                                  │
│       ▼                                                                  │
│  AbstractGeoData (Protocol)       Adds read capabilities                │
│  ├── values: ndarray              Array data                            │
│  ├── fill_value_default           Nodata value                          │
│  └── load(): GeoTensor            Read all data                         │
│       │                                                                  │
│       ├──────────────────────┬──────────────────────┐                   │
│       ▼                      ▼                      ▼                   │
│  RasterioReader         GeoTensor              Custom Readers           │
│  (Lazy file access)     (In-memory)           (User-defined)           │
│                                                                          │
│  GeoData = Union[AbstractGeoData, GeoTensor]  ← Common type alias       │
└─────────────────────────────────────────────────────────────────────────┘
```

Two splits to notice:

- **Vertical:** `GeoDataBase` (just enough to *describe* a raster — useful for window math without ever loading bytes) vs `GeoData` (can materialise). The `FakeGeoData` dataclass below is the canonical `GeoDataBase`-only object.
- **Horizontal:** `RasterioReader` (lazy, file-backed), `GeoTensor` (in-memory ndarray subclass), and any third-party reader that conforms.

In the file, `AbstractGeoData = GeoData` is set as a back-compat alias — older code refers to `AbstractGeoData`; new code should prefer `GeoData`.

---

## 3. The protocol contract

What you must supply to be a `GeoData`:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    REQUIRED PROPERTIES                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Property              Type                  Description                 │
│  ──────────           ──────                ───────────                 │
│  transform            rasterio.Affine       6-element affine matrix     │
│  crs                  Any (CRS-like)        EPSG code, WKT, or CRS obj  │
│  shape                Tuple[int, ...]       (C, H, W) or (H, W)         │
│  values               ndarray               Raster data array           │
│  fill_value_default   number                Nodata/fill value           │
│                                                                          │
│  Required Methods:                                                       │
│  ─────────────────                                                       │
│  load() → GeoTensor   Read all data into memory                         │
│                                                                          │
│  Derived Properties (computed from above):                              │
│  ──────────────────────────────────────────                             │
│  width                shape[-1]             Number of columns           │
│  height               shape[-2]             Number of rows              │
│  bounds               From transform+shape  (minx, miny, maxx, maxy)    │
│  res                  From transform        (xres, yres) pixel size     │
│  footprint            Polygon               Bounding polygon in CRS     │
└─────────────────────────────────────────────────────────────────────────┘
```

`GeoData` provides default implementations for the **derived** properties on top of the required five — so a custom reader only has to wire up the required ones, and gets `bounds` / `res` / `footprint` for free.

The base class deliberately raises `NotImplementedError` for `dtype`, `dims`, and `fill_value_default` rather than inventing a default — these are sensor-specific and any subclass that ignores them is going to bite users elsewhere.

---

## 4. Reading the source

The whole file is short enough to read in one sitting. Key landmarks:

- **`GeoDataBase`** — [abstract_reader.py:147](../../../../tmp/georeader_src/georeader/abstract_reader.py#L147). Pure `Protocol` with `transform`, `crs`, `shape` and computed `width`/`height`. No `load`. Any class that has these three attrs satisfies it without inheriting (structural typing via `typing.Protocol`).

- **`FakeGeoData`** — [abstract_reader.py:169](../../../../tmp/georeader_src/georeader/abstract_reader.py#L169). A `@dataclass` placeholder used for window/bounds math when you don't want to allocate the array. Three fields: `crs`, `transform`, `shape` (optional). Used heavily inside `read.py` to do "what window would I read?" calculations before deciding whether to actually fetch bytes.

- **`GeoData`** — [abstract_reader.py:188](../../../../tmp/georeader_src/georeader/abstract_reader.py#L188). The full reader interface. Defines `load(boundless=True)`, `read_from_window(window, boundless)`, plus the derived `bounds` / `res` / `footprint` / `values` (which delegates to `load`).

- **`AbstractGeoData = GeoData`** — [abstract_reader.py:249](../../../../tmp/georeader_src/georeader/abstract_reader.py#L249). Back-compat alias.

- **`same_extent(geo1, geo2, precision=1e-3)`** — [abstract_reader.py:252](../../../../tmp/georeader_src/georeader/abstract_reader.py#L252). The canonical "are these two rasters geographically interchangeable" check: transform-almost-equal, CRS-compare, last-two-dims-equal. Used by `GeoTensor`'s binary arithmetic and by `mosaic` / `read` before broadcasting operations.

---

## 5. `FakeGeoData` — describing a grid without owning one

```python
fake = FakeGeoData(
    crs="EPSG:4326",
    transform=Affine.translation(-122.5, 37.5) * Affine.scale(0.001, -0.001),
    shape=(3, 1000, 1000),
)
```

Why this exists:

1. **Pre-flight calculations.** `window_from_bounds(fake, bounds, crs_bounds)` lets you ask "if I had a raster with this transform and shape, what window would my AOI be?" — before opening any file.
2. **Designing output grids.** When mosaicking or reprojecting, you build a `FakeGeoData` describing the *target* grid, then pass it to `read.read_reproject_like(src, dst=fake)`.
3. **Testing.** Cheap fixtures for window-math tests.

`shape` is `Optional[Tuple[int, ...]]` — if you only need `transform` + `crs` for a coordinate calc, you can leave it as `None`. `width` and `height` then raise `ValueError("Shape is not defined")` if accessed, which is correct: you said you didn't have one.

---

## 6. `same_extent` — the equality predicate

```python
def same_extent(geo1: GeoData, geo2: GeoData, precision: float = 1e-3) -> bool:
    return (
        geo1.transform.almost_equals(geo2.transform, precision=precision)
        and window_utils.compare_crs(geo1.crs, geo2.crs)
        and (geo1.shape[-2:] == geo2.shape[-2:])
    )
```

Three observations on the design:

- **`transform.almost_equals(precision=1e-3)`** — not exact. Floating-point transforms drift through reprojection round-trips; an exact comparison would be too strict.
- **`compare_crs`** is in `window_utils` — it normalises across "EPSG:4326" / `4326` / `pyproj.CRS.from_epsg(4326)` so the user doesn't have to.
- **Only the last two dims** are checked. A 3-band stack and a 5-band stack at the same footprint are "same-extent" — what matters is the spatial grid.

Used by `GeoTensor.__add__` etc. to refuse `gt1 + gt2` when extents disagree (see [Chapter 1 §12](01_geotensor.md)).

---

## 7. How this protocol shows up downstream

Almost every public function in `georeader.read` and `georeader.window_utils` is annotated `data: GeoData` or `data: GeoDataBase`. Concretely:

| Caller-side type | What you typically pass |
|---|---|
| `GeoDataBase` | `FakeGeoData`, `GeoTensor`, `RasterioReader` — anything with the three structural attrs |
| `GeoData` / `AbstractGeoData` | `GeoTensor`, `RasterioReader`, custom readers — anything that can materialise |
| `GeoTensor` (concrete) | only when an in-memory ndarray is genuinely required |

This means most of your geotoolz operators should be typed `data: GeoData` (not `GeoTensor`) so they accept lazy readers transparently — the `load()` happens once, inside the operator, and is cheap when the operator is the leaf.

---

## 8. Sharp edges

- **`Protocol` doesn't enforce at runtime.** `isinstance(obj, GeoDataBase)` works only if you've decorated `GeoDataBase` with `@runtime_checkable` (this module does *not*). So duck-typing is by static checker (mypy/ty) or by `try/except AttributeError`. In practice: be liberal in what you accept, raise loud errors in `__init__`/factory functions if a required attribute is missing.
- **`load(boundless=True)` is the universal handshake.** Every reader must implement it. The default contract: return a `GeoTensor` of the same shape, padding with `fill_value_default` for any out-of-bounds region.
- **Subclasses, not Protocol implementations.** Despite the docstring talking about "Protocol", `GeoData` is a regular base class with `NotImplementedError` stubs — `RasterioReader` and the hyperspectral readers actually subclass it. The `Protocol` form is only `GeoDataBase`. That asymmetry is historical; treat both as duck-typed contracts in your own code.
- **Last two dims are spatial.** This is enforced nowhere in the protocol — it's a convention, and the rest of the package relies on it. A reader returning `(H, W, C)` ("channels-last") would silently break things.

---

## 9. The shape of a custom reader (what's in the next chapter)

The next chapter unpacks `RasterioReader` — the canonical `GeoData` subclass. Reading it is the fastest way to understand what a real implementation of this protocol looks like and what conveniences (`.isel`, `.read_from_window`, multi-file stacks, overviews, VSI cloud paths) sit on top of the bare contract above.

Next chapter: [03_rasterio_reader.md](03_rasterio_reader.md).
