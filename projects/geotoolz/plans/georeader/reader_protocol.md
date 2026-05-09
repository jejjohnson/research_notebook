---
title: Reader Protocol
subject: georeader design
subtitle: "`_ReaderMeta` / `SyncReader` Protocols + `RasterioReader` refactor"
short_title: Protocol
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, georeader, protocol, refactor
---

> **Parent:** [README.md](README.md)
> **Scope:** lock the Protocol surface every reader honours; refactor the existing `RasterioReader` and `GeoData` / `GeoDataBase` to conform.
> **Status:** ready to implement once the open questions in the [parent](README.md#open-questions) are decided.

---

## Why this issue exists

The new reader (`AsyncGeoTIFFReader`) needs a shared interface to slot into. The existing `RasterioReader` and the `GeoData` / `GeoDataBase` Protocols in `abstract_reader.py` are close to that shape but not aligned. This issue lands the Protocol surface and brings the existing reader into compliance — without breaking any current caller. The same Protocol leaves room for future readers (sync raw-byte readers, sensor-specific readers) without further refactoring.

Done before Issues 2 and 3 because they implement against the Protocols this issue defines.

---

## Primer for newcomers

> **ELI5.** A Python Protocol is like a **job description**: if you can do the listed tasks, you're qualified — regardless of which company you trained at. Three different reader classes can fill the same "reader" job because they all do the listed tasks (have the right methods), even though they're built completely differently inside.

### Python Protocols (the typing kind)

**What it is.** A `typing.Protocol` is a class that lists method signatures and attributes — and any other class with the same shape satisfies it, without needing to inherit. It's how Python expresses "if it walks like a duck and quacks like a duck, it's a duck" with type-checker support.

**How it works.** Define `class Foo(Protocol): def bar(self) -> int: ...`. Any class with a `bar() -> int` method is now a `Foo`, no `class MyClass(Foo)` declaration required. Add `@runtime_checkable` to make `isinstance(x, Foo)` work at runtime too. The static type-checker (`mypy` / `ty`) verifies conformance at the call site.

**What this means for us.** The reader classes — `RasterioReader` (sync, GDAL-backed) and `AsyncGeoTIFFReader` (async, GDAL-free), plus any future sensor-specific or raw-byte readers — don't share a base class. They each satisfy `_ReaderMeta` (and `SyncReader` or `AsyncReader`) structurally. User code typed `def f(reader: SyncReader)` accepts any conforming reader with no isinstance checks. This is the seam that makes the readers swappable — same interface, independent implementations.

```{mermaid}
classDiagram
    class _ReaderMeta {
        <<Protocol>>
        crs
        transform
        bounds
        shape
        dtype
        nodata
    }
    class SyncReader {
        <<Protocol>>
        read_window()
        read_bounds()
        load()
    }
    class AsyncReader {
        <<Protocol>>
        async read_window()
        async read_bounds()
        async load()
    }
    class RasterioReader
    class AsyncGeoTIFFReader

    _ReaderMeta <|-- SyncReader
    _ReaderMeta <|-- AsyncReader
    SyncReader <.. RasterioReader : satisfies
    AsyncReader <.. AsyncGeoTIFFReader : satisfies
```

### The metadata-vs-read split

**What it is.** Every reader has cheap metadata (CRS, transform, shape, dtype) and expensive bytes (the actual pixel data). The Protocol design splits these into two layers: `_ReaderMeta` (metadata only) and `SyncReader` (`_ReaderMeta` + read methods).

**How it works.** A reader's `__init__` reads only the file header — enough to populate `crs` / `transform` / `shape` / etc. That's the `_ReaderMeta` surface. Calling `read_window(window)` fetches actual pixel bytes; that's the `SyncReader` surface on top. The split exists because many functions (window math, bounds queries, intersection checks) only need metadata and shouldn't pay I/O cost.

**What this means for us.** `FakeGeoData` (an existing class in `abstract_reader.py`) is a `_ReaderMeta`-only object — it carries metadata for window calculations without owning data. After this refactor, that pattern is formalised as the Protocol layer. Functions that take `_ReaderMeta` are guaranteed I/O-free; functions that take `SyncReader` may issue reads.

### The three bytes paths in `RasterioReader`

**What it is.** `RasterioReader` wraps `rasterio.open(...)`, which delegates to GDAL. Underneath GDAL is some library that fetches the actual bytes. The refactor exposes three options.

**How it works.** Three constructor knobs:

- **`opener=None`, `fs=None`** (default): GDAL VSI uses libcurl in C. Fastest sync option, no Python in the byte-fetching loop. Works for `s3://`, `gs://`, `az://`, `https://`.
- **`fs=fsspec_filesystem`**: GDAL calls back into a Python file-like object via fsspec for each byte range. Slower (Python ↔ C trip per range) but covers backends GDAL doesn't speak natively (FTP, SFTP, GitHub).
- **`opener=callable`**: same shape as fsspec but with a user-supplied callback. Lets advanced users wire in obstore or custom HTTP clients.

A small helper, `_resolve_open_kwargs`, is the only Python code that knows which path is active.

```{mermaid}
flowchart TD
    Start[RasterioReader<br/>__init__]
    Start --> Q{opener=? fs=?}
    Q -->|both None default| GDAL[GDAL VSI<br/>libcurl in C]
    Q -->|fs=fsspec_fs| Fsspec[Python file-like<br/>via fsspec]
    Q -->|opener=callable| Custom[Python adapter<br/>e.g. obstore-aware]
    GDAL --> Cloud[(S3 / GCS / Azure / HTTP)]
    Fsspec --> Cloud
    Custom --> Cloud
```

**What this means for us.** Most users land on the default and never think about it. Users who need a niche backend (custom auth, MinIO endpoint, GitHub-hosted fixtures) flip `fs=` and keep the rest of their pipeline unchanged. Users who want maximum cloud throughput skip `RasterioReader` entirely and use [`AsyncGeoTIFFReader`](reader_async_geotiff.md).

---

## Deliverables

1. **`_ReaderMeta` Protocol** — 10-property metadata surface, in `georeader/abstract_reader.py`.
2. **`SyncReader` Protocol** — extends `_ReaderMeta` with sync read methods.
3. **`RasterioReader` refactor** — implements `SyncReader`; adds `opener=` / `fs=` / `rio_open_kwargs=` constructor knobs and the bytes-path triage.
4. **`GeoData` / `GeoDataBase` alignment** — current Protocols stay (back-compat); they're redefined as `SyncReader` / `_ReaderMeta` aliases or supersets.
5. **`GeoTensor` Protocol conformance** — `GeoTensor` already morally satisfies `_ReaderMeta`; declare it formally so the type-checker agrees.
6. **Tutorial updates** — [Ch. 2](../../georeader_tutorial/02_abstract_reader.md) and [Ch. 3](../../georeader_tutorial/03_rasterio_reader.md) reflect the new surface.

`AsyncReader` is **deliberately not** in this issue — it lives in [Issue 3](reader_async_geotiff.md). Defining it here would force scope creep into a refactor that's otherwise sync-only.

---

## `_ReaderMeta` Protocol

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
```

Generalises today's `GeoDataBase` (3 properties) into the full 10-property surface a downstream consumer needs to know about a raster *without* reading any pixels. Anything satisfying `_ReaderMeta` can be passed to `window_from_bounds`, `figure_out_transform`, `same_extent`, or any of the other coordinate-math helpers in [`window_utils`](../../georeader_tutorial/04_window_utils.md).

---

## `SyncReader` Protocol

```python
class SyncReader(_ReaderMeta, Protocol):
    """Sync read interface — RasterioReader (and any future sync reader)."""

    def read_window(self, window: Window) -> GeoTensor: ...
    def read_bounds(
        self,
        bounds: tuple[float, float, float, float],
        *,
        target_resolution: tuple[float, float] | None = None,
        target_crs: pyproj.CRS | str | None = None,
    ) -> GeoTensor: ...
    def read_geoslice(self, slice_: GeoSlice) -> GeoTensor: ...
    def load(self) -> GeoTensor: ...
    def close(self) -> None: ...

    def __enter__(self) -> "SyncReader": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None: ...
```

Method shapes are the canonical sync reader surface; any future sync reader follows them so user code can swap concrete readers without changing call sites. `read_bounds` accepts optional `target_resolution=` and `target_crs=` because cross-CRS reads are common; readers that can't reproject (or can but slowly) document the cost in their docstring.

`close` and the context-manager methods are required by the Protocol but tolerate no-ops — readers backed by `obstore` (which pools connections) typically implement `close()` as a no-op.

---

## `RasterioReader` refactor

The existing class today has constructor:

```python
RasterioReader(paths, allow_different_shape=False, window_focus=None,
               fill_value_default=None, stack=True, indexes=None,
               overview_level=None, check=True, rio_env_options=None)
```

It stays. New keyword-only knobs are added; the new methods are added alongside the existing ones:

```python
class RasterioReader(SyncReader):
    """Sync, GDAL-backed reader. The default in georeader.

    Reads happen via rasterio.open(...).read(window=...). The bytes path *under*
    the rasterio call has three modes — see "Inside RasterioReader" below:

      1. opener=None and fs=None  → GDAL VSI (libcurl in C); the default.
                                     Cloud paths /vsis3/, /vsigs/, /vsiaz/, /vsicurl/.
      2. opener=callable          → GDAL calls the callable for each byte range.
      3. fs=fsspec_filesystem     → shortcut: equivalent to opener=fs.open.

    On-the-fly reprojection in read_bounds() is done via rasterio.warp.WarpedVRT.
    """
    path_or_url: str                                      # alias for paths[0] when single
    indexes: tuple[int, ...] | None
    _opener: "Callable[[str, str], BinaryIO] | None"
    _fs: "fsspec.AbstractFileSystem | None"
    _rio_open_kwargs: dict

    def __init__(
        self,
        paths,                                            # existing
        # ... all existing kwargs preserved ...
        *,
        opener: "Callable[[str, str], BinaryIO] | None" = None,    # new
        fs: "fsspec.AbstractFileSystem | None" = None,              # new
        rio_open_kwargs: dict | None = None,                        # new
    ): ...

    # internal — bytes-path triage
    def _resolve_open_kwargs(self) -> dict:
        """Translate the constructor's opener/fs knobs into rasterio.open kwargs."""
        kwargs = dict(self._rio_open_kwargs or {})
        if self._opener is not None:
            kwargs["opener"] = self._opener
        elif self._fs is not None:                       # fs= shortcut
            kwargs["opener"] = self._fs.open
        # else: no opener key → rasterio uses GDAL VSI for cloud paths
        return kwargs

    # metadata — straight passthrough to rasterio dataset attrs
    @property
    def crs(self) -> pyproj.CRS: ...
    # ... (the rest of _ReaderMeta surface)

    # new sync-reader methods
    def read_window(self, window: Window) -> GeoTensor:
        """ds.read(indexes=..., window=...) → GeoTensor with windowed transform."""
        ...
    def read_bounds(self, bounds, *, target_resolution=None, target_crs=None) -> GeoTensor:
        """Wrap in WarpedVRT if target_crs differs from native; window the
        VRT to the requested bounds; read."""
        ...
    def read_geoslice(self, slice: GeoSlice) -> GeoTensor:
        """Convenience: read_bounds(slice_.bounds, target_resolution=slice_.resolution,
                                    target_crs=slice_.crs)."""
        ...
    def load(self) -> GeoTensor: ...
    def close(self) -> None: ...
    def __enter__(self) -> "RasterioReader": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None: ...

    # back-compat — existing methods kept, possibly delegating to the new ones
    def read_from_window(self, window, boundless: bool = True): ...      # existing
    # load() with the existing boundless= behaviour stays valid for callers that
    # already pass it; the new no-arg load() is what SyncReader requires.
```

### The three bytes paths

The `opener=` / `fs=` knobs route bytes through one of three paths: GDAL VSI (default, fastest), fsspec (for niche backends), or a custom obstore callback. The diagram and per-path comparison table live in [`geostack.md` §"What's actually inside `RasterioReader`"](../geostack.md#whats-actually-inside-rasterioreader). `_resolve_open_kwargs` (above) is the only Python code that knows which path is active; after it returns, GDAL takes over.

### Usage examples

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
# at this point you'd use AsyncGeoTIFFReader instead
def obstore_opener(path: str, mode: str) -> "BinaryIO":
    """Return a file-like wrapping obstore range fetches."""
    ...
reader = RasterioReader("s3://bucket/scene.tif", opener=obstore_opener)
```

### Credential handling across the three paths

The refactor doesn't change the existing GDAL-VSI credential pattern. It does add two paths where credentials can live in user objects rather than process env vars — useful for tests, multi-account isolation in one process, and refreshable tokens. Where credentials live in each path:

| Path | Credential locus |
|---|---|
| **GDAL VSI** (`opener=None`, `fs=None`; default) | Process environment variables (`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`, `AZURE_STORAGE_*`). Set once at app startup via `os.environ[...] = ...` or via a config-file helper like `mars_data_ops.fs_access_from_config(...)`. The today-pattern documented in [Tutorial Ch. 3 §9](../../georeader_tutorial/03_rasterio_reader.md). |
| **fsspec** (`fs=fsspec_fs`) | The `fs` object's construction — `fsspec.filesystem("s3", key=..., secret=...)`. Per-reader, no env vars needed. Multi-account isolation comes free: two readers with two `fs` instances see two credential sets. |
| **opener=callable** | Whatever the callable closes over. Most flexible, most user-managed; this is where refreshable-token implementations would live until the package ships a typed credential surface. |

A typed `Credential` Protocol that unifies these three paths is proposed separately in [`plans/types/credentials.md`](../types/credentials.md). The wiring on `RasterioReader` (`credential=` kwarg, refresh-on-401, auto-rewrite for SAS fallback) is in [`reader_rasterio.md`](reader_rasterio.md). Both designs are downstream of this refactor — Issue 1 just needs to not paint into a corner that prevents them.

---

## `GeoData` / `GeoDataBase` alignment

The current Protocols in `abstract_reader.py`:

- `GeoDataBase` — `transform`, `crs`, `shape`, `width`, `height` (3 required + 2 derived).
- `GeoData` (= `AbstractGeoData`) — adds `values`, `load(boundless=True)`, `read_from_window(window, boundless)`, `bounds`, `res`, `dtype`, `dims`, `fill_value_default`, `footprint(crs=None)`.

After this issue:

- `GeoDataBase` continues to exist as an alias for the relevant subset of `_ReaderMeta` (or a strict back-compat Protocol).
- `GeoData` continues to exist as a superset of `SyncReader` (it has extra methods: `footprint`, `read_from_window`, etc.). New code should prefer `SyncReader`; existing code keeps working.
- `AbstractGeoData = GeoData` continues to be exported for back-compat.

Concretely:

- No method is removed.
- No method's signature changes.
- Two new Protocols (`_ReaderMeta`, `SyncReader`) are *added*; existing types satisfy them by structural typing without changes.

This ensures every current caller of `GeoData` keeps working.

---

## `GeoTensor` Protocol conformance

`GeoTensor` already exposes:

- `crs`, `transform`, `bounds`, `shape`, `dtype`, `res` — directly.
- `nodata` — as `fill_value_default` (a property alias may be added).
- `count`, `height`, `width` — as derived properties.
- `read_from_window`, `load` — already implemented (Ch. 1 §10).

Declaring `GeoTensor` as `_ReaderMeta`-conformant is a typing-only change. May need to add a `path_or_url` attribute (e.g., `None` or a synthetic identifier) and an `indexes` attribute to satisfy the Protocol exactly.

---

## Tutorial updates

After this issue lands:

- **[Ch. 2](../../georeader_tutorial/02_abstract_reader.md)** — describes `_ReaderMeta` / `SyncReader` Protocols. The current `GeoDataBase` / `GeoData` description stays as the back-compat layer (with a note that new code should prefer the new Protocols).
- **[Ch. 3](../../georeader_tutorial/03_rasterio_reader.md)** — adds a new section on the `opener=` / `fs=` constructor knobs and the three-bytes-paths triage. Cross-links to the parent design doc.

Tutorial updates land alongside the implementation PR for this issue, not before — the tutorial follows the package state.

---

## Acceptance criteria

- `_ReaderMeta` and `SyncReader` Protocols exported from `georeader.abstract_reader`.
- `RasterioReader` instances satisfy `SyncReader` per static type-check (mypy / ty).
- `GeoTensor` instances satisfy `_ReaderMeta` per static type-check.
- All existing tests pass without modification.
- New tests for `RasterioReader.read_window`, `read_bounds(target_crs=...)`, `read_geoslice`, `load()`.
- New test for `RasterioReader("s3://...", fs=fsspec_fs)` opening successfully.
- New test for `RasterioReader("s3://...", opener=obstore_opener)` opening successfully.
- `GeoData` / `GeoDataBase` Protocols continue to be exported and import-compatible.
- Tutorial Ch. 2 and Ch. 3 updated.

---

## Issue-specific open questions

In addition to the [parent design's open questions](README.md#open-questions), this issue should resolve:

1. **`path_or_url` on multi-file readers.** Today's `RasterioReader` accepts `paths: list[str]` for time stacks. The Protocol expects `path_or_url: str`. Options: expose `paths[0]` as `path_or_url`, or relax the Protocol to `str | list[str]`.
2. **`indexes` type.** Today's reader uses `list[int]`; the Protocol uses `tuple[int, ...] | None`. Pick one and reconcile.
3. **`shape` shape.** Today's reader exposes `(T, C, H, W)` for stacked time-series; the Protocol expects `(count, height, width)` 3-tuple. Either widen the Protocol to allow longer tuples, or have the reader expose only the 3D view through the Protocol surface (with the 4D shape via a separate property).
