---
title: Reader Protocol
subject: georeader design
subtitle: "Add `AsyncGeoData` Protocol; widen `RasterioReader` bytes paths"
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
> **Status:** **revised 2026-05-09** — collapsed from a "build a parallel `_ReaderMeta` / `SyncReader` / `AsyncReader` taxonomy" design into "keep today's `GeoData` / `GeoDataBase`; add only `AsyncGeoData`". See §"Why the rewrite" below.
> **Scope:** add a single new Protocol (`AsyncGeoData`) so [`AsyncGeoTIFFReader`](reader_async_geotiff.md) has a typed seam to slot into. Optionally widen [`RasterioReader`](reader_rasterio.md) with `opener=` / `fs=` / `rio_open_kwargs=` knobs to expose its three bytes paths. Don't redefine the metadata surface; today's `GeoDataBase` and `GeoData` already do that.

---

## Why this issue exists

`AsyncGeoTIFFReader` (the new async COG reader; see [Issue 2](reader_async_geotiff.md)) needs a typed Protocol to satisfy. Today's `GeoData` / `GeoDataBase` Protocols cover the *sync* surface — properties + sync `load(boundless=True)` + sync `read_from_window(window, boundless)`. We need an async mirror for the new reader.

That's the entire ambition of this issue: add `AsyncGeoData`. The earlier draft also proposed adding `_ReaderMeta` and `SyncReader` Protocols *alongside* `GeoData` / `GeoDataBase` and renaming the surface — that scope has been removed (see §"Why the rewrite").

A second, smaller scope item is widening `RasterioReader`'s constructor with `opener=` / `fs=` / `rio_open_kwargs=` knobs so users can route bytes through GDAL VSI / fsspec / a custom callback explicitly. This is genuinely additive — no method renames, no Protocol churn — and lets `RasterioReader` remain the canonical sync reader without forcing users to monkey-patch `rasterio.Env` to reach niche backends.

---

## Why the rewrite

The earlier draft of this doc proposed a parallel taxonomy:

- `_ReaderMeta` Protocol (10 properties + `path_or_url` + `indexes`).
- `SyncReader` Protocol (extends `_ReaderMeta` + sync read methods).
- `AsyncReader` Protocol (extends `_ReaderMeta` + async read methods, optionally lifted to Issue 1).
- "Keep `GeoData` / `GeoDataBase` as back-compat aliases."

On review (2026-05-09), three problems:

1. **Two names for every concept.** Every property would have a `GeoDataBase`-shaped name and a `_ReaderMeta`-shaped name. We'd explain back-compat in every doc forever.
2. **Most of `_ReaderMeta` is already in `GeoData` / `GeoDataBase`.** `crs`, `transform`, `shape`, `width`, `height` are in `GeoDataBase`; `bounds`, `res`, `dtype`, `fill_value_default` are in `GeoData` (with default implementations on top of the required three). The only genuinely new fields proposed (`path_or_url`, `indexes`) are *reader-construction details* that leak file-backed-reader concerns onto an abstract surface — `GeoTensor` shouldn't have to fake them.
3. **Async is the only real gap.** Today's Protocols have no async surface. That's the actual problem.

So: drop `_ReaderMeta` and `SyncReader` from the plan. Add `AsyncGeoData` (mirror of `GeoData` with `async` read methods). Document the `opener=` / `fs=` constructor widening for `RasterioReader` separately as an additive change.

If we want a clean rename later (`GeoData` → `SyncReader`, `GeoDataBase` → `ReaderMeta`), that's a one-line deprecation alias upstream in `spaceml-org/georeader` proper — *not* a parallel layer in our plan. Out of scope for this issue.

---

## Primer for newcomers

> **ELI5.** A Python Protocol is like a **job description**: if you can do the listed tasks, you're qualified — regardless of which company you trained at. Today, `GeoData` is the sync-reader job description. We're adding `AsyncGeoData` as the async-reader job description. `RasterioReader` keeps doing the sync job; `AsyncGeoTIFFReader` shows up to do the async one.

### Python Protocols (the typing kind)

**What it is.** A `typing.Protocol` is a class that lists method signatures and attributes — and any other class with the same shape satisfies it, without needing to inherit. It's how Python expresses "if it walks like a duck and quacks like a duck, it's a duck" with type-checker support.

**How it works.** Define `class Foo(Protocol): def bar(self) -> int: ...`. Any class with a `bar() -> int` method is now a `Foo`, no `class MyClass(Foo)` declaration required. Add `@runtime_checkable` to make `isinstance(x, Foo)` work at runtime too. The static type-checker (`mypy` / `ty`) verifies conformance at the call site.

**What this means for us.** `RasterioReader` (sync, GDAL-backed) satisfies `GeoData` today. `AsyncGeoTIFFReader` (async, GDAL-free) satisfies `AsyncGeoData` after this issue. User code typed `def f(reader: AsyncGeoData)` accepts any conforming async reader — no isinstance checks, no shared base class. This is the seam that makes the two readers swappable per workload.

```{mermaid}
classDiagram
    class GeoDataBase {
        <<Protocol>>
        crs
        transform
        shape
        width
        height
    }
    class GeoData {
        <<Protocol>>
        load(boundless)
        read_from_window(window, boundless)
        values
        bounds, res, dtype
        fill_value_default
    }
    class AsyncGeoData {
        <<Protocol>>
        async load()
        async read_window()
        async read_bounds()
    }
    class RasterioReader
    class GeoTensor
    class AsyncGeoTIFFReader

    GeoDataBase <|-- GeoData
    GeoDataBase <|-- AsyncGeoData
    GeoData <.. RasterioReader : satisfies
    GeoData <.. GeoTensor : satisfies
    AsyncGeoData <.. AsyncGeoTIFFReader : satisfies
```

### The metadata-vs-read split

**What it is.** Every reader has cheap metadata (CRS, transform, shape, dtype) and expensive bytes (the actual pixel data). The Protocol design splits these into two layers: `GeoDataBase` (metadata only) and `GeoData` / `AsyncGeoData` (`GeoDataBase` + read methods).

**How it works.** A reader's `__init__` (or `await open(...)` for async) reads only the file header — enough to populate `crs` / `transform` / `shape` / etc. That's the `GeoDataBase` surface. Calling `read_from_window(window)` or `await read_window(window)` fetches actual pixel bytes; that's the `GeoData` / `AsyncGeoData` layer on top. The split exists because many functions (window math, bounds queries, intersection checks) only need metadata and shouldn't pay I/O cost.

**What this means for us.** `FakeGeoData` (an existing dataclass in `abstract_reader.py`) is a `GeoDataBase`-only object — it carries metadata for window calculations without owning data. Functions typed `data: GeoDataBase` are guaranteed I/O-free; functions typed `data: GeoData` may issue sync reads; functions typed `data: AsyncGeoData` may issue async reads.

### The three bytes paths in `RasterioReader`

**What it is.** `RasterioReader` wraps `rasterio.open(...)`, which delegates to GDAL. Underneath GDAL is some library that fetches the actual bytes. The optional widening exposes three options.

**How it works.** Three constructor knobs:

- **`opener=None`, `fs=None`** (default): GDAL VSI uses libcurl in C. Fastest sync option, no Python in the byte-fetching loop. Works for `s3://`, `gs://`, `az://`, `https://`.
- **`fs=fsspec_filesystem`**: GDAL calls back into a Python file-like object via fsspec for each byte range. Slower (Python ↔ C trip per range) but covers backends GDAL doesn't speak natively (FTP, SFTP, GitHub).
- **`opener=callable`**: same shape as fsspec but with a user-supplied callback. Lets advanced users wire in custom HTTP clients.

A small helper, `_resolve_open_kwargs`, is the only Python code that knows which path is active.

```{mermaid}
flowchart TD
    Start[RasterioReader<br/>__init__]
    Start --> Q{opener=? fs=?}
    Q -->|both None default| GDAL[GDAL VSI<br/>libcurl in C]
    Q -->|fs=fsspec_fs| Fsspec[Python file-like<br/>via fsspec]
    Q -->|opener=callable| Custom[Python adapter]
    GDAL --> Cloud[(S3 / GCS / Azure / HTTP)]
    Fsspec --> Cloud
    Custom --> Cloud
```

**What this means for us.** Most users land on the default and never think about it. Users who need a niche backend (custom auth, MinIO endpoint, GitHub-hosted fixtures) flip `fs=` and keep the rest of their pipeline unchanged. Users who want maximum cloud throughput skip `RasterioReader` entirely and use [`AsyncGeoTIFFReader`](reader_async_geotiff.md), which routes through `obstore` (no GDAL).

---

## Deliverables

### Required

1. **`AsyncGeoData` Protocol** — added to `georeader/abstract_reader.py`. Mirrors `GeoData`'s sync surface with `async` read methods. ~30 LOC.
2. **`GeoTensor` Protocol conformance check** — `GeoTensor` already satisfies `GeoData` morally; add a static-type-check confirming this so the type-checker agrees. (No code change to `GeoTensor` expected.)
3. **Tutorial update** — [Ch. 2](../../georeader_tutorial/02_abstract_reader.md) gains a small section describing `AsyncGeoData` alongside the existing `GeoData` / `GeoDataBase` writeup.

### Optional (additive — bundle if convenient, otherwise defer)

4. **`RasterioReader` constructor widening** — add `opener=`, `fs=`, `rio_open_kwargs=` keyword-only knobs. No breaking changes; defaults reproduce today's behaviour. See §"`RasterioReader` widening" below.
5. **Tutorial update for the bytes-path triage** — [Ch. 3](../../georeader_tutorial/03_rasterio_reader.md) gains a section on the three bytes paths if (4) lands.

What this issue does **not** ship:

- A `_ReaderMeta` Protocol. Today's `GeoDataBase` already plays this role.
- A `SyncReader` Protocol. Today's `GeoData` already plays this role.
- A rename of `GeoData` / `GeoDataBase`. If we want one, do it upstream in `spaceml-org/georeader` proper as a separate PR with deprecation aliases — not as a parallel layer in our plan.
- New `path_or_url` / `indexes` fields on the abstract surface. Those are reader-construction details; they live on the concrete reader classes only.

---

## `AsyncGeoData` Protocol

```python
from typing import Optional, Protocol, Union

import numpy as np
import rasterio
import rasterio.windows
from shapely.geometry import Polygon

from georeader.abstract_reader import GeoDataBase
from georeader.geotensor import GeoTensor


class AsyncGeoData(GeoDataBase, Protocol):
    """Async mirror of :class:`GeoData`.

    Concrete async readers (today: :class:`AsyncGeoTIFFReader`) satisfy
    this Protocol. User code typed against ``AsyncGeoData`` accepts any
    conforming async reader without isinstance checks.

    Inherits the metadata surface (``transform``, ``crs``, ``shape``,
    ``width``, ``height``) from :class:`GeoDataBase`. Adds async read
    methods + the same derived properties (``bounds``, ``res``,
    ``dtype``, ``fill_value_default``, ``footprint``) as
    :class:`GeoData`.
    """

    async def load(self, boundless: bool = True) -> GeoTensor:
        raise NotImplementedError

    async def read_from_window(
        self,
        window: rasterio.windows.Window,
        boundless: bool = True,
    ) -> Union["AsyncGeoData", GeoTensor]:
        raise NotImplementedError

    @property
    def res(self) -> tuple[float, float]:
        from georeader import window_utils
        return window_utils.res(self.transform)

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def fill_value_default(self):
        raise NotImplementedError

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        from georeader import window_utils
        return window_utils.window_bounds(
            rasterio.windows.Window(
                row_off=0, col_off=0,
                height=self.shape[-2], width=self.shape[-1],
            ),
            self.transform,
        )

    def footprint(self, crs: Optional[str] = None) -> Polygon:
        from georeader import window_utils
        pol = window_utils.window_polygon(
            rasterio.windows.Window(
                row_off=0, col_off=0,
                height=self.shape[-2], width=self.shape[-1],
            ),
            self.transform,
        )
        if (crs is None) or window_utils.compare_crs(self.crs, crs):
            return pol
        return window_utils.polygon_to_crs(pol, self.crs, crs)
```

Note that `AsyncGeoData.values` is **not** present (unlike `GeoData.values`, which materialises sync via `self.load()`). An async-equivalent would have to be a coroutine, but properties can't be async. Callers that want the array call `await reader.load()` explicitly. Documenting this in the Protocol docstring is enough.

The `footprint`, `res`, `bounds` properties are duplicated from `GeoData` because Python Protocols don't compose default implementations cleanly through inheritance. Concrete readers can override; the defaults match `GeoData`'s behaviour.

---

## `RasterioReader` widening *(optional — bundle if convenient)*

The existing class today has constructor:

```python
RasterioReader(paths, allow_different_shape=False, window_focus=None,
               fill_value_default=None, stack=True, indexes=None,
               overview_level=None, check=True, rio_env_options=None)
```

It stays. New keyword-only knobs are added:

```python
class RasterioReader(GeoData):
    """Sync, GDAL-backed reader. The default in georeader.

    Reads happen via rasterio.open(...).read(window=...). The bytes
    path *under* the rasterio call has three modes — see the docstring
    on the new keyword-only ``opener`` / ``fs`` / ``rio_open_kwargs``
    args, and the per-path comparison table in
    plans/geostack.md §"What's actually inside RasterioReader".

      1. opener=None and fs=None  → GDAL VSI (libcurl in C); the default.
                                     Cloud paths /vsis3/, /vsigs/, /vsiaz/.
      2. opener=callable          → GDAL calls the callable for each byte range.
      3. fs=fsspec_filesystem     → shortcut: equivalent to opener=fs.open.

    On-the-fly reprojection in read_bounds() is done via
    rasterio.warp.WarpedVRT.
    """

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
```

### The three bytes paths

The `opener=` / `fs=` knobs route bytes through one of three paths: GDAL VSI (default, fastest), fsspec (for niche backends), or a custom obstore-aware callback. The diagram and per-path comparison table live in [`geostack.md` §"What's actually inside `RasterioReader`"](../geostack.md#whats-actually-inside-rasterioreader). `_resolve_open_kwargs` (above) is the only Python code that knows which path is active; after it returns, GDAL takes over.

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

# For high-concurrency async fan-out, skip RasterioReader entirely
# and use AsyncGeoTIFFReader (which routes through obstore + async-tiff).
reader = await AsyncGeoTIFFReader.open("s3://bucket/scene.tif")
```

### Credential handling across the three paths

The widening doesn't change the existing GDAL-VSI credential pattern. It does add two paths where credentials can live in user objects rather than process env vars — useful for tests, multi-account isolation in one process, and refreshable tokens. Where credentials live in each path:

| Path | Credential locus |
|---|---|
| **GDAL VSI** (`opener=None`, `fs=None`; default) | Process environment variables (`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`, `AZURE_STORAGE_*`). Set once at app startup via `os.environ[...] = ...` or via a config-file helper like `mars_data_ops.fs_access_from_config(...)`. The today-pattern documented in [Tutorial Ch. 3 §9](../../georeader_tutorial/03_rasterio_reader.md). |
| **fsspec** (`fs=fsspec_fs`) | The `fs` object's construction — `fsspec.filesystem("s3", key=..., secret=...)`. Per-reader, no env vars needed. Multi-account isolation comes free: two readers with two `fs` instances see two credential sets. |
| **opener=callable** | Whatever the callable closes over. Most flexible, most user-managed; this is where refreshable-token implementations would live until the package ships a typed credential surface. |

A typed `Credential` Protocol that unifies these three paths is proposed separately in [`plans/types/credentials.md`](../types/credentials.md). The wiring on `RasterioReader` (`credential=` kwarg, refresh-on-401, auto-rewrite for SAS fallback) is in [`reader_rasterio.md`](reader_rasterio.md). Both designs are downstream of this issue — Issue 1 just needs to not paint into a corner that prevents them.

---

## `GeoTensor` Protocol conformance

`GeoTensor` already exposes:

- `crs`, `transform`, `bounds`, `shape`, `dtype`, `res` — directly.
- `fill_value_default` — directly.
- `width`, `height` — as derived properties.
- `read_from_window`, `load` — already implemented (Tutorial Ch. 1 §10).

Declaring `GeoTensor` as `GeoData`-conformant is a typing-only change. May need a small alignment if the type-checker objects to one signature; otherwise no code change.

---

## Acceptance criteria

- `AsyncGeoData` Protocol exported from `georeader.abstract_reader`.
- [`AsyncGeoTIFFReader`](reader_async_geotiff.md) instances satisfy `AsyncGeoData` per static type-check.
- `GeoTensor` instances satisfy `GeoData` per static type-check.
- All existing tests pass without modification.
- Tutorial Ch. 2 updated with an `AsyncGeoData` section.
- (If §"`RasterioReader` widening" is bundled): new tests for `RasterioReader("s3://...", fs=fsspec_fs)` and `RasterioReader("s3://...", opener=callable)`; Tutorial Ch. 3 updated.

---

## Issue-specific open questions

In addition to the [parent design's open questions](README.md#open-questions), this issue should resolve:

### 1. Should `AsyncGeoData` add a `values_async()` method?

`GeoData.values` is a sync property that materialises via `self.load()`. Properties can't be async, so the natural async equivalent would be `await reader.values_async()`. Tentative: don't add it — `await reader.load()` is fine and `.values` on the *returned* `GeoTensor` works the way users expect.

### 2. Upstream rename of `GeoData` / `GeoDataBase`?

If we wanted `GeoData` → `SyncReader` and `GeoDataBase` → `ReaderMeta` for naming consistency with `AsyncReader`-shaped names, that should happen in `spaceml-org/georeader` proper as a one-line deprecation alias, not as a parallel layer here. Out of scope for this issue. Flagging because the original design tried to do it, and we should be intentional about *not* doing it here.

### 3. Are `path_or_url` / `indexes` ever lifted onto a Protocol?

No. They're reader-construction details. `GeoTensor` shouldn't have to fake them. `FakeGeoData` shouldn't have to declare them. They stay on the concrete reader classes only.

### 4. Should `AsyncGeoData` be `@runtime_checkable`?

`GeoDataBase` and `GeoData` are not currently runtime-checkable (see [Tutorial Ch. 2 §8](../../georeader_tutorial/02_abstract_reader.md)). Tentative: keep `AsyncGeoData` non-runtime-checkable too for symmetry. If we ever flip them, do it together upstream.
