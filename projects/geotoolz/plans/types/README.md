---
title: Core types
subject: Core types
subtitle: Cross-cutting types consumed by multiple designs
short_title: Core types
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, types, protocol
---

> **Status:** index of cross-cutting type designs.
> **Scope:** types and small dataclasses that are consumed by *more than one* of the major designs (reader, catalog, geotoolz operators) and therefore deserve their own home rather than being defined inside whichever design happens to need them first.
> **Audience:** anyone touching a type that flows between layers.

---

## Primer for newcomers

> **ELI5.** Some Lego pieces have **unique shapes that only fit in one castle**.
> Others are the **standard 2×4 brick** that fits in everything.
> This directory is for the standard 2×4 bricks of the design — small, reusable, with one shape that lots of places need.

### What's a "type" in this directory?

**What it is.** A *type* in this directory is a Python construct — usually a `Protocol`, occasionally a `@dataclass` — that defines a *shape* (what attributes/methods/fields a value must have) without owning the implementation.
Types live here when more than one design references them.

**How it works.** Three flavours show up:

- **`typing.Protocol`** — structural typing.
  Any class with the right method signatures satisfies it; no inheritance required.
  Used for `Credential` and the reader Protocols (`GeoData` / `GeoDataBase` / `AsyncGeoData`).
  The static type-checker (`mypy` / `ty`) verifies conformance at the call site.
- **`@dataclass`** — auto-generated `__init__` / `__repr__` / `__eq__`.
  Used for `GeoSlice`.
  Often `frozen=True` to make instances immutable and hashable.
- **Concrete subclasses** of a Protocol — `AzureSASCredential`, etc. Live alongside the Protocol they implement, in the same design doc.

**What this means for us.** Code that takes a `Protocol` parameter (`def f(reader: GeoData)`) accepts any conforming object — no shared base class, no inheritance dance.
Code that takes a `@dataclass` parameter (`def f(slice_: GeoSlice)`) gets all the dataclass machinery (immutable fields, equality, repr) for free.
The patterns are deliberately small and uniform across the directory.

### When does a type land here vs in a design subdir?

**What it is.** The criterion for promoting a type to `plans/types/`.
Three conditions, all of which must hold.

**How it works.** A type lives here when (1) it's a public surface — users construct or pattern-match against it directly, (2) it's consumed by more than one of the major designs, and (3) it's small enough to specify in one document.
Types that fail any of these stay scoped to their owning design — e.g., `GeoData` / `GeoDataBase` / `AsyncGeoData` live in `georeader/reader_protocol.md` because they're the *subject* of that design, not just incidental to it.
Same logic kept `GeoCatalog` in geodatabase and `Operator` in geotoolz.

**What this means for us.** This directory grows slowly.
Most "types" stay in their owning subdir; the ones that end up here are the ones that flow *between* layers (`GeoSlice`, `Credential`).
The `bytestore.md` doc is here too but is **not a Protocol of our own** — it documents the decision to defer cloud byte transport to upstream [`obspec`](https://github.com/developmentseed/obspec) (see [`bytestore.md`](bytestore.md)).

```{mermaid}
flowchart LR
    GS[GeoSlice] --> Reader[Reader designs]
    GS --> Cat[Geodatabase]
    GS --> Op[geotoolz operators]
    Cred[Credential] --> Reader
    OB["obspec.AsyncStore<br/>(external — not a type we own)"] --> Reader

    style GS fill:#e1f5ff,stroke:#0288d1
    style Cred fill:#e1f5ff,stroke:#0288d1
    style OB fill:#f5f5f5,stroke:#999,stroke-dasharray: 5 5
```

---

## What goes here

A type lands in this directory when **all three** are true:

1. **It's a public surface** — users construct or pattern-match against it directly, not just an internal helper.
2. **It's consumed by more than one design** — moves between layers (reader → catalog → operator) rather than being scoped to a single subsystem.
3. **It's small enough to specify in one document** — a dataclass, a Protocol, or a small family of related primitives.
   Big subsystems (the reader Protocol surface, the catalog Protocol) get their own design dirs.

The georeader-side types that *aren't* here, and why:

- **`GeoTensor`** — already a real implemented type in `georeader/geotensor.py`; documented in [Tutorial Ch. 1](../../georeader_tutorial/01_geotensor.md).
  No design doc needed.
- **`GeoData` / `GeoDataBase` / `AsyncGeoData`** — all live in [Reader reconciliation](../georeader/README.md) because they're the subject of that design, not just incidental to it.
- **`GeoCatalog` Protocol** — lives in [Geodatabase](../geodatabase/README.md) for the same reason.
- **`obspec.AsyncStore`** — the cloud-byte transport surface.
  It's *not* ours — we defer to upstream [`obspec`](https://github.com/developmentseed/obspec) (DevSeed).
  The note at [`bytestore.md`](bytestore.md) documents this decision and the small `geotoolz.io.open_store(url)` helper we ship.

If a type starts in another design and grows into something multiple designs reference, **promote it here** — the cleanup is the same shape as the GeoSlice promotion that motivated this directory.

---

## Current designs

| Design | Type(s) covered |
|---|---|
| [`geoslice.md`](geoslice.md) | `GeoSlice` dataclass + the sampler/stitch family (`random_sampler`, `grid_sampler`, `stitch`) that produces and consumes `GeoSlice`. |
| [`credentials.md`](credentials.md) | `Credential` Protocol + per-cloud subclasses (`AzureSASCredential`, `AzureManagedIdentityCredential`, `AWSStaticCredential`, `AWSProfileCredential`, `GCSServiceAccountCredential`) + `from_config(...)` factory + `to_obstore_*_store()` adapter helpers. Replaces the env-var-soup pattern that every project currently re-implements. |
| [`bytestore.md`](bytestore.md) | One-page passthrough note: cloud byte transport is `obspec.AsyncStore` (upstream, not ours). We ship a small `geotoolz.io.open_store(url)` factory and nothing else. **Not** a Protocol of our own. |

---

## Future candidates

These are types that *might* land here as the geotoolz ecosystem grows.
Listed for orientation, not commitment:

- **`Operator` Protocol** — if a shared base class for `geotoolz` operators turns out to be reused by other libraries (e.g., a sibling `xr_toolz`-shaped library), it'd live here.
  Today it's scoped to [`geotoolz.md`](../geotoolz/geotoolz.md).
- **A `Chip` or `Window` reconciliation type** — if `GeoSlice`, `rasterio.windows.Window`, and `slices.create_windows` outputs end up needing a unified shape.
- **A `Sensor` / `Mission` metadata struct** — if sensor-preset operators ([geotoolz.md §1.2](../geotoolz/geotoolz.md)) need to share a structured description of band layout, calibration constants, etc. Today this lives ad-hoc inside each reader (e.g., `BANDS_S2`, `BANDS_S2_L2A`).

When a candidate becomes a real design, it lands here as a sibling to `geoslice.md`.

---

## Conventions

- **One file per type family.** A "type family" can include a small number of closely-coupled types — e.g., `GeoSlice` plus the three samplers and `stitch` that produce/consume it — but not unrelated types stuffed together for filing convenience.
- **Same design-doc skeleton as the other designs:** Status / Scope / Motivation / Goals / Non-goals / Constraints / The type itself / Connections to other designs / Open questions / Alternatives.
- **Keep concrete enough to implement.** The whole point of pulling a type out is that it gets the same attention as a subsystem — meaning a real Protocol or dataclass spec, not a sketch.

---

## Open questions, gotchas, and warnings

The cross-cutting types are the load-bearing ones — when they're wrong, every downstream design ripples.
Things to watch:

- **`Credential.apply()` per-call vs `apply_to_os_environ()` mutation.** The design distinguishes pure (returns a dict) vs mutating (writes `os.environ`) modes.
  Per-reader isolation only works if every reader consumes the per-call dict; the env-var mode is a backwards-compat hatch, not the recommended path.
  Audit each reader on the reconciliation path to make sure it accepts the per-call dict.
- **`obspec` API stability.** `obspec` is the upstream Protocol (DevSeed); `obstore` is the reference implementation.
  Both are pre-1.0. Our `geotoolz.io.open_store(...)` returns `obspec.AsyncStore`, so any breaking change there ripples through `AsyncGeoTIFFReader`.
  Pin a minor range and bump deliberately.
- **`GeoSlice` time-axis convention.** `GeoSlice.interval` is a `pd.Interval`; the slice can be spatial-only (no interval) or spatial+temporal. Document the discriminated-union behaviour clearly so samplers and stitchers don't trip over the optional time dimension.
- **Sampler API stability.** `random_geo_sampler` / `grid_geo_sampler` are bigger than they look — they couple `GeoSlice`, the catalog, and the inference loop.
  Treat their signatures as v0.1 public API; bump `_v2` if the shape needs to change rather than mutating in place.
- **`stitch_predictions` reduction semantics.** Average vs first-write-wins vs max-vote — pick a default, document the alternatives, expose a `method=` knob.
  Without this, every user reinvents stitching.
- **DuckDB credential bridge.** Reading cloud GeoParquet from DuckDB needs `httpfs` configured with credentials.
  The `Credential` Protocol should provide a `to_duckdb_secret()` adapter so users don't configure auth twice.
  Out-of-scope for v0.1, in-scope for the geodatabase Phase 2 design — flag the dependency.
- **Future-candidate types are not commitments.** `Operator` Protocol, `Chip`/`Window`, `Sensor`/`Mission` are listed for orientation.
  Each lands here only when a second design references it.
  Resist promoting speculative types — they're cheap to add later, expensive to retract.
