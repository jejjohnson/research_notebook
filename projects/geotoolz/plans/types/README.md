# Core types

> **Status:** index of cross-cutting type designs.
> **Scope:** types and small dataclasses that are consumed by *more than one* of the major designs (reader, catalog, geotoolz operators) and therefore deserve their own home rather than being defined inside whichever design happens to need them first.
> **Audience:** anyone touching a type that flows between layers.

---

## What goes here

A type lands in this directory when **all three** are true:

1. **It's a public surface** — users construct or pattern-match against it directly, not just an internal helper.
2. **It's consumed by more than one design** — moves between layers (reader → catalog → operator) rather than being scoped to a single subsystem.
3. **It's small enough to specify in one document** — a dataclass, a Protocol, or a small family of related primitives. Big subsystems (the reader Protocol surface, the catalog Protocol) get their own design dirs.

The georeader-side types that *aren't* here, and why:

- **`GeoTensor`** — already a real implemented type in `georeader/geotensor.py`; documented in [Tutorial Ch. 1](../../georeader_tutorial/01_geotensor.md). No design doc needed.
- **`GeoData` / `GeoDataBase` / `_ReaderMeta` / `SyncReader` / `AsyncReader`** — all live in [Reader reconciliation](../georeader/README.md) because they're the subject of that design, not just incidental to it.
- **`GeoCatalog` Protocol** — lives in [Geodatabase](../geodatabase/README.md) for the same reason.
- **`ByteStore`** — lives in [`reader_lazy_cog.md`](../georeader/reader_lazy_cog.md) for the same reason.

If a type starts in another design and grows into something multiple designs reference, **promote it here** — the cleanup is the same shape as the GeoSlice promotion that motivated this directory.

---

## Current designs

| Design | Type(s) covered |
|---|---|
| [`geoslice.md`](geoslice.md) | `GeoSlice` dataclass + the sampler/stitch family (`random_sampler`, `grid_sampler`, `stitch`) that produces and consumes `GeoSlice`. |

---

## Future candidates

These are types that *might* land here as the geotoolz ecosystem grows. Listed for orientation, not commitment:

- **`Operator` Protocol** — if a shared base class for `geotoolz` operators turns out to be reused by other libraries (e.g., a sibling `xr_toolz`-shaped library), it'd live here. Today it's scoped to [`geotoolz.md`](../geotoolz/geotoolz.md).
- **A `Chip` or `Window` reconciliation type** — if `GeoSlice`, `rasterio.windows.Window`, and `slices.create_windows` outputs end up needing a unified shape.
- **A `Sensor` / `Mission` metadata struct** — if sensor-preset operators ([geotoolz.md §1.2](../geotoolz/geotoolz.md)) need to share a structured description of band layout, calibration constants, etc. Today this lives ad-hoc inside each reader (e.g., `BANDS_S2`, `BANDS_S2_L2A`).

When a candidate becomes a real design, it lands here as a sibling to `geoslice.md`.

---

## Conventions

- **One file per type family.** A "type family" can include a small number of closely-coupled types — e.g., `GeoSlice` plus the three samplers and `stitch` that produce/consume it — but not unrelated types stuffed together for filing convenience.
- **Same design-doc skeleton as the other designs:** Status / Scope / Motivation / Goals / Non-goals / Constraints / The type itself / Connections to other designs / Open questions / Alternatives.
- **Keep concrete enough to implement.** The whole point of pulling a type out is that it gets the same attention as a subsystem — meaning a real Protocol or dataclass spec, not a sketch.
