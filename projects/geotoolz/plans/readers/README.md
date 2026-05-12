---
title: Sensor readers
subject: Sensor readers
subtitle: Per-sensor reader designs in georeader
short_title: Sensor readers
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, readers, sensors
---

> **Status:** index of per-sensor reader designs.
> **Scope:** the design of the *reader classes* for specific sensor families being migrated into `georeader` from `rs_tools`.
> Each per-sensor design specifies the file format, metadata parsing, calibration, and how the reader fits into one of `georeader`'s existing patterns (S2-style affine `GeoData`, or PRISMA-style raw-arrays-plus-`lons`/`lats`).
> **Audience:** anyone implementing a new sensor reader in `georeader`, or trying to figure out which sensor goes with which existing pattern.

---

## What goes here

Per-sensor reader designs land here when:

1. **The sensor is a candidate for migration into `georeader`** — typically from `rs_tools` or a similar prototyping repo.
2. **The reader follows one of two existing patterns** ([Track A or Track B below](#the-two-tracks)) but has sensor-specific quirks worth documenting before implementation.
3. **The work fits into one or more focused issues** rather than a single trivial PR.

The companion to this directory is [`georeader/`](../georeader/) — the **reader-protocol reconciliation** design, which keeps today's `GeoData` / `GeoDataBase` Protocols and adds an `AsyncGeoData` Protocol that all readers (current and future) will conform to.
That design is about the *interface*; this directory is about *concrete sensor implementations* that satisfy it.

---

## The two tracks

`georeader` already has two patterns for reader implementations, distinguished by the geometry of the data on disk:

| Track | Geometry | Existing examples | Pattern |
|---|---|---|---|
| **A** — clean affine | Sensor data sits on a regular grid in a standard CRS; the file format ships an affine transform (or one is recoverable from metadata). | Sentinel-2 SAFE, Landsat | Reader subclasses [`GeoData`](../../georeader_tutorial/02_abstract_reader.md). Reads route through `read.read_from_bounds`. |
| **B** — irregular geolocation | Sensor data is a raw array with per-pixel `lons` / `lats`; no honest affine. | EMIT, PRISMA, EnMAP (curvilinear-but-orthorectified) | Reader exposes raw arrays + geolocation; downstream calls [`griddata.read_to_crs`](../../georeader_tutorial/07_griddata.md) to resample to a regular grid. |

**Most new sensors fit one of these two patterns.** The per-sensor design docs in this directory specify which track applies and what sensor-specific glue is needed.

---

## Current designs

| Design | Sensors | Track |
|---|---|---|
| [`geostationary.md`](geostationary.md) | GOES-R ABI, MSG SEVIRI, MTG-FCI, Himawari AHI | A for ABI/FCI (clean `+proj=geos`), B for SEVIRI/AHI (irregular file formats) |
| [`modis.md`](modis.md) | MODIS (Aqua/Terra), VIIRS (S-NPP / NOAA-20/21), planned: AVHRR, Sentinel-3 OLCI/SLSTR, AVIRIS-NG | B (curvilinear scanners; per-pixel `lons`/`lats` is the only honest description) |

Both designs share structure: User Story / Motivation / Mathematics / Target API / Example Use Cases / Subtasks / Open Design Questions.
They reference each other directly because the geostationary readers were designed first and the MODIS family followed the same template.

---

## Future candidates

These are sensor families that *might* land here as the migration progresses:

- **Landsat L2 surface reflectance** — Track A, but with ARD / Collection-2-level metadata quirks.
- **Sentinel-3 OLCI / SLSTR** — currently mentioned in `modis.md` as a future Track-B addition.
- **AVHRR** — same; Track B, polar-orbit scanner.
- **AVIRIS-NG** — airborne hyperspectral; per-pixel geolocation, similar to PRISMA.
- **PACE / OCI** — future ocean-color hyperspectral; geometry TBD.
- **Commercial high-res** — Planet, Maxar, etc., if licensing makes them in scope.

When a candidate becomes a real design, it lands here as a sibling to the existing files.

---

## Connections to other designs

| Design | How sensor readers touch it |
|---|---|
| [Reader reconciliation](../georeader/README.md) | All readers (existing and new) conform to the `GeoData` (sync) or `AsyncGeoData` (async) Protocol surface defined there. The Protocol locks the interface; this directory specifies the per-sensor implementations. |
| [Geodatabase](../geodatabase/README.md) | A `GeoCatalog` indexes files; the reader is what's used to *open* one row's file. The `reader_class=...` kwarg on `CatalogPipeline` selects which reader (and therefore which sensor pattern) is in use. |
| [Core types — `GeoSlice`](../types/geoslice.md) | Once a reader satisfies `GeoData.read_geoslice(slice)` (or the `AsyncGeoData` async equivalent), it slots into the sampler/loader pipeline regardless of which track it follows. |
| [`geotoolz.md`](../geotoolz/geotoolz.md) | Sensor-preset operators in `geotoolz.presets.*` (S2, EMIT, EnMAP, ...) wrap the readers specified in this directory. |

---

## Conventions

- **One file per sensor family**, not per individual sensor — closely related sensors with shared geometry (e.g., GOES-R ABI + MTG-FCI; MODIS + VIIRS) are co-documented in one file.
- **Same skeleton across files** so readers cross-comparing two sensor designs can find the same sections in the same places.
- **Track membership flagged in the file header** so a quick scan tells you which existing pattern applies.
- **Open Design Questions section** at the end — sensor-specific quirks that need a decision before implementation.
