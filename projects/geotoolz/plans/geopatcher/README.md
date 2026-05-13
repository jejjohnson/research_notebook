---
title: geopatcher
subject: geopatcher design
subtitle: A four-axis Patcher framework over the geotoolz Field/Domain layer
short_title: geopatcher
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, geopatcher, patching, locality, streaming, geotoolz
---

> **Status:** design proposal — three companion documents below (`design.md`, `examples.md`, `scaling.md`).
> **Shipping shape:** incubated as `geotoolz.patch` inside the `geotoolz` library — *not* a standalone package at v0.1. Graduation to a standalone `geopatcher` package is future work, gated on API stability and a real external user that wants the patcher without the operator algebra.
> **Scope:** a backend-agnostic *Patcher* abstraction that controls **locality** in the geotoolz stack — how a global field is split into local patches, how operators consume them, and how local outputs are merged back into a global field.
> **Audience:** anyone composing operators that need a notion of *receptive field* (CNNs, FNOs, GPs, GNNs, neural processes), anyone building inference pipelines that stream over fields that don't fit in RAM, and anyone writing per-tile training datasets on top of [`georeader`](../georeader/README.md) / [`geotoolz.catalog`](../geodatabase/README.md).

---

## Summary

`geopatcher` is the **locality layer** of the geotoolz stack.
Where the [`Field` / `Domain` adapter layer](../georeader/README.md) settles "*what backend the data lives on*" and [`geotoolz`](../geotoolz/README.md) settles "*what operator runs on it*," `geopatcher` settles the third orthogonal question: "*what slice of the data does the operator see at once, and how do local outputs become a global field?*"

The whole framework is one object with four pluggable axes:

| Axis | Controls | Examples |
|------|----------|----------|
| **`PatchGeometry`** | Shape + scale of the neighbourhood (and the domain topology) | `Rectangular`, `SphericalCap`, `KNNGraph`, `RadiusGraph`, `PolygonIntersection` |
| **`Sampler`** | Where anchors are placed (overlap/redundancy is emergent) | `RegularStride`, `JitteredStride`, `Random`, `PoissonDisk`, `Explicit` |
| **`Window`** | Boundary treatment (spectral leakage, edge artefacts) | `Boxcar`, `Hann`, `Tukey`, `Gaussian` |
| **`Aggregation`** | Local predictions → global field | `OverlapAdd`, `Mean`, `WeightedSum`, `InvVarWeightedMean`, `ByIndex`, `Learned` |

The `Patcher` composes the four axes and, in the current proposal, exposes a tiny surface: `split(field) → Iterator[Patch]` and `merge(patches, domain) → Field`.
The operator sits *outside* the Patcher — that's the whole point. Patching handles locality; the operator handles modelling. Swap either independently.

`geopatcher` is the geotoolz-native realisation of what `xrpatcher` does for xarray cubes (referenced in [`geotoolz.md` §5](../geotoolz/geotoolz.md)) — same composition idea, but anchored on the `Field` / `Domain` protocols so it covers rasters, gridded cubes, points, polygons, and (later) graphs/meshes through one Protocol-dispatched surface.

**Inside `geotoolz`, this is the canonical home for every concrete sampler, window, and aggregation primitive.** Earlier drafts of the catalog plan proposed putting `grid_geo_sampler` / `random_geo_sampler` / `stitch_predictions` into `georeader.samplers`; that was a layering inversion. `georeader` (the separate upstream library) owns the substrate (reader Protocols, `GeoTensor`, byte paths). `geotoolz` (this library) owns everything else, with two incubation submodules: `geotoolz.catalog` (file indexing) and `geotoolz.patch` (the patching/sampling algebra specified here). `geotoolz.ops.GridSampler` / `ApplyToChips` / `CatalogPipeline` become thin operator wrappers around `geotoolz.patch.Patcher` rather than bespoke samplers. The ownership table is spelled out in [`design.md` §1 "Ownership: who lives where in the geotoolz stack"](design.md#1-patching-the-locality-layer).

### Why incubate as a submodule rather than ship as a standalone package?

Two libraries (`georeader` + `geotoolz`) are cheaper to maintain than four, and the patching abstraction needs to earn its keep against real operator graphs before it commits to a stable public API. Living as `geotoolz.patch` means:

- **API can churn freely** during the v0.1–v0.3 phase. No external semver promise.
- **`Operator`/`Sequential`/`Graph` are the forcing function** for whether the four-axis Patcher actually fits geoscience workflows. Bad abstractions get spotted faster when their primary user lives in the same repo.
- **One install, one dep graph.** Users who reach for `geotoolz` for chip-iteration get the patcher; users who only want operators don't pay extra.
- **Graduation is a planned event, not a perpetual ambition.** When the API stabilises and a real external user wants the patcher without the operator algebra, extract `geotoolz.patch → geopatcher` with `from geotoolz.patch import *` re-export shims + `DeprecationWarning`. Same pattern Keras used for layer/optimizer subpackages, and `sklearn.experimental` uses today.

---

## What's in the design

Three companion documents make up this plan:

| File | What's in it |
|---|---|
| [`design.md`](design.md) | The four-axis framework end-to-end. §1 Patching (the four axes + pseudocode + data-flow diagram). §2 Models (parameter sharing × pooling × predict scope — complete / no / partial pooling). §3 Algorithm families by data domain (spectral / graph / kernel / spectral-on-manifold). §4 Backends — the `Field` / `Domain` adapter layer (`RasterioField`, `XarrayField`, `GeoPandasField`, `XvecField`, …) with single-dispatch `PatchGeometry.neighborhood`. §5 Time as a peer axis — `TimePatcher` + `SpatioTemporalPatcher` with product vs coupled coupling. §6 The unified stack diagram. |
| [`examples.md`](examples.md) | A 7-case gallery of `(Geometry, Domain)` pairings with the resulting `Patch[AnchorT, IndicesT, DataT]` types: Rectangular×Raster (tiling a satellite scene), Rectangular×Grid (patching ERA5), SphericalCap×Grid (global SST around stations), RadiusGraph×Point (Argo float neighbourhoods), RadiusGraph×Vector (admin polygons near a facility), KNNGraph×Point (fixed-k GNN training), PolygonIntersection×Raster (facility-level methane aggregation). Closes with the **uniform vs ragged** taxonomy that drives downstream batching. |
| [`scaling.md`](scaling.md) | The streaming story. §1 The five memory pressure points in `split → operate → merge`. §2 What the framework gives you for free (lazy `Field.select`, `Sampler.anchors` iterators). §3 Four extensions: `Patcher.split` as `Iterator`, disk-backed `Aggregation` (Zarr/memmap), hierarchical Patcher-of-Patchers, `streaming_safe` ClassVar. §4 The streaming asymmetry — *output* preallocability is the bottleneck, not input. §5 Two workflows: training-time `PatchDataset` + inference-time disk-backed reconstruction. §6 A full family of streaming-compatible aggregations (exact monoidal + approximate sketches + `InvVarWeightedMean` for Bayesian merge). |

---

## Why a Patcher abstraction (and why now)

Three reasons make a unified Patcher worth doing in the geotoolz stack:

1. **Every operator that uses locality reimplements patching.** CNNs slide windows. FNOs cut tiles. Local GPs build kdtree radius queries. GNNs assemble k-nearest neighbours. Each of these is a `(geometry, sampler, window, aggregation)` choice with one of the four axes varied — but no shared object captures the composition. Promoting the pattern into `geopatcher` collapses dozens of bespoke "tile this thing" loops into one configurable Patcher and lets the operator stay focused on modelling.

2. **The natural locality knob is the same across operator families.** `Rectangular` on a raster, `RadiusGraph` on points, and `SphericalCap` on a global grid all answer the same question — *what does the operator see at once?* — for very different data geometries. A typed `PatchGeometry` with single-dispatch over `Domain` makes the same Patcher composition serve CNN, GP, and GNN workflows without per-operator glue.

3. **Streaming, lazy reads, and hierarchical composition are all properties of the Patcher**, not of any individual operator. Once the Patcher returns an `Iterator[Patch]` and aggregations can write into a disk-backed store, every operator that consumes the Patcher inherits these properties for free. That's how `geopatcher` makes the same composition pattern carry from a notebook through to petabyte-scale reconstruction.

The framework slots into the existing geotoolz layering exactly between the `Field` / `Domain` substrate (consumed via `Field.select`) and the `Operator` layer (consumed via the `(patch → output)` callable). It needs *no new substrate*, *no new operator surface* — only a small protocol over the existing ones.

---

## Where it sits in the geotoolz stack

```
   Operator (geotoolz / xrtoolz)         ◄── what to compute
        │
        │ consumes Patch[AnchorT, IndicesT, DataT]
        ▼
   Patcher (this design)                 ◄── what slice it sees + how outputs merge
        │   ├── PatchGeometry            (dispatch over Domain)
        │   ├── Sampler                  (anchor placement)
        │   ├── Window                   (boundary treatment)
        │   └── Aggregation              (local → global; streaming-safe property)
        │
        │ produces patches over
        ▼
   Field / Domain (georeader + georeader bridges)
        ├── RasterioField  ─┐
        ├── GeoTensorField  ├──→ RasterDomain
        ├── RioXarrayField ─┘
        ├── XarrayField      ──→ GridDomain
        ├── GeoPandasField   ──→ VectorDomain
        └── XvecField        ──→ PointDomain
        │
        ▼
   GeoCatalog (geodatabase)              ◄── which files to feed the Field in
```

The Patcher is the missing middle. Without it, the operator either bakes in its own patching (CNN-style, FNO-style) or pretends the field fits in RAM. With it, the same operator composes against any backend, any locality scheme, and any scale.

---

## Connections to other designs

`geopatcher` sits above the substrate layer and below the operator layer, so the cross-design touchpoints are concrete:

| Design | What `geopatcher` consumes / produces |
|---|---|
| [Reader reconciliation](../georeader/README.md) | `Field` / `Domain` protocols are the substrate the Patcher splits over. Lazy `Field.select(indexer)` is the source of *streaming for free* (`scaling.md` §1). New `Field` adapters slot in by registering with the `singledispatch` in `PatchGeometry.neighborhood`. |
| [Geodatabase](../geodatabase/README.md) | A `GeoCatalog` selects *which files become a `Field`*; the Patcher then splits that `Field` into patches. The two compose: catalog → field → patcher → operator → aggregation. Hierarchical Patcher-of-Patchers (`scaling.md` §3) is the natural unit of distribution over a `GeoCatalog`. |
| [Core types — `GeoSlice`](../types/geoslice.md) | `Rectangular × RasterDomain` produces a rasterio `Window`; `Rectangular × GridDomain` produces a `dict[str, slice]`. A future thin shim can produce a `GeoSlice` directly, making Patcher anchors interchangeable with the existing sampler primitives in `GeoSlice` land. |
| [`geotoolz`](../geotoolz/README.md) | The operator graph is the *consumer* of `Patch[…]`. `geotoolz.inference.ApplyToChips`-style chip iteration is the v0.1 special case of `Patcher(Rectangular × RegularStride × Hann × OverlapAdd)`. `geotoolz.sampling.GridSampler` is the special case of `Sampler` for raster grids. The general Patcher subsumes both. |
| [`xrtoolz` / `xrpatcher`](https://github.com/jejjohnson/xr_toolz) | `xrpatcher` is the xarray-native peer: same four-axis composition idea, but specialised to `Dataset` / `DataArray`. `geopatcher` is the geotoolz-native version unified across raster, grid, point, and vector domains through `Field` / `Domain`. The two libraries share the composition pattern; the substrates differ. |
| Spatiotemporal tutorials | The [data tutorial](../../tutorials/spatiotemporal_data.md) and [operators tutorial](../../tutorials/spatiotemporal_operators.md) lay the conceptual ground — canonical `[S, T, V, X]` tensor, coordinates, geometry, fit-scope × predict-scope dependency game, and a KNN/Radius preview. `geopatcher` is where those concepts become a concrete framework with code. |

---

## Open questions

The design is largely complete in the three companion documents; the architectural questions still open are:

1. **API surface for `Patcher.split` — `list` or `Iterator`?** `scaling.md` §3 argues for `Iterator` as the default with `list(patcher.split(field))` as the eager opt-in. This is a small protocol-level decision but ripples through every consumer; lock it in before v0.1.
2. **Disk-backed aggregation: framework-managed Zarr vs user-provided.** `scaling.md` §3 proposes `OverlapAdd(streaming=True, target_path=..., chunks=...)`. The alternative is to pass in an opened Zarr store. The framework-managed path is friendlier for casual users; the user-provided path is essential for Dask/distributed writers. Likely ship both, with the framework-managed one as the documented default.
3. **`streaming_safe` — warning vs hard error vs configurable.** Currently a warning. A global `strict` flag would let CI-like contexts fail fast without per-call API noise. `scaling.md` §3.4 flags this open.
4. **`SpatioTemporalPatcher` coupling modes.** `design.md` §5 names `"product"` and `"coupled"`. Coupled-anchor semantics (explicit `(space, time)` tuples) need a concrete data type — likely `list[SpatioTemporalAnchor]`. Defer until the first event-triggered use case (Argo profiles, methane plume detections, storm tracks) lands.
5. **Where does `geopatcher` live — its own package, a `geotoolz.patch` submodule, or in `xr_toolz`?** Same question that opened with `xrpatcher`. Tentative pick: a standalone `geopatcher` package, with adapters in `geotoolz` and `xrtoolz` that wire it up to each carrier. Worth revisiting once a real user lands.
6. **Distributed reconstruction.** Zarr writes parallelise across workers via Dask; the hierarchical Patcher-of-Patchers is the natural unit of distribution. Worth documenting as a *recipe* on top of the framework rather than baking distribution into the Patcher class.
7. **Two-pass / global-context operators.** Global normalisation, attention across all patches, and learned aggregations need a global statistic before the per-patch update can fire. `scaling.md` §4 sketches the two-pass pattern; whether a dedicated `TwoPassPatcher` earns its keep depends on how often this comes up in real workflows.

---

## Why this README is a thin wrapper

The detailed design lives in [`design.md`](design.md), [`examples.md`](examples.md), and [`scaling.md`](scaling.md) — together they cover the four axes, the substrate dispatch, the type signatures across every `(Geometry, Domain)` pairing, the streaming asymmetry, and the family of streaming-compatible aggregations.

This README is a navigation entry point and the place where other designs in the geotoolz plan tree cross-link to `geopatcher`. If you're cross-referencing it from another plan, prefer linking to a specific section of one of the three companion documents — readers want to land on the section that's relevant to their question, not navigate two levels down.
