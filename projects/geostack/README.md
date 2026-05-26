---
title: "Geostack — composable operators on real RS imagery"
---

# Geostack worked examples

A self-contained, **chronologically ordered** notebook walkthrough of the
[`pipekit`](https://github.com/jejjohnson/pipekit) +
[`geotoolz`](https://github.com/jejjohnson/geotoolz) +
[`geopatcher`](https://github.com/jejjohnson/geopatcher) stack. Each
notebook is a single self-contained slice that builds on the previous
one. The arc:

| # | Notebook | Substrate | What it shows |
|---|---|---|---|
| 01 | [Composition core walkthrough](notebooks/01_composition_core.ipynb) | scalars | `Operator`, `Sequential`, `Graph`, `Fanout`, `Branch`, `Switch`, `Tap`, `Snapshot`, `ShapeTrace`, `Identity`, `Const`, `Lambda`, `Sink`, `ModelOp`, pickling — the entire composition algebra against plain Python ints. **No GIS setup**; read this first if `Operator` is new. |
| 02 | [Pipeline idioms](notebooks/02_pipeline_idioms.ipynb) | scalars + small numpy | The pipekit idiom gallery: `Profile`, `Histogram`, `Try`, `Coalesce`, `Retry`, `Cache`, `AssertShape`/`AssertDType`/`AssertHasAttribute`, `Quarantine`, plus build-your-own recipes for the few primitives not yet in pipekit (Spy, Diff, Provenance, Subsample, ApplyToBands). |
| 03 | [Operators on Sentinel-2 — Lake Tahoe](notebooks/03_operators_lake_tahoe.ipynb) | real S2 L2A (MPC) | First real-data notebook. STAC search → `GeoTensor` → named ops from `gz.radiometry` / `gz.indices` / `gz.cloud` / `gz.mask`. Ends with a fully instrumented pipeline (`AssertShape | Tap | Profile.wrap | Snapshot | ShapeTrace | AssertDType`). |
| 04 | [Image processing — Caldor fire](notebooks/04_image_processing_caldor.ipynb) | pre/post Sentinel-2 (MPC) | Full `gz.radiometry` display chain (`ToFloat32 | DNToReflectance | PercentileClip | Gamma`), `gz.cloud.MaskFromSCL`, then `gz.Fanout({"ndvi", "nbr", "ndwi"})` from one reflectance carrier. Closes with a USGS-classed **dNBR severity map** over the 2021 burn scar. |
| 05 | [Patching — grid → process → stitch](notebooks/05_patching_grids.ipynb) | real S2 | `geopatcher.SpatialPatcher` + `gz.patch_ops.{GridSampler, ApplyToChips, Stitch}` — the canonical three-op tiled-inference pipeline. Compares `SpatialHann` vs `SpatialBoxcar` windows against the full-scene reference. |
| 06 | [ML patches — augmentations + inference](notebooks/06_ml_patches_augment.ipynb) | real S2 | Same patcher machinery, but the per-chip op is `gz.ModelOp(model, method="predict")` and the stitch is `SpatialHardVote(n_classes=3)`. Demonstrates `gz.augment.Compose([RandomFlip, RandomRotate90, BrightnessJitter, GaussianNoise])` for training-chip augmentation, plus `SpatialJitteredStride` for jittered training-chip sampling. |
| 07 | [Deployment shapes](notebooks/07_deployment_shapes.ipynb) | mixed | The capstone: thirteen deployment patterns (notebook exploration, ETL, FastAPI, tile server, orchestrator, regulatory artifact, benchmark, audit, hot-reload, streaming, …) showing where the same operator algebra fits across production contexts. |

The order is **pedagogical**, not strictly historical: 01–02 establish
the algebra, 03–04 ground it in real imagery and the named-op surface,
05–06 add the patcher / ML layer, 07 sketches deployment. Notebooks
01–02 run against plain scalars and need no MPC access; 03–06 fetch
Sentinel-2 from Microsoft Planetary Computer (anonymous read — no
auth); 07 mixes both.

## Deep dives

Once the applied walkthrough is comfortable, two deep-dive families go
underneath the surface of the stack:

### `notebooks/patching/` — `geopatcher`

| # | Notebook | What it shows |
|---|---|---|
| 01 | [Intro — sliding-window inference](notebooks/patching/01_intro.ipynb) | The four `SpatialPatcher` axes (`Geometry`, `Sampler`, `Window`, `Aggregation`) end-to-end on a single raster. |
| 02 | [Geometries gallery](notebooks/patching/02_geometries.ipynb) | All five geometry types: `Rectangular`, `SphericalCap`, `KNNGraph`, `RadiusGraph`, `PolygonIntersection`. |
| 03 | [Samplers gallery](notebooks/patching/03_sampling.ipynb) | Where anchors go: `RegularStride`, `JitteredStride`, `Random`, `PoissonDisk`, `Explicit`. |
| 04 | [Field backends](notebooks/patching/04_backends.ipynb) | One Patcher, five `Field` adapters: `RasterField`, `XarrayField`, `RioXarrayField`, `XvecField`, `GeoPandasField`, `DaskField`. |
| 05 | [Temporal + spatiotemporal](notebooks/patching/05_time.ipynb) | `TemporalPatcher` along the time axis, then `SpatioTemporalPatcher` composing space × time. |
| 06 | [Streaming reconstruction](notebooks/patching/06_streaming.ipynb) | Zarr accumulator → real GeoTIFFs without materialising the full grid in memory. |

#### Framework recipes (`notebooks/patching/recipes/`)

| Recipe | Bridge |
|---|---|
| [Grain MapDataset](notebooks/patching/recipes/grain.ipynb) | `SpatialPatcher` → JAX `grain.MapDataset` |
| [JAX vmap](notebooks/patching/recipes/jax_vmap.ipynb) | `SpatialPatcher` → `jax.vmap` batched inference |
| [torch Dataset](notebooks/patching/recipes/torch_dataset.ipynb) | `SpatialPatcher` → `torch.utils.data.Dataset` |

### `notebooks/catalog/` — `geocatalog`

| # | Notebook | What it shows |
|---|---|---|
| 01 | [Intro — build → query → load](notebooks/catalog/01_intro.ipynb) | Build a catalog from a real Sentinel-2 L2A archive on MPC (eight scenes over Lake Tahoe), query it, mosaic and time-stack the matches. |
| 02 | [Backends](notebooks/catalog/02_backends.ipynb) | Raster, xarray, and vector catalog backends. |
| 03 | [Set algebra](notebooks/catalog/03_set_algebra.ipynb) | `query`, intersect, union — composable catalog operations. |
| 04 | [DuckDB at scale](notebooks/catalog/04_duckdb.ipynb) | DuckDB-backed catalogs over millions of items. |
| 05 | [Catalog ↔ Patch bridge](notebooks/catalog/05_patching_bridge.ipynb) | `CatalogDomain` plugs the multi-file archive into the same `SpatialPatcher` pipeline. |

The applied walkthrough (01–07 at the top) shows *what* the stack does
on real data; the deep dives show *every knob* on each axis with
worked examples.

## Layout

```
projects/geostack/
├── pyproject.toml          # standalone "geostack" package
├── README.md               # this file
├── src/geostack/           # shared real-data loaders (data.py)
├── tests/                  # smoke tests for the loaders
└── notebooks/
    ├── 01_composition_core.ipynb
    ├── 02_pipeline_idioms.ipynb
    ├── 03_operators_lake_tahoe.ipynb
    ├── 04_image_processing_caldor.ipynb
    ├── 05_patching_grids.ipynb
    ├── 06_ml_patches_augment.ipynb
    ├── 07_deployment_shapes.ipynb
    ├── patching/           # 6 deep dives + recipes/ (3 framework bridges)
    └── catalog/            # 5 deep dives
```

Each `*.ipynb` ships with an executed copy (figures inline). To
re-execute a single notebook against fresh MPC data:

```bash
pixi run -e geostack jupyter nbconvert --to notebook --execute --inplace \
    projects/geostack/notebooks/03_operators_lake_tahoe.ipynb
```

## Reproducing

The parent `research_notebook` pixi file defines a `geostack` feature /
environment that bundles all the deps (geotoolz, geopatcher,
geocatalog, planetary-computer, pystac-client, rioxarray, matplotlib,
ipykernel, nbconvert, scipy, duckdb, pyogrio, netcdf4, xvec, …).

```bash
# One-time install
pixi install -e geostack

# Re-execute scoped subsets (each task targets one notebook tier).
pixi run -e geostack execute-geostack            # applied walkthrough (01–07)
pixi run -e geostack execute-geostack-patching   # patching/ deep dives + recipes
pixi run -e geostack execute-geostack-catalog    # catalog/ deep dives

# Convenience: applied + patching + catalog in one shot.
pixi run -e geostack execute-geostack-all

# Smoke-test the geostack.data loaders against MPC / GBIF / Natural Earth.
pixi run -e geostack test-geostack

# Or run a single notebook
pixi run -e geostack jupyter nbconvert --to notebook --execute --inplace \
    projects/geostack/notebooks/03_operators_lake_tahoe.ipynb
```

For non-pixi users, the standalone `pyproject.toml` here pins the same
deps; `uv pip install -e projects/geostack` (or `pip install -e .`)
into an activated venv works equivalently.

## Why these scenes?

- **Lake Tahoe (10SGJ), June 14 2024 (0.01 % cloud)** — peak-green
  Sierra window. NDVI and NDWI light up cleanly, the lake provides a
  hard-water signal, and the surrounding granite + forest gives
  contrast. Used by notebooks 03, 05, 06.
- **Caldor fire core (10SGH), Aug 9 vs Nov 27 2021** — the second S2
  scene ever to cross the Sierra Nevada crest. Pre / post bracketing
  shows a dramatic dNBR severity gradient. Used by notebook 04.

Both AOIs are on Microsoft Planetary Computer's anonymous Sentinel-2
L2A read path — no API key, no signed URLs to manage — so the
reproduction is one `pixi run` from a clean clone.

## Cross-references back to geotoolz

The notebooks link out to the `geotoolz` docs for concept pages and
API reference (Concepts, Define an operator, Branching pipelines,
Integration with geocatalog & geopatcher, Core API). Those links point
at the canonical source on
[github.com/jejjohnson/geotoolz](https://github.com/jejjohnson/geotoolz/tree/main/docs).
This project is the **applied** companion; `geotoolz/docs` is the
**library reference**.
