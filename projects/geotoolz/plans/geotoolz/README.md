---
title: geotoolz
subject: geotoolz design
subtitle: Operator-composition library overview
short_title: geotoolz
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, geotoolz, operators, remote-sensing
---

> **Status:** design proposal — full report in [`geotoolz.md`](geotoolz.md).
> **Scope:** a composable Operator library for remote sensing, sitting on top of `georeader.GeoTensor`. Sibling to `xr_toolz`, targeted at a different community and a different substrate.
> **Audience:** anyone building RS pipelines that compose preprocessing, inference, and evaluation steps; anyone consuming a `georeader` `GeoCatalog` and wanting to apply per-tile operators at scale.

---

## Summary

`geotoolz` is the **operator-composition layer** above `georeader`. Where `georeader` reads bytes and produces `GeoTensor`s, `geotoolz` lets users *compose* those reads with classification heads, spectral-index calculations, masking, mosaicking, sampling, inference, and writing — declared as a `Sequential` of operators or a `Graph` of named ops, executed eagerly or driven by a Hydra config.

It's the RS-shaped sibling to `xr_toolz` (the climate-side composition library): same architectural patterns, separate codebase, different substrate. `GeoTensor` is the shared currency at the bottom; the operator surface is built fresh for each library's audience.

---

## What's in the design

The full design report is [`geotoolz.md`](geotoolz.md). It runs through the architecture in 10 numbered sections; the high-level outline:

| § | Topic |
|---|---|
| 0 | Inputs the design draws on (`xr_toolz` patterns, `georeader` substrate, `jej_vc_snippets` empirical vocabulary) |
| 1 | Inventory — two-tier model (jaxtyped Array primitives → Operator), 12 module surface (radiometry, indices, cloud, compositing, sampling, inference, ...). The collapse from xr_toolz's three tiers is possible because `GeoTensor` is an `np.ndarray` subclass with `__array_ufunc__`. |
| 2 | User story — five personas, the goal arc from "load S2 → mask clouds → NDVI → save COG" to "Hydra-YAML pipeline" |
| 3 | Motivation — why a separate library (not extending `xr_toolz`), why re-implement the composition core, comparison with TorchGeo / xarray-spatial / stackstac |
| 4 | Mathematics — Operator delegation chain, dual-mode `__call__` (eager vs Graph), split-object stateful pattern, sampler stride, compositing reductions, matched filter, pansharpening, Lee speckle |
| 5 | Coupling with the ecosystem — `georeader`, scipy/scikit-image, `xrpatcher`, `georeader.catalog`, ML frameworks, Hydra-zen, `xr_toolz` |
| 6 | Proposed API surface — module layout, `Operator` base class, `Sequential` / `Graph`, `ModelOp`, signature conventions, Hydra YAML compatibility, dependencies, versioning |
| 7 | Sharp edges — twelve flagged-pitfalls-with-mitigations |
| 8 | End-to-end examples — RGB viz, NDVI, cloud-free composites, atmospheric correction, tiled inference, hyperspectral retrievals, catalog-driven processing, Hydra YAML, `Graph` branching, sensor presets |
| 9 | Verdict — concrete v0.1 deliverables, two-month roadmap, what to defer |
| 10 | `xr_toolz` coexistence — substrate-split rule, what cross-pollinates and what doesn't |

---

## Connections to other designs

`geotoolz` sits on top of the rest of the plan tree. The cross-design touchpoints:

| Design | What `geotoolz` consumes / produces |
|---|---|
| [Reader reconciliation](../georeader/README.md) | `geotoolz.catalog_ops.CatalogPipeline` accepts a `reader_class=...` kwarg pulling from any `SyncReader` / `AsyncReader`. The strategy injection is the central swappability seam. |
| [Geodatabase](../geodatabase/README.md) | `CatalogPipeline(catalog, op).run()` consumes a `GeoCatalog`. The pipeline iterates the catalog, applies the operator per row, writes outputs. |
| [Core types — `GeoSlice`](../types/geoslice.md) | `geotoolz.sampling.GridSampler` wraps `grid_sampler`; `geotoolz.inference.ApplyToChips` consumes the iterator and uses `stitch` for the inverse step. The `Stitch` operator in `geotoolz` is a direct re-export of the primitive specified there. |
| [Sensor readers](../readers/README.md) | Sensor-preset operators (`presets.s2.S2_L2A_RGB`, `presets.emit.EMIT_METHANE_MF`, etc.) wrap the per-sensor readers from those designs. |

---

## Open questions

The full design report flags twelve sharp edges in §7, architectural questions in §3 and §10, and a comprehensive risks/gotchas section in §11. The ones most worth surfacing here:

1. **`Operator` base class — shared with `xr_toolz` or re-implemented?** The current decision (§0) is to re-implement (~300 LOC, freedom to specialise). Worth revisiting if a third sibling library appears.
2. **Hydra-zen as a hard dep or extra?** §6 currently scopes it to an optional `[hydra]` extra. Some users want it as a baseline.
3. **`presets/legacy/` for SPOT VGT and Proba-V?** §1.2 doesn't list them. [Tutorial Ch. 17](../../georeader_tutorial/17_legacy_sensors.md) suggests they'd live there if added.
4. **Async operator support.** The current `Operator` base class is sync. An `AsyncOperator` for use with [`AsyncGeoTIFFReader`](../georeader/reader_async_geotiff.md) is plausible but not designed. **Pick a design before v0.1** — see §11.2.
5. **`coordax` stability for the JAX path.** The future-work JAX bridge depends on `coordax` (NeuralGCM, research-grade). Spike before committing the v0.5 work; have a fallback (equinox + jaxtyping directly).
6. **`georeader 2.0` (`feature/geotensor_npapi`) merge timing.** The two-tier model assumes the ndarray-subclass GeoTensor lands. If it stalls, geotoolz blocks.
7. **`_src` privacy decision.** Primitives are currently "semi-public" — likely a footgun for non-ufunc primitives passed `GeoTensor` inputs. Decide truly-private vs deliberately-public namespace (§11.3 recommends truly-private for v0.1).
8. **Sensor preset scope reduction for v0.1.** 80 operators in 4 months is tight. Recommendation: cut Himawari-AHI HSD, MTG-FCI, SEVIRI-HRIT to v0.5+; ship MODIS + ABI as v0.1 sensor proofs.
9. **Array-API compliance from day one or as a retrofit?** If JAX path matters, factoring primitives through the array-API standard at v0.1 is cheap; v0.5 retrofit is expensive. See §11.3.

The full risks/gotchas list lives in [`geotoolz.md` §11](geotoolz.md#11-open-questions-gotchas-and-risks) — covers strategic risks, implementation gotchas to test in CI, and scope honesty.

---

## Why this README is a thin wrapper

The detailed design lives in [`geotoolz.md`](geotoolz.md) — 800+ lines covering everything from architecture decisions to per-operator math to a 15-example use-case gallery. That file is the source of truth; this README is a navigation entry point that other designs link to.

If you're cross-referencing `geotoolz` from another plan, prefer linking to a specific section of [`geotoolz.md`](geotoolz.md) where possible (`#1-2-module-surface`, `#5-2-numpy-scipy-scikit-image-scikit-learn-compute`, etc.) rather than to this README — readers want to land on the section that's relevant to their question, not navigate two levels down.
