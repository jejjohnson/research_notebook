---
title: geotoolz design report
subject: geotoolz design
subtitle: Composable Operator library on top of GeoTensor
short_title: Design report
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

> **Scope:** a composable Operator library for remote sensing — preprocess, infer, and evaluate satellite imagery on top of `georeader.GeoTensor`.
> Sibling to `xr_toolz`, targeted at a different community and a different substrate.
> Functions, classes, and presets, all RS-shaped.
> **Status:** design proposal. Assumes `georeader` 2.0 (the `np.ndarray`-subclass `GeoTensor` with `__array_ufunc__`) exists as today.
> Borrows architectural patterns from `xr_toolz` but re-implements the composition core to avoid coordination tax.

---

## 0. What I read

| Source | Takeaway |
| --- | --- |
| `xr_toolz/docs/design/vision.md`, `architecture.md`, `decisions.md`, `boundaries.md` | A working three-tier (Array / Tensor / Operator) stack, the Operator/Sequential/Graph machinery, the split-object stateful pattern (D2), dual-mode `__call__` (D3), Hydra-zen integration. **Patterns to borrow, code to re-implement.** |
| `xr_toolz` decisions D7 / D9 / D10 / D11 | Two-layer module pattern (functions + Operator wrappers); domain-stub consolidation; viz operators returning Figure / Axes; per-module array-tier entry. |
| `georeader/geotensor.py` (post `feature/geotensor_npapi`) | `GeoTensor(np.ndarray)` with `__array_ufunc__`, `__array_finalize__`, time as a first-class dim, math dunders. The substrate. |
| `georeader.read`, `mosaic`, `reflectance`, `griddata`, `vectorize`, `rasterize`, `slices` | Existing georeader has the I/O + GeoTensor construction + reflectance physics. `geotoolz` sits on top, never re-implements. |
| `jej_vc_snippets/remote_sensing/` (dataset, sampler, transforms_pipelines, predictions, spectral_unmixing, etc.) | The empirical RS workflow vocabulary — the operators users have been writing as snippets for years and want a library home for. |
| Prior reports in this thread (catalog Phase 1/2, sampler design, `geo_toolz` v1/v2/v3 attempts) | The catalog work goes in `georeader`. The pure-numpy library idea is dropped: numpy primitives become per-Operator implementation details inside `geotoolz._src/`. |

> **Decision recorded here:** the composition core (`Operator`, `Sequential`, `Graph`, `ModelOp`) is **re-implemented in `geotoolz`**, not shared with `xr_toolz`. ~300 LOC, gives each library freedom to specialize for its substrate and audience.

---

## 1. Inventory: what `geotoolz` is

A composable Operator library targeted at remote sensing.
Sibling to `xr_toolz` at the architectural level; orthogonal at the substrate level.

### 1.1 The two-tier model

`geotoolz` ships **two tiers**, where `xr_toolz` ships three.
The collapse is possible because `GeoTensor` is an `np.ndarray` *subclass* with `__array_ufunc__` / `__array_finalize__` — it transparently carries metadata through ufuncs, slicing, copies, and views.
The intermediate "tensor function" tier `xr_toolz` needs (because `xarray.DataArray` is *composition* over an ndarray, not a subclass) collapses into the Array tier here.

| Tier | Location | Input | Output | Coordinate semantics |
| --- | --- | --- | --- | --- |
| **A — Array (primitive)** | `geotoolz.<module>._src.array` | `Float[ndarray, "..."]` (jaxtyped) | `Float[ndarray, "..."]` | `axis=` |
| **B — Operator** | `geotoolz.<module>` (public) | `GeoTensor` (or two for multi-input) | `GeoTensor` / scalar / `Figure` | constructor `axis=` / `band=` |

**Tier A — primitives.** Pure `np.ndarray → np.ndarray` functions, jaxtyped at the signature so callers see `Float[ndarray, "*batch bands H W"]`.
Ufunc-pure where possible; for non-ufunc upstream calls (`skimage`, `scipy.ndimage`, `sklearn`) primitives return a plain `ndarray` — **no subclass dance inside the primitive**.
Semi-public: importable, jaxtyped, documented, but the canonical user-facing surface is the Operator.

**Tier B — Operators.** The carrier-aware boundary.
There are **three cases** an Operator's `_apply` may need to handle, and each has a distinct pattern:

1. **Shape-preserving + ufunc-pure** (e.g. `NDVI`, `Mask`, arithmetic, slicing): delegate straight to the primitive.
   `__array_ufunc__` round-trips the metadata automatically — `_apply` is one line.
2. **Shape-preserving + non-ufunc** (e.g. `LeeSpeckle` via `scipy.ndimage`, `MaskFromQABits` via boolean ops, classification via `sklearn.predict`): strip → run → re-wrap.
   `out = primitive(np.asarray(gt)); return gt._wrap(out)`.
   One line of dance per Operator that needs it.
3. **Shape- or metadata-changing** (`Resize`, `Regrid`, `Pansharpen`, multi-input fusion across CRSs, dim-reducing ops like `MeanComposite`): the primitive returns a plain `ndarray` and the Operator **constructs a fresh `GeoTensor`** with a *new* `transform` / `crs` / `dims`.
   `_wrap` does **not** apply — `_wrap` preserves metadata, but the metadata is precisely what changes here.
   Use `georeader`'s `GeoTensor` constructor directly.

Operators carry config (Hydra-friendly) and compose via `Sequential` and `Graph`.
The three cases are not exotic edge cases — Case 3 covers most of `compositing`, `pansharpen`, `sampling`, and any reprojection-flavoured op.
Treat all three as first-class patterns from day one.

**Why two tiers, not three.** `xr_toolz` has a middle "tensor function" tier because lifting `DataArray ↔ ndarray` is real work (extract `.values`, run, re-attach `.coords` / `.dims`).
For `GeoTensor` that work collapses to `gt._wrap(out)` — one line at the Operator boundary.
A dedicated tier-of-functions for that one line would be ceremony without value.

**What jaxtyping does (and doesn't) buy us.** Tier A's `Float[ndarray, "..."]` annotations communicate *shape contracts* — IDE-visible, optional runtime-checkable, lower barrier than typing every primitive against `GeoTensor`.
They do **not** communicate georef-awareness.
The Operator type signature (`GeoTensor → GeoTensor`) remains the canonical place where georef-awareness is asserted.
Primitives are "discipline plus jaxtyping"; Operators are "the typed contract".

**Tests.** Tier A for analytic ground truth (math is right on numpy arrays, shapes match jaxtyping); Tier B for config round-trip + GeoTensor metadata propagation.
No middle tier means no third place to test.

### 1.2 Module surface

Twelve modules, organised by RS workflow stage.
Roughly 80 Operators in steady state; v0.1 ships ~25 (the asterisks below).

| Group | Module | Operators |
| --- | --- | --- |
| **Composition core** | `core` | `Operator`*, `Sequential`*, `Graph`*, `Input`*, `Node`*, `Tap`, `ApplyToEach`, `Augment` |
| **Radiometry & viz** | `radiometry` | `ToFloat32`*, `MinMax`*, `PercentileClip`*, `Gamma`*, `ZScore`, `LogStretch`, `LinearToDB`, `DBToLinear`, `SRFBin` |
| **Atmospheric correction** | `correction` | `TOAToBOA`, `DarkObjectSubtraction`, `Py6SCorrection` (optional extra) |
| **Spectral indices** | `indices` | `NDVI`*, `NDWI`*, `MNDWI`, `NDMI`, `EVI`, `SAVI`, `BSI`, `NormalizedDifference`*, `AppendIndex`* |
| **Cloud & QA** | `cloud` | `MaskClouds`*, `MaskFromQABits`*, `ApplyMask`*, `CloudSEN12` (via `ModelOp`) |
| **Compositing** | `compositing` | `MedianComposite`, `MaxNDVIComposite`, `CloudFreeComposite`, `MeanComposite` |
| **Pansharpening** | `pansharpen` | `BroveyPansharpen`, `GramSchmidtPansharpen`, `HCSPansharpen` |
| **SAR** | `sar` | `LeeSpeckle`, `FrostSpeckle`, `RatioPolarimetric` |
| **Hyperspectral** | `hyperspectral` | `MatchedFilter`, `ACEDetector`, `RXDetector`, `LinearUnmixing` |
| **Sampling & inference** | `sampling` | `RandomSampler`*, `GridSampler`*, `Stitch`* |
| | `inference` | `ModelOp`*, `ApplyToChips`*, `BatchedModelOp` |
| **Catalog-driven** | `catalog_ops` | `CatalogPipeline`, `WriteCOG`, `WriteParquet` |
| **Sensor presets** | `presets.s2` | `S2_L2A_RGB`, `S2_L2A_NDVI`, `S2_L1C_TO_BOA_NDVI`, `S2_QA60_CLOUD_MASK` |
| | `presets.landsat` | `L8_BOA_NDVI`, `L8_QA_PIXEL_CLOUD_MASK` |
| | `presets.emit` | `EMIT_METHANE_MF`, `EMIT_REFLECTANCE` |
| | `presets.enmap` | `ENMAP_TO_S2_BANDS` (SRF binning preset) |

### 1.3 What it explicitly is *not*

- **Not `xr_toolz`.** Different substrate (`GeoTensor` vs `xr.Dataset`).
  Different audience.
  Same architectural patterns, separately implemented.
- **Not `georeader`.** No I/O, no CRS plumbing, no reader classes, no catalog construction.
  Those are `georeader`'s job.
  `geotoolz` consumes `GeoTensor`s and produces `GeoTensor`s.
- **Not a numpy primitives library.** Tier A primitives live inside `_src/array.py` per module — semi-public (importable, jaxtyped, documented), but the **canonical** surface is the Operator.
  `from geotoolz.indices._src.array import ndvi` works for users who want pure-numpy + jaxtyping; `gz.indices.NDVI(...)(gt)` is what most users will write.
  Primitives are "discipline + shape contracts"; Operators are "the typed georef-aware contract".
- **Not framework-coupled.** `ModelOp` doesn't import torch / JAX / sklearn — it duck-types via `getattr(model, "predict")` or `model(arr)`.
  Optional `[ml]` extra for torch/sklearn presets if needed.
- **Not a kitchen sink.** Curated.
  RS-shaped.
  Sensor presets are pinned to product versions to bound the maintenance surface.

---

## 2. User story

**Personas:**

1. **The RS researcher** wants "load S2 → mask clouds → atmospheric correct → NDVI → save COG" as one composable pipeline.
   Today they write 40 lines of glue code per analysis.
2. **The ML engineer** wants tiled inference: chip a scene, run a model on each chip, stitch back, save a georeferenced output.
   Today they write a custom dataloader + inference loop per project.
3. **The hyperspectral retrieval scientist** wants a matched filter operator that takes a target spectrum and a hyperspectral `GeoTensor`.
   Today they copy code from a paper supplement.
4. **The Hydra-driven pipeline author** wants to declare the whole workflow in YAML and have it reproducibly run.
   Today they hand-roll a class registry per project.
5. **The "I just want to make a quick figure" user** wants `S2_L2A_RGB(brightness=2.0)(gt)` and gets a sensible RGB visualisation in three tokens.

**Goal arc:**

```python
import geotoolz as gz

# 1. RGB visualisation in three tokens.
viz = gz.presets.s2.S2_L2A_RGB(brightness=2.0)(gt)

# 2. NDVI in two operators composed.
ndvi = gz.Sequential([
    gz.cloud.MaskClouds(qa_band="QA60", bits=[10, 11]),
    gz.indices.NDVI(red_idx=2, nir_idx=3),
])(gt)

# 3. Cloud-free monthly composite over a year.
composite = gz.Sequential([
    gz.cloud.MaskClouds(qa_band="QA60", bits=[10, 11]),
    gz.compositing.CloudFreeComposite(time_axis=0, merge="last"),
])(timeseries_gt)                                     # GeoTensor (T, B, H, W) → (B, H, W)

# 4. Tiled inference: chip → predict → stitch.
infer = gz.inference.ApplyToChips(
    sampler=gz.sampling.GridSampler(chip_size=(256, 256), stride=(224, 224)),
    chip_op=gz.inference.ModelOp(model, batch_size=8),
    stitcher=gz.sampling.Stitch(method="average"),
)
prediction = infer(gt)                                # GeoTensor (1, H, W)

# 5. Methane plume detection on EMIT.
plumes = gz.presets.emit.EMIT_METHANE_MF(target=ch4_signature)(emit_gt)
```

The unifying line: **every RS workflow is a `Sequential` of operators, declared in code or YAML, executed eagerly on `GeoTensor`s.**

---

## 3. Motivation

### 3.1 Why a separate library, not "extend `xr_toolz`"

| Concern | Force |
| --- | --- |
| **Substrate is different.** RS users have `GeoTensor` (numpy-shaped, integer axes, transform + crs). Climate users have `xr.Dataset` (named dims, coord arrays, dask chunking). The Operator's `__call__` signature is genuinely different. | strong |
| **Audience is different.** RS researchers think in bands, sensors, retrievals, indices, QA bits, atmospheric correction, mosaicking. Climate researchers think in dims, climatologies, anomalies, EOFs, regridding, ESM evaluation. Different vocabulary, different examples, different presets. | strong |
| **Workflows are different.** RS is pipeline-shaped: ingest → preprocess → infer → save. Climate is analysis-shaped: clean → diagnose → compare → score. Different operators rise to the top. | strong |
| **Forcing one library to serve both** means picking awkward compromises (rename `axis` to `dim`? carry both? ban `axis`?). Every previous attempt in this thread (`geo_toolz` numpy / curated wrapper / from-scratch) hit this wall. | strong |
| **Coordination cost of one library + two adapters** is higher than the cost of two libraries with one shared substrate underneath. | medium |

### 3.2 Why re-implement the composition core

`Operator`, `Sequential`, `Graph`, `Input`, `Node`, `ModelOp` are ~300 LOC total. Sharing them through a third package costs a release process, a coordination tax, and a temptation to add features for one library that the other doesn't want.
**Re-implementing is cheaper and freer.**

What `geotoolz` will diverge on:

- **`Operator.__call__` signature.** xr_toolz's is `(ds: xr.Dataset)`. geotoolz's is `(gt: GeoTensor)` with native chip-batching semantics for samplers and tilers.
- **Default `dim` keyword.** xr_toolz: `dim="time"`. geotoolz: `axis=0` (with optional `dim_name="time"` for human-readable repr).
- **Multi-input arity.** RS commonly pairs imagery with masks or labels; geotoolz's Operator accepts `(image, mask)` or `(image, ref)` directly without going through Graph for this case.
- **ModelOp batching.** geotoolz's `ModelOp` defaults to chip-batched inference; xr_toolz's defaults to dataset-flattened inference.

These small differences add up.
Sharing across libraries would mean both diverge from a least-common-denominator.

### 3.3 Compared with adjacent libraries

- **TorchGeo.** Heavy torch dep, ML-first, `BoundingBox` query model.
  `geotoolz` is torch-optional and focused on Operator composition; the audiences overlap but the abstraction tier differs.
- **xarray-spatial.** xarray-bound.
  NDVI etc. but no operator framework, no inference, no presets.
- **`stackstac` / `odc-stac`.** Catalog + xarray composition.
  Different abstraction tier.
- **`xr_toolz`** itself.
  Wrong substrate for RS workflows.
- **plain rasterio + custom code.** What users do today.
  Verbose, no composition, snippet-shaped.

`geotoolz` slots between torchgeo and xarray-spatial: more opinionated than xarray-spatial, less framework-coupled than torchgeo.

---

## 4. Mathematics

### 4.1 The two-tier delegation chain (worked examples: `NDVI`, `LeeSpeckle`)

**Ufunc-pure case (the common one).** `NDVI` is arithmetic — closed under ufuncs — so the Operator delegates straight to the jaxtyped primitive and `__array_ufunc__` handles the GeoTensor wrap automatically:

```python
import numpy as np
from numpy import ndarray
from jaxtyping import Float

# Tier A — array primitive (jaxtyped, semi-public)
def ndvi(
    arr: Float[ndarray, "*batch bands H W"],
    *, axis: int, red_idx: int, nir_idx: int, eps: float = 1e-6,
) -> Float[ndarray, "*batch H W"]:
    take = lambda i: np.take(arr, i, axis=axis)
    return (take(nir_idx) - take(red_idx)) / (take(nir_idx) + take(red_idx) + eps)

# Tier B — operator
class NDVI(Operator):
    def __init__(self, *, red_idx, nir_idx, axis=0, eps=1e-6):
        self.red_idx, self.nir_idx, self.axis, self.eps = red_idx, nir_idx, axis, eps
    def _apply(self, gt: GeoTensor) -> GeoTensor:
        # Ufunc-pure primitive: GeoTensor metadata round-trips automatically
        # via __array_ufunc__ / __array_finalize__. Pass the GeoTensor straight in.
        return ndvi(gt, axis=self.axis, red_idx=self.red_idx,
                    nir_idx=self.nir_idx, eps=self.eps)
    def get_config(self):
        return {"red_idx": self.red_idx, "nir_idx": self.nir_idx,
                "axis": self.axis, "eps": self.eps}
```

The Operator carries config (Hydra-friendly) and delegates straight to the primitive.
**No middle tier, no logic duplicated.**

**Non-ufunc case.** Anything that calls `scipy.ndimage`, `skimage`, or `sklearn` strips the GeoTensor subclass — those upstream libraries call `np.asarray()` internally.
The Operator's `_apply` is the wrap boundary:

```python
from scipy.ndimage import uniform_filter

# Tier A — non-ufunc primitive (jaxtyped, returns plain ndarray)
def lee_speckle(
    arr: Float[ndarray, "*batch H W"], *, window: int,
) -> Float[ndarray, "*batch H W"]:
    mean = uniform_filter(arr, size=window)
    var  = uniform_filter(arr ** 2, size=window) - mean ** 2
    noise_var = float(var.mean())
    weight = var / (var + noise_var + 1e-12)
    return mean + weight * (arr - mean)

# Tier B — operator handles the strip / run / wrap dance explicitly
class LeeSpeckle(Operator):
    def __init__(self, *, window: int = 7): self.window = window
    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = lee_speckle(np.asarray(gt), window=self.window)  # plain ndarray
        return gt._wrap(out)                                    # one-line re-wrap
    def get_config(self): return {"window": self.window}
```

The wrap is one line, lives in exactly one place per Operator, never propagates upstream.

**Shape- or metadata-changing case.** When the output's shape, transform, or CRS differs from the input — the third case from §1.1 — `_wrap` does **not** apply.
The Operator constructs a fresh `GeoTensor`:

```python
from skimage.transform import resize as _skimage_resize

# Tier A — shape-changing primitive (jaxtyped, returns plain ndarray)
def resize(
    arr: Float[ndarray, "*batch H W"],
    *, target_shape: tuple[int, int], order: int = 1,
) -> Float[ndarray, "*batch Hp Wp"]:
    """Bilinear resample. Output spatial dims differ from input."""
    return _skimage_resize(arr, target_shape, order=order, preserve_range=True)

# Tier B — operator constructs a fresh GeoTensor with a new transform
class Resize(Operator):
    def __init__(self, *, target_shape: tuple[int, int], order: int = 1):
        self.target_shape, self.order = target_shape, order
    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = resize(np.asarray(gt), target_shape=self.target_shape, order=self.order)
        # New transform = old transform * scale, where scale comes from the resample factor.
        # `georeader` provides the GeoTensor constructor; concrete API depends on the
        # final georeader 2.0 surface, but the shape is "build a fresh carrier from
        # (array, transform, crs, fill_value)".
        sy = gt.shape[-2] / self.target_shape[0]
        sx = gt.shape[-1] / self.target_shape[1]
        new_transform = gt.transform * gt.transform.scale(sx, sy)
        return GeoTensor(
            values=out, transform=new_transform, crs=gt.crs,
            fill_value_default=gt.fill_value_default,
        )
    def get_config(self): return {"target_shape": self.target_shape, "order": self.order}
```

Same shape applies to `Pansharpen` (changes spatial resolution to the pan grid), `Regrid` (changes CRS + transform), `MeanComposite` over time (drops the time axis from `dims`), and any multi-input fusion that lands on a target grid.
The Operator owns the metadata construction; `georeader` provides the constructor; primitives stay numpy-typed.

**Tests.** Tier A for analytic ground truth (math is right on numpy arrays, shapes match jaxtyping).
Tier B for config round-trip *and* GeoTensor metadata propagation (transform, CRS, fill_value preserved through both ufunc-pure and non-ufunc operators).

### 4.2 The dual-mode `Operator.__call__` (Graph mode)

Borrowed from `xr_toolz` D3 verbatim:

```python
class Operator:
    def __call__(self, *args, **kwargs):
        if any(isinstance(a, Node) for a in args):
            # Graph-building mode: record the call, return a Node.
            return Node(operator=self, parents=args)
        return self._apply(*args, **kwargs)
    def _apply(self, *args, **kwargs):
        raise NotImplementedError
```

Same operator works eagerly in `Sequential` and symbolically in `Graph`.
No parallel hierarchy.

### 4.3 The split-object pattern for stateful operators (D2)

Stateful ops (`PercentileClip` learning thresholds from data, atmospheric correction learning AOD from a calibration scene) split into two operators:

```python
# Stage 1: compute state
percentiles = CalculatePercentiles(p=(2, 98), axis=(-2, -1))(reference_gt)
# percentiles is a small ndarray, saveable to disk.

# Stage 2: apply state
clip = ApplyPercentileClip(percentiles=percentiles)
clipped = clip(gt)
```

Every operator in a `Sequential` is `GeoTensor → GeoTensor`, always.
State is an artifact, not a hidden field.

### 4.4 Sampler stride math (carried from earlier reports)

For each tile of size `(H, W)`, chip `(h, w)`, stride `(s_y, s_x)`:

$$
n_y = \left\lceil \frac{H - h}{s_y} \right\rceil + 1, \qquad
n_x = \left\lceil \frac{W - w}{s_x} \right\rceil + 1
$$

Final-row/column chips are shifted (not truncated) to keep chip size exact.
`GridSampler` returns a sequence of slice tuples; `ApplyToChips` iterates over them, applies a chip op, hands the chip + slice to `Stitch`.

### 4.5 Compositing reductions

For a stacked time series `X ∈ ℝ^(T × B × H × W)`:

| Operator | Rule |
| --- | --- |
| `MeanComposite` | $C_{b,h,w} = \frac{1}{T'} \sum_{t \in V} X_{t,b,h,w}$ where $V$ = valid timesteps |
| `MedianComposite` | $C_{b,h,w} = \mathrm{median}_{t \in V} X_{t,b,h,w}$ |
| `MaxNDVIComposite` | for each $(h,w)$: $t^* = \arg\max_t \mathrm{NDVI}(X_t)$; output $X_{t^*}$ |
| `CloudFreeComposite` | for each $(h,w)$: most recent $t$ with `mask_t = 0` |

The `valid` set is computed from a paired QA `GeoTensor` passed alongside the imagery.
Multi-input Operator: `CloudFreeComposite()(imagery_gt, mask_gt)`.

### 4.6 Matched filter (hyperspectral)

Standard target detector:

$$
\mathrm{MF}(x) = \frac{(x - \mu)^\top \Sigma^{-1} t}{t^\top \Sigma^{-1} t}
$$

with `μ` = scene mean spectrum, `Σ` = scene covariance, `t` = target spectrum.
Pure numpy at the array tier; one-line einsum reshape.
Sibling detectors (`ACE`, `RX`) reuse the same covariance compute via the split-object pattern (compute Σ once, share across detectors).

### 4.7 Pansharpening (Brovey, condensed)

$$
\mathrm{R}'_{h,w} = \mathrm{R}_{h,w} \cdot \frac{\mathrm{P}_{h,w}}{\mathrm{R}_{h,w} + \mathrm{G}_{h,w} + \mathrm{B}_{h,w} + \epsilon}
$$

Resamples the multispectral bands to the pan grid first (via `skimage.transform.resize` — a non-ufunc, shape-changing call wrapped at the Operator boundary, where the Operator constructs a fresh `GeoTensor` with the pan-grid transform), then applies the per-channel ratio.
Same pattern for Gram-Schmidt and HCS, different math.

### 4.8 Lee speckle filter (SAR)

Adaptive local-statistics filter:

$$
\hat{x} = \bar{x} + W \cdot (y - \bar{x}), \quad W = \frac{\sigma_x^2}{\sigma_x^2 + \sigma_n^2}
$$

with `ȳ`, `σ_y²` from a sliding window, `σ_n²` an estimate of speckle noise.
`scipy.ndimage.uniform_filter` for the local statistics; pure numpy for the rest.

---

## 5. Coupling with the ecosystem

### 5.1 `georeader` — primary substrate

`geotoolz` operators take and return `GeoTensor`.
`__array_ufunc__` handles metadata round-trips for ufunc-pure operations transparently; for non-ufunc upstream calls (`skimage`, `scipy.ndimage`, `sklearn`), the Operator's `_apply` does the strip / run / `_wrap` dance at the Operator boundary.
**Zero changes required to `georeader`.**

### 5.2 `numpy / scipy / scikit-image / scikit-learn` — compute

All Tier A primitives consume `np.ndarray` (jaxtyped) and return `np.ndarray`.
`geotoolz` consumes the upstream libraries; never re-implements.
Hard deps.

### 5.3 `xrpatcher` — chip extraction (optional)

`geotoolz.sampling.GridSampler` can use `xrpatcher` internally for the windowing logic, or a hand-rolled stride math (the §4.4 formula).
`xrpatcher` is an optional `[patcher]` extra; the hand-rolled path is the default.

### 5.4 `georeader.catalog` — large-scale processing

`geotoolz.catalog_ops.CatalogPipeline(catalog, op)` iterates a catalog, applies an Operator per row, writes outputs.
The catalog itself (build, query, intersect, persist) is **`georeader`'s job**, not `geotoolz`'s.
`geotoolz` is a consumer.

### 5.5 ML frameworks (`torch` / `jax` / `sklearn`) — optional, via `ModelOp`

`ModelOp(model)` calls `getattr(model, "predict")`, `getattr(model, "__call__")`, or a user-provided method name.
Never imports a framework.
Optional `[ml]` extra ships convenience subclasses (`SklearnModelOp`, `TorchModelOp`) that set sensible defaults but stay duck-typed.

### 5.6 `Hydra` / `hydra-zen` — config-driven pipelines

Every Operator's `get_config()` returns JSON. `hydra-zen.builds(NDVI, red_idx=2, nir_idx=3)` round-trips.
YAML pipelines compose without ceremony.
Optional `[hydra]` extra ships pre-built configs for the standard operators.

### 5.7 `xr_toolz` — sibling library, no dep

`geotoolz` and `xr_toolz` share architectural patterns (D2, D3, D11) but no code.
Users with both substrates install both.
See [§10](#10-xr_toolz-coexistence).

---

## 6. Proposed API surface

### 6.1 Module layout

```text
geotoolz/
├── pyproject.toml                # numpy, scipy, scikit-image, scikit-learn, georeader
├── src/geotoolz/
│   ├── __init__.py               # __version__; re-export Operator, Sequential, Graph, Tap
│   ├── core/                     # Operator, Sequential, Graph, Input, Node, Tap, ApplyToEach
│   │   ├── __init__.py
│   │   └── _src/
│   │       ├── operator.py       # Operator base, __call__ dual-mode, get_config
│   │       ├── sequential.py     # Sequential, __or__ pipe sugar
│   │       ├── graph.py          # Graph, Input, Node, topological sort
│   │       └── utils.py          # axis helpers, jaxtyping shape helpers
│   ├── radiometry/{__init__.py, _src/{array.py, operators.py}}
│   ├── correction/...            # TOAToBOA, DarkObjectSubtraction, Py6S
│   ├── indices/...               # NDVI, NDWI, MNDWI, NDMI, EVI, SAVI, BSI, NormalizedDifference, AppendIndex
│   ├── cloud/...                 # MaskClouds, MaskFromQABits, ApplyMask, CloudSEN12
│   ├── compositing/...           # MedianComposite, MaxNDVIComposite, CloudFreeComposite, MeanComposite
│   ├── pansharpen/...            # Brovey, GramSchmidt, HCS
│   ├── sar/...                   # LeeSpeckle, FrostSpeckle, RatioPolarimetric
│   ├── hyperspectral/...         # MatchedFilter, ACEDetector, RXDetector, LinearUnmixing
│   ├── sampling/...              # RandomSampler, GridSampler, Stitch
│   ├── inference/...             # ModelOp, ApplyToChips, BatchedModelOp
│   ├── catalog_ops/...           # CatalogPipeline, WriteCOG, WriteParquet
│   └── presets/
│       ├── s2.py, landsat.py, emit.py, enmap.py, modis.py
└── tests/                        # mirroring src/, test_array.py + test_operators.py per module
```

### 6.2 The Operator base class (the ~80 LOC core)

```python
class Operator:
    """Base class for geotoolz operators.

    Single-input ops map GeoTensor → GeoTensor. Multi-input ops accept multiple
    positional arguments. Reductions may return scalars / lower-dim GeoTensors /
    plain ndarrays. Terminal viz/IO ops may return Figure / None.

    Subclasses implement `_apply(*args, **kwargs)`; the base class handles the
    eager-vs-graph dispatch.
    """

    def __call__(self, *args, **kwargs):
        if any(isinstance(a, Node) for a in args):
            return Node(operator=self, parents=args)
        return self._apply(*args, **kwargs)

    def _apply(self, *args, **kwargs):
        raise NotImplementedError

    def get_config(self) -> dict:
        """JSON-serialisable dict of constructor args."""
        return {}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_config().items())
        return f"{self.__class__.__name__}({params})"

    def __or__(self, other: "Operator") -> "Sequential":
        if isinstance(other, Sequential):
            return Sequential([self, *other.operators])
        return Sequential([self, other])
```

### 6.3 `Sequential` and `Graph`

```python
class Sequential(Operator):
    def __init__(self, operators: list[Operator]):
        self.operators = operators
    def _apply(self, gt):
        for op in self.operators:
            gt = op(gt)
        return gt
    def get_config(self):
        return {"operators": [
            {"class": op.__class__.__name__, "config": op.get_config()}
            for op in self.operators
        ]}

class Input:
    def __init__(self, name: str): self.name = name; self.parents = (); self.operator = None

class Node:
    def __init__(self, operator: Operator, parents: tuple): self.operator = operator; self.parents = parents

class Graph(Operator):
    def __init__(self, inputs: dict[str, Input], outputs: dict[str, Node]):
        self.inputs, self.outputs = inputs, outputs
        self._order = self._topological_sort()
    def _apply(self, **kwargs):
        cache = {id(node): kwargs[name] for name, node in self.inputs.items()}
        for node in self._order:
            args = tuple(cache[id(p)] for p in node.parents)
            cache[id(node)] = node.operator._apply(*args)
        return {name: cache[id(node)] for name, node in self.outputs.items()}
```

Same shape as xr_toolz's, separately implemented.

### 6.4 `ModelOp` — framework-agnostic inference

```python
class ModelOp(Operator):
    """Wrap any callable as an Operator.

    Calls `getattr(model, method)(arr)` or `model(arr)`. Never imports
    torch / jax / sklearn. Optional batched inference for chips.
    """
    def __init__(self, model, *, method: str = "__call__", batch_size: int | None = None):
        self.model, self.method, self.batch_size = model, method, batch_size
    def _apply(self, gt):
        arr = gt.values  # plain ndarray
        fn = getattr(self.model, self.method) if self.method != "__call__" else self.model
        if self.batch_size is None:
            out = fn(arr)
        else:
            out = self._batched(fn, arr)
        return GeoTensor(values=out, transform=gt.transform, crs=gt.crs)
    def _batched(self, fn, arr):
        # split arr along axis 0 into batches, concatenate results
        ...
```

### 6.5 Function signature conventions

- Every Operator constructor takes keyword-only args (after `self`).
- `axis=` for the integer axis convention; `dim_name=` (optional) for human-readable repr.
- Single-input `_apply(self, gt)`; multi-input `_apply(self, gt_a, gt_b)` (unambiguous from arity).
- `get_config()` returns JSON. Rich state (e.g. fitted percentiles, a model) is a constructor argument; users save / load it themselves.
- No `**kwargs` in public constructors.
  Every option named.

### 6.6 Hydra-zen YAML compatibility

```yaml
# conf/preprocess.yaml
preprocess:
  _target_: geotoolz.core.Sequential
  operators:
    - _target_: geotoolz.cloud.MaskClouds
      qa_band: "QA60"
      bits: [10, 11]
    - _target_: geotoolz.radiometry.PercentileClip
      p_min: 2
      p_max: 98
      axis: [-2, -1]
    - _target_: geotoolz.indices.NDVI
      red_idx: 2
      nir_idx: 3
```

```python
from hydra import compose, initialize
from hydra.utils import instantiate

with initialize(config_path="conf"):
    cfg = compose(config_name="preprocess")
pipeline = instantiate(cfg.preprocess)
result = pipeline(gt)
```

### 6.7 Dependencies

```toml
[project]
dependencies = [
    "numpy >= 1.26",
    "scipy >= 1.12",
    "scikit-image >= 0.22",
    "scikit-learn >= 1.4",
    "georeader >= 2.0",
]

[project.optional-dependencies]
ml = ["torch", "scikit-learn"]
hydra = ["hydra-zen"]
patcher = ["xrpatcher"]
catalog = ["geopandas", "duckdb"]
viz = ["matplotlib"]
atmos = ["py6s"]
```

Five hard deps.
Everything else opt-in.

### 6.8 Versioning and stability

Pre-1.0 (`0.x`).
Operator constructor signatures are the public surface.
Tier A primitive signatures are *semi-public* — jaxtyping annotations document the shape contract, but signatures may evolve until 1.0. Sensor presets are pinned with version suffixes (`S2_L2A_RGB_v1`) so users can stay on an old preset across breaking sensor changes.

---

## 7. Sharp edges

1. **Don't share `Operator` with `xr_toolz`.** They diverge on call signature and chip semantics.
   The 300-LOC duplication is cheaper than the coordination tax.
2. **`GeoTensor` subclass round-trip discipline.** For non-ufunc upstream calls (skimage, scipy.ndimage, sklearn), the **Operator's `_apply`** does the wrap — `out = primitive(np.asarray(gt)); return gt._wrap(out)`.
   One line per Operator that needs it; never inside the Tier A primitive.
   For shape-changing operations (resize, zoom, regridding), the primitive returns a plain ndarray and the Operator constructs a fresh `GeoTensor` with the new transform — `georeader` provides the construction helpers.
   **Footgun to document:** users importing primitives directly from `_src/array` get a plain `ndarray` back when the primitive is non-ufunc, even if they passed a `GeoTensor`.
   Per-primitive docstrings should mark "ufunc-pure" vs "non-ufunc" so the behaviour at primitive level is predictable.
3. **Time-axis convention.** GeoTensor's `dims` may be `("time", "band", "y", "x")` or `("band", "time", "y", "x")` depending on how it was loaded.
   **Operators take `axis=` (int), never assume time is at axis 0.** The user passes `axis=` to match their tensor.
4. **Multi-input arity is positional.** `CloudFreeComposite()(imagery, mask)` — the first GeoTensor is the imagery, the second is the mask.
   Document per Operator.
   Don't do `(image=..., mask=...)` keyword form; mixing positional and keyword args in `__call__` makes the dual-mode dispatch (eager vs graph) fragile.
5. **`Sequential` strict on intermediate types.** Every step except the last must return a `GeoTensor`.
   Terminal ops (`WriteCOG`, viz operators) are allowed only at the end.
   Sequential validates and raises a clear `TypeError`.
6. **Sensor presets are version-pinned.** When ESA changes Sentinel-2 product spec (it has, multiple times), the old `S2_L2A_RGB_v1` keeps working; `S2_L2A_RGB_v2` is a new operator.
   Users opt in.
7. **`ModelOp` doesn't import frameworks.** Imports happen in user code, not in `geotoolz`.
   `from torch import nn` is the user's job.
   The optional `[ml]` extra is for *examples* and sensor-preset model wrappers, not for `ModelOp` itself.
8. **Stateful operators use the split-object pattern.** No `fit` / `transform` duality on Operator.
   `CalculatePercentiles → ApplyPercentileClip(percentiles)`.
   State is an artifact.
9. **Catalog ops live on georeader.** `geotoolz.catalog_ops.CatalogPipeline(catalog, op)` consumes a `georeader.catalog.GeoCatalog`.
   Don't reimplement catalog construction here.
10. **Don't absorb every snippet.** Curated operator surface.
    If a user has a one-off RS computation, they write it as a one-off function or a custom `Operator` subclass — not as a PR adding it to `geotoolz`.
11. **No `print` in operators.** Use `warnings.warn` for soft issues; raise for hard.
    Operators are building blocks; they shouldn't decorate stdout.
12. **Float precision.** Default `float32` for image-bandwidth math, `float64` for stats and FFT. Per Operator, documented.

---

## 8. End-to-end examples

`GeoTensor`-centric, using the proposed Operator surface.
All examples assume `import geotoolz as gz` and `import numpy as np`.

| § | Example | Pattern |
| --- | --- | --- |
| 8.1 | A. RGB visualisation in three lines | viz |
| 8.1 | B. NDVI with cloud-masked Sentinel-2 | indices + cloud |
| 8.2 | C. Cloud-free monthly composite over a year | compositing |
| 8.2 | D. Max-NDVI annual composite | compositing |
| 8.3 | E. TOA → BOA with dark-object subtraction | atmospheric correction |
| 8.4 | F. Tiled inference with model + stitch | sampling + inference |
| 8.4 | G. Patch-based cloud detection writing aligned QA | inference + catalog |
| 8.5 | H. Methane plume detection on EMIT | hyperspectral |
| 8.5 | I. Hyperspectral linear unmixing | hyperspectral |
| 8.6 | J. Catalog-driven processing across hundreds of tiles | catalog_ops |
| 8.7 | K. Hydra YAML pipeline | composition |
| 8.7 | L. Branching analysis via Graph | composition |
| 8.8 | M. Sentinel-2 RGB preset in one import | presets |

### 8.1 Simple pipelines

#### A. RGB visualisation

```python
gt = georeader.read_from_bounds(reader, bounds=AOI)            # (B, H, W) uint16

viz = gz.Sequential([
    gz.radiometry.ToFloat32(),
    gz.radiometry.PercentileClip(p_min=2, p_max=98, axis=(-2, -1)),
    gz.radiometry.MinMax(),
    gz.radiometry.Gamma(g=1.2),
])(gt)                                                         # GeoTensor (B, H, W) ∈ [0, 1]
plt.imshow(viz[:3].transpose(1, 2, 0).values)
```

#### B. NDVI with cloud-masked Sentinel-2

```python
ndvi_pipeline = gz.Sequential([
    gz.cloud.MaskClouds(qa_band_idx=-1, bits=[10, 11]),
    gz.indices.NDVI(red_idx=2, nir_idx=3),
])
veg = ndvi_pipeline(s2_gt)                                     # GeoTensor (H, W)
georeader.save_cog(veg, "/out/ndvi.tif")
```

### 8.2 Compositing

#### C. Cloud-free monthly composite

```python
ts = georeader.read_timeseries(reader, bounds=AOI, t=YEAR)     # GeoTensor (T, B, H, W)

# Build a paired QA timeseries (one band per timestep)
qa = georeader.read_timeseries(qa_reader, bounds=AOI, t=YEAR)  # GeoTensor (T, 1, H, W)

# Cloud-free composite: most recent valid pixel per location
composite = gz.compositing.CloudFreeComposite(
    time_axis=0, qa_bits=[10, 11], merge="last",
)(ts, qa)                                                       # GeoTensor (B, H, W)
georeader.save_cog(composite, "/out/cloud_free_2023.tif")
```

#### D. Max-NDVI annual composite

```python
mosaic = gz.Sequential([
    gz.cloud.MaskClouds(qa_band_idx=-1, bits=[10, 11]),
    gz.compositing.MaxNDVIComposite(time_axis=0, red_idx=2, nir_idx=3),
])(ts)                                                          # GeoTensor (B, H, W)
```

### 8.3 Atmospheric correction

#### E. TOA → BOA with dark-object subtraction

```python
boa_pipeline = gz.Sequential([
    gz.cloud.MaskClouds(qa_band_idx=-1, bits=[10, 11]),
    gz.correction.DarkObjectSubtraction(percentile=1, axis=(-2, -1)),
    gz.correction.TOAToBOA(sun_zenith=sza, atmosphere="midlatitude_summer"),
])
boa = boa_pipeline(toa_gt)                                      # GeoTensor (B, H, W) reflectance
```

`TOAToBOA` is an Operator wrapping georeader's reflectance physics — it doesn't re-implement, it composes.

### 8.4 Tiled inference

#### F. Model inference + stitch

```python
model = load_my_unet()                                          # any callable / torch model

tiled_inference = gz.inference.ApplyToChips(
    sampler=gz.sampling.GridSampler(chip_size=(256, 256), stride=(224, 224)),
    chip_op=gz.inference.ModelOp(model, batch_size=8),
    stitcher=gz.sampling.Stitch(method="average"),
)
prediction = tiled_inference(s2_gt)                             # GeoTensor (1, H, W)
georeader.save_cog(prediction, "/out/segmentation.tif")
```

#### G. Cloud detection writing aligned QA across a catalog

```python
catalog = georeader.catalog.open_catalog("s3://bucket/s2_eu.parquet")

per_tile = gz.Sequential([
    gz.inference.ApplyToChips(
        sampler=gz.sampling.GridSampler(chip_size=(512, 512)),
        chip_op=gz.inference.ModelOp(cloudsen12_model),
        stitcher=gz.sampling.Stitch(method="average"),
    ),
    gz.catalog_ops.WriteCOG(path_template="{filepath_stem}.cloud.tif"),
])

gz.catalog_ops.CatalogPipeline(catalog, per_tile).run()         # writes one COG per row
```

### 8.5 Hyperspectral

#### H. Methane plume detection on EMIT

```python
emit_gt = georeader.readers.emit.load(scene_path)               # (B=285, H, W) reflectance

mf_pipeline = gz.Sequential([
    gz.cloud.ApplyMask(mask_op=gz.cloud.MaskFromQABits(...)),
    gz.hyperspectral.MatchedFilter(target=ch4_signature, axis=0),
])
score = mf_pipeline(emit_gt)                                    # GeoTensor (H, W) detection score
georeader.save_cog(score, "/out/methane_mf.tif")
```

#### I. Linear spectral unmixing

```python
endmembers = np.load("/data/endmembers_4_classes.npy")          # (4, B) reference spectra

unmix = gz.hyperspectral.LinearUnmixing(endmembers=endmembers, method="fcls", axis=0)
fractions = unmix(emit_gt)                                      # GeoTensor (4, H, W) class fractions
```

### 8.6 Catalog-driven

#### J. Apply a pipeline across hundreds of tiles

```python
catalog = georeader.catalog.open_catalog("/cat/s2_T29SND.parquet")

pipeline = gz.Sequential([
    gz.cloud.MaskClouds(qa_band_idx=-1, bits=[10, 11]),
    gz.indices.NDVI(red_idx=2, nir_idx=3),
    gz.catalog_ops.WriteCOG(path_template="/out/ndvi/{tile_id}_{date}.tif"),
])

gz.catalog_ops.CatalogPipeline(
    catalog,
    pipeline,
    n_workers=4,                                                # process multiple tiles in parallel
).run()
```

`CatalogPipeline` is a Tier B Operator that wraps the catalog iteration.
The actual catalog (build / query / persistence) lives in `georeader.catalog`.

### 8.7 Composition patterns

#### K. Hydra YAML pipeline

```yaml
# conf/methane_retrieval.yaml
pipeline:
  _target_: geotoolz.core.Sequential
  operators:
    - _target_: geotoolz.cloud.ApplyMask
      mask_op:
        _target_: geotoolz.cloud.MaskFromQABits
        qa_band_idx: -1
        bits: [10, 11]
    - _target_: geotoolz.hyperspectral.MatchedFilter
      target: ${load_target_spectrum}
      axis: 0
```

```python
from hydra import compose, initialize
from hydra.utils import instantiate

with initialize(config_path="conf"):
    cfg = compose(config_name="methane_retrieval")
pipeline = instantiate(cfg.pipeline)
score = pipeline(emit_gt)
```

#### L. Branching analysis via `Graph`

```python
img = gz.core.Input("image")
ref = gz.core.Input("reference")

cleaned = gz.cloud.MaskClouds(qa_band_idx=-1, bits=[10, 11])(img)
ndvi    = gz.indices.NDVI(red_idx=2, nir_idx=3)(cleaned)
ndwi    = gz.indices.NDWI(green_idx=1, nir_idx=3)(cleaned)
rmse    = gz.metrics.RMSE(axis=(-2, -1))(ndvi, ref)             # multi-input

g = gz.core.Graph(
    inputs={"image": img, "reference": ref},
    outputs={"ndvi": ndvi, "ndwi": ndwi, "rmse": rmse},
)
result = g(image=s2_gt, reference=ref_gt)                       # dict of GeoTensors / scalars
```

### 8.8 Sensor presets

#### M. One-line Sentinel-2 RGB

```python
from geotoolz.presets.s2 import S2_L2A_RGB_v1

viz = S2_L2A_RGB_v1(brightness=2.0)(s2_gt)                      # GeoTensor (3, H, W) viz-ready
plt.imshow(viz.transpose(1, 2, 0).values)
```

The preset is just a `Sequential` with sensible defaults baked in.
Users override via constructor args.

---

## 9. Verdict

`geotoolz` is **the RS-shaped sibling to `xr_toolz`**.
Same architectural patterns, different substrate, different audience, different opinionated surface.
Two libraries, two use cases, that's it.

What needs to happen to ship v0.1:

- [ ] Cut the new repo.
  `src/` layout, five hard deps.
- [ ] Re-implement the composition core: `Operator`, `Sequential`, `Graph`, `Input`, `Node`, `Tap`.
  Borrow patterns from `xr_toolz`; ~300 LOC; no shared dep.
- [ ] Author the v0.1 modules: `radiometry`, `indices`, `cloud`, `sampling`, `inference`, plus the core. ~25 Operators.
  **2 weeks.**
- [ ] One `_src/{array, operators}.py` per module — preserves the two-tier discipline from day one.
  Tier A primitives jaxtyped (`jaxtyping >= 0.2`) at the signature.
- [ ] Per-Operator `_wrap` discipline (Operators handle non-ufunc primitive output explicitly) + `GeoTensor` round-trip tests covering both ufunc-pure and non-ufunc primitives.
- [ ] Hydra-zen smoke tests (every Operator's `get_config()` round-trips through `builds()`).
- [ ] One end-to-end example notebook per category in §8 (six notebooks).

v0.2 — `compositing`, `correction`, `catalog_ops`.
**2 weeks.** v0.3 — `pansharpen`, `sar`, `hyperspectral` (matched filter, ACE, RX, unmixing).
**2 weeks.** v0.4 — `presets.s2`, `presets.landsat`, `presets.emit`, `presets.enmap`.
**1 week per sensor.**

What to defer past v0.4:

- A full atmospheric correction story (Py6S/MODTRAN integration).
  Optional `[atmos]` extra; ship a stub `Py6SCorrection` operator.
- Distributed processing (Dask integration on the `CatalogPipeline` parallelism). v0.5+.
- xrpatcher integration as the default sampler. v0.5+.
- A "geotoolz / xr_toolz interop" doc page. v0.4, when both have shipped enough to point at concrete patterns.

The total cost: roughly **two months** for v0.1–v0.4. The payoff: every RS workflow that today is 30–100 lines of glue code becomes a `Sequential` of 3–6 Operators, declared in code or YAML, composable, testable, and pinnable to a sensor product version.

---

## 10. `xr_toolz` coexistence

`geotoolz` and `xr_toolz` are siblings.
This section codifies the boundary so users know which to reach for.

### 10.1 The substrate-split rule

| If your data is | Reach for |
| --- | --- |
| `xr.Dataset` / `xr.DataArray` (named dims, coord arrays, dask chunks) | `xr_toolz` |
| `georeader.GeoTensor` (numpy subclass with transform + crs) | `geotoolz` |
| Raw `np.ndarray` (no metadata) | `numpy` / `scipy` / `scikit-image` directly |

For users with both substrates: install both.
Conversion between substrates is `georeader.dataarray.to_dataarray(gt)` / `from_dataarray(da)` — already in georeader.

### 10.2 Shared architectural patterns

Most architecture is shared; **the tier model is the one place they diverge**.

| Pattern | `geotoolz` | `xr_toolz` |
| --- | --- | --- |
| **Tier model** | **Two-tier** (Array → Operator) | **Three-tier** (Array → DataArray → Operator) |
| Split-object pattern for stateful operations | ✓ | ✓ |
| Dual-mode `__call__` (eager vs Graph-symbolic) | ✓ | ✓ |
| `Sequential` and `Graph` composition | ✓ | ✓ |
| `get_config()` for Hydra round-trip | ✓ | ✓ |
| No framework deps in core; ML via duck-typed `ModelOp` | ✓ | ✓ |

**Why the tier asymmetry.** `GeoTensor` is an `np.ndarray` *subclass* with `__array_ufunc__` — the wrap from `ndarray` back to the carrier is one line (`gt._wrap(out)`) and lives at the Operator boundary.
`xarray.DataArray` is *composition* over an ndarray — lifting `DataArray ↔ ndarray` requires `.values` extraction and explicit `.coords`/`.dims` re-attach, real enough work to deserve its own tier.
The substrate dictates the tier count; both libraries land on the same pattern *given their substrate*.

The shared pattern docs live once (in `xr_toolz` or in a third "design conventions" doc); each library's own docs reference them and call out their tier-count specialisation.
Code is independent.

### 10.3 What does NOT cross-pollinate

- **Operator implementations.** `xr_toolz.indices.NDVI` doesn't exist — climate users don't need NDVI as an Operator.
  If they do, they call `geotoolz.indices.NDVI` after converting to `GeoTensor`.
- **`xr_toolz.kinematics.Coriolis(lat=...)`** stays in xr_toolz for climate users (operates on a `Dataset` with named lat/lon).
- **`geotoolz.kinematics.geostrophic_velocity(...)`** is a leaf operator if RS workflows need it (operates on a `GeoTensor` with integer axes).
  Re-implementation is fine — different signatures, different defaults.

### 10.4 Conversion patterns

Cross-substrate workflows (load with georeader, run climate analysis, write back) look like:

```python
gt = georeader.read_from_bounds(...)                          # GeoTensor
da = georeader.dataarray.to_dataarray(gt)                     # xr.DataArray
result_da = xr_toolz.detrend.RemoveClimatology(clim)(da)
result_gt = georeader.dataarray.from_dataarray(result_da)     # GeoTensor
georeader.save_cog(result_gt, "/out/result.tif")
```

`georeader` is the bridge.
Both libraries treat it as substrate, not as dep tier.

### 10.5 Documentation cross-references

A short doc page in each library: "I have *X* data, should I use this library or the other?" with a decision tree.
Linked from both READMEs.
Half-day of doc work, prevents the most common new-user confusion.

### 10.6 The endpoint

Two libraries, two communities, one shared substrate library underneath, one shared design vocabulary on top.
Each library focused, each community served, no awkward unifying compromise.
**That's the right shape, and that's the recommendation.**

---

## 11. Open questions, gotchas, and risks

The architecture is sound; several execution-level concerns deserve flags.
None are blockers; all are things to manage actively.
Strategic risks first, implementation gotchas second, scope honesty third.

### 11.1 Strategic risks

**`georeader 2.0` (`feature/geotensor_npapi`) is critical-path.** The two-tier model assumes the ndarray-subclass `GeoTensor` with `__array_ufunc__` lands cleanly upstream.
If that branch stalls, `geotoolz` blocks — the wrap discipline depends on it.
Track the merge as a blocker on v0.1; if it slips, the contingency is to ship `geotoolz` against a vendored `GeoTensor` until upstream catches up.

**`coordax` is research-grade.** The future-work JAX path leans on `coordax` (NeuralGCM, Google), which is early and has no stability commitment.
Pin known-good versions; spike before committing the JAX path; have a fallback (`equinox` + `jaxtyping` directly, reinventing the small bit of labelled-array machinery you actually need).
Treat the JAX path as **v0.5+**, not v0.1, until coordax stabilises.

**80-operator scope vs roadmap pace.** v0.1–v0.4 budgets ~2 months for ~80 operators at ~1.5 day/operator quality.
Realistic estimate is 5–6 months at 1 FTE; if it's a side project, double the timeline.
**Mitigation:** cut sensor presets to v0.5+ for low-priority sensors (Himawari-AHI HSD, MTG-FCI, SEVIRI-HRIT); ship MODIS + ABI as the v0.1 sensor proofs and let the rest accumulate as need arises.

**Sensor preset maintenance is a forever commitment.** ESA / NASA / EUMETSAT product specs change every 1–2 years; each change spawns a new `_v2` preset.
Budget ~5–10 preset bumps/year as steady-state maintenance, not one-time dev cost.
Document a version-bump runbook so this doesn't become tribal knowledge.

### 11.2 Implementation gotchas (test these in CI)

**`__array_function__` (NEP-18) coverage in `GeoTensor`.** `__array_ufunc__` covers ufuncs only.
Functions like `np.fft.fft2`, `np.linalg.svd`, `np.einsum`, `np.percentile`, and many `np.linalg.*` go through `__array_function__`, not `__array_ufunc__`.
Verify `GeoTensor` implements both, or document which numpy submodules strip the subclass.
Add a CI test that round-trips metadata through every numpy module the operators touch (`fft`, `linalg`, basic ufuncs, reductions, indexing).

**`gt._wrap(out)` semantic edge cases.** Spec the rule per case so 80 operators don't each invent their own:
- **Scalar output** (`tensor.mean()`): wrap into 0-d GeoTensor with original transform, or return scalar?
- **Dim-reducing output** (`tensor.max(axis=-1)`): preserve transform but drop the axis from `dims`?
- **Multi-input with divergent metadata** (`composite(image, mask)` where mask has no transform): which input's metadata wins?
- **Boolean indexing** (`tensor[tensor > 0]`): returns 1-D, transform meaningless.
  Disallow at Operator level?

Pick rules now; document in a `_wrap` design-doc entry that the Operator authors reference.

**ndarray subclass fragility outside numpy.** `GeoTensor` survives numpy + scipy + skimage + matplotlib.
It does **not** survive PyTorch (`torch.from_numpy` strips it), JAX (`jnp.asarray` strips it), or Dask without explicit `meta=` plumbing.
This bounds the "GeoTensor flows everywhere" mental model — outside numpy-land, conversion is **explicit**.
Document the supported boundary in the `GeoTensor` user docs.

**Async ↔ sync Operator boundary.** `Operator._apply` is sync; the async readers (`AsyncGeoTIFFReader`, the `AsyncGeoData` Protocol) are async.
Mixing them inside `_apply` means `asyncio.run()` per Operator call — one event loop per invocation, expensive at batch scale.
Pick a design before v0.1: (a) introduce an `AsyncOperator` family with `async def _apply` and an `AsyncSequential` runner; (b) restrict async to the `CatalogPipeline` boundary (sync operators on already-fetched data, async only at the read step — probably the cleanest); (c) sync wrapper that reuses an event loop across calls.

**Pickling discipline for production.** "Operator graph as FastAPI handler" depends on pickling working.
`@dataclass`-shaped operators pickle fine; lambdas, closures, unbound methods break silently.
Add a CI test that pickles every example operator graph in §8 and unpickles cleanly.
If the test ever fails, the failing example is the bug.

**Three-cases-not-two for `_apply`.** §1.1 / §4.1 lay out the three cases (ufunc-pure, non-ufunc shape-preserving, shape/metadata-changing).
The third covers most of `compositing`, `pansharpen`, `sampling`, and any reprojection-flavoured op.
Treat it as first-class from day one — shape-changing operators construct a fresh `GeoTensor`, they don't `_wrap`.

### 11.3 Scope honesty

**`_src` privacy is fuzzy.** Currently semi-public ("importable, jaxtyped, documented").
Users will import primitives directly and hit subclass-stripping for non-ufunc primitives.
Decide before v0.1:
- **(a)** Make `_src` truly private (single-underscore module names, "do not import" docstrings, optional `__all__` enforcement).
  Cleanest.
- **(b)** Expose primitives at a deliberate public namespace (`geotoolz.indices.array.ndvi`).
  Honest but doubles the API surface.
- **(c)** Keep the current "semi-public" path and document per-primitive ufunc-pure vs non-ufunc behaviour.
  Maximum footgun risk.

**Recommendation:** (a) for v0.1; promote to (b) only if real users ask for primitives as a first-class import path.

**Array-API compliance for primitives.** True backend-agnostic primitives use the [array-API standard](https://data-apis.org/array-api/) (`xp.add`, not `np.add`), portable across numpy / JAX / PyTorch. Current primitives use `np.*` directly — JAX requires rewrites.
**Cost-now-or-later trade-off:** factoring through array-API at v0.1 is small (some primitives become slightly less readable); doing it as a v0.5 retrofit is expensive.
If the JAX path matters, eat the cost now.

**`ModelOp` is inference-only.** Duck-typed `__call__` works for inference across torch / JAX / sklearn.
It does *not* unify training (different optimizers, gradient APIs, loss surfaces, batch semantics).
State this explicitly in `ModelOp` docs so users don't expect a training abstraction.

**"Same operator everywhere" has limits.** Holds for: numpy, Dask-orchestrated batch, distributed JAX, FastAPI, Airflow, Ray.
Does *not* hold for: Sedona/Spark (different paradigm — would need SQL emission), streaming engines (Flink/Beam, not streaming-aware), edge inference (`ModelOp` ships the model, not the operator graph).
Keep the in-scope vs out-of-scope list explicit in the public motivation doc ([`geostack_notes.md`](../../geostack_notes.md) "Honest research-to-prod scope").

**Aggregate learning curve.** Two-layer ladder + jaxtyping + Hydra-zen + GeoTensor wrap discipline + dual-mode `__call__` + split-object stateful pattern + sensor presets is more than the "Keras simple" pitch suggests.
Tutorial gallery (§8) should introduce concepts one at a time, not in a single kitchen-sink example.
*"What you need to know to write your first NDVI pipeline" → "What you need to know to ship a Hydra-driven catalog inference run"* — a curriculum, not a manual.

**Sub-pickle interop risks.** `Sequential` exported as ONNX, used as a `torch.utils.data.Dataset`, tracked by `mlflow` — these are integration questions that *will* come up.
Scope which integrations are first-class vs out-of-scope before v0.1, even if the answer is "first-class only mlflow at v0.4, the rest is user-driven".
