---
title: geotoolz.sklearn design doc
subject: geotoolz design
subtitle: Wrapping scikit-learn estimators as Operators over GeoTensor
short_title: sklearn bridge
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, geotoolz, sklearn, operators, machine-learning, remote-sensing
---

> **Status:** draft (2026-05-10) — first design pass on the sklearn bridge.
> **Scope:** how every scikit-learn estimator (and every `BaseEstimator`-protocol third party — `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`) lands in a `geotoolz` Operator graph through one `PixelTable` bridging type, a small family of shape adapters (spatial-pixelwise, spatiotemporal-pixelwise, chipwise), and four estimator wrappers (Classifier, Regressor, Transformer, Clusterer) — with first-class NaN strategy taxonomy and an explicit fit-out-of-graph discipline.
> **Audience:** anyone composing geospatial ML pipelines that lean on the sklearn estimator surface — pixelwise classifiers, transformers, clusterers, gradient-boosted regressors, or any third-party library following the BaseEstimator protocol.

## Goals

Make every scikit-learn estimator (and every third-party library following the `BaseEstimator` protocol — `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`, custom estimators) usable inside a `geotoolz` operator graph **as a single Operator**, without per-estimator wrapping code.

Concretely:

- One bridging type (`PixelTable`) for sklearn’s `(n_samples, n_features)` shape.
- A small family of **shape adapters** for the standard majorisation patterns (spatial pixelwise, spatiotemporal pixelwise, chipwise).
- A small family of **estimator wrappers** — Classifier, Regressor, Transformer, Clusterer — that compose the shape adapter with `.predict` / `.transform`.
- **First-class NaN handling** with a rich strategy taxonomy, since NaN is the default state of geospatial data and sklearn estimators are intolerant of it.
- **Fitting is not in the operator graph.** Fit utilities are separate; pre-fitted estimators are the artifact that flows into pipelines.

## Non-goals

- Re-implementing sklearn algorithms in JAX or on GPU.
- Training inside the operator graph (state, multi-pass).
- `sktime` / time-series-specific estimators — covered by a follow-on `geotoolz.sktime` integration with the same shape vocabulary.
- Bridging sklearn to xarray (that’s `xr_toolz`’s job).

## Design philosophy

**Thin glue, not reimplement.** Wrap, don’t fork.
Every operator in this integration should be ≤ 100 lines.
The hard work is in defining the right *shape adapters* and the right *NaN policy* — once those are correct, the estimator wrappers are trivial.

**One bridge type, many adapters.** Sklearn’s universe is 2D `(N, F)`.
The adapters convert between geotoolz’s spatial / spatiotemporal carriers and that shape.
Adapters are first-class Operators — composable, configurable, and exposed as primitives for users who need custom shape work.

**Honest about state.** Estimators are stateful.
The operator graph isn’t.
Resolve by treating the fitted estimator as a **pre-computed artifact** (loaded from `.joblib`, passed in at construction time, hashed into provenance), not as state that the graph manages.

-----

## Carrier types

### `GeoTensor` (recap)

The primary carrier.
Two shape conventions:

- **Spatial-only:** `(C, H, W)` — bands × rows × cols.
  Carries `transform` and `crs`.
- **Spatiotemporal:** `(C, T, H, W)` — bands × time × rows × cols.
  Carries `transform`, `crs`, and `interval` (the temporal range from `GeoSlice`).

(Some upstream readers also produce `(T, C, H, W)` frame-major arrays; the adapters should accept both via an explicit `time_axis` parameter, default 1.)

### `PixelTable` (new — bridges to sklearn)

Companion carrier for sklearn’s `(n_samples, n_features)` view.
Holds:

```python
@dataclass(frozen=True)
class PixelTable:
    values:    np.ndarray              # (N, F)
    height:    int                     # spatial H, for inversion
    width:     int                     # spatial W, for inversion
    t_size:    int | None = None       # temporal extent, if any
    transform: Affine | None = None    # carried through for inversion
    crs:       CRS | None = None
    feature_layout: tuple[str, ...] = ()  # e.g. ("c",) or ("t", "c")
    sample_layout:  tuple[str, ...] = ()  # e.g. ("h", "w") or ("h", "w", "t")
    mask:      np.ndarray | None = None   # (N,) bool — True where row is valid
```

The `feature_layout` and `sample_layout` are how the inverse adapter knows which axis a flattened dimension came from.
The `mask` is how NaN-aware strategies carry validity through.

A `PixelTable` is **not** a `GeoTensor` — different invariants (rows aren’t spatially localised, no per-pixel transform).
Keeping them distinct prevents accidentally calling a spatial operator on a flattened pixel table.

-----

## Shape adapters

The heart of the integration.
Adapters are first-class Operators with config- round-trippable parameters.
The naming convention is `To{X}Major` / `From{X}Major`.

### Spatial pixelwise

The standard land-cover / scene-classification pattern.
Each pixel is a sample; the bands form the feature vector.

```python
class ToPixelMajor(Operator):
    """(C, H, W) → PixelTable with (H*W, C)."""
    def _apply(self, gt: GeoTensor) -> PixelTable:
        c, h, w = gt.shape
        values = np.asarray(gt).transpose(1, 2, 0).reshape(h * w, c)
        return PixelTable(
            values=values, height=h, width=w,
            transform=gt.transform, crs=gt.crs,
            feature_layout=("c",), sample_layout=("h", "w"),
        )

class FromPixelMajor(Operator):
    """PixelTable (H*W, C') → GeoTensor (C', H, W)."""
    def _apply(self, pt: PixelTable) -> GeoTensor:
        n, c = pt.values.shape
        values = pt.values.reshape(pt.height, pt.width, c).transpose(2, 0, 1)
        return GeoTensor(values=values, transform=pt.transform, crs=pt.crs)
```

### Spatiotemporal pixelwise

The standard crop-classification-from-time-series pattern.
Each pixel is a sample; its full timeseries is flattened into the feature vector.

```python
ToTemporalPixelMajor(time_handling="features")
# (C, T, H, W) → PixelTable (H*W, T*C)
# feature_layout=("t", "c"), sample_layout=("h", "w")
```

Two other temporal majorisations are supported because they correspond to distinct downstream tasks:

```python
# Time as samples — each (pixel, timestep) pair is a sample. For temporal
# anomaly detection, change-point classifiers, per-step regressors.
ToTemporalPixelMajor(time_handling="samples")
# (C, T, H, W) → PixelTable (H*W*T, C)
# feature_layout=("c",), sample_layout=("h", "w", "t")

# Per-pixel timeseries — for `sktime`-style estimators that expect each row
# to *be* a timeseries (panel format).
ToTemporalPixelMajor(time_handling="panel")
# (C, T, H, W) → PanelTable (H*W, C, T)
```

The `panel` mode produces a third carrier (`PanelTable`) used only by `sktime`-compatible estimators.
Most sklearn workflows live in `features` mode.

**Often the right move is to reduce time first**, keeping the sklearn wrapper time-agnostic.
A `TimeAggregate` operator (mean, percentile, harmonic amplitude) collapses `(C, T, H, W) → (C', H, W)`; then a plain `PixelwiseX` runs on the spatial result.
This pushes temporal feature engineering into a dedicated operator and keeps the sklearn integration simple.
The dedicated adapters are there for cases where temporal structure matters as a feature to the estimator itself.

### Chipwise / scenewise

The whole chip is one sample, all pixels are features.
For scene-level classification, image-level regression, embedding-based retrieval.

```python
ToChipMajor()
# (C, H, W) → PixelTable (1, C*H*W) with sample_layout=("chip",)
# or for a batch of chips, (B, C*H*W)
```

### Bandwise (rare but real)

Each band is a sample; pixels are features.
For spectral clustering across bands.

```python
ToBandMajor()
# (C, H, W) → PixelTable (C, H*W)
```

### The shape-adapter contract

Every `To{X}Major` adapter:

1. Preserves enough metadata in the output `PixelTable` for a `From{X}Major` to reconstruct the original carrier shape.
2. Records the layout explicitly (`feature_layout`, `sample_layout`) so downstream operators can introspect and validate.
3. Round-trips through YAML (`get_config()` returns a flat dict of primitives).
4. Optionally carries a validity mask through, enabling NaN policies that need to skip rows.

-----

## Estimator wrappers

Four core wrappers, one per sklearn estimator type.
Each composes a shape adapter (configurable, default `ToPixelMajor`) with the estimator’s `.predict` / `.transform` / `.predict_proba` method.

```python
gz.sklearn.PixelwiseClassifier(estimator=clf)          # .predict, integer labels
gz.sklearn.PixelwiseRegressor(estimator=reg)           # .predict, continuous
gz.sklearn.PixelwiseTransformer(estimator=pca)         # .transform → multi-band
gz.sklearn.PixelwiseClusterer(estimator=kmeans)        # .predict, cluster IDs
gz.sklearn.PixelwiseProba(estimator=clf)               # .predict_proba → C bands
gz.sklearn.PixelwiseDecision(estimator=svm)            # .decision_function
```

Each accepts the same kwargs:

```python
gz.sklearn.PixelwiseClassifier(
    estimator=clf,
    adapter=gz.sklearn.ToPixelMajor(),    # configurable; default by name
    nan_policy=gz.sklearn.NanPolicy.default("classifier"),
    output_dtype="uint8",
)
```

For temporal estimators, swap the adapter:

```python
gz.sklearn.PixelwiseClassifier(
    estimator=time_series_clf,
    adapter=gz.sklearn.ToTemporalPixelMajor(time_handling="features"),
)
```

For sklearn objects that don’t fit the type-named wrappers (custom estimators with non-standard methods), there’s a generic escape hatch:

```python
gz.sklearn.SklearnOp(
    estimator=custom,
    method="predict",           # which method to call
    adapter=ToPixelMajor(),
    nan_policy=NanPolicy(...),
)
```

The type-named wrappers are sugar over `SklearnOp` with sensible defaults (`output_dtype`, `nan_policy` defaults appropriate for the method).

-----

## NaN handling

The single most important design surface in this integration.
Geospatial data is mostly NaN somewhere (clouds, sensor gaps, out-of-swath, masking), and sklearn estimators are intolerant of it.
The strategy needs to be **explicit, configurable per-axis, and visible in the operator graph’s YAML config**.

### Strategy taxonomy

|Strategy   |Behaviour                                                       |When to use                                  |
|-----------|----------------------------------------------------------------|---------------------------------------------|
|`raise`    |Error on any NaN in input                                       |Production ETL where NaN is a bug            |
|`warn`     |Log NaN presence, then apply fallback strategy                  |Debug / staging                              |
|`propagate`|Pass NaN through to estimator (usually crashes)                 |Custom NaN-tolerant estimators               |
|`mask`     |Skip NaN rows; output `fill_value` at masked positions          |Standard for inference                       |
|`drop`     |Drop NaN rows entirely; output array has fewer rows             |Fitting-time only                            |
|`impute`   |Fill NaN with strategy (mean, median, zero, constant)           |When NaN is sparse and imputation is harmless|
|`partial`  |Per-sample: if NaN fraction below threshold, impute; else mask  |Mixed clean/sparse data                      |
|`fill`     |Replace NaN with a sentinel value before estimator              |When estimator handles a known sentinel      |
|`sentinel` |Treat a specific value (not NaN) as missing, then apply strategy|When upstream uses `-9999` etc.              |

The strategies aren’t mutually exclusive — `partial` composes `impute` and `mask`; `sentinel` composes a value-to-NaN conversion with another strategy.

### `NanPolicy` — a first-class config object

```python
@dataclass(frozen=True)
class NanPolicy:
    on_input:         str = "mask"          # strategy applied to features (X)
    on_output:        str = "propagate"     # strategy for estimator-produced NaN
    on_label:         str = "drop"          # fitting only: strategy for labels (y)
    fill_value:       Any = np.nan          # written where masked rows would be
    impute_strategy:  str = "mean"          # "mean" | "median" | "zero" | "constant"
    impute_constant:  float = 0.0
    threshold:        float | None = None   # for "partial": NaN fraction threshold
    sentinel_value:   float | None = None   # for "sentinel": value to treat as NaN
    on_violation:     str = "raise"         # what `raise` / `warn` do
```

Default policies per estimator type:

```python
NanPolicy.default("classifier")
# NanPolicy(on_input="mask", on_output="propagate", fill_value=-1, on_label="drop")

NanPolicy.default("regressor")
# NanPolicy(on_input="mask", on_output="propagate", fill_value=np.nan, on_label="drop")

NanPolicy.default("transformer")
# NanPolicy(on_input="mask", on_output="propagate", fill_value=np.nan, on_label=None)
```

Sensible shortcuts:

```python
PixelwiseClassifier(estimator=clf, nan="mask")        # str shortcut → default mask policy
PixelwiseClassifier(estimator=clf, nan=NanPolicy(on_input="impute", impute_strategy="median"))
PixelwiseClassifier(estimator=clf, nan="raise")       # strict ETL
```

### Where NaN policy applies

Three points in the lifecycle where NaN appears, each handled distinctly:

**1. Input NaN (features X).** Most common case.
Pixels with missing bands.
Resolved by `nan_policy.on_input`.

```python
# (C, H, W) input → some pixels have NaN in some bands
PixelwiseClassifier(estimator=clf, nan=NanPolicy(on_input="mask"))
# Output: (1, H, W) with -1 (fill_value) at any NaN pixel.
```

**2. Label NaN (labels y, fit-time only).** Missing or invalid labels.
Resolved by `nan_policy.on_label`.
Typically `drop` (drop rows where label is missing — can’t fit a model on unlabeled samples).

**3. Output NaN (estimator-produced).** Some regressors and transformers produce NaN under degenerate conditions.
Resolved by `nan_policy.on_output`.
Usually `propagate` is right — the estimator’s NaN is meaningful and should flow through.

### Composition: NaN handling as an Operator

For complex cases, NaN handling is itself an Operator chain that can be lifted out of the wrapper:

```python
gz.Sequential([
    gz.sklearn.NanReplace(sentinel=-9999, with_=np.nan),  # sentinel → NaN
    gz.sklearn.NanImpute(strategy="median"),              # fill NaN
    gz.sklearn.PixelwiseClassifier(estimator=clf, nan="raise"),  # now strict
])
```

Equivalent to wrapping the policy inside the classifier, but more explicit and easier to debug (intermediate carriers can be Snapshot-tapped).

-----

## Fitting (outside the graph)

Deliberate design: **fitting is a separate ceremony.** The operator graph is stateless; the fitted estimator is the artifact that flows into it.
This keeps graphs YAML-reproducible and avoids the “did this graph fit on the right data?” ambiguity.

### Batch fit

```python
clf = gz.sklearn.fit_pixelwise(
    estimator=Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200)),
    ]),
    X_pipeline=feature_pipeline,            # Operator producing GeoTensor features
    y_pipeline=label_pipeline,              # Operator producing GeoTensor labels
    catalog=training_catalog,
    sampler=gz.sampling.RandomSampler(chip_size=(256, 256), length=200),
    n_pixels=500_000,                       # subsample down from accumulated chips
    nan_policy=NanPolicy(on_input="drop", on_label="drop"),
    sample_weight_band=None,                # optional: which band gives sample weights
    random_state=42,
)
joblib.dump(clf, "/models/crop_rf.joblib")
```

The X and y pipelines are themselves Operators — they could be sourced from two different readers (Sentinel-2 features, a labeled land-cover raster for labels), and the helper handles spatial alignment via the shared `GeoSlice`.

### Incremental fit

For estimators that support `partial_fit` (SGD-based learners, MiniBatchKMeans), or when the training set is too large for memory:

```python
clf = gz.sklearn.fit_pixelwise_incremental(
    estimator=SGDClassifier(loss="log_loss"),
    X_pipeline=feature_pipeline,
    y_pipeline=label_pipeline,
    catalog=training_catalog,
    sampler=gz.sampling.GridSampler(chip_size=(256, 256)),
    chunk_size=10_000,
    classes=np.arange(N_CLASSES),           # required for partial_fit on classifiers
    epochs=3,
    nan_policy=NanPolicy(on_input="drop", on_label="drop"),
)
```

### Time-aware fitting

When the X pipeline produces `(C, T, H, W)` and the estimator is temporal (or expects flattened-time features):

```python
clf = gz.sklearn.fit_pixelwise(
    estimator=RandomForestClassifier(),
    X_pipeline=temporal_feature_pipeline,
    y_pipeline=label_pipeline,
    adapter=gz.sklearn.ToTemporalPixelMajor(time_handling="features"),
    ...
)
```

The same adapter is then used at inference time, ensuring train/inference shape parity.

### Sample weighting

Geospatial labels often come with confidence scores or pixel-area weights.
The helper accepts a `sample_weight_band` (the band index in the label GeoTensor that carries weights):

```python
fit_pixelwise(..., y_pipeline=label_pipeline, sample_weight_band=1)
# Band 0 of y_pipeline output → labels; band 1 → sample_weight passed to estimator
```

### Persistence

The fit helpers return the *fitted estimator object*, not an Operator.
The user is responsible for `joblib.dump` (or ONNX export).
At inference time:

```python
clf = joblib.load("/models/crop_rf.joblib")
pipeline = gz.Sequential([
    feature_pipeline,
    gz.sklearn.PixelwiseClassifier(estimator=clf, nan="mask"),
])
```

The Operator’s `get_config()` records the estimator’s class and parameters (via `estimator.get_params()`), plus a hash of the estimator pickle.
The pickle path itself is *not* serialised — it’s a runtime injection.
This matches how `ModelOp` handles PyTorch checkpoints: the artifact is referenced by hash, not embedded.

-----

## sklearn Pipeline interop

The composition story. sklearn’s own `Pipeline`, `ColumnTransformer`, `FeatureUnion`, and any third-party estimator following the protocol drop in *as a single Operator*:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Build sklearn Pipeline normally
sklearn_pipe = Pipeline([
    ("scaler",  StandardScaler()),
    ("pca",     PCA(n_components=8)),
    ("clf",     XGBClassifier(n_estimators=200, tree_method="hist")),
])
sklearn_pipe.fit(X_train, y_train)
joblib.dump(sklearn_pipe, "crop_full.joblib")

# Geotoolz side: one Operator wraps the whole sklearn graph
gz.sklearn.PixelwiseClassifier(estimator=joblib.load("crop_full.joblib"))
```

Why this matters: every estimator on PyPI that follows the `BaseEstimator` protocol (and that’s most of them) is supported without per-estimator code in `geotoolz`.
The wrapping cost stays constant as the sklearn ecosystem grows.

The estimator hash includes the full nested config, so an `xgboost` version bump shows up in provenance metadata — important for reproducibility audits.

-----

## Serialisation and reproducibility

The trickiest part of the integration.
Two challenges:

**Pickle drift.** Sklearn pickles aren’t guaranteed stable across versions — a `RandomForestClassifier` pickle from sklearn 1.3 may not load in 1.5. Mitigations:

- Pin `scikit-learn` (and `numpy`, `scipy`) in the `poetry.lock` that ships with regulatory artifacts.
- For long-lived production models, prefer ONNX export over joblib: `skl2onnx.convert_sklearn(clf, ...)` produces a stable format readable by any ONNX runtime.
- The `PixelwiseClassifier` should accept *either* a Python estimator object *or* an ONNX runtime session; behaviour is identical from the Operator side.

**Provenance.** The artifact pattern from [the regulatory use case](../examples/usecases.md#9-pinned--hashed-regulatory-artifact) must extend to include estimator pickles.
The pinned artifact contains:

- pipeline YAML (geotoolz graph)
- estimator pickles or ONNX files referenced by the YAML
- combined hash covering both
- `poetry.lock` covering sklearn + dependencies

Otherwise the YAML alone is insufficient — you can’t re-run a graph that references `/models/crop_rf.joblib` without that file plus the sklearn version that produced it.

-----

## Compatibility utilities (reference)

The full surface of the integration.
Roughly 15 operators / utilities; the estimator wrappers are the user-facing API, the rest are primitives and helpers.

### Shape adapters

- `gz.sklearn.ToPixelMajor()` — `(C, H, W) → PixelTable (H*W, C)`
- `gz.sklearn.FromPixelMajor()` — inverse
- `gz.sklearn.ToTemporalPixelMajor(time_handling=...)` — `(C, T, H, W) → PixelTable`
  - `time_handling="features"` → `(H*W, T*C)`
  - `time_handling="samples"`  → `(H*W*T, C)`
  - `time_handling="panel"`    → `PanelTable (H*W, C, T)`
- `gz.sklearn.FromTemporalPixelMajor()` — inverse (output shape determined by estimator’s per-row output)
- `gz.sklearn.ToChipMajor()` — `(C, H, W) → (1, C*H*W)` for scene-level
- `gz.sklearn.ToBandMajor()` — `(C, H, W) → (C, H*W)` for spectral clustering

### Estimator wrappers

- `gz.sklearn.PixelwiseClassifier(estimator=..., ...)` — `.predict` → integer label band
- `gz.sklearn.PixelwiseRegressor(estimator=..., ...)` — `.predict` → continuous band
- `gz.sklearn.PixelwiseTransformer(estimator=..., ...)` — `.transform` → C’ bands
- `gz.sklearn.PixelwiseClusterer(estimator=..., ...)` — `.predict` → cluster IDs
- `gz.sklearn.PixelwiseProba(estimator=..., ...)` — `.predict_proba` → per-class bands
- `gz.sklearn.PixelwiseDecision(estimator=..., ...)` — `.decision_function`
- `gz.sklearn.ChipwiseClassifier(estimator=..., ...)` — scene-level variants
- `gz.sklearn.SklearnOp(estimator=..., method=..., ...)` — generic escape hatch

### NaN handling

- `gz.sklearn.NanPolicy(...)` — config object, defaults per estimator type
- `gz.sklearn.NanReplace(sentinel=..., with_=np.nan)` — sentinel ↔ NaN conversion
- `gz.sklearn.NanImpute(strategy=...)` — standalone imputer Operator

### Fitting helpers (not Operators)

- `gz.sklearn.fit_pixelwise(...)` — batch fit
- `gz.sklearn.fit_pixelwise_incremental(...)` — `partial_fit` loop

### ONNX bridge

- `gz.sklearn.OnnxEstimator(session, input_name=..., output_name=...)` — wraps an ONNX runtime session in an sklearn-compatible protocol (`.predict`, `.transform`) so the same `PixelwiseX` wrappers work over ONNX inference

-----

## Honest tradeoffs

**CPU-bound.** Sklearn is numpy/Cython; the wrappers inherit that.
For high-throughput inference, GPU-accelerated estimators (XGBoost with `device= 'cuda'`, RAPIDS cuML) drop in via the same wrappers — but plain sklearn algorithms don’t get GPU acceleration just by living in a geotoolz pipeline.

**Memory cost of `ToPixelMajor`.** A `(C, H, W)` chip with `H=W=512, C=10` is 10 MB; the corresponding `PixelTable` is also 10 MB (just reshaped).
Fine.
But `ToTemporalPixelMajor` with `time_handling="features"` on `(C=10, T=12, H=512, W=512)` produces a `(262144, 120)` table — 250 MB. For full-scene inference, prefer chip-based inference via `ApplyToChips` rather than whole-scene `ToTemporalPixelMajor`.

**Pickle drift.** sklearn pickles aren’t a long-term archive format.
Treat joblib files as deployment artifacts (months), not regulatory artifacts (years).
For multi-year reproducibility, mandate ONNX export.

**No in-graph fitting.** A real cost — users can’t write `gz.Sequential([ preprocess, FitAndPredict(estimator=clf) ])` and have the graph learn from the data flowing through.
Workaround: fit utilities are first-class and ergonomic; the regulatory-artifact pattern includes both the YAML and the fitted estimator.
The win is that operator graphs stay deterministic, which is worth more than the lost convenience.

**Adapter naming surface.** `ToPixelMajor` vs `ToTemporalPixelMajor` vs `ToChipMajor` is fine at 4-5 names; if it grows to 10, refactor toward a single `Reshape(samples=..., features=...)` primitive with the named versions as thin sugar.

**sklearn Pipeline introspection.** A wrapped sklearn `Pipeline` reports as one Operator in `geotoolz`‘s `get_config()`, even though it contains multiple sklearn stages.
That’s correct for the operator-graph view (the fitted estimator is an atomic artifact) but means `Profile().wrap(op)` can’t time individual sklearn stages.
For that, profile inside sklearn directly.

## Open questions

- **Should `PanelTable` exist now or wait for `geotoolz.sktime`?** The three-carrier proliferation (`GeoTensor`, `PixelTable`, `PanelTable`) is a real cost.
  Pushing the `panel` case into the sktime integration would let `geotoolz.sklearn` stay with two carriers.
- **`NanPolicy` config nesting in YAML.** A `NanPolicy` dataclass nested inside a `PixelwiseClassifier` config nested inside a `Sequential` is three levels of YAML. Worth a flat-string shorthand (`nan: "mask:fill=-1"` parsed into a `NanPolicy`)?
- **Validation across the adapter / estimator boundary.** Should the wrapper verify at construction time that the adapter’s output feature count matches the estimator’s `n_features_in_`?
  Pro: catches train/serve shape mismatches early.
  Con: requires `__init__`-time introspection of a fitted estimator, which makes the construction less robust.