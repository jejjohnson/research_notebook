---
title: geotoolz.sklearn examples
subject: geotoolz examples
subtitle: A gallery of sklearn Operator patterns over GeoTensor
short_title: sklearn examples
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, geotoolz, sklearn, examples, machine-learning, remote-sensing
---

> **Status:** companion to [`design_sklearn.md`](design_sklearn.md) — worked examples across the three fit/predict mental models.
> **Audience:** anyone wiring scikit-learn (or an `xgboost` / `lightgbm` / `catboost` / `imbalanced-learn` estimator) into a `geotoolz` pipeline for classification, regression, transformation, clustering, temporal modelling, cross-validation, or incremental fitting on raster data.

Companion to the [design doc](design_sklearn.md).
Examples cover classification, regression, transformations, sklearn Pipeline composition, unsupervised clustering, temporal models, cross-validation, and incremental fitting — across the three fit/predict mental models below.

## Contents

- [Mental models](#mental-models)
- [Patch-generation strategies](#patch-generation-strategies)
- [1. Crop classification (global fit, global predict)](#1-crop-classification-global-fit-global-predict)
- [2. Chipwise inference for large scenes (global fit, local predict)](#2-chipwise-inference-for-large-scenes-global-fit-local-predict)
- [3. Forest biomass regression](#3-forest-biomass-regression)
- [4. Pixel-wise PCA dimensionality reduction](#4-pixel-wise-pca-dimensionality-reduction)
- [5. Full sklearn Pipeline composition](#5-full-sklearn-pipeline-composition)
- [6. Per-tile classifier (local fit, local predict)](#6-per-tile-classifier-local-fit-local-predict)
- [7. Time-series crop classification](#7-time-series-crop-classification)
- [8. Spatial cross-validation with `GridSearchCV`](#8-spatial-cross-validation-with-gridsearchcv)
- [9. Unsupervised land-cover clustering](#9-unsupervised-land-cover-clustering)
- [10. Incremental fitting for large training sets](#10-incremental-fitting-for-large-training-sets)

-----

## Mental models

Three common fit/predict relationships in geospatial ML, each with different implications for the operator graph:

**Fit globally + predict globally.** Sample training pixels across the full catalog (multiple scenes, multiple regions, multiple seasons).
Fit one estimator.
Apply that estimator to *anything* in the catalog at inference.
The standard supervised-learning case — most appropriate when the underlying relationship between features and labels is stationary across the catalog.

**Fit globally + predict locally (patches).** Same global fit, but inference runs patch-by-patch via `ApplyToChips`.
The patches are an implementation detail of inference (memory, tile-server latency, parallelism), not a modelling choice.
The estimator doesn’t know it’s being run on patches — the predictions are stitched back into the full output.

**Fit locally + predict locally.** Train a separate estimator per region (per MGRS tile, per season, per ecoregion, per scene).
Each estimator only predicts on its corresponding region.
Use when spatial heterogeneity is real — different atmospheric conditions, different crop calendars, different sensor calibrations — and a global model would underperform.
Implementation: a registry of fitted estimators, dispatched via `Switch` on metadata.

A fourth pattern — **fit locally + predict globally** (transfer learning, domain adaptation) — exists but isn’t covered here; it’s an active research area, not a stable pattern.

## Patch-generation strategies

When the data comes in patches — for fitting *or* inference — there are several ways to produce them.
The right choice depends on data scale, where the workflow lives, and whether `fit` and `predict` need the same patches.

**1. `GeoCatalog` + sampler (geotoolz-native).** Lazy, catalog-driven.
Each patch is read independently from cloud storage via a `GeoSlice`.
Scales to planetary catalogs because no scene is ever fully materialised.
The default choice for *training over a catalog* and for *inference over a catalog*.

```python
sampler = gz.sampling.GridSampler(chip_size=(256, 256))
for sl in sampler(catalog):
    with georeader.RasterioReader.open(sl.source_uri) as r:
        gt = r.read_geoslice(sl)
    process(gt)
```

**2. Read whole + `xrpatcher` (eager).** Read a full scene into xarray, then chop it into patches.
One read, many patches — better I/O when patches are densely sampled from one scene.
The right choice for *single-scene workflows* (experimentation, viz, quick prototypes).

```python
import xrpatcher
ds = xr.open_dataset(scene_uri)
patches = xrpatcher.XRDAPatcher(ds, patches=dict(x=256, y=256), strides=dict(x=224, y=224))
for patch_ds in patches:
    gt = GeoTensor.from_xarray(patch_ds)
    process(gt)
```

**3. `ApplyToChips` (geotoolz, inference-only).** Wraps an in-memory GeoTensor as chips for *inference*.
The right choice when you already have a full scene loaded and want to run a chipwise model over it.
Not for fitting (no label coupling).

```python
infer = gz.inference.ApplyToChips(
    sampler=gz.sampling.GridSampler(chip_size=(256, 256), stride=(224, 224)),
    chip_op=pipeline,
    stitcher=gz.sampling.Stitch(method="average"),
)
pred = infer(scene_gt)
```

Heuristics:

|Task                                     |Strategy                    |
|-----------------------------------------|----------------------------|
|Training over many scenes                |GeoCatalog + `RandomSampler`|
|Validation on a held-out scene catalog   |GeoCatalog + `GridSampler`  |
|Inference on a single scene (full output)|`ApplyToChips`              |
|Inference across a catalog               |`CatalogPipeline`           |
|Single-scene experimentation             |Read whole + `xrpatcher`    |

The examples below cite which strategy they use, with the rationale.

-----

## 1. Crop classification (global fit, global predict)

**Setting.** Multi-class crop classification from Sentinel-2 monthly composites across an agricultural region (multiple MGRS tiles, one growing season).
One Random Forest fit on pixels sampled from across the training catalog; applied to held-out scenes at inference.

**Patch strategy.** `GeoCatalog + RandomSampler` for training (lazy, catalog-scale).
Per-scene prediction at inference (each scene read whole in this example; chipwise inference shown in the next example).

```python
import geotoolz as gz
import georeader
import joblib
from sklearn.ensemble import RandomForestClassifier

# === Training catalog: S2 chips with co-located crop labels ===
train_catalog = georeader.catalog.open_catalog("s3://crop/train_2024.parquet")

feature_pipeline = gz.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.correction.TOAToBOA(sun_zenith_band=-2),
    gz.indices.AppendIndex(gz.indices.NDVI(red_idx=2, nir_idx=3), name="ndvi"),
    gz.indices.AppendIndex(gz.indices.NDWI(green_idx=1, nir_idx=3), name="ndwi"),
])

label_pipeline = gz.Sequential([
    gz.catalog_ops.ReadLabel(label_uri_field="label_uri", band="crop_class"),
])

# === Fit globally ===
clf = gz.sklearn.fit_pixelwise(
    estimator=RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1),
    X_pipeline=feature_pipeline,
    y_pipeline=label_pipeline,
    catalog=train_catalog,
    sampler=gz.sampling.RandomSampler(chip_size=(256, 256), length=500),
    n_pixels=1_000_000,
    nan_policy=gz.sklearn.NanPolicy(on_input="drop", on_label="drop"),
    random_state=42,
)
joblib.dump(clf, "/models/crop_rf_2024.joblib")

# === Predict globally ===
predict_pipeline = gz.Sequential([
    feature_pipeline,
    gz.sklearn.PixelwiseClassifier(
        estimator=joblib.load("/models/crop_rf_2024.joblib"),
        nan="mask",                       # NaN pixels → fill_value
        output_dtype="uint8",
    ),
])

test_catalog = georeader.catalog.open_catalog("s3://crop/test_2024.parquet")
gz.catalog_ops.CatalogPipeline(
    test_catalog,
    gz.Sequential([
        predict_pipeline,
        gz.catalog_ops.WriteCOG(path_template="s3://out/crop_2024/{tile}.tif"),
    ]),
    n_workers=8,
).run()
```

**Notes on this example.**

- `nan_policy` differs between fit and predict.
  Fitting *drops* NaN rows (you can’t learn from incomplete pixels).
  Inference *masks* them and writes `fill_value` to the output (`-1` for classifiers by default).
  The fit and predict pipelines aren’t required to share NaN strategy.
- `feature_pipeline` is identical in both paths.
  The cloud mask and atmospheric correction are part of the model’s contract — if they differ between train and serve, the model degrades silently.
  Keeping one shared object is the train/serve skew prevention from the [use-cases doc](../examples/usecases.md#3-ml-training-and-inference).

-----

## 2. Chipwise inference for large scenes (global fit, local predict)

**Setting.** Same crop classifier as Example 1, but inference now runs on full Sentinel-2 tiles (~10,980 × 10,980 pixels, ~1 GB in memory).
Whole-scene inference would OOM on a workstation; chipwise inference holds memory constant.

**Patch strategy.** `ApplyToChips` over a full-scene GeoTensor.
The same model from Example 1 — no retraining.

```python
clf = joblib.load("/models/crop_rf_2024.joblib")

# Chip operator: features + classifier. Same as Example 1's predict_pipeline.
chip_op = gz.Sequential([
    feature_pipeline,
    gz.sklearn.PixelwiseClassifier(estimator=clf, nan="mask", output_dtype="uint8"),
])

# Wrap as a chipwise inference Operator.
infer = gz.inference.ApplyToChips(
    sampler=gz.sampling.GridSampler(chip_size=(512, 512), stride=(480, 480)),
    chip_op=chip_op,
    stitcher=gz.sampling.Stitch(method="majority"),  # vote per pixel across overlap
)

# Apply to full-scene GeoTensor.
with georeader.RasterioReader.open("s3://s2/tile_29SND_2024-08-12.tif") as r:
    scene_gt = r.read_geoslice(scene_slice)

prediction = infer(scene_gt)
georeader.save_cog(prediction, "/out/29SND_2024-08-12_crop.tif")
```

**Notes.**

- Overlap (`stride=480 < chip_size=512`) costs ~10% more reads for smoother boundaries.
  For classification, `Stitch(method="majority")` votes per pixel; for probability outputs use `method="average"`.
- The classifier itself doesn’t change.
  This is purely an inference-pattern shift driven by memory constraints, not a modelling decision.
  The estimator artifact is the same.
- For *production* tile-server use (Example 7 of the [use-cases doc](../examples/usecases.md#7-tile-server-zxy--png)), go further and tile at z/x/y level instead of pre-computing a full scene.

-----

## 3. Forest biomass regression

**Setting.** Aboveground biomass (continuous, Mg/ha) from Sentinel-1 + Sentinel-2 fused features.
Regression rather than classification — different default NaN policy, different output dtype, different metrics.

**Patch strategy.** `GeoCatalog + RandomSampler` for training.
Per-scene prediction at inference (same as Example 1).

```python
import geotoolz as gz
from sklearn.ensemble import GradientBoostingRegressor

# === Fused feature pipeline: optical + radar ===
feature_pipeline = gz.Graph(
    inputs={"optical": gz.core.Input("optical"), "sar": gz.core.Input("sar")},
    outputs={"features": gz.fusion.AlignAndStack(target_grid="optical")(
        gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11])(gz.core.Input("optical")),
        gz.sar.LinearToDB()(gz.sar.LeeSpeckle(window=7)(gz.core.Input("sar"))),
    )},
)

# === Fit ===
reg = gz.sklearn.fit_pixelwise(
    estimator=GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05),
    X_pipeline=feature_pipeline,
    y_pipeline=gz.catalog_ops.ReadLabel(label_uri_field="agb_uri", band="agb_mg_ha"),
    catalog=biomass_train_catalog,
    sampler=gz.sampling.RandomSampler(chip_size=(256, 256), length=300),
    n_pixels=500_000,
    nan_policy=gz.sklearn.NanPolicy(
        on_input="drop",
        on_label="drop",
    ),
)
joblib.dump(reg, "/models/agb_gbr.joblib")

# === Predict ===
predict = gz.Sequential([
    feature_pipeline,
    gz.sklearn.PixelwiseRegressor(
        estimator=joblib.load("/models/agb_gbr.joblib"),
        nan="mask",
        fill_value=np.nan,                # ← regression default; classification used -1
        output_dtype="float32",
    ),
])
```

**Notes.**

- `PixelwiseRegressor` rather than `PixelwiseClassifier`.
  The wrapper calls `.predict()` either way, but defaults differ: `fill_value=np.nan` (vs `-1` for classifiers), `output_dtype="float32"` (vs `uint8`).
- Regression is far more sensitive to feature scale than tree-based classification.
  If using a linear regressor (Ridge, ElasticNet), prefer a sklearn `Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])` (Example 5).
- Uncertainty: `GradientBoostingRegressor` with `loss="quantile"` can give prediction intervals — wrap with `PixelwiseRegressor` for each quantile separately, or write a small custom wrapper that produces a 3-band output (5th, 50th, 95th).

-----

## 4. Pixel-wise PCA dimensionality reduction

**Setting.** Hyperspectral data (~200 bands).
Reduce to 10 principal components for downstream classification or visualisation.
The transformer case — output is multi-band, not single-band.

**Patch strategy.** `GeoCatalog + RandomSampler` over a representative subset of scenes for fitting PCA. Apply per-scene at inference.

```python
from sklearn.decomposition import PCA

# === Fit PCA on a representative sample ===
pca = gz.sklearn.fit_pixelwise(
    estimator=PCA(n_components=10, whiten=True),
    X_pipeline=hyperspectral_pipeline,
    y_pipeline=None,                       # unsupervised
    catalog=hyperspectral_catalog,
    sampler=gz.sampling.RandomSampler(chip_size=(64, 64), length=100),
    n_pixels=200_000,
    nan_policy=gz.sklearn.NanPolicy(on_input="drop"),
)
joblib.dump(pca, "/models/pca_10.joblib")

# === Apply as preprocessing in any pipeline ===
preprocess = gz.Sequential([
    hyperspectral_pipeline,
    gz.sklearn.PixelwiseTransformer(
        estimator=joblib.load("/models/pca_10.joblib"),
        nan="mask",
    ),
    # Output: (10, H, W) GeoTensor — 10 PCA components
])

# Compose with anything downstream
classify = gz.Sequential([
    preprocess,
    gz.sklearn.PixelwiseClassifier(estimator=clf_on_pca, nan="mask"),
])
```

**Notes.**

- The transformer’s output is `(n_components, H, W)` — geotoolz’s `FromPixelMajor` reads `n_components` from the table shape.
- For genuine preprocessing in a *training pipeline*, prefer composing PCA into the sklearn `Pipeline` itself (Example 5) — keeps everything in one fitted artifact.
- Beware of fitting PCA on the wrong subset: a sample drawn from cloud-free pixels of forested scenes won’t capture variance you need for shadow or urban classes.
  The sample distribution should mirror the prediction distribution.

-----

## 5. Full sklearn Pipeline composition

**Setting.** Soil moisture regression from Sentinel-1 backscatter + DEM features.
The estimator is a multi-stage sklearn Pipeline: imputation → scaling → polynomial feature expansion → Ridge regression.
The killer interop feature — the whole thing is **one Operator** on the geotoolz side.

**Patch strategy.** `GeoCatalog + RandomSampler` for training.
The Pipeline is one estimator from geotoolz’s perspective.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

sklearn_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("poly",    PolynomialFeatures(degree=2, interaction_only=True)),
    ("reg",     Ridge(alpha=1.0)),
])

# Fit the whole Pipeline as one estimator
fitted = gz.sklearn.fit_pixelwise(
    estimator=sklearn_pipe,
    X_pipeline=sm_feature_pipeline,
    y_pipeline=gz.catalog_ops.ReadLabel(label_uri_field="sm_uri", band="vwc"),
    catalog=sm_catalog,
    sampler=gz.sampling.RandomSampler(chip_size=(256, 256), length=400),
    n_pixels=800_000,
    nan_policy=gz.sklearn.NanPolicy(
        on_input="propagate",          # let SimpleImputer handle NaN
        on_label="drop",
    ),
)
joblib.dump(fitted, "/models/sm_pipeline.joblib")

# Inference — the whole sklearn Pipeline is one Operator
predict = gz.Sequential([
    sm_feature_pipeline,
    gz.sklearn.PixelwiseRegressor(
        estimator=joblib.load("/models/sm_pipeline.joblib"),
        nan="propagate",                # SimpleImputer at the start handles it
    ),
])
```

**Notes.**

- `nan="propagate"` here because the first stage of the sklearn Pipeline is `SimpleImputer`, which is itself a NaN handler.
  Setting `nan="mask"` would mask NaN *before* it reaches the imputer, defeating the imputer’s purpose.
  This is a real subtlety: NaN policy on the wrapper interacts with NaN handling inside the sklearn Pipeline.
- The wrapped sklearn Pipeline serialises (via joblib) as one artifact — `n_features_in_` on the Pipeline gives the *original* feature count (before polynomial expansion), so the geotoolz wrapper’s shape check works correctly.
- Provenance: `fitted.get_params(deep=True)` records every nested estimator’s config. Hash that, store in the regulatory artifact, and you have full reproducibility of the multi-stage Pipeline.

-----

## 6. Per-tile classifier (local fit, local predict)

**Setting.** Crop classification across an MGRS grid spanning multiple ecoregions and climate zones.
A global model underperforms because growing seasons, atmospheric profiles, and dominant crops vary by region.
Train one classifier per MGRS tile; dispatch on tile ID at inference.

**Patch strategy.** `GeoCatalog + RandomSampler` for fitting *each tile’s model* separately.
`Switch` for dispatch at inference.

```python
import geotoolz as gz
from sklearn.ensemble import RandomForestClassifier

# === Fit one model per tile ===
tile_models = {}
for tile_id in train_catalog.tiles():
    tile_catalog = train_catalog.filter(tile=tile_id)
    clf = gz.sklearn.fit_pixelwise(
        estimator=RandomForestClassifier(n_estimators=200, n_jobs=-1),
        X_pipeline=feature_pipeline,
        y_pipeline=label_pipeline,
        catalog=tile_catalog,
        sampler=gz.sampling.RandomSampler(chip_size=(256, 256), length=100),
        n_pixels=300_000,
        nan_policy=gz.sklearn.NanPolicy(on_input="drop", on_label="drop"),
    )
    joblib.dump(clf, f"/models/crop_rf_{tile_id}.joblib")
    tile_models[tile_id] = clf

# === Dispatch via Switch at inference ===
classifier_dispatch = gz.core.Switch(
    key=lambda gt: gt.metadata["mgrs_tile"],
    cases={
        tile_id: gz.sklearn.PixelwiseClassifier(
            estimator=clf, nan="mask", output_dtype="uint8",
        )
        for tile_id, clf in tile_models.items()
    },
    default=gz.core.Raise("no model for tile: {key}"),
)

predict = gz.Sequential([feature_pipeline, classifier_dispatch])

# Catalog-driven inference — each scene gets routed to its tile's classifier
gz.catalog_ops.CatalogPipeline(test_catalog, predict, n_workers=8).run()
```

**Notes.**

- `Switch` reads `gt.metadata["mgrs_tile"]` — the reader must populate this from the catalog row’s metadata.
  Most readers do, but verify with `gz.core.Tap(lambda gt: print(gt.metadata))` if dispatch isn’t firing.
- Storage cost: one pickle per tile (could be 100+ models for a country-scale catalog).
  Lazy-loading helps — keep `tile_models[tile_id] = path` and load inside `Switch` on first use, cached thereafter.
- The harder question: how many tiles share enough signal that they could use *one* model?
  Clustering tiles by some similarity metric (mean spectral signature, dominant land cover) before fitting can collapse the model registry.
  Worth the engineering only if you have many tiles.

-----

## 7. Time-series crop classification

**Setting.** Crop classification from Sentinel-2 monthly composites (`(C=10, T=12, H, W)`) — twelve months of features per pixel.
The estimator sees each pixel’s full year as a `120`-dim feature vector and predicts one crop class per pixel.

**Patch strategy.** `GeoCatalog + RandomSampler`, but the catalog rows reference *temporal stacks* (not individual scenes).
Readers materialise `(C, T, H, W)` GeoTensors.

```python
import geotoolz as gz
from sklearn.ensemble import RandomForestClassifier

# === Temporal feature pipeline ===
temporal_features = gz.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),    # acts per-timestep
    gz.correction.TOAToBOA(sun_zenith_band=-2),
    gz.temporal.LinearGapFill(),                       # interpolate cloud-masked gaps
])

# === Fit with temporal adapter ===
clf = gz.sklearn.fit_pixelwise(
    estimator=RandomForestClassifier(n_estimators=400, max_depth=15, n_jobs=-1),
    X_pipeline=temporal_features,
    y_pipeline=label_pipeline,
    catalog=temporal_catalog,
    sampler=gz.sampling.RandomSampler(chip_size=(256, 256), length=300),
    adapter=gz.sklearn.ToTemporalPixelMajor(time_handling="features"),  # (C,T,H,W) → (H*W, T*C)
    n_pixels=500_000,
    nan_policy=gz.sklearn.NanPolicy(on_input="drop", on_label="drop"),
)
joblib.dump(clf, "/models/crop_ts_rf.joblib")

# === Predict ===
predict = gz.Sequential([
    temporal_features,
    gz.sklearn.PixelwiseClassifier(
        estimator=joblib.load("/models/crop_ts_rf.joblib"),
        adapter=gz.sklearn.ToTemporalPixelMajor(time_handling="features"),
        nan="mask",
        output_dtype="uint8",
    ),
])
```

**Notes.**

- **Adapter parity between fit and predict is essential.** The fit step uses `ToTemporalPixelMajor(time_handling="features")` to produce `(H*W, T*C)` training data; predict must use the *same* adapter so feature order matches.
  Setting `time_handling="samples"` at predict would feed `(C,)` features to a model trained on `(T*C,)` — silent shape mismatch caught only by the `n_features_in_` check.
- `LinearGapFill` fills cloud-masked timesteps so the estimator sees a complete time series.
  Alternative: keep NaN, use a NaN-tolerant classifier like `HistGradientBoostingClassifier`.
- For genuinely temporal models (DTW-based classifiers, shapelet learners), use `time_handling="panel"` with `PanelTable` and an `sktime`-compatible estimator.
  That path lives in the (future) `geotoolz.sktime` integration.
- Reducing time first (e.g., harmonic features, percentile composites) and fitting a non-temporal classifier on the reduced features is often competitive with deep temporal models — and uses the same standard `PixelwiseClassifier` path.

-----

## 8. Spatial cross-validation with `GridSearchCV`

**Setting.** Hyperparameter tuning for the crop classifier.
Standard k-fold CV leaks information geographically — nearby pixels are spatially correlated, so a pixel in fold 1 and a pixel in fold 2 from the same field are nearly identical, inflating CV scores.
Need **spatial block CV**: group pixels by a coarse spatial block, then `GroupKFold` ensures all pixels in one block stay in one fold.

**Patch strategy.** `GeoCatalog + RandomSampler` for training.
CV machinery lives inside the sklearn estimator passed to `fit_pixelwise`.

```python
import geotoolz as gz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold

# === Fit_pixelwise extracts a spatial-block group ID per pixel ===
# `block_band` is a band in the y_pipeline output carrying the spatial block ID.
block_label_pipeline = gz.Sequential([
    label_pipeline,
    gz.spatial.BlockID(block_size_m=10_000),     # 10 km blocks → block ID band
])

# === Wrap classifier in GridSearchCV with spatial folds ===
grid = GridSearchCV(
    estimator=RandomForestClassifier(n_jobs=-1),
    param_grid={
        "n_estimators": [200, 400, 600],
        "max_depth":    [10, 20, None],
        "min_samples_leaf": [1, 5, 20],
    },
    cv=GroupKFold(n_splits=5),
    scoring="f1_macro",
    n_jobs=1,                    # n_jobs handled by RF, avoid nested parallelism
    verbose=2,
)

# === Fit — fit_pixelwise passes the block ID as `groups=` ===
fitted_grid = gz.sklearn.fit_pixelwise(
    estimator=grid,
    X_pipeline=feature_pipeline,
    y_pipeline=block_label_pipeline,
    catalog=train_catalog,
    sampler=gz.sampling.RandomSampler(chip_size=(256, 256), length=300),
    n_pixels=500_000,
    nan_policy=gz.sklearn.NanPolicy(on_input="drop", on_label="drop"),
    group_band=1,                # band 1 of y_pipeline is the block ID
)

print(f"Best params: {fitted_grid.best_params_}")
print(f"Best CV score: {fitted_grid.best_score_:.3f}")

# Best estimator is itself fitted on the full training set
joblib.dump(fitted_grid.best_estimator_, "/models/crop_rf_tuned.joblib")
```

**Notes.**

- The `group_band` argument is how `fit_pixelwise` passes per-sample group IDs to `estimator.fit(X, y, groups=...)`.
  The block ID is treated as an extra label channel — packaged into the `y_pipeline` output as a second band, and extracted by the fit helper.
- Block size is a modelling choice.
  Too small (1 km) and CV still leaks; too large (50 km) and you have too few blocks for meaningful k-fold.
  Rule of thumb: block size ≈ 5–10× the spatial autocorrelation range of the target, often 5–20 km for vegetation.
- For *temporal* CV (time-series classification with seasonal effects), pair with `TimeSeriesSplit` or block by year/season instead of space.
  Same mechanism — just different group definition.
- The fitted `GridSearchCV` object is itself an sklearn estimator.
  The whole search wraps as one Operator at inference time — same composition story as Example 5.

-----

## 9. Unsupervised land-cover clustering

**Setting.** No labels available.
Want a coarse land-cover segmentation from Sentinel-2 features as an exploratory product or pre-labelling step for active learning.
Use MiniBatchKMeans for scalability.

**Patch strategy.** `GeoCatalog + RandomSampler` for fitting on a sample.
`CatalogPipeline` for inference over the whole catalog.

```python
from sklearn.cluster import MiniBatchKMeans

# === Unsupervised fit ===
kmeans = gz.sklearn.fit_pixelwise(
    estimator=MiniBatchKMeans(n_clusters=12, batch_size=10_000, random_state=42),
    X_pipeline=feature_pipeline,
    y_pipeline=None,                       # unsupervised — no labels needed
    catalog=catalog,
    sampler=gz.sampling.RandomSampler(chip_size=(256, 256), length=200),
    n_pixels=500_000,
    nan_policy=gz.sklearn.NanPolicy(on_input="drop"),
)
joblib.dump(kmeans, "/models/landcover_kmeans_12.joblib")

# === Predict cluster IDs per pixel ===
predict = gz.Sequential([
    feature_pipeline,
    gz.sklearn.PixelwiseClusterer(
        estimator=joblib.load("/models/landcover_kmeans_12.joblib"),
        nan="mask",
        output_dtype="uint8",
    ),
])

# === Bonus: cluster centroids as inspection ===
centroids = kmeans.cluster_centers_           # (12, n_features)
plot_centroid_spectra(centroids, feature_names=["B02", "B03", "B04", "B08", "NDVI", "NDWI"])
```

**Notes.**

- `y_pipeline=None` for unsupervised.
  `fit_pixelwise` skips the label alignment step entirely.
- `MiniBatchKMeans` is the right choice for large catalogs — `KMeans` fits in one shot and won’t scale past a few million pixels.
- The cluster IDs are arbitrary integers.
  For an interpretable land-cover product, label each cluster post-hoc (inspect a sample of pixels from each cluster, assign semantic names).
- Pair with [`Histogram`](../examples/tips_n_tricks.md#histogram) on the output to see cluster frequencies per scene — useful for detecting class imbalance and rare clusters.

-----

## 10. Incremental fitting for large training sets

**Setting.** Training data is genuinely larger than memory — billions of labeled pixels across a continental catalog.
Batch-fit any single estimator on all of it is infeasible.
Use `partial_fit`-capable estimators streamed through the catalog.

**Patch strategy.** `GeoCatalog + GridSampler` (deterministic coverage, multiple epochs) for fitting.
Standard per-scene prediction at inference.

```python
import geotoolz as gz
from sklearn.linear_model import SGDClassifier

clf = gz.sklearn.fit_pixelwise_incremental(
    estimator=SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        learning_rate="optimal",
        random_state=42,
    ),
    X_pipeline=feature_pipeline,
    y_pipeline=label_pipeline,
    catalog=continental_catalog,
    sampler=gz.sampling.GridSampler(chip_size=(256, 256)),
    chunk_size=10_000,                     # pixels per partial_fit call
    classes=np.arange(N_CLASSES),          # required for SGD classifier
    epochs=3,
    nan_policy=gz.sklearn.NanPolicy(on_input="drop", on_label="drop"),
    shuffle=True,                          # shuffle chips within each epoch
)
joblib.dump(clf, "/models/crop_sgd.joblib")
```

**Implementation sketch** (helper internals):

```python
def fit_pixelwise_incremental(*, estimator, X_pipeline, y_pipeline,
                               catalog, sampler, chunk_size, classes,
                               epochs, nan_policy, shuffle=True, **_):
    for epoch in range(epochs):
        slices = list(sampler(catalog))
        if shuffle: random.shuffle(slices)
        buf_X, buf_y = [], []
        for sl in slices:
            X_chip = X_pipeline(sl.load())
            y_chip = y_pipeline(sl.load())
            X, y = _to_pixel_major(X_chip), _to_pixel_major(y_chip)
            X, y = nan_policy.apply(X, y)
            buf_X.append(X); buf_y.append(y)
            if sum(len(b) for b in buf_X) >= chunk_size:
                Xc = np.concatenate(buf_X)[:chunk_size]
                yc = np.concatenate(buf_y)[:chunk_size]
                estimator.partial_fit(Xc, yc, classes=classes)
                buf_X, buf_y = [], []
        # flush final partial chunk
        if buf_X:
            estimator.partial_fit(np.concatenate(buf_X), np.concatenate(buf_y), classes=classes)
    return estimator
```

**Notes.**

- `classes=np.arange(N_CLASSES)` is required for classifiers — `partial_fit` can’t infer the class set from a single chunk.
  For regressors and most transformers, omit it.
- Epochs matter for SGD-based learners. 1 epoch is usually under-trained; 3–5 epochs is typical. Shuffle the slice order between epochs.
- Inference uses the standard `PixelwiseClassifier`.
  The wrapper doesn’t care whether the estimator was batch-fit or incrementally-fit — same `.predict()` interface.
- For incremental learning that genuinely *adapts* over time (online learning with concept drift), the `partial_fit` loop runs continuously rather than for fixed epochs.
  Out of scope for this fit helper; build on the helper’s internals directly.

-----

## Cross-cutting observations

A few patterns recur across these examples worth naming explicitly:

**Adapter parity.** The fit-time adapter and the inference-time adapter must match. The most common silent bug — using `ToTemporalPixelMajor(...="features")` to fit and `ToPixelMajor()` to predict, or different `time_handling` modes between fit and serve — produces shape-mismatched feature vectors that sklearn either rejects (best case) or silently predicts garbage on.
Always pass the adapter as a named variable shared between fit and predict pipelines.

**NaN policy asymmetry between fit and predict.** Fit-time policies usually *drop* NaN rows because you can’t learn from incomplete labels.
Predict-time policies usually *mask* NaN rows because the output must be a complete raster with NaN written at masked positions.
Defaults reflect this; verify when changing them.

**Pre-fitted estimator as the artifact.** Across all ten examples, the estimator pickle is the durable artifact, not the operator graph.
The graph is the *instructions* for how to apply the estimator; the estimator is the *content*.
Both need to land in the regulatory-artifact bundle for full reproducibility, but their lifecycles differ — the graph YAML may change across deployments while the estimator stays pinned.

**Feature pipeline shared across fit and predict.** The same `feature_pipeline` Operator object appears in both paths in every example.
This is the train/serve skew prevention from the [use-cases doc](../examples/usecases.md#3-ml-training-and-inference) — violated, models silently degrade in production.

**The right patch strategy is decided once per workflow.** Mixing strategies (GeoCatalog for training, xrpatcher for inference, ApplyToChips for tile serving) is fine *as long as the chip operator is identical*.
The patch strategy determines I/O performance and scalability; the operator graph determines correctness.
Keep them orthogonal.