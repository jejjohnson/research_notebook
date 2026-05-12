---
title: geotoolz.skimage examples
subject: geotoolz examples
subtitle: A gallery of skimage Operator patterns over GeoTensor
short_title: skimage examples
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, geotoolz, skimage, examples, remote-sensing
---

> **Status:** companion to [`design_skimage.md`](design_skimage.md) — concrete worked examples for the skimage Operator bridge.
> **Audience:** anyone reaching for skimage from a `geotoolz` pipeline (preprocessing, feature engineering, viz post-processing) and looking for the canonical recipe before writing their own wrapper.

Companion to the [design doc](design_skimage.md).
Examples cover denoising, morphology, feature extraction, segmentation, restoration, edge detection, and ML-feature engineering — across the patterns where skimage fits naturally into geospatial workflows.

## Contents

- [How skimage fits in](#how-skimage-fits-in)
- [Workflow patterns](#workflow-patterns)
- [1. Speckle reduction for SAR (per-band Lee filter)](#1-speckle-reduction-for-sar-per-band-lee-filter)
- [2. Cloud-mask post-processing (morphology)](#2-cloud-mask-post-processing-morphology)
- [3. Adaptive contrast enhancement (CLAHE for viz)](#3-adaptive-contrast-enhancement-clahe-for-viz)
- [4. Edge detection for field boundaries](#4-edge-detection-for-field-boundaries)
- [5. Superpixel segmentation (object-based image analysis)](#5-superpixel-segmentation-object-based-image-analysis)
- [6. Inpainting cloud-masked gaps](#6-inpainting-cloud-masked-gaps)
- [7. Multi-scale features for ML](#7-multi-scale-features-for-ml)
- [8. Texture features (GLCM) for land-cover classification](#8-texture-features-glcm-for-land-cover-classification)
- [9. Blob detection for object localisation](#9-blob-detection-for-object-localisation)
- [10. Image registration / co-alignment](#10-image-registration--co-alignment)

-----

## How skimage fits in

skimage is decades of community work on image processing — filtering, morphology, segmentation, restoration, registration, feature extraction.
For geospatial data, it lives in three places naturally:

**Preprocessing** — speckle reduction, contrast enhancement, gap-filling.
Runs before downstream analysis or ML.

**Feature engineering for ML** — texture statistics, multi-scale features, edge maps.
The outputs become input bands for a classifier or regressor.

**Post-processing** — morphological cleanup of classification masks, region labelling, vectorisation prep.
Runs after a model produces a raw output.

The three wrappers (`PerBand`, `MultiBand`, `Func`) cover every skimage function.
The examples below show which wrapper fits each task and why.

## Workflow patterns

A short orientation for *where in a pipeline* skimage typically sits:

**Single-scene work.** Most skimage examples below operate on a single GeoTensor at a time — they’re stateless and chip-friendly.
Drop them into any `Sequential` and they compose like any other Operator.

**Chipwise inference.** Wrap a skimage-heavy pipeline in `ApplyToChips` for big scenes.
Watch the edge effects — most skimage filters have boundary modes that produce artefacts at chip seams.
Use overlap (stride < chip_size) and stitch with averaging where possible.

**Catalog-scale ETL.** Drop the same skimage Operators into a `CatalogPipeline` to run across many scenes.
The only constraint: skimage functions are CPU-bound, so wall-clock time scales with worker count.

**As pre/post processing around an ML model.** Common pattern — skimage preprocesses features, ML model classifies, skimage cleans up the output.
The whole thing is one `Sequential`.

-----

## 1. Speckle reduction for SAR (per-band Lee filter)

**Setting.** Sentinel-1 SAR data is dominated by multiplicative speckle noise.
Lee filtering or its variants smooth speckle while preserving edges.
A required preprocessing step before most SAR analysis.

**Wrapper.** `PerBand` — Lee filter is 2D and operates per-polarisation.

```python
import geotoolz as gz
from skimage.restoration import denoise_tv_chambolle

# Lee filter not in skimage core; use TV-Chambolle as a close cousin
speckle_filter = gz.skimage.PerBand(
    denoise_tv_chambolle,
    weight=0.1,
    n_iter_max=200,
    dtype_in="float32",
    dtype_out="float32",
)

sar_pipeline = gz.Sequential([
    gz.sar.LinearToDB(),          # convert linear backscatter to dB
    speckle_filter,                # filter in dB space (additive noise)
    gz.sar.DBToLinear(),           # back to linear if needed downstream
])
```

For the actual Lee filter (a fast sliding-window mean/variance estimator):

```python
def lee_filter(arr, size=7):
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(arr, size)
    sqr_mean = uniform_filter(arr ** 2, size)
    var = sqr_mean - mean ** 2
    overall_var = arr.var()
    weights = var / (var + overall_var)
    return mean + weights * (arr - mean)

# Wrap with PerBand — works for any 2D function, not just skimage's
gz.skimage.PerBand(lee_filter, size=7)
```

**Notes.**

- `dtype_in="float32"` is critical here.
  `denoise_tv_chambolle` expects floats and silently does bad things on integer input.
  The wrapper makes this conversion explicit and YAML-round-trippable.
- The pipeline runs the filter in dB space (where speckle is additive) rather than linear space (where it’s multiplicative).
  Mathematically cleaner — TV-Chambolle assumes additive noise.
- For very large SAR scenes, this is per-pixel CPU-bound.
  Push to GPU via `cucim.skimage.restoration.denoise_tv_chambolle` for a ~10–50× speedup; same wrapper, just the import changes.

-----

## 2. Cloud-mask post-processing (morphology)

**Setting.** Sentinel-2’s QA60 cloud mask is noisy at the boundaries — isolated cloudy pixels in clear regions, isolated clear pixels in cloudy regions, and ragged edges.
Morphological opening removes small false-positive clouds; closing fills small holes in cloud bodies; a final dilation expands the mask conservatively to catch unmasked cloud shadows.

**Wrapper.** `PerBand`, accessed via the `Morphology` sugar namespace.

```python
import geotoolz as gz

cloud_cleanup = gz.Sequential([
    gz.cloud.MaskFromQABits(qa_band=-1, bits=[10, 11]),  # raw mask
    gz.skimage.Morphology.opening(radius=2, shape="disk"),   # remove small islands
    gz.skimage.Morphology.closing(radius=3, shape="disk"),   # fill small holes
    gz.skimage.Morphology.dilation(radius=5, shape="disk"),  # conservative buffer
])

# Use the cleaned mask in downstream pipelines
classification = gz.Sequential([
    feature_pipeline,
    gz.cloud.ApplyMask(cloud_cleanup),
    classifier,
])
```

**Notes.**

- The sugar namespace produces `PerBand` Operators under the hood — the YAML config records them as `_target_: geotoolz.skimage.Morphology.opening` with `radius=2, shape="disk"`.
  Loader resolves the sugar to the underlying `PerBand(morphology.opening, footprint=disk(2))`.
- Order matters.
  Opening *then* closing is “remove specks, then fill holes” — the typical cleanup.
  Closing *then* opening is “fill holes, then remove specks” which produces a different result.
  Document the intent in comments.
- For cloud shadows specifically, the `dilation` step is asymmetric — you want to expand the mask in the sun direction.
  A directional structuring element (`shape="line"` with an angle) does this; for most workflows the symmetric disk is sufficient.

-----

## 3. Adaptive contrast enhancement (CLAHE for viz)

**Setting.** Sentinel-2 true-colour visualisation.
Histogram-stretched RGB looks flat — shadows are black, highlights are washed out.
CLAHE (Contrast Limited Adaptive Histogram Equalization) gives much better visual contrast for browse imagery and tile-server output.

**Wrapper.** `PerBand` with explicit dtype conversion.

```python
import geotoolz as gz
from skimage import exposure

true_color_clahe = gz.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.radiometry.SelectBands(indices=[3, 2, 1]),    # NIR, R, G — false color
    gz.radiometry.PercentileClip(p_min=2, p_max=98),
    gz.core.Rescale(in_range=(0, 10_000), out_range=(0, 1), dtype="float32"),
    gz.skimage.PerBand(
        exposure.equalize_adapthist,
        clip_limit=0.03,
        kernel_size=128,
        dtype_in="float01",
        dtype_out="float32",
    ),
    gz.viz.ToUint8RGB(),       # final cast for PNG encoding
])
```

**Notes.**

- The explicit `Rescale` before CLAHE is intentional. `equalize_adapthist` with `dtype_in="float01"` would call `img_as_float`, which divides by `dtype.max` — for `uint16` Sentinel-2 reflectance with max value ~10000, that gives values around `0.15`.
  CLAHE on tiny values produces no visible contrast.
  The explicit `Rescale(in_range=(0, 10_000))` does the right thing.
- This pipeline drops into a tile server (the [use-cases doc Example 7](../examples/usecases.md#7-tile-server-zxy--png)) as a layer config. Latency is dominated by CLAHE’s per-tile computation — for sub-second tile rendering, pre-compute and cache.

-----

## 4. Edge detection for field boundaries

**Setting.** Agricultural field-boundary extraction.
Edges in NDVI or NIR bands often align with field margins (different crops, harvest dates, soil types).
Sobel or Canny edge detection produces a per-pixel boundary-likelihood map.

**Wrapper.** `PerBand` for Sobel; `Func` for Canny (which is 2D-only with strict input requirements).

```python
import geotoolz as gz

# Simple: Sobel magnitude per band
sobel_pipeline = gz.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.indices.NDVI(red_idx=2, nir_idx=3),
    gz.skimage.Filter.sobel(),
])

# More refined: Canny edges on a smoothed NDVI
canny_pipeline = gz.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.indices.NDVI(red_idx=2, nir_idx=3),
    gz.skimage.Filter.gaussian(sigma=1.5),
    gz.skimage.Func(
        skimage.feature.canny,
        requires="2d",
        sigma=1.0,
        low_threshold=0.1,
        high_threshold=0.2,
        dtype_in="float32",
    ),
])
```

**Notes.**

- `Filter.sobel()` is the sugar; it expands to `PerBand(filters.sobel)` under the hood.
- Canny is `Func` rather than `PerBand` because the input must be a single 2D array — the upstream `NDVI` already produces a single-band output, so no looping is needed.
  Using `PerBand` here would still work (loop over one band) but `Func` is more honest.
- For *vectorising* the resulting edge raster into polygons, hand the output to `rasterio.features.shapes` or `geopandas`. skimage stops at raster edges; geotoolz doesn’t (yet) bridge to vector.

-----

## 5. Superpixel segmentation (object-based image analysis)

**Setting.** Object-based image analysis (OBIA) — group similar neighbouring pixels into segments, then classify the segments rather than individual pixels.
Reduces noise, captures spatial context, and produces output that’s easier to vectorise.
SLIC (Simple Linear Iterative Clustering) is the standard fast segmenter.

**Wrapper.** `Func` with `requires="3d_hwc"` because SLIC consumes a multi-channel image and returns a single-channel integer label map.

```python
import geotoolz as gz
from skimage import segmentation

slic_segmenter = gz.skimage.Func(
    segmentation.slic,
    requires="3d_hwc",                       # SLIC wants HWC input
    n_segments=400,
    compactness=10,
    sigma=1.0,
    start_label=1,
    channel_axis=-1,
    dtype_in="float01",
    dtype_out="int32",
)

obia_pipeline = gz.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.radiometry.SelectBands(indices=[1, 2, 3, 7]),   # G, R, NIR, SWIR
    gz.radiometry.PercentileClip(p_min=2, p_max=98),
    gz.core.Rescale(in_range=(0, 10_000), out_range=(0, 1)),
    slic_segmenter,                                     # → integer-label GeoTensor
])
```

For pixel-wise classification informed by segments:

```python
gz.Sequential([
    feature_pipeline,
    gz.core.Fanout({
        "features": gz.core.Identity(),
        "segments": slic_segmenter,
    }),
    # Aggregate per-segment features (e.g., mean per segment per band)
    gz.spatial.AggregateBySegments(reduction="mean"),
    gz.sklearn.PixelwiseClassifier(estimator=segment_clf, ...),
])
```

**Notes.**

- `requires="3d_hwc"` triggers the wrapper to transpose CHW → HWC before applying SLIC, then output is already 2D (single label map per pixel), no inverse transpose needed.
- Segment labels are **arbitrary integers** that change between runs (depend on internal initialisation order).
  Don’t expect stable labels across re-runs even with the same input — use `start_label` and a fixed random seed if reproducibility matters.
- For very large scenes, SLIC is memory-heavy.
  `slic_zero` is a faster variant for the same wrapper signature; `felzenszwalb` is another alternative with different boundary characteristics.

-----

## 6. Inpainting cloud-masked gaps

**Setting.** A Sentinel-2 NDVI mosaic has small holes from masked clouds.
For visualisation or for downstream filters that don’t tolerate NaN, fill the holes via biharmonic inpainting — a physically-motivated smooth interpolation that respects boundary values.

**Wrapper.** `Func` with `requires="2d"` (inpainting operates on a 2D array with a mask).

```python
import geotoolz as gz
from skimage import restoration

class InpaintNaN(Operator):
    """Biharmonic inpaint over NaN pixels."""
    def _apply(self, gt):
        arr = np.asarray(gt).astype(np.float32)
        mask = np.isnan(arr)
        if not mask.any():
            return gt
        # inpaint per band
        filled = np.stack([
            restoration.inpaint_biharmonic(arr[i], mask[i])
            for i in range(arr.shape[0])
        ], axis=0)
        return GeoTensor(values=filled, transform=gt.transform, crs=gt.crs)
```

Or more idiomatically using `NanFill`:

```python
gz.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.indices.NDVI(red_idx=2, nir_idx=3),
    gz.skimage.NanFill(method="biharmonic"),   # fills NaN gaps
])
```

**Notes.**

- Biharmonic inpainting is slow (~seconds per band for a 1024×1024 chip).
  For dense gap-filling at scale, use a faster method (`method="local_mean"`) and accept the quality trade-off.
- Use sparingly — inpainting *invents* data.
  For scientific analysis, keep NaN visible and propagate.
  For visualisation or as a preprocessing step for filters that crash on NaN, inpainting is correct.
- Pair with [`Snapshot`](../examples/tips_n_tricks.md#snapshot) to inspect before/after — the `NanFill` should produce visually-plausible output; if it doesn’t, something’s wrong upstream.

-----

## 7. Multi-scale features for ML

**Setting.** Build an ML feature stack from multi-scale image statistics — Gaussian-blurred versions at multiple sigmas, gradient magnitudes, local intensity — to feed a pixel-wise classifier. skimage has a built-in `multiscale_basic_features` that produces all of these in one call.

**Wrapper.** `MultiBand` with `channel_axis=0` (skimage’s function uses channel axis for multi-channel input).

```python
import geotoolz as gz
from skimage import feature

multiscale_features = gz.skimage.MultiBand(
    feature.multiscale_basic_features,
    channel_axis=0,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=1.0,
    sigma_max=16.0,
    num_sigma=5,
)

feature_pipeline = gz.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.radiometry.SelectBands(indices=[1, 2, 3, 7]),
    multiscale_features,           # → many feature bands
])

classify = gz.Sequential([
    feature_pipeline,
    gz.sklearn.PixelwiseClassifier(estimator=clf, ...),
])
```

**Notes.**

- The output has many more channels than the input — `multiscale_basic_features` with `num_sigma=5` and three feature types produces ~45 bands from a 4-band input.
  Downstream `PixelwiseClassifier` automatically picks this up (sklearn’s `n_features_in_` matches the flattened feature count).
- Fit the classifier on the *same* multi-scale feature definition you’ll use at inference.
  As with all skimage preprocessing, parity between fit and serve matters — the feature pipeline goes in the YAML, the classifier pickle references it through provenance.
- Memory cost is real: 45 bands × `(512, 512)` chip × `float32` = ~45 MB per chip just for features.
  For `RandomForestClassifier`, this is fine.
  For deep models, consider feature subsetting or smaller sigmas.

-----

## 8. Texture features (GLCM) for land-cover classification

**Setting.** Gray-Level Co-occurrence Matrix (GLCM) statistics are classical texture descriptors — useful for separating urban from bare soil, or forest from shrubland, where spectral signatures alone are ambiguous.
Compute GLCM properties (contrast, homogeneity, energy, correlation) in sliding windows over each band.

**Wrapper.** Custom `Func` because GLCM doesn’t fit any single skimage function — it’s `greycomatrix` + `greycoprops` over a sliding window.

```python
import geotoolz as gz
from skimage.feature import graycomatrix, graycoprops
from numpy.lib.stride_tricks import sliding_window_view

def glcm_features(band, window=16, distances=[1], angles=[0, np.pi/2]):
    """Per-pixel GLCM contrast and homogeneity over a window."""
    band_u8 = (band * 255 / band.max()).astype(np.uint8)
    out_contrast = np.zeros_like(band, dtype=np.float32)
    out_homog = np.zeros_like(band, dtype=np.float32)
    # naive sliding window — not optimised, illustrative only
    for i in range(window//2, band.shape[0] - window//2):
        for j in range(window//2, band.shape[1] - window//2):
            patch = band_u8[i-window//2:i+window//2, j-window//2:j+window//2]
            glcm = graycomatrix(patch, distances, angles, levels=256, symmetric=True)
            out_contrast[i, j] = graycoprops(glcm, "contrast").mean()
            out_homog[i, j] = graycoprops(glcm, "homogeneity").mean()
    return np.stack([out_contrast, out_homog], axis=0)  # (2, H, W)

# Wrap — per-band returning a 2-channel per-band output
glcm_op = gz.skimage.Func(
    glcm_features,
    requires="2d",
    window=16,
    dtype_in="float01",
    dtype_out="float32",
)
```

For practical use, prefer a vectorised GLCM implementation (e.g., via `numba` or `cucim` GPU acceleration) — the naive version above is illustrative but slow.

**Notes.**

- GLCM is genuinely slow without optimisation.
  For catalog-scale work, use `cucim.skimage.feature` on GPU, or pre-aggregate windows at lower resolution.
- The output is a multi-channel feature stack (`contrast`, `homogeneity`, optionally `energy`, `correlation`).
  Drop the whole thing into a feature pipeline alongside spectral bands.
- For texture-aware classification, combine GLCM features with a per-pixel classifier (Example 7 pattern) — the multi-scale + texture combination often outperforms spectral-only models for built-up area mapping.

-----

## 9. Blob detection for object localisation

**Setting.** Detect localised circular or near-circular features in a raster — solar panels, methane point sources, isolated water bodies, dust plumes.
`blob_log` (Laplacian of Gaussian) finds local maxima at multiple scales.

**Wrapper.** `Func` with `requires="2d"`; output is a list of `(y, x, sigma)` tuples, not a raster — need a small custom Operator to handle that conversion.

```python
import geotoolz as gz
from skimage import feature

class BlobDetector(Operator):
    """Detect blobs in a single band; return a GeoTensor mask of blob centres."""
    def __init__(self, band_idx=0, min_sigma=2, max_sigma=10, threshold=0.05):
        self.band_idx = band_idx
        self.min_sigma, self.max_sigma = min_sigma, max_sigma
        self.threshold = threshold

    def _apply(self, gt):
        band = np.asarray(gt[self.band_idx]).astype(np.float32)
        blobs = feature.blob_log(
            band,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            threshold=self.threshold,
        )
        mask = np.zeros_like(band, dtype=np.float32)
        for y, x, sigma in blobs:
            mask[int(y), int(x)] = sigma   # store sigma as proxy for size
        return GeoTensor(
            values=mask[np.newaxis],       # (1, H, W)
            transform=gt.transform, crs=gt.crs,
        )

# Or, in a pipeline:
gz.Sequential([
    matched_filter_pipeline,             # produces methane likelihood
    BlobDetector(min_sigma=2, max_sigma=10, threshold=0.1),
])
```

**Notes.**

- The custom Operator handles the list-of-tuples-to-raster conversion that doesn’t fit any of the three wrappers cleanly.
  This is the *right* pattern when the function’s return shape doesn’t match the input.
- The output isn’t a feature stack — it’s a sparse raster with one non-zero value per detected blob.
  Downstream consumers should treat it as a point-list-encoded-as-raster, not a continuous field.
- For vector output (`{lat, lon, sigma, intensity}` rows in a GeoDataFrame), the Operator could write directly to a parquet sidecar via a custom terminal Operator — that’s a `Sink`-pattern variant.

-----

## 10. Image registration / co-alignment

**Setting.** Two images of the same scene from different dates are slightly misaligned (sub-pixel to a few pixels) due to orbital geometry and reprojection.
Phase cross-correlation gives a global translation estimate that can be applied as a sub-pixel shift.

**Wrapper.** `Func` with `requires="2d"`; return value is a `(shift_y, shift_x)` tuple, not a raster — custom Operator.

```python
import geotoolz as gz
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

class Coregister(Operator):
    """Align target to reference using phase cross-correlation."""
    def __init__(self, *, reference_band=0, upsample_factor=10):
        self.reference_band = reference_band
        self.upsample_factor = upsample_factor

    def _apply_pair(self, ref_gt, tgt_gt):
        ref = np.asarray(ref_gt[self.reference_band]).astype(np.float32)
        tgt = np.asarray(tgt_gt[self.reference_band]).astype(np.float32)
        offset, _, _ = phase_cross_correlation(
            ref, tgt, upsample_factor=self.upsample_factor,
        )
        # apply shift to every band of target
        shifted = np.stack([
            shift(tgt_gt[i], offset, mode="nearest")
            for i in range(tgt_gt.shape[0])
        ], axis=0)
        return GeoTensor(values=shifted, transform=tgt_gt.transform, crs=tgt_gt.crs)
```

Use inside a `Graph` for paired inputs:

```python
ref = gz.core.Input("reference")
tgt = gz.core.Input("target")
aligned = Coregister(reference_band=0)._apply_pair(ref, tgt)

g = gz.core.Graph(
    inputs={"reference": ref, "target": tgt},
    outputs={"aligned_target": aligned},
)
out = g(reference=ref_gt, target=tgt_gt)
```

**Notes.**

- Phase cross-correlation estimates *global translation* only.
  For more complex misalignment (rotation, scale, local distortion), use `skimage.transform.AffineTransform` with feature-matching or move to GDAL’s reprojection tooling.
- The `upsample_factor=10` gives sub-pixel accuracy (~0.1 pixel).
  Higher values give more precision but slow down the FFT-based correlation.
- Co-registration is a *prerequisite* for many time-series and change-detection workflows.
  Running it as the first step of a pipeline (against a fixed reference scene) ensures all subsequent ops see aligned data.

-----

## Cross-cutting observations

Several patterns recur across these examples worth naming:

**dtype handling lives at the wrapper boundary.** Every example specifies `dtype_in` and/or `dtype_out` explicitly.
Inputs from real readers come in `uint16` (S2 reflectance), `float32` (SAR backscatter), `int16` (DEMs); skimage functions want various things.
Setting these in the wrapper keeps the conversion visible in YAML and easy to debug.

**Sugar for the common cases, escape hatch for the rest.** Examples 2, 4, and 5 use the sugar namespaces (`Morphology`, `Filter`); Examples 6, 9, and 10 drop to custom Operators or `Func` because the function doesn’t fit a one-line invocation.
The split is right — sugar covers ~80% of real use; the wrappers cover the rest; custom Operators cover the genuinely-novel cases.

**NaN is the boundary between “works” and “produces garbage near edges”.** Every filter and morphology op has implicit NaN behaviour that’s almost never what you want.
Either propagate NaN intentionally, fill before filtering (Example 6 pattern), or mask out filter outputs near NaN boundaries explicitly.
The default `propagate` is honest but rarely operationally correct — set `nan_policy` deliberately.

**Output shape parity.** Examples 1–4, 7–8 preserve `(C, H, W)` — input shape and output shape match (modulo band count).
Examples 5, 9–10 *change* the shape semantics — segmentation produces label maps, blob detection produces sparse rasters, registration applies transformations.
Document this in the Operator’s docstring; downstream consumers care.

**Per-band looping is the bottleneck for hyperspectral.** Examples that use `PerBand` over 200-band hyperspectral data run 200 Python iterations.
Move to `cucim` GPU acceleration (`import cucim.skimage as skimage`) for a near-drop-in speedup, or use vectorised numpy/scipy equivalents (`scipy.ndimage.gaussian_filter` vectorises over `axes=`).

**Chip seams are the silent failure mode.** Running `ApplyToChips` over a skimage filter without overlap produces visible seams at chip boundaries.
For *any* spatial filter, use `stride < chip_size` and stitch with averaging where possible.
Catch the seams early with [`Diff`](../examples/tips_n_tricks.md#diff) against a whole-scene reference run on small test scenes.