---
title: geotoolz.skimage design doc
subject: geotoolz design
subtitle: Wrapping scikit-image as Operators over GeoTensor
short_title: skimage bridge
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, geotoolz, skimage, operators, remote-sensing
---

> **Status:** draft (2026-05-10) — first design pass on the skimage bridge.
> **Scope:** how every `scikit-image` function lands in a `geotoolz` Operator graph through a small family of wrapping Operators (`PerBand`, `MultiBand`, `Func`), axis-aware `HWC`/`CHW` adapters, dtype safety, and NaN handling.
> **Audience:** anyone composing geospatial preprocessing or feature-engineering pipelines that lean on skimage's filtering, morphology, segmentation, restoration, registration, or feature-extraction surface.

## Goals

Make every scikit-image function usable inside a `geotoolz` operator graph **as a single Operator**, without per-function wrapping code. scikit-image is decades of community work on filtering, morphology, segmentation, restoration, feature extraction, and registration — wrapping it well gives geotoolz an enormous algorithmic surface for a few hundred lines of glue.

Concretely:

- A small family of **wrapping Operators** (`PerBand`, `MultiBand`, `Func`) that adapt skimage’s calling conventions to `GeoTensor`.
- **Axis-aware adapters** that handle skimage’s `HWC`-vs-`CHW` channel conventions transparently.
- **dtype safety** — skimage functions are picky about input ranges (`float [0, 1]`, `uint8 [0, 255]`); the wrappers handle conversion in and back.
- **NaN handling** — most skimage filters propagate NaN destructively over spatial footprints; provide masking and restoration strategies.
- **Sugar for the common cases** (`Morphology.opening`, `Filter.gaussian`) without exploding the API surface.

## Non-goals

- Re-implementing skimage in JAX or on GPU. (`cucim` is the GPU-accelerated drop-in if needed — works through the same wrappers.)
- Stateful estimators. skimage is almost entirely stateless functions; the rare exceptions (e.g., trained denoising models) are out of scope.
- 3D volumetric processing. skimage supports it; geotoolz currently doesn’t carry a Z axis.
  Revisit if voxel data becomes a real case.

## Design philosophy

**Thin glue, not reimplement.** Three wrappers (`PerBand`, `MultiBand`, `Func`) plus a handful of conventions cover the entire skimage surface.
Resist the urge to ship `geotoolz.skimage.Gaussian`, `geotoolz.skimage.Sobel`, `geotoolz.skimage.Canny`, etc. — that’s an N-ary wrapping anti-pattern that doesn’t scale to skimage’s hundreds of functions.

**Functions are first-class.** The wrapped object is a Python function, not a class.
`get_config()` records the function by fully-qualified name so YAML round-trip works; the loader resolves it on import.

```yaml
_target_: geotoolz.skimage.PerBand
fn: skimage.filters.gaussian
kwargs:
  sigma: 2.0
  preserve_range: true
```

**Explicit shape contracts.** skimage functions have wildly different shape expectations — 2D-only, 3D with `channel_axis`, ND with `axis`.
The wrappers make the contract explicit at construction; mismatches fail loud rather than producing weirdly-shaped output.

-----

## Carrier types

### `GeoTensor` (no new carriers)

Unlike the sklearn integration (which needs `PixelTable` for `(N, F)` shape), skimage stays in image space.
Inputs and outputs are both `GeoTensor`.
The wrappers handle internal shape juggling (axis transposes, band looping) but never expose a non-`GeoTensor` carrier to users.

Shape conventions for skimage:

- **Per-band** functions expect `(H, W)` 2D arrays.
  Wrapper loops over the band axis.
- **Multi-channel** functions accept `channel_axis=` and operate on either `(H, W, C)` (HWC, the skimage default for ≥0.19) or `(C, H, W)` (CHW) depending on the value.
  Wrapper handles axis convention.
- **ND** functions (`scipy.ndimage`-style) operate on arbitrary dimensions via an `axis=` parameter.
  Wrapper passes through.

geotoolz uses **CHW** internally.
The wrappers translate.

-----

## Wrapping Operators

Three core wrappers.
The user-facing API is small; everything else is composed from these.

### `PerBand` — for 2D-only functions

```python
class PerBand(Operator):
    """Apply a 2D skimage function independently to each band."""
    def __init__(self, fn: Callable, *, dtype_in: str | None = None,
                 dtype_out: str | None = None, **kwargs):
        self.fn = fn
        self.kwargs = kwargs
        self.dtype_in, self.dtype_out = dtype_in, dtype_out

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = _to_dtype(np.asarray(gt), self.dtype_in)
        bands_out = [self.fn(arr[i], **self.kwargs) for i in range(arr.shape[0])]
        out = np.stack(bands_out, axis=0)
        out = _to_dtype(out, self.dtype_out)
        return GeoTensor(values=out, transform=gt.transform, crs=gt.crs)

    def get_config(self):
        return {
            "fn":        f"{self.fn.__module__}.{self.fn.__name__}",
            "dtype_in":  self.dtype_in,
            "dtype_out": self.dtype_out,
            **self.kwargs,
        }
```

Usage:

```python
from skimage import filters, morphology

gz.skimage.PerBand(filters.gaussian, sigma=2.0)
gz.skimage.PerBand(morphology.opening, footprint=morphology.disk(3))
gz.skimage.PerBand(filters.median, footprint=morphology.disk(5))
```

### `MultiBand` — for functions with a `channel_axis`

```python
class MultiBand(Operator):
    """Apply a multi-channel skimage function, handling channel-axis convention."""
    def __init__(self, fn: Callable, *, channel_axis: int = 0,
                 dtype_in: str | None = None, dtype_out: str | None = None,
                 **kwargs):
        self.fn = fn
        self.channel_axis = channel_axis
        self.kwargs = kwargs
        self.dtype_in, self.dtype_out = dtype_in, dtype_out

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = _to_dtype(np.asarray(gt), self.dtype_in)
        out = self.fn(arr, channel_axis=self.channel_axis, **self.kwargs)
        out = _to_dtype(out, self.dtype_out)
        return GeoTensor(values=out, transform=gt.transform, crs=gt.crs)
```

Usage:

```python
from skimage import color, feature

gz.skimage.MultiBand(color.rgb2lab, channel_axis=0)
gz.skimage.MultiBand(feature.multiscale_basic_features, channel_axis=0,
                    intensity=True, edges=True, texture=True)
```

### `Func` — generic escape hatch

For functions that don’t fit the per-band or multi-band patterns — segmentation, registration, anything that operates on the array as a whole and returns something with a different shape or dtype.

```python
class Func(Operator):
    """Generic wrapper for any skimage function."""
    def __init__(self, fn: Callable, *, requires: str = "2d",
                 dtype_in: str | None = None, dtype_out: str | None = None,
                 **kwargs):
        # requires: "2d" | "3d_hwc" | "3d_chw" | "nd"
        self.fn, self.requires, self.kwargs = fn, requires, kwargs
        self.dtype_in, self.dtype_out = dtype_in, dtype_out

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = _prepare_shape(np.asarray(gt), self.requires)
        arr = _to_dtype(arr, self.dtype_in)
        out = self.fn(arr, **self.kwargs)
        out = _to_dtype(out, self.dtype_out)
        out = _restore_shape(out, gt.shape, self.requires)
        return GeoTensor(values=out, transform=gt.transform, crs=gt.crs)
```

Usage:

```python
gz.skimage.Func(skimage.segmentation.slic, requires="3d_hwc",
               n_segments=200, compactness=10)
gz.skimage.Func(skimage.exposure.equalize_adapthist, requires="2d",
               clip_limit=0.03, dtype_in="float01", dtype_out="float32")
```

### Why three wrappers, not one

A single `Func(...)` with smart dispatch could cover everything, but the explicit `PerBand` / `MultiBand` naming signals intent and lets static readers of YAML configs see *how* the function consumes the array.
A `PerBand(filters.gaussian, sigma=2)` is unambiguously per-band; the same expressed as `Func(filters.gaussian, requires="2d", sigma=2, _loop="band")` hides the semantics behind a parameter.

-----

## Channel-axis conventions

skimage 0.19+ standardised on a `channel_axis` parameter.
Earlier versions used `multichannel=True` (deprecated).
The wrappers always pass `channel_axis` explicitly; users supplying older `multichannel`-style calls get a clear error.

geotoolz convention: **CHW** (`channel_axis=0`).
For skimage functions that strongly prefer HWC, `MultiBand` includes a `transpose_to_hwc=True` mode that runs `channel_axis=-1` internally:

```python
gz.skimage.MultiBand(some_hwc_function, transpose_to_hwc=True)
# Internally: CHW → HWC, apply, HWC → CHW
```

`OnHWC` is a sugar Operator for the common case:

```python
class OnHWC(Operator):
    """Transpose CHW → HWC, apply inner Operator, transpose back."""
    def __init__(self, inner: Operator):
        self.inner = inner
    def _apply(self, gt):
        arr = np.asarray(gt).transpose(1, 2, 0)
        gt_hwc = GeoTensor(values=arr, transform=gt.transform, crs=gt.crs)
        out_hwc = self.inner(gt_hwc)
        out = np.asarray(out_hwc).transpose(2, 0, 1)
        return GeoTensor(values=out, transform=gt.transform, crs=gt.crs)
```

-----

## dtype safety

The single most common skimage footgun: passing a `uint16` Sentinel-2 array to a function that expects `float [0, 1]` and getting either silent garbage or an obscure error.
The wrappers solve this via `dtype_in` and `dtype_out` parameters that handle conversion transparently.

### Supported dtype conventions

|Token      |Meaning             |Conversion                         |
|-----------|--------------------|-----------------------------------|
|`"float01"`|float in `[0, 1]`   |`arr / arr.max()` or `img_as_float`|
|`"float32"`|float32 (no rescale)|`arr.astype(np.float32)`           |
|`"float64"`|float64 (no rescale)|`arr.astype(np.float64)`           |
|`"uint8"`  |uint8 in `[0, 255]` |`img_as_ubyte`                     |
|`"uint16"` |uint16 (no rescale) |`arr.astype(np.uint16)`            |
|`None`     |no conversion       |unchanged                          |

Usage:

```python
gz.skimage.PerBand(
    skimage.exposure.equalize_adapthist,
    clip_limit=0.03,
    dtype_in="float01",        # rescale input to [0, 1]
    dtype_out="float32",       # cast output back
)
```

### Why this matters

Three reasons to handle dtype in the wrapper rather than asking users to do it inline:

1. **Round-trip discipline.** `dtype_in="float01"` is a YAML-serialisable field; an inline `(arr / 10_000.0)` call inside a `Lambda` is not.
2. **Composition correctness.** Operators downstream expect `GeoTensor` of the original dtype; the wrapper restores it.
3. **Error visibility.** Wrong dtype produces a `WrappedSkimageError` at the wrapper boundary, not an obscure failure deep inside skimage.

### Rescale strategies

`"float01"` is intentionally vague — does `max` mean per-band or global?
Per-array or per-dataset?
The default is `img_as_float` (skimage’s own convention: integer types divided by their dtype’s max, float types passed through).
Custom scaling via a separate `Rescale` Operator:

```python
gz.core.Rescale(in_range=(0, 10_000), out_range=(0, 1), dtype="float32")
```

Composes naturally:

```python
gz.Sequential([
    gz.core.Rescale(in_range=(0, 10_000), out_range=(0, 1)),
    gz.skimage.PerBand(skimage.exposure.equalize_adapthist, clip_limit=0.03),
    gz.core.Rescale(in_range=(0, 1), out_range=(0, 10_000), dtype="uint16"),
])
```

For workflows where dtype handling is non-trivial (different rescaling per band, dataset-derived rescale ranges from training statistics), keep the rescaling explicit as separate Operators — don’t bury it in `dtype_in`.

-----

## NaN handling

skimage filters propagate NaN destructively over their spatial footprints.
A single NaN pixel in a 5×5 Gaussian footprint corrupts 25 output pixels.
Mishandled, a small cloud mask becomes a much larger artefact in the filtered output.

The integration provides three primary strategies, configured via a `nan_policy` parameter on every wrapper:

|Strategy     |Behaviour                                                  |Use case                                                  |
|-------------|-----------------------------------------------------------|----------------------------------------------------------|
|`propagate`  |Pass NaN through (default for most skimage functions)      |When NaN propagation is fine                              |
|`mask`       |Fill NaN with `fill_value` before, restore NaN after       |Filters where input NaN is invalid                        |
|`interpolate`|Fill NaN with interpolated values before, restore NaN after|When NaN regions are small and noise-free output is needed|
|`raise`      |Error on any NaN                                           |Strict ETL                                                |
|`warn`       |Log NaN presence, then propagate                           |Debug / staging                                           |

### Why `propagate` is the default

skimage’s documented behaviour is NaN-propagating.
Defaulting to anything else creates surprise.
Users who want NaN-aware filtering opt in explicitly:

```python
gz.skimage.PerBand(
    filters.gaussian, sigma=2.0,
    nan_policy="mask", fill_value=0.0,
)
```

### `interpolate` strategy in detail

For smoothing filters (Gaussian, median, denoise) over a scene with small cloud-masked regions, the right behaviour is usually: fill NaN with sensible values *before* filtering, then *restore* NaN at the original locations so masked pixels stay masked but don’t bleed into their neighbours.

```python
class NanPolicy:
    @staticmethod
    def apply_interpolate(arr, fn, **kwargs):
        mask = np.isnan(arr)
        if not mask.any():
            return fn(arr, **kwargs)
        # cheap interpolation: replace NaN with local mean
        filled = arr.copy()
        filled[mask] = _local_mean_fill(arr, mask)
        out = fn(filled, **kwargs)
        out[mask] = np.nan
        return out
```

For high-quality interpolation use `scipy.interpolate.griddata` or `skimage.restoration.inpaint_biharmonic`; for speed use a local mean.
The choice is exposed via `interpolate_method`:

```python
gz.skimage.PerBand(
    filters.gaussian, sigma=2.0,
    nan_policy="interpolate", interpolate_method="biharmonic",
)
```

### Composition: NaN handling as separate Operators

Same pattern as the sklearn design — NaN handling can be lifted out:

```python
gz.Sequential([
    gz.skimage.NanFill(method="biharmonic"),
    gz.skimage.PerBand(filters.gaussian, sigma=2.0, nan_policy="propagate"),
    gz.skimage.NanRestore(),                          # restores NaN at masked positions
])
```

The standalone `NanFill` / `NanRestore` Operators are more flexible — useful when the same fill should serve multiple downstream filters without recomputing.

-----

## Sugar for the common cases

While the *anti-pattern* is shipping `Gaussian`, `Sobel`, `Canny`, etc., **a small set of sugar Operators** for genuinely-common patterns saves users from importing skimage modules and constructing footprints by hand.

### `Morphology` namespace

The most common morphological operations with footprint shorthands:

```python
gz.skimage.Morphology.opening(radius=3, shape="disk")
gz.skimage.Morphology.closing(radius=5, shape="square")
gz.skimage.Morphology.erosion(radius=2, shape="disk")
gz.skimage.Morphology.dilation(radius=2, shape="disk")
gz.skimage.Morphology.tophat(radius=10, shape="disk")
gz.skimage.Morphology.bottomhat(radius=10, shape="disk")
```

Implementation:

```python
class Morphology:
    @staticmethod
    def opening(*, radius: int, shape: str = "disk"):
        return PerBand(
            morphology.opening,
            footprint=_footprint(shape, radius),
        )
    # ... and so on
```

`_footprint("disk", 3)` returns `morphology.disk(3)`, etc.

### `Filter` namespace

```python
gz.skimage.Filter.gaussian(sigma=2.0)
gz.skimage.Filter.median(radius=3)
gz.skimage.Filter.sobel()
gz.skimage.Filter.scharr()
```

### Naming convention

Sugar Operators live under namespaced static classes (`Morphology`, `Filter`, `Color`, `Feature`).
They are *not* their own Operator classes — they are factory functions returning `PerBand` / `MultiBand` / `Func` instances.
This keeps the wrapper count fixed at three and lets sugar evolve without schema migration.

```yaml
# YAML output of Morphology.opening(radius=3, shape="disk") is identical to PerBand
_target_: geotoolz.skimage.PerBand
fn: skimage.morphology.opening
kwargs:
  footprint: [[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]]
```

Or, if footprint serialisation is unwieldy, use a normalised form:

```yaml
_target_: geotoolz.skimage.Morphology.opening
radius: 3
shape: disk
```

The latter is more readable but requires the loader to know about each sugar factory.
The former is mechanical and works for anything.
**Recommendation:** support the readable form for the sugar namespaces; fall through to `_target_: PerBand` form for arbitrary functions.

-----

## Compatibility utilities (reference)

The full surface — roughly 15 operators and utilities.
The three core wrappers carry most of the work; the rest are conventions, sugar, and NaN helpers.

### Core wrappers

- `gz.skimage.PerBand(fn, *, dtype_in, dtype_out, nan_policy, **kwargs)` — 2D-only functions applied per band
- `gz.skimage.MultiBand(fn, *, channel_axis, dtype_in, dtype_out, nan_policy, **kwargs)` — multi-channel functions
- `gz.skimage.Func(fn, *, requires, dtype_in, dtype_out, nan_policy, **kwargs)` — generic escape hatch

### Axis convention helpers

- `gz.skimage.OnHWC(inner_op)` — runs inner op in HWC space
- `gz.skimage.Transpose(order)` — explicit axis reordering Operator

### dtype helpers

- `gz.core.Rescale(in_range, out_range, dtype)` — explicit linear rescaling (lives in `core`, not `skimage`-specific)

### NaN helpers

- `gz.skimage.NanFill(method="biharmonic" | "local_mean" | "constant", fill_value=...)` — fill NaN
- `gz.skimage.NanRestore()` — restore NaN from stashed mask metadata

### Sugar namespaces

- `gz.skimage.Morphology.{opening, closing, erosion, dilation, tophat, bottomhat}`
- `gz.skimage.Filter.{gaussian, median, sobel, scharr, prewitt, laplace}`
- `gz.skimage.Color.{rgb2lab, lab2rgb, rgb2hsv, hsv2rgb, rgb2gray}`
- `gz.skimage.Segmentation.{slic, watershed, felzenszwalb, quickshift}`
- `gz.skimage.Feature.{canny, hog, blob_log, blob_dog, peak_local_max}`

### Sugar conventions

- Footprint shapes: `"disk" | "square" | "diamond" | "octagon"` (mapping to `skimage.morphology` factory functions)
- Default `nan_policy="propagate"` for filters; `"raise"` for segmentation (which produces nonsense on NaN inputs)

-----

## Tradeoffs and out of scope

**CPU-bound.** skimage is numpy/Cython.
The wrappers inherit that.
For GPU acceleration, swap `import skimage` for `import cucim.skimage` — same function names, GPU-backed, drops through the same wrappers without code changes.
Worth documenting prominently.

**NaN-aware filtering is approximate.** “Fill before, restore after” is not mathematically rigorous — a Gaussian convolution that should integrate over NaN pixels with weight 0 isn’t what `fill→filter→mask` produces.
For rigorous treatment, use `astropy.convolution.convolve` (NaN-aware convolution) via a custom `Func` wrapper.
Document that the built-in strategies are pragmatic, not rigorous.

**Per-band looping is slow for hyperspectral.** A `PerBand` Gaussian over a 200-band hyperspectral cube runs 200 Python-level iterations.
For genuinely per-band hyperspectral work, prefer wrapping `scipy.ndimage` functions that vectorise via `axis=` parameters, or push to GPU via `cucim`.
The fallback works but isn’t optimal.

**Boundary modes vary across functions.** skimage’s `filters.gaussian` defaults to `mode="nearest"`; `filters.median` uses `mode="mirror"`; `morphology.opening` uses no padding at all (output shape shrinks).
The wrappers don’t normalise this — users must know each function’s defaults.
Document a “common boundary gotchas” appendix.

**Segmentation output dtype.** `slic`, `watershed`, etc. return integer label maps with no spatial information of their own.
The wrapper restores `transform` and `crs` from the input, but the label numbering is arbitrary and not stable across runs (changes with seed, sample order).
Don’t expect a `slic` segmentation from yesterday to match today’s pixel-for-pixel even on identical input.

**Footprint serialisation.** `morphology.disk(5)` returns an 11×11 numpy array; serialising that into YAML is ugly.
Sugar factories (`Morphology.opening( radius=5, shape="disk")`) sidestep this; raw `PerBand(morphology.opening, footprint=disk(5))` requires the YAML to embed the footprint or reference a factory by name.
Default: sugar for the common cases, explicit footprint arrays for custom shapes.

## Open questions

- **`OnHWC` placement.** Should it live in `geotoolz.skimage` (its main use is skimage), or in `geotoolz.core` (it’s a generic axis adapter)?
  Probably `core`, since it’s useful beyond skimage.
- **dtype `float01` semantics.** `img_as_float` divides by dtype max, which is wrong for Sentinel-2 reflectance (stored as `uint16` with values up to ~10000, not 65535).
  Document that `float01` is “naive rescaling for skimage’s conventions” and that proper reflectance normalisation needs explicit `Rescale`.
- **Whether to ship a `cucim` parallel namespace.** `gz.skimage_gpu.PerBand` wrapping `cucim.skimage.filters.gaussian`?
  Or just document the `import cucim.skimage as skimage` swap pattern?
  The swap pattern is cleaner; ship documentation, not a parallel namespace.