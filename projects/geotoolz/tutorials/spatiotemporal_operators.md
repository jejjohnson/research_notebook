---
title: Spatiotemporal operators
subject: geotoolz tutorial
subtitle: The dependency game on raster `[samples, time, variables, space]` data
short_title: Spatiotemporal operators
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: tutorial, geotoolz, spatiotemporal, operators, raster, xarray, einx, sklearn, skimage
---

> **Status:** Draft, raster-focused.
> **Scope:** A conceptual tutorial — pseudocode only, no executable cells.
> **Audience:** Researchers building operators on geophysical / EO tensors and trying to decide *which axes their operator depends on.*
> **Companion:** [Spatiotemporal data](spatiotemporal_data.md) covers the data side — shapes, coordinates, and geometry.
> Read it first if you want the build-up of the canonical tensor `[S, T, V, X]`, the coord taxonomy (raw vs feature, static vs dynamic), or the geometry primitives (point / line / polygon / multi-polygon / raster).
> This tutorial assumes that material as background and focuses on **raster** operators.
> Graph and mesh operators are deferred to a future tutorial.

The single hardest question when wrapping an operator for spatiotemporal data is not "what does it compute" but "what does it depend on."
The same algorithm — say, a linear regression — becomes a *climatology*, a *bias correction*, a *detrending*, or a *kriging covariance* depending entirely on which axes you fit over and which axes you keep free.
This tutorial walks the design space.

We use a single running data shape — `[samples, time, variables, space]` — and ask the same question of every operator: *which axes does it reduce, and which axes does its parameter tensor span?*
Answer that, and the rest follows: how to reshape, how to wrap it for sklearn / skimage / a graph operator, and whether it preserves physical locality.

The framing is at heart *physical*, not algorithmic.
Every axis of the data tensor carries a different scale (ensemble realizations, time, state variables, space), and each geophysical phenomenon couples only a subset of those scales over a characteristic range.
ENSO couples `X` and `T` globally across the equatorial Pacific; a methane plume couples them locally over a few kilometres and a few hours; a per-pixel NDVI trend couples `T` over decades and `X` not at all.
Picking an operator's scope is therefore picking a hypothesis about which scales the physics is stationary in — and getting it wrong does not just slow training, it produces a parameter tensor that cannot represent the phenomenon at all.

---

## 1. The canonical tensor `[S, T, V, X]`

Almost every geophysical / EO array can be flattened into four conceptual axes.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│         S   ── samples       (scenes, events, batch)        │
│         T   ── time          (timesteps within a sample)    │
│         V   ── variables     (channels: T, p, q, R, G, NIR) │
│         X   ── space         (pixels, points, mesh nodes)   │
│                                                             │
│              ┌──────────────────────────┐                   │
│              │       arr[S, T, V, X]    │                   │
│              └──────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

`X` is "space" in the abstract sense — it can be:

- a single flat axis of points (`N`),
- a structured spatial grid (`H, W` or `D, H, W`),
- an unstructured mesh (nodes + an adjacency / KDTree off-array).

For most of the tutorial we write `X` as a single axis and note where structure matters.

We type our arrays with [jaxtyping](https://github.com/google/jaxtyping) and name dimensions explicitly with [xarray](https://docs.xarray.dev/) so that every reshape is auditable.

```python
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Bool, Array
import xarray as xr
import einx

Field = Float[Array, "S T V X"]
```

The named-dims version, for the same physical quantity:

```python
da: xr.DataArray  # dims = ("sample", "time", "variable", "space")
```

`einx.rearrange` patterns are written against these names, not positional indices, so the reshape *reads* as the intent.

For the build-up of how real datasets reach this canonical shape — univariate time series, the three constructions of `S` (ensembling, windowing, patching), the coord pack that travels alongside, and the geometric primitives that underlie it all — see the data-side companion [Spatiotemporal data](spatiotemporal_data.md).
This tutorial assumes the input is a `Raster`-topology tensor with a static `(lat, lon)` coord pack; the operator-side dependency game then has its cleanest shape.

---

### 1.1 What an "operator" depends on

An operator has two scopes that are independent and worth naming.

**Fit scope.**
Which axes does the operator *reduce over* when estimating its parameters?
A linear regression's slope is a number; what makes it "global" or "local" is whether that slope was computed by reducing over all pixels (global), or just one pixel's time series (local), or just one window (windowed-local).

**Predict scope.**
Which axes does the *output* span?
A clustering model can be fit globally but predict per-pixel labels.
EOF can be fit globally and predict either per-sample mode amplitudes or per-pixel modes — different scopes, same fit.

The shape of the parameter tensor tells you both at once.
A parameter tensor of shape `(K, V)` means *fit reduced `T`, `S`, `X`; predict ranges over whatever survives.*
A parameter tensor of shape `(2, V, X)` (slope + intercept per pixel per variable) means *fit reduced `S, T`; predict is per-pixel.*

This framing is what the rest of the tutorial uses.
*The operator's parameter tensor shape, not its algorithm, dictates how to wrap it.*

---

### 1.2 Why local vs global is a physics question

Geophysical processes carry *characteristic length and time scales* — `L_c` and `T_c` — that set the range over which they are coherent.
These scales are not preferences; they are imposed by the underlying physics: the Rossby radius of deformation, gravity-wave dispersion, photochemical lifetimes, plant phenology, advective transport speeds, soil-hydraulic conductivity, and so on.
A satellite scene is therefore not a featureless tensor.
It is a superposition of phenomena, each one stationary over its own range of scales and nonstationary outside it.

This matters because an operator's parameter tensor *is itself an assumption about stationarity*.
A parameter tensor of shape `(K, V)` says: *the relationship I'm encoding does not vary with `S`, `T`, or `X`*.
A parameter tensor of shape `(2, V, X)` says: *the relationship varies with `X`, but is stationary in `S` and `T`*.
If the underlying physics is *not* stationary in the axes your operator assumed away, the operator cannot represent it — no amount of training data fixes the structural mistake.

A few characteristic geophysical scales worth keeping in mind:

| Phenomenon                                | `L_c` (space)         | `T_c` (time)        | Natural fit scope                |
|-------------------------------------------|-----------------------|---------------------|----------------------------------|
| ENSO / NAO / PDO modes                    | 1,000–10,000 km       | months–decades      | global in `X`, global in `T`     |
| Synoptic weather systems (cyclones)       | ~1,000 km             | days                | global in `X`, local in `T`      |
| Tropical cyclones                         | 100–1,000 km          | days                | local in `X`, local in `T`       |
| Mesoscale ocean eddies                    | 10–100 km             | weeks               | local                            |
| Mesoscale convective systems              | ~100 km               | hours               | local in `X`, local in `T`       |
| Methane / CO₂ / NO₂ plumes                | 0.1–10 km             | hours               | local                            |
| Sea-surface salinity (E−P pattern)        | 1,000s of km          | seasons             | global background + local eddies |
| Vegetation phenology (biome scale)        | 10–1,000 km           | seasons             | local-by-biome                   |
| Agricultural fields, plant canopies       | 10–100 m              | weeks               | local                            |
| Soil-moisture variability                 | 10 m – 10 km          | days                | local                            |
| Topographic illumination correction       | 30–100 m              | hours               | local                            |
| Land-surface diurnal temperature cycle    | 100 m – 10 km         | 24 h                | local                            |
| Barotropic tides                          | 1,000s of km          | hours               | global in `X`, sinusoidal in `T` |
| Global mean sea-level rise                | planet-scale          | decades             | global                           |
| Ice-sheet mass-balance trends             | 100 km – continental  | years–decades       | local-pooled to global           |

Three rules of thumb fall out of this table, and they govern every later section of the tutorial:

1. **An operator that aims at a phenomenon must be global along axes where the phenomenon has scales comparable to the data extent.**
   ENSO cannot be captured by a local fit — there is no smaller subdomain that contains it.
2. **An operator must be local along axes where the phenomenon's scale is small compared to the data extent.**
   A methane plume detector that pools the whole scene to estimate its background has just averaged the plume into the surroundings.
3. **When two phenomena overlap (background + perturbation, climate + weather, biome + pixel), the operator must be a *composition*: global in the scales of the background, local in the scales of the perturbation.**
   This is why most useful EO pipelines are not single operators but `Sequential` of operators with mismatched scopes — the *composition* expresses the scale separation.

This last point is also why pure machine-learning "scaling laws" can mislead in geoscience.
More data does not help if the *scope* is wrong, because the parameter tensor is in the wrong shape to represent the phenomenon.
Picking `(fit_scope, predict_scope)` is therefore a physical modelling choice, on the same footing as picking a closure scheme in a turbulence model — the parameter tensor's axes encode an assumption about which scales the physics is stationary in.

---


## 2. The reshape game

Every ecosystem has a preferred shape and a fixed assumption about what each axis means.
You can almost always get into that shape with one `einx.rearrange`, but the *cost* of the reshape — what locality you lose — is the part that matters.

### 2.1 sklearn — `(N, F)`

sklearn's iron rule: rows are iid samples, columns are features.
There is no notion of geometry; once you flatten in, sklearn assumes everything is exchangeable.

```python
def to_sklearn(arr: Float[Array, "S T V X"]) -> Float[Array, "S F"]:
    return einx.rearrange("s t v x -> s (t v x)", arr)
```

```
[S, T, V, X]                         [S, F=T·V·X]
┌──────────┐                         ┌──────────────┐
│ ▓ ▓ ▓ ▓  │   einx.rearrange        │ ▓▓▓▓▓▓▓▓▓▓   │
│ ▓ ▓ ▓ ▓  │   "s t v x -> s (t v x)"│ ▓▓▓▓▓▓▓▓▓▓   │
│ ▓ ▓ ▓ ▓  │ ──────────────────────► │ ▓▓▓▓▓▓▓▓▓▓   │
└──────────┘                         └──────────────┘
```

The cost: every spatial / temporal neighborhood is gone, and sklearn cannot recover it.
This is fine for *exchangeable* operators (PCA, scaling, isotonic calibration) and a footgun for anything that depends on adjacency.

**Geophysical reading.**
"Sample" here means an independent realization of the geophysical process — a different scene over a different region, or a different ensemble member of a Monte Carlo simulation.
Two adjacent Sentinel-2 tiles are *not* iid samples (they share atmosphere, illumination, and surface), and sklearn cannot tell.
Treating them as iid is the silent assumption behind almost every "machine-learning baseline" failure on EO benchmarks: the reported test score is overoptimistic because the operator's iid assumption was violated by the very spatial autocorrelation the operator threw away in the reshape.

Alternate sklearn flavor — "each pixel is a sample, each timestep is a feature" — for per-pixel temporal classifiers:

```python
def per_pixel_timeseries(arr: Float[Array, "S T V X"]) -> Float[Array, "N F"]:
    return einx.rearrange("s t v x -> (s x) (t v)", arr)
```

This one preserves *time* as a feature axis but discards both `S`-vs-`X` distinction and spatial geometry.
It is the standard pre-processing for **per-pixel land-cover classification from time-series stacks** (the TIMESAT / BFAST style of analysis), where each pixel's 30-year Landsat record becomes a 30-dimensional feature vector and the operator never knows two adjacent pixels exist.

### 2.2 skimage — `(H, W)` or `(H, W, C)`

skimage's iron rule: a single 2-D (or 2-D + channel) image.
No time, no batch — those are *your* job to vectorise over.

```python
def to_skimage(arr: Float[Array, "S T V H W"]) -> Float[Array, "B V H W"]:
    return einx.rearrange("s t v h w -> (s t) v h w", arr)
```

```
[S, T, V, H, W]                      [B=S·T, V, H, W]
                                     ┌──────┐
   per-sample, per-time              │  H W │  per V (channel)
   image stack                       │      │
                                     └──────┘
                                     repeated B times
```

The cost: time is hidden inside the batch axis and the operator cannot use it.
This is appropriate when *each frame is processed independently* — denoising, morphology, edge detection.
It is *wrong* for ops that need temporal context (optical flow, change detection); for those, see §2.4.

**Geophysical reading.**
The skimage shape is the natural one for **scene-level radiometric and geometric corrections** — cloud masking from a single Sentinel-2 scene, gradient-based edge detection on a single SAR backscatter image, water-body extraction from a single Landsat tile.
These operations exploit local pixel adjacency (the `(H, W)` axes) and treat the scene as a self-contained image; bringing time in would only add noise unless you have a specific multi-temporal algorithm in mind.

### 2.3 Channel-first / ConvLSTM — `(B, V, T, H, W)`

PyTorch convention; time and channel both kept, geometry preserved.

```python
def to_convlstm(arr: Float[Array, "S T V H W"]) -> Float[Array, "S V T H W"]:
    return einx.rearrange("s t v h w -> s v t h w", arr)
```

No reduction at all — just a transpose.
This is the "lose nothing, decide later" reshape, and the default for any operator that genuinely uses all four named axes.

**Geophysical reading.**
This is the right shape for *anything that couples space and time at comparable scales* — convective-precipitation nowcasting, sea-ice-drift prediction, mesoscale-eddy tracking in altimetry, plume-evolution forecasting from hyperspectral cubes.
The operator's parameter tensor will typically span `V` (channels), have *learnable* spatial-temporal kernels of fixed extent (the receptive field is the operator's `L_c`), and be applied as a sliding window over `(T, H, W)`.
If your operator's receptive field is smaller than the phenomenon's `L_c`, you are under-resolving; larger, and you are bleeding across phenomena of different physical origin.

### 2.4 Spatial EOF / PCA — `(N, F)` with geometry packed into features

A classic in climate: each *timestep* is a sample; the spatial field is the feature vector.

```python
def to_eof(arr: Float[Array, "S T V H W"]) -> Float[Array, "N F"]:
    return einx.rearrange("s t v h w -> (s t) (v h w)", arr)
```

Here `S·T` becomes the "iid samples" axis (which it isn't, strictly — but EOF tolerates it) and `V·H·W` becomes a feature vector whose covariance structure is precisely what we want PCA to find.

**Geophysical reading.**
This is the reshape behind every standard climate-mode analysis: EOFs of monthly SST anomalies (ENSO, PDO, AMO), EOFs of geopotential-height anomalies (NAO, AO, PNA), EOFs of soil-moisture anomalies for drought monitoring.
The reshape *encodes* the physical hypothesis that the variability lives in a low-rank spatial pattern that recurs across many timesteps.
If the variability is instead localised in space and stationary in time (e.g., a persistent plume), EOF will waste its leading modes representing the constant background and discover nothing.

### 2.5 Cheat sheet

| Target ecosystem            | Pattern                                       | What's lost                  |
|-----------------------------|-----------------------------------------------|------------------------------|
| sklearn (whole-sample)      | `"s t v x -> s (t v x)"`                      | all geometry, all time order |
| sklearn (per-pixel time-series) | `"s t v x -> (s x) (t v)"`                | space ↔ sample distinction   |
| skimage (per-frame)         | `"s t v h w -> (s t) v h w"`                  | time as a context axis       |
| ConvLSTM                    | `"s t v h w -> s v t h w"`                    | nothing                      |
| EOF / spatial PCA           | `"s t v h w -> (s t) (v h w)"`                | sample/time iid-ness         |
| Per-pixel regression        | `"s t v h w -> (s h w) t v"` then per-row fit | spatial structure of fit     |

The pattern is the operator's *contract* with its ecosystem.
If two of your operators have different contracts, they cannot compose without a re-reshape between them.
This is what motivates `geotoolz.Operator` having a single shape protocol.

---

(operators-dependency-game)=
## 3. The dependency game — global vs local

We now answer the central question: *which axes does my operator's parameter tensor span?*

Frame every operator as a pair `(fit_scope, predict_scope)`.
Each can be *global* (reduce that axis during fit / predict) or *local* (keep that axis varying).
That gives a 2×2.

|                       | **predict global** (one summary)            | **predict local** (one output per cell)  |
|-----------------------|---------------------------------------------|------------------------------------------|
| **fit global**        | §3.1 EOF / climatology                      | §3.2 KMeans land cover, quantile mapping |
| **fit local**         | §3.4 Variogram pooling, global Moran's I    | §3.3 Per-pixel detrending, harmonic fit  |

The diagonal is common.
The off-diagonals are where most of the design judgment lives.

---

### 3.1 Fit global, predict global — *climatology and EOF*

**Real-world example: EOF analysis of SST anomalies.**
The leading empirical orthogonal function of monthly sea-surface-temperature anomalies *is* the El Niño Southern Oscillation pattern.
Fit reduces over all samples and time to find spatial modes; predict projects a new field onto those modes and returns a low-dimensional mode-amplitude vector.

Parameter tensor: `(K, V, H, W)` — `K` spatial modes shared by every sample.
Output of `predict`: `(S, T, K)` — one amplitude per mode per sample-time.
Nothing varies per pixel after the fit; the operator is *global in space and time*.

```python
def fit(arr: Float[Array, "S T V H W"], k: int) -> Float[Array, "K V H W"]:
    anomaly = arr - arr.mean(axis=(0, 1), keepdims=True)         # remove mean over S, T
    flat = einx.rearrange("s t v h w -> (s t) (v h w)", anomaly)
    _, _, vt = jnp.linalg.svd(flat, full_matrices=False)
    return einx.rearrange(
        "k (v h w) -> k v h w", vt[:k], v=arr.shape[2], h=arr.shape[3], w=arr.shape[4]
    )

def predict(
    arr: Float[Array, "S T V H W"], modes: Float[Array, "K V H W"],
) -> Float[Array, "S T K"]:
    return einx.dot("s t v h w, k v h w -> s t k", arr, modes)
```

**Other instances:**

- **Long-term climatology mean** over a region — the 30-year monthly mean of MERRA-2 temperature at each grid point that every anomaly product subtracts.
- **Whole-field z-score normalisation** (`mean`, `std` are global scalars per variable) — the standard pre-processing for any climate-mode index.
- **Regression of a global index on another global index** — NAO on Iberian winter rainfall, ENSO3.4 on East-African short rains, AMO on Sahelian decadal drought.
- **Spherical-harmonic decomposition** of geopotential or gravity (GRACE), where the parameter tensor is the harmonic coefficients shared across all timesteps.
- **Global tide-model fits** (TPXO, FES) — a single set of harmonic constants fit globally, applied to predict tidal height at any `(lat, lon, t)`.

**Physical reasoning.**
This cell is the natural home for any phenomenon whose `L_c` and `T_c` are *comparable to the data extent*.
ENSO has spatial extent of roughly half the Pacific basin and a recurrence time of 2–7 years; there is no smaller subdomain that contains it, and there is no shorter temporal window that resolves it.
A regional fit would never see the mode — it would estimate its leading EOF as a local SST gradient and miss the planetary teleconnection entirely.
The same logic applies to NAO, the seasonal cycle of total-column water vapour, the global mean energy budget, or anything whose spatial scale is the domain itself.

The *parameter tensor* (`K` shared spatial modes) is in some sense the *physics* of these phenomena: the leading modes of climate variability are the slow manifold of the coupled atmosphere-ocean system, and the fact that they are recoverable as a low-rank decomposition reflects an underlying separation of timescales.
When you reach for this cell, you are asserting that the phenomenon is *spatially stationary in pattern, temporally stationary in statistics*, and that a single parameter tensor is enough to encode it.

```
fit:          [S, T, V, H, W] ──reduce S,T──► params[K, V, H, W]
predict:      [S, T, V, H, W] ──project────► out[S, T, K]
                                  ↑
                          params are shared
                          across every (s, t) and every pixel
```

---

### 3.2 Fit global, predict local — *KMeans land cover, quantile mapping*

**Real-world example: KMeans land-cover clustering from a multi-temporal Sentinel-2 cube.**
Fit pools every pixel from every scene into one giant bag of `V`-dimensional points and learns `K` cluster centers.
Predict assigns each pixel of a new scene to its nearest center, producing a label *per pixel*.

Parameter tensor: `(K, V)` — `K` centers in variable-space.
Output of `predict`: `(S, T, H, W)` — one label per pixel.
The fit is global; the predict is local-in-space because each pixel is scored independently against the *same* centers.

```python
def fit(arr: Float[Array, "S T V H W"], k: int) -> Float[Array, "K V"]:
    pixels = einx.rearrange("s t v h w -> (s t h w) v", arr)
    return kmeans(pixels, k=k).centers  # (K, V)

def predict(
    arr: Float[Array, "S T V H W"], centers: Float[Array, "K V"],
) -> Int[Array, "S T H W"]:
    d2 = einx.reduce(
        "s t v h w, k v -> s t k h w", arr, centers,
        op=lambda a, c: (a - c) ** 2, reduce="sum", axis="v",
    )
    return einx.argmin("s t [k] h w", d2)
```

**Other instances:**

- **Quantile-mapping bias correction** for climate model output: empirical CDFs fit on all historical (model, obs) pairs, applied per pixel × per day to correct CMIP projections before regional impact analysis.
- **Random-forest pixel classifier** trained once on the full archive, then scored per-pixel on new scenes — the workhorse of the ESA WorldCover and Dynamic World land-cover products.
- **Pretrained foundation-model embeddings** (Prithvi, SatMAE, Clay, AnySat) — weights frozen and global, output is a per-patch latent that downstream pipelines consume.
- **Global radiance-to-reflectance regression** trained on the full collocation set, applied at each pixel — the structure of every atmospheric-correction algorithm (6S, Py6S, LUT-based ACOLITE).
- **Look-up-table retrievals**: trace-gas retrievals (CH₄, NO₂, CO₂) where the LUT is built once from radiative-transfer simulations across the global parameter space, then queried per-pixel against the observed radiance.
- **SMAP / SMOS soil-moisture retrieval**: a single forward model (τ-ω, single-channel algorithm) fit to validation data globally, applied to each L-band brightness-temperature pixel.

**Physical reasoning.**
This cell is the right home for any phenomenon where the *physical law connecting the variables is universal*, but its *spatial expression is per-pixel*.
A forest is a forest in the Amazon, in Borneo, and in the Congo, and its reflectance in `V`-space (R, G, NIR, SWIR) is approximately the same — the cluster centers are a property of *land cover as a category*, not of any particular scene.
But the *spatial arrangement* of categories is per-scene, set by historical land use, soil, topography, and management.
Pairing a global parameter tensor with a local predict is exactly the right algebra: the parameter tensor encodes the universal physics, the per-pixel application encodes the local realisation.

The same structure underwrites *every retrieval* in remote sensing.
Atmospheric correction, methane plume detection, snow-cover mapping, sea-surface salinity from SMOS — they all rest on a forward model that is approximately stationary in `(X, T)` and a per-pixel inversion.
When the assumption breaks (an aerosol regime not seen in training, a novel cloud type for the cloud mask, a previously unobserved cover type), the global parameter tensor cannot adapt and the retrieval silently fails at those pixels — the typical failure mode is bias, not noise, because the operator's prior is wrong rather than uncertain.

```
fit:          [S, T, V, H, W] ──reduce S,T,H,W──► params[K, V]
predict:      [S, T, V, H, W] ──per-pixel score─► out[S, T, H, W]
                                  ↑
                          one set of params,
                          scored independently at every pixel
```

---

### 3.3 Fit local, predict local — *per-pixel detrending*

The most common cell in EO time-series analysis.
Each pixel gets its own parameter set, derived from its own history.

**Real-world example: per-pixel temporal detrending of NDVI.**
For every pixel, fit a slope-and-intercept across time; subtract the linear trend.
You end up with a slope map (often the deliverable on its own — "where is greening happening?") and a residual cube (the deliverable as an operator).

Parameter tensor: `(2, V, H, W)` — slope + intercept per pixel per variable.
Output of `predict`: `(S, T, V, H, W)` — detrended cube.
Both fit and predict are local in space; the operator is *embarrassingly parallel over `(H, W)`*.

```python
def fit(arr: Float[Array, "S T V H W"]) -> Float[Array, "2 V H W"]:
    t = jnp.arange(arr.shape[1], dtype=arr.dtype)
    X = jnp.stack([jnp.ones_like(t), t], axis=1)            # (T, 2)
    pinv = jnp.linalg.pinv(X)                               # (2, T)

    y = einx.rearrange("s t v h w -> t (s v h w)", arr)
    coef = pinv @ y                                          # (2, S·V·H·W)
    coef = einx.rearrange(
        "two (s v h w) -> two s v h w", coef,
        s=arr.shape[0], v=arr.shape[2], h=arr.shape[3], w=arr.shape[4],
    )
    return coef.mean(axis=1)                                 # average over S → (2, V, H, W)

def predict(
    arr: Float[Array, "S T V H W"], coef: Float[Array, "2 V H W"],
) -> Float[Array, "S T V H W"]:
    t = jnp.arange(arr.shape[1], dtype=arr.dtype)
    trend = einx.dot("t, v h w -> t v h w", t, coef[1]) + coef[0][None]  # (T, V, H, W)
    return arr - trend[None]
```

**Other instances:**

- **Per-pixel STL decomposition** of sea-ice extent, NDVI, or surface temperature, exposing trend, seasonality, and remainder per cell.
- **Per-pixel harmonic / Fourier fit** — annual + semi-annual cycle, leaves a residual that highlights anomalies (the basis of CCDC continuous change detection).
- **Local linear regression / LOESS** for surface fitting — DEM smoothing, sea-surface-height detrending against geoid.
- **KNN regression** where every query point has its own neighbor set — gap-filling missing pixels in MODIS LST stacks.
- **Per-station rainfall-IDF fitting** in hydrology — every gauge gets its own extreme-value distribution because rainfall extremes depend on local orography and proximity to moisture sources.
- **Per-pixel diurnal cycle fitting** in geostationary LST (MSG, ABI) — every pixel's daytime peak amplitude depends on its surface thermal inertia, which varies on the field scale.
- **Per-pixel snow phenology** (start-of-season, melt-out date) from MODIS or VIIRS — strongly modulated by aspect and elevation, varies on hectometre scales.
- **Per-station / per-buoy bias calibration** for in-situ networks (Argo, GTS) — each instrument has its own systematic offsets.

**Physical reasoning.**
This cell is forced on you whenever the *parameters themselves vary on a scale comparable to the pixel grid*.
NDVI's seasonal cycle phase and amplitude depend on biome, latitude, hydroclimate, and management; the cycle of a tropical-evergreen pixel is near-flat, while a boreal-deciduous pixel has a sharp summer peak with an envelope set by snow-cover dates.
A *global* trend fit would mix these regimes and produce a "mean" that mostly tracks biome composition rather than greening — the right scope for the question "is this pixel getting greener" is per-pixel because the *baseline* is per-pixel.

A second physical motivation is *measurement-instrument heterogeneity*.
Two adjacent gauges in an IDF network may sit in radically different precipitation regimes (windward vs leeward of a ridge), and pooling them would bias the extreme-value parameters toward the milder regime.
Per-station fitting is not a methodological convenience — it is the only way to respect the actual spatial nonstationarity of the rainfall process.

The risk on the other side is *sample starvation*: per-pixel fits with too few timesteps are noisy.
Real systems often add a regularisation term that couples nearby pixels (a spatial prior, a Markov-random-field smoother, a hierarchical pooling of slopes toward a regional mean) — at which point the operator slides from "purely local" toward "windowed-local" or even "hierarchical local + global pooling" (§3.4).
The boundary between §3.3 and §3.4 is gradual, not sharp, and the right choice depends on the ratio of `T` (samples per pixel) to the parameter count.

```
fit:          [S, T, V, H, W] ──reduce S,T,─────► params[2, V, H, W]
                                  (per pixel)
predict:      [S, T, V, H, W] ─per-pixel subtract► out[S, T, V, H, W]
                                  ↑
                          parameter tensor varies per pixel —
                          same shape grain as the input
```

---

(operators-local-global)=
### 3.4 Fit local, predict global — *variogram pooling, global Moran's I*

The rare cell, and the one that most operator wrappers get wrong (they often refuse to express it).
The fits are local — each one uses a small window or neighborhood — but the *deliverable* is a single global object.

**Real-world example: empirical variogram pooling for kriging.**
For each spatial window, compute the empirical variogram (variance as a function of lag).
Average the per-window variograms into one pooled variogram and fit a single covariance model (`nugget`, `sill`, `range`).
That single model is then used everywhere for kriging interpolation.

Parameter tensor: `(3,)` — `(nugget, sill, range)`.
The *intermediate* parameter tensor (per-window variograms) has shape `(N_windows, L)`, but it is *pooled* before becoming the operator's parameters.

```python
def fit(
    arr: Float[Array, "S T V H W"], patch: int, n_lags: int,
) -> Float[Array, "3"]:
    patches = einx.rearrange(
        "s t v (nh ph) (nw pw) -> (s t v nh nw) ph pw",
        arr, ph=patch, pw=patch,
    )
    local = jax.vmap(empirical_variogram, in_axes=(0, None))(patches, n_lags)   # (N, L)
    pooled = local.mean(axis=0)                                                  # (L,)
    return fit_spherical(pooled)                                                 # (3,)
```

`predict` here is *kriging* using that single covariance — the operator's "global" output is the covariance object itself; downstream interpolation is a separate operator (and lives in §3.2 — fit global, predict local).

**Other instances of fit-local / predict-global:**

- **Global Moran's I** from local-Moran statistics — measure per-pixel spatial autocorrelation, aggregate to a single global index + significance test for whether spatial clustering is present at all.
- **"Fraction of land showing significant greening"** — per-pixel Mann-Kendall trend tests, aggregated to one scalar deliverable for an IPCC-style report or national greenhouse-gas inventory.
- **Hierarchical / multilevel temperature-trend models** — fit per-station slopes, pool to a single population-level slope estimate via partial pooling (the Berkeley Earth, HadCRUT, GISTEMP families all do a version of this).
- **Stacked regional emulators** — train per-region surrogates of a numerical model, fit a single meta-weighting via out-of-sample errors to make a global ensemble emulator.
- **Spatial-scan hotspot detection** for epidemiology / pollution — per-window likelihood-ratio statistics, aggregated to a global "is there clustering" p-value (Kulldorff scan).
- **National / basin-scale methane budgets** — per-plume emission estimates from TROPOMI or EMIT scenes, aggregated to one annual flux per country or per oil-and-gas basin (Permian, Sahara, North Sea).
- **Ice-sheet mass balance** — per-tile altimetry trends (ICESat-2, CryoSat-2) aggregated to a single dM/dt for Greenland or Antarctica, which then drives a single sea-level-rise contribution.
- **GRACE basin water-storage anomalies** — per-mascon trends pooled into a basin-aggregated groundwater-depletion estimate.
- **Permafrost active-layer-thickness sensitivity** — per-borehole thaw-depth trends, pooled to one circumpolar carbon-release rate.

```
fit:          [S, T, V, H, W] ──windowed reduce──► local_params[N_windows, L]
                                                   │
                                          pool / aggregate
                                                   ▼
                                              params[L_global]
predict:      anything ──single global object───► out[…]
                                  ↑
                          parameter tensor is small;
                          the fit work was local but the deliverable is one thing
```

**Physical reasoning.**
This cell exists because *the world is spatially nonstationary but the deliverable is one number*.
Soil and topography differ across a continent — the variogram of a rugged catchment is not the variogram of a flat agricultural region — but a kriging interpolator needs a *single* covariance function, and a national emissions inventory needs *one* annual flux.
A purely global fit would be biased toward the dominant subregion (the biggest basin, the largest land class), under-representing everything else.
A purely local fit gives you a *map* of locally-fitted parameters, but it does not collapse that map to the scalar your stakeholder asked for.

Pooling local fits into a global one is a *robust-statistics trick* with a clean physical interpretation: each window contributes information about short-range variability, the pooling averages out region-specific quirks, and the deliverable inherits the robustness of the mean.
This is how the climate-science community produces IPCC headline numbers ("warming since 1850 of 1.1 K"), how the methane community converts per-plume detections into national flux estimates, and how mass-balance studies turn per-pixel altimetry into a contribution to sea-level rise.

The conceptual difference from §3.1 ("fit global, predict global") is *what stationarity you assume*.
§3.1 assumes the parameter tensor itself is the right shape for the phenomenon and that pooling can happen at the data level.
§3.4 assumes the data is too heterogeneous to pool directly, but the *summary statistic* of locally-fit parameters is still meaningful at the global scale.
The first is the right call when the physics has a low-rank global pattern (ENSO); the second is the right call when the physics is heterogeneous but the bookkeeping target is global (the national methane budget).

---

### 3.5 The axis-triage decision

You can derive the right cell mechanically:

1. **List the axes** of your input tensor (here `S, T, V, X`).
2. **Mark each axis** with one of:
   - **F** — reduced during fit (operator is global in this axis),
   - **K** — kept during fit (operator's parameters span this axis),
   - **P** — reduced during predict (operator collapses this axis at inference),
   - **B** — broadcast / passed through during predict.
3. The marking of `(F vs K)` tells you the fit cell; `(P vs B)` tells you the predict cell.

Worked examples from above:

| Operator                          | S    | T    | V    | X    | Cell                          |
|-----------------------------------|------|------|------|------|-------------------------------|
| Spatial EOF                       | F, P | F, P | K, B | K, P | global / global (modes)       |
| KMeans land cover                 | F, B | F, B | K, P | F, B | global / local                |
| Per-pixel detrend                 | F, B | F, B | K, B | K, B | local-in-X / local-in-X       |
| Variogram pooling                 | F, — | F, — | F, — | K-windowed → F | local / global      |

If two operators in a pipeline don't agree on what's `K` and what's `F`, they don't compose without a `Reduce` or `Broadcast` operator between them.
This is how `geotoolz.Sequential` and the upcoming `Reduce` / `Broadcast` markers fall out of the design naturally.

---

## 4. Appendix — beyond rasters: locality is geometric, not logistical (KNN vs radius)

> **Status: preview of future non-raster tutorial.**
> The rest of this tutorial assumes raster input, where neighborhoods are fixed stencils and the question of "what is a neighbor?" is decided by the grid.
> This appendix previews how the same dependency-game framework extends to non-raster geometries (point sets, meshes, graphs), where the operator must *define* its neighborhood explicitly.
> The full treatment of graph and mesh operators belongs to a later tutorial; this section is here because the design implication for `geotoolz`'s `Operator` signature is already worth flagging.
> The corresponding design proposal — a four-axis `Patcher` framework over the `Field` / `Domain` substrate, with `KNNGraph` and `RadiusGraph` as first-class `PatchGeometry` choices — lives in the [`geopatcher` plan](../plans/geopatcher/README.md).

The hardest part of the dependency game on non-raster data is that *"local"* is not a single concept.
Two operators can both claim to be local in `X` and yet have entirely different physical receptive fields, because they disagree on *how to define a neighborhood*.
The two canonical definitions are **K-nearest-neighbors** (count-bounded) and **radius neighbors** (distance-bounded), and choosing between them is a *physical* statement about the phenomenon you are modelling — not a logistical convenience.

On a raster, both strategies collapse to the same fixed-stencil convolution — the regular tessellation makes the neighborhood unambiguous.
The choice only becomes interesting once you leave the raster, which is what this appendix is about.

This section walks through the distinction, shows why it matters under resampling, and lays out the pseudocode of an operator parametrised by a `Neighborhood` strategy.

---

### 4.1 The two strategies

**K-nearest-neighbors (`KNN`).**
For each query point, find the `k` reference points whose distances are smallest.
Receptive field has *variable spatial extent* — small in dense regions, large in sparse ones.
Number of neighbors is fixed.

**Radius neighbors (`Radius`).**
For each query point, find all reference points within distance `r`.
Receptive field has *fixed spatial extent* — exactly `r` everywhere.
Number of neighbors is variable (zero in sparse regions, many in dense ones).

```
   point cloud (uniform)               point cloud (clustered)

                                                  •••
   •   •   •   •   •                            ••• •••
                                                  •••
   •   •  ★   •   •                              •★•   •     •
                                                  •••
   •   •   •   •   •                            ••• •••
                                                  •••

   KNN(k=4):                            KNN(k=4):
   ┌───┐                                  ┌─┐
   │ ★ │  receptive field                 │★│  receptive field
   └───┘  matches grid                    └─┘  collapses in cluster

   Radius(r=R):                         Radius(r=R):
   ┌───┐                                ┌───┐
   │ ★ │  same R                        │ ★ │  same R
   └───┘                                └───┘
```

Both strategies satisfy "the operator uses local information."
But they answer different questions:

- **KNN** asks *"who are my `k` most similar (closest) neighbors?"* — density-adaptive.
- **Radius** asks *"who is physically within distance `r` of me?"* — geometry-faithful.

---

### 4.2 The mesh-invariance test

The cleanest way to see which strategy preserves physical locality is to *resample the same field at two densities* and ask whether the operator's effective receptive field stays the same.

Consider a methane plume sampled from a hyperspectral cube.
The plume has a physical extent `L_c ≈ 1 km`.
You build a smoothing operator that averages over each pixel's neighborhood.

- At native 30 m resolution, `KNN(k=8)` covers ≈ 240 m — smaller than the plume — and smooths within the plume.
- At resampled 5 m resolution (e.g., after super-resolution), `KNN(k=8)` covers ≈ 40 m — now the operator smooths over a region inside the plume that is structurally meaningful, *but the receptive field has shrunk*.

The plume hasn't changed.
The physics hasn't changed.
Your operator's effective receptive field changed *because the data density changed*.
This is the silent failure mode of KNN-based operators in EO pipelines: they appear to work on one resolution, then fail validation on another, with no warning that the issue is geometric.

Radius-based operators do not suffer this failure mode.
A `Radius(r=200 m)` operator covers 200 m of physical space at every resolution; the *number* of pixels it averages changes, but the spatial scale of the smoothing does not.

```
   data density × 1                     data density × 4

   • • • • •                            •••••••••••••••
   • • ★ • •     KNN(k=8) ≈ 2Δ          •••••••••••••••
   • • • • •                            •••••••★•••••••
                                        •••••••••••••••
                                        •••••••••••••••

                                        KNN(k=8) ≈ Δ/2  ←── shrunk!


   Radius(r=R)                          Radius(r=R)
       ┌─┐                                  ┌─┐
       │★│   R fixed                        │★│   R fixed
       └─┘                                  └─┘
```

This is the same lesson the operator-learning community learnt the hard way moving from grid-based CNNs to mesh-free graph neural operators.
A vanilla GNN with `KNN` edges is *not* mesh-invariant; remesh the same physics with different node density, and the network's effective stencil changes.
The fix — adopted by Graph Neural Operators (GNO), Mesh GraphNets, and the Equivariant-style PDE solvers — is to use radius graphs, so that the operator's receptive field is determined by the *physics* (the correlation length of the PDE solution) rather than by the *grid* (where you happened to sample).

---

### 4.3 Pseudocode — a `Neighborhood` strategy and an operator built on it

We make the neighborhood definition an explicit, swappable object — not a flag, not a string, not an `if/else`.
The operator consumes the neighborhood; the *choice* of `KNN` vs `Radius` is a physical statement made at construction time.

```python
import equinox as eqx
from jaxtyping import Float, Int, Array

# ----- strategies -----

class Neighborhood(eqx.Module):
    """Strategy interface: maps a query set to a neighbor index."""

    def find(
        self,
        query: Float[Array, "Q D"],
        ref:   Float[Array, "N D"],
    ) -> Int[Array, "Q K"]:
        raise NotImplementedError


class KNN(Neighborhood):
    k: int

    def find(self, query, ref):
        # squared pairwise distance — illustrative; production code uses a KDTree
        d2 = einx.reduce("q [d], n [d] -> q n",
                         query[:, None] - ref[None], op="sqsum")
        return einx.argmin("q [n] -> q k", d2, k=self.k)


class Radius(Neighborhood):
    r: float
    k_max: int   # we still return a fixed-shape index for jit

    def find(self, query, ref) -> Int[Array, "Q K"]:
        d2 = einx.reduce("q [d], n [d] -> q n",
                         query[:, None] - ref[None], op="sqsum")
        within = d2 < self.r ** 2                          # (Q, N) bool

        # top-k_max within-radius hits, ties broken by distance.
        # Out-of-radius slots are filled with -1 sentinels that downstream
        # code must mask before indexing.
        masked = jnp.where(within, d2, jnp.inf)
        idx    = einx.argmin("q [n] -> q k", masked, k=self.k_max)
        # mark any slot whose chosen neighbor is out of radius as -1:
        chosen_d2 = einx.get_at("q [n], q k -> q k", masked, idx)
        return jnp.where(chosen_d2 < jnp.inf, idx, -1)
```

Now an operator that *consumes* a neighborhood — say, a local-mean smoother — written in the coord-aware `(data, coords) → (data, coords)` signature from [Spatiotemporal data §3.6](spatiotemporal_data.md#coords-dependency-game):

```python
class LocalMean(eqx.Module):
    neighborhood: Neighborhood

    def __call__(
        self,
        data:   Float[Array, "N V"],
        coords: dict[str, Array],          # must contain "lat", "lon" (or generic "pos")
    ) -> tuple[Float[Array, "N V"], dict[str, Array]]:
        positions = einx.rearrange(
            "n, n -> n two", coords["lat"], coords["lon"],
        )                                                            # (N, 2)
        idx = self.neighborhood.find(positions, positions)           # (N, K)
        neighbors = einx.get_at("[n] v, q k -> q k v", data, idx)    # (N, K, V)
        smoothed  = einx.mean("n [k] v", neighbors)
        return smoothed, coords          # coords pass through unchanged
```

The operator *demands* coords in its signature — there is no version of `LocalMean` that doesn't.
The two strategies are then a one-line configuration choice:

```python
density_adaptive = LocalMean(neighborhood=KNN(k=16))
mesh_invariant   = LocalMean(neighborhood=Radius(r=200.0, k_max=64))
```

Two operators, same algorithm, different *physical* contracts.
The first is appropriate for problems where similarity-in-feature-space is the locality (anomaly detection in `V`-space, manifold-based gap-filling).
The second is appropriate for problems where the geometry sets the scale (PDE solvers, plume smoothing, kriging, spatially-stationary GP priors).

**Static vs dynamic coords** (the [Spatiotemporal data §3.3](spatiotemporal_data.md#coords-static-dynamic) distinction) maps cleanly onto neighborhood reuse:

```python
# Static coords — KDTree / radius index built ONCE, reused across all timesteps
static_op = LocalMean(neighborhood=Radius(r=200.0, k_max=64))
neighborhood_idx = static_op.neighborhood.find(positions_static, positions_static)
# … apply per timestep with the precomputed idx

# Dynamic coords — KDTree / radius index rebuilt per timestep
def step(data_t, coords_t):
    return LocalMean(neighborhood=Radius(r=200.0, k_max=64))(data_t, coords_t)

result = jax.vmap(step)(data, coords_per_t)
```

The structural cost difference (one build vs `T` builds) is exactly the asymmetry [Spatiotemporal data §3.3](spatiotemporal_data.md#coords-static-dynamic) flagged: static coords amortise neighborhood construction, dynamic coords cannot.
Operators that consume `Neighborhood` strategies should be written so the index is *cacheable* on static-coord tensors — otherwise dynamic-coord workloads silently inflate cost by a factor of `T`.

---

### 4.4 Which to pick — a physical decision rule

The choice is dictated by *what your operator is approximating*.

**Use radius when the operator is a discretisation of a continuous physical kernel.**
Convolution against a Gaussian, Green's function for a PDE, kriging covariance, atmospheric-transport response, point-spread-function deconvolution.
These kernels have a *fixed length scale* — they live in the metric of the underlying space, not in the metric of your sampling.
Examples: GraphCast and similar weather-model emulators use radius graphs because the dynamics are local in physical space; mesh-free PDE solvers like Smoothed Particle Hydrodynamics use kernel functions with fixed support radius for the same reason.

**Use KNN when the operator is a smoother in feature space, or when sampling density is intentionally uniform.**
The classic case is anomaly detection in `V`-space (k-NN density estimation), or gap-filling on a regular grid where `KNN(k=4)` and `Radius(r=Δx)` are interchangeable.
Examples: cosine-similarity retrieval over satellite embeddings; collaborative-filtering-style spatial gap-fill on a regular Landsat grid; nearest-neighbour resampling in image warping.

**A useful tie-breaker.**
If you can write down the operator's correlation length scale in *physical units* (km, days), use `Radius`.
If you can only write it down in *count units* (k = 16), you are implicitly assuming the sampling density is meaningful — fine on a regular grid, dangerous everywhere else.

---

### 4.5 Geophysical examples where the choice matters

- **Methane plume detection in TROPOMI / EMIT scenes.**
  Plume extent is ~ 1–10 km, set by wind speed and source strength.
  A KNN-based smoother trained on a 7 km TROPOMI grid will silently change behaviour when applied to a 30 m EMIT scene; a radius-based smoother does not.
- **Ocean drifter / Argo float interpolation.**
  Sampling density varies by factor of 10× across the ocean.
  Radius-based interpolation respects the eddy length scale; KNN-based interpolation produces tight stencils in the data-rich tropics and balloons in the Southern Ocean, biasing the resulting maps.
- **Mesh-free PDE surrogates (DeepONet, FNO-Graph, GNO).**
  Radius graphs make the operator mesh-invariant; KNN graphs make it mesh-dependent and break under refinement.
- **Soil-moisture downscaling.**
  Sensor footprints are 1–40 km but topographic / vegetation drivers vary at 10–100 m.
  The correct local kernel for downscaling has a physical scale (soil-correlation length); use `Radius`.
- **Aftershock-rate estimation in seismology.**
  Each main shock has an Omori-law decay in time and a fault-extent in space; aggregating aftershocks by KNN biases toward dense seismic networks (urban) and away from sparse ones (offshore); aggregating by radius is geophysically correct.
- **EnKF / EnSRF localisation.**
  Ensemble Kalman filters localise covariance with a *physical* localisation radius (Gaspari-Cohn, typically a few hundred km in the atmosphere).
  Using KNN here would silently amplify ensemble-size sensitivity.

---

### 4.6 How this plugs back into the dependency game

`KNN` and `Radius` do not change the §3 cell — they parametrise *what "local" means inside it*.

- A per-pixel local regression (§3.3) can use either strategy to define the neighborhood whose data fits the per-pixel model.
- A windowed variogram (§3.4) uses `Radius` implicitly — the window is a fixed spatial extent.
- A KMeans clustering (§3.2) uses *neither* — its locality is in `V`-space, with no spatial structure at all.
- An EOF (§3.1) is *not local in `X`* by construction; the neighborhood is the whole field.

So the picture is two-dimensional: §3 chooses *which axes* are local; §4 chooses *what locality means* on those axes.
A robust operator wrapper carries both: a scope marker per axis, and (where applicable) a `Neighborhood` strategy for the local axes.

---

## 5. A worked operator — four scopes, one algorithm

> **Stub.**
> Carry one algorithm — *anomaly detection on temperature* — through all four cells:
>
> 1. **Global / global**: climatology subtraction; the "anomaly" is the residual.
> 2. **Global / local**: quantile-mapped exceedance against a global CDF.
> 3. **Local / local**: per-pixel z-score against the pixel's own historical mean and std.
> 4. **Local / global**: per-window heatwave statistics aggregated to a regional indicator.
>
> Each gets pseudocode + a side-by-side parameter-tensor shape diagram, so the reader sees the same algorithm assuming four different scope contracts.

---

## 6. Cheat sheet

> **Stub.**
> One-page reference: axis-triage table, the `einx.rearrange` patterns from §2, the four cells of §3, and a "footguns" callout (sklearn flatten kills geometry, KNN-as-locality, mixing fit/predict scopes in `Sequential`).
