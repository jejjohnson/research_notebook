---
title: Spatiotemporal data
subject: geotoolz tutorial
subtitle: Shapes, coordinates, and geometry of the world that arrives at your operator
short_title: Spatiotemporal data
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: tutorial, geotoolz, data, shapes, coordinates, geometry, xarray, einx, jaxtyping
---

> **Status:** Draft.
> **Scope:** A conceptual tutorial — pseudocode only, no executable cells.
> **Audience:** Anyone wrapping geophysical / EO data for downstream operators and trying to think about *what arrives at the operator and what travels alongside it.*

Operators get all the attention.
But every operator is downstream of a more basic question: *what does the data look like, what coordinates does it carry, and what geometric topology does it sit on?*
This tutorial is the data-side companion to [Spatiotemporal operators](spatiotemporal_operators.md).

The two tutorials share a vocabulary — the canonical tensor `[S, T, V, X]`, the named-dim convention, and `einx` / `xarray` / `jaxtyping` for typing.
This one walks how the data *gets* to that shape, what coordinates it brings with it, and how the underlying geometry (point, line, polygon, raster, multi-polygon) governs which operators can run on it.

The framing is at heart *physical*: every axis of the data tensor carries a different scale, every coordinate is a hook for an operator to bind to, and every geometric primitive is a different statement about what the world delivered.
Getting the data shape, coordinates, and geometry right is upstream of every operator-side choice — and in practice, it is where most pipelines live or die.

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
from jaxtyping import Float, Array
import xarray as xr
import einx

Field = Float[Array, "S T V X"]
```

The named-dims version, for the same physical quantity:

```python
da: xr.DataArray  # dims = ("sample", "time", "variable", "space")
```

`einx.rearrange` patterns are written against these names, not positional indices, so the reshape *reads* as the intent.

---

## 2. Building up to `[S, T, V, X]` — the data you actually meet

Real datasets rarely arrive at `[S, T, V, X]`.
They start as a single time series, or a single spatial field, and the canonical tensor is *constructed* — by stacking, ensembling, windowing, or patching.
This section walks the construction so that every later mention of "the `S` axis" or "the `T` axis" has a concrete geophysical referent.

We use six base shapes, ordered by how many physical axes they expose, and then three operations that synthesise an `S` axis on top.

---

### 2.1 Stage 0 — `[T]` — univariate time series

One variable, one location, sampled over time.
The simplest geophysical signal there is.

```
        time →
        ┌────────────────────────────┐
arr     │ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ │
        └────────────────────────────┘
```

```python
arr:    Float[Array, "T"]
coords = {
    "time": Float[Array, "T"],     # datetime64 or seconds-since-epoch
}
da:     xr.DataArray               # dims = ("time",)
# geometry: point (one fixed location, see §4.2)
```

**Examples.**
A single tide gauge's hourly sea level (Brest, Sydney), one Argo float's daily SST record, the Mauna Loa monthly CO₂ time series, a borehole's groundwater level, a single PM₂.₅ sensor, one eddy-covariance flux tower's NEE record.
Anything you'd plot as a single line chart.

**Operators that live here.**
Trend estimation, change-point detection, harmonic decomposition, autoregressive modelling, peaks-over-threshold extremes — all of classical univariate time-series analysis.

---

### 2.2 Stage 1 — `[T, V]` — multivariate time series

Same location, same time grid, several variables.

```
        time →
   T   ┌────────────────────────────┐
   p   │ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ │
   q   │ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ │
   u   │ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ │
       └────────────────────────────┘
```

```python
arr:    Float[Array, "T V"]
coords = {
    "time":     Float[Array, "T"],
    "variable": Str[Array,   "V"],   # variable names (or band IDs, depths)
}
da:     xr.DataArray                 # dims = ("time", "variable")
# geometry: point
```

**Examples.**
A weather station record (T, RH, P, wind, precip), a hyperspectral pixel time series (B1…B224), a multi-gas flask-sample record (CO₂, CH₄, N₂O), a multi-depth soil-moisture profile at one station, a single MODIS pixel through all spectral bands.

**Operators that live here.**
Cross-correlation of variables, vector autoregression (VAR), canonical correlation analysis, multivariate extreme-event detection, copula-based dependence modelling.
The `V` axis is where channel-mixing happens; `T` is where temporal context lives.

---

### 2.3 Stage 2 — `[X]` — univariate spatial field

One variable, one snapshot in time, sampled over space.

```
              W →
       H   ┌─────────────────┐
       ↓   │ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ │
           │ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ │
           │ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ │
           └─────────────────┘
```

```python
arr:    Float[Array, "H W"]                # structured grid
arr:    Float[Array, "N"]                  # unstructured mesh / point cloud
coords = {
    "lat": Float[Array, "H"],              # static — fixed for the lifetime of the data
    "lon": Float[Array, "W"],
    # unstructured variant:
    # "lat": Float[Array, "N"], "lon": Float[Array, "N"]
}
da:     xr.DataArray                       # dims = ("y", "x")  or  ("lat", "lon")
# geometry: raster (structured) or point set (unstructured)
```

**Examples.**
A single Landsat NDVI map, an SRTM 30 m DEM tile, a snapshot of MERRA-2 surface temperature at one timestamp, a one-orbit TROPOMI XCH₄ scene, an ICESat-2 single-track elevation profile.

**Operators that live here.**
Edge detection, morphology, segmentation, spatial smoothing, kriging interpolation, hot-spot detection (Getis-Ord G*, local Moran's I).
This is `skimage`-shaped territory.

---

### 2.4 Stage 3 — `[V, X]` — multivariate spatial field

Several variables, one snapshot, sampled over space.
The canonical "remote-sensing scene".

```
                 W →
       V   ┌─────────────────┐
  channel  │ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ │  per channel
   stack   │ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ │
           │ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ │
           └─────────────────┘
```

```python
arr:    Float[Array, "V H W"]
coords = {
    "variable": Str[Array,   "V"],
    "lat":      Float[Array, "H"],
    "lon":      Float[Array, "W"],
}
da:     xr.DataArray             # dims = ("band", "y", "x")
# geometry: raster
```

**Examples.**
A multispectral scene (Landsat 8: 11 bands), a hyperspectral cube (PRISMA, EnMAP, EMIT: 200+ bands), a multi-variable ERA5 surface snapshot (T2, U10, V10, TP, SSR), a single SAR polarimetric image (VV + VH).

**Operators that live here.**
Spectral unmixing, supervised classification, principal-component compression, band-ratio indices (NDVI, NDWI, NBR), atmospheric correction, cloud masking.

---

### 2.5 Stage 4 — `[T, X]` — univariate spatiotemporal

One variable, a sequence of snapshots — the basic geophysical "video".

```
         time →
                                   each frame is a flat field

  t1: ┌────────┐     t2: ┌────────┐     t3: ┌────────┐
      │ ▓ ▓ ▓ ▓│         │ ▓ ▓ ▓ ▓│         │ ▓ ▓ ▓ ▓│   ...
      │ ▓ ▓ ▓ ▓│         │ ▓ ▓ ▓ ▓│         │ ▓ ▓ ▓ ▓│
      └────────┘         └────────┘         └────────┘
```

```python
arr:    Float[Array, "T H W"]
coords = {
    "time": Float[Array, "T"],
    "lat":  Float[Array, "H"],     # static if regular grid
    "lon":  Float[Array, "W"],
    # dynamic-grid variant (e.g. swath data):
    # "lat": Float[Array, "T H W"], "lon": Float[Array, "T H W"]
}
da:     xr.DataArray               # dims = ("time", "y", "x")
# geometry: raster (structured) or point cloud per t (swath)
```

**Examples.**
A monthly NDVI time series over Africa, daily SST over the Pacific, hourly precipitation-radar mosaics over a river basin, an Argo gridded temperature stack at one depth level, daily MODIS LST.

**Operators that live here.**
Per-pixel trend, change detection, optical flow, object tracking, Kalman-style state estimation, persistence forecasts, video segmentation.

---

### 2.6 Stage 5 — `[T, V, X]` — multivariate spatiotemporal

The most common datacube in modern Earth-system science.

```
       V  ┌──┐   ┌──┐   ┌──┐
  per    │T1│   │T2│   │T3│   ...      a stack of (T, V) snapshots
 channel └──┘   └──┘   └──┘             each cell is a 2-D field
```

```python
arr:    Float[Array, "T V H W"]
coords = {
    "time":     Float[Array, "T"],
    "variable": Str[Array,   "V"],
    "lat":      Float[Array, "H"],     # static
    "lon":      Float[Array, "W"],
}
da:     xr.DataArray                   # dims = ("time", "variable", "y", "x")
# geometry: raster
```

**Examples.**
ERA5 reanalysis (decades of hourly data, dozens of variables, global grid), CMIP6 model output, a satellite analysis-ready datacube (ARD), a multi-channel SAR time series, an atmospheric-chemistry assimilation product (CAMS), a multispectral monthly mosaic (Sentinel-2 L2A monthly composites).

**Operators that live here.**
Almost anything you'd run on the canonical tensor *except* operations along an ensemble axis.
This is `xarray`'s and `Pangeo`'s native shape; most operational EO pipelines never leave it.

---

### 2.7 Stage 6 — `[S, T, V, X]` — the canonical tensor

The `S` axis is the *bookkeeping* axis: independent (or quasi-independent) realisations of a `[T, V, X]` cube.

```python
arr:    Float[Array, "S T V H W"]
coords = {
    "member":   Int[Array,   "S"],     # ensemble member ID, window start, or patch position
    "time":     Float[Array, "T"],
    "variable": Str[Array,   "V"],
    "lat":      Float[Array, "H"],
    "lon":      Float[Array, "W"],
}
da:     xr.DataArray                   # dims = ("member", "time", "variable", "y", "x")
# geometry: raster (× S)
```

`S` is rarely a single natural data axis.
It is almost always *manufactured* by one of three operations: **simulation**, **windowing**, or **patching**.
Each construction creates an `S` axis with different semantics — *and reshapes the coord pack along with it*.
The dependency-game logic in [Spatiotemporal operators §3](spatiotemporal_operators.md) hinges on which construction you used.

---

### 2.8 `S` from simulation — ensembles

Run a numerical model (or a stochastic forecast) `N` times with perturbed initial conditions, parameters, or boundary forcings.
The `S` axis indexes the realisations.

```
                                arr[s=0, T, V, X]
   simulation runs              arr[s=1, T, V, X]
       ↓                        arr[s=2, T, V, X]
                                   …
       run 0                    arr[s=N, T, V, X]
       run 1
       run 2          S becomes a *probability* axis —
        …             reductions over S compute mean,
       run N          variance, quantiles, EnKF analysis.
```

**Real instances.**
ECMWF / GFS / NCEP ensemble forecasts (51 / 31 / 21 members), GraphCast and Pangu-Weather ensembles, the CMIP6 multi-model archive (`S` indexes models *and* runs), EnKF / EnSRF data-assimilation ensembles, Lagrangian particle back-trajectory ensembles (FLEXPART, HYSPLIT), perturbed-parameter climate-sensitivity sweeps.

```python
def from_ensemble(
    runs: list[Float[Array, "T V H W"]],
) -> Float[Array, "S T V H W"]:
    return einx.rearrange("s ... -> s ...", jnp.stack(runs))
```

**Semantics: exchangeable.**
The members are realisations of the *same* physics under stochastic perturbation; reductions over `S` are well-defined moments.
This is the `S` axis the dependency-game logic implicitly assumes.

**Coord transform: a new axis appears.**
The `time`, `lat`, `lon` arrays are *shared* across all members — they do not gain an `S` dimension.
A single new coord, `member` (shape `[S]`), is added.
Operators reducing over `S` (ensemble mean, spread, EnKF analysis) only ever touch this one new coord.

---

### 2.9 `S` from windowing — local-in-time slicing

Slice a long `[T, V, X]` cube into overlapping or non-overlapping temporal windows.
The `S` axis indexes window starts.

```
   one long [T, V, X] series:
      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
      └──┘                              window 1
         └──┘                           window 2  (stride s)
            └──┘                        window 3
               └──┘                     window 4   ──►  arr[S, T_w, V, X]
                  └──┘                                      ↑       ↑
                                                      window idx  window length
```

**Real instances.**
30-day rolling NDVI windows for phenology classification, 1-second seismic-waveform windows for earthquake / explosion classification, daily-rolling rainfall windows for flood-onset detection, attention-context windows for transformer-based forecasts, lookback windows for autoregressive sea-ice forecasts.

```python
def windowed(
    arr: Float[Array, "T V H W"], win: int, stride: int,
) -> Float[Array, "S T_w V H W"]:
    # einx supports strided unfolding via constraint patterns;
    # production code may use jax.numpy.lib.stride_tricks.sliding_window_view.
    return einx.rearrange(
        "(s_pos win) v h w -> s_pos win v h w",
        sliding_view(arr, axis=0, window=win, stride=stride),
        win=win,
    )
```

**Semantics: chronologically ordered, not exchangeable.**
Adjacent windows share data; reductions over `S` are *not* iid means.
Operators must either treat `S` as a sequence axis (RNN / transformer), enforce non-overlap, or honestly inflate the effective-sample-size correction.

**Coord transform: `time` gains an `S` axis.**
Each window has its own time stamps, so `coords["time"]` becomes shape `[S, T_w]` (and overlapping windows share values across rows — a redundant-coord fact that downstream code must respect).
`lat` and `lon` remain unchanged because windowing is purely temporal.
Window-relative time (e.g., "lag-from-window-start") is a useful *derived* feature, often added as a second time-coord.

---

### 2.10 `S` from patching — local-in-space slicing

Slice a single large `[T, V, H, W]` field into overlapping or non-overlapping spatial patches.
The `S` axis indexes patch positions.

```
   one large [V, H, W] field:                arr[S, T, V, H_p, W_p]
   ┌──────────────────────────┐                     ↑      ↑     ↑
   │  ┌──┐ ┌──┐ ┌──┐ ┌──┐    │   patches            S    patch  patch
   │  └──┘ └──┘ └──┘ └──┘    │   ─────►        position idx     size
   │  ┌──┐ ┌──┐ ┌──┐ ┌──┐    │
   │  └──┘ └──┘ └──┘ └──┘    │
   └──────────────────────────┘
```

**Real instances.**
64×64 patches for CNN training on Sentinel-2 (the BigEarthNet / EuroSAT standard), 256×256 patches for FNO / U-Net training on ERA5, sliding-window plume-detection crops on hyperspectral cubes, patch-based denoisers (BM3D, Noise2Self), tile-based foundation-model embedding (Prithvi: 224×224 tiles).

```python
def patched(
    arr: Float[Array, "T V H W"], patch: int,
) -> Float[Array, "S T V H_p W_p"]:
    return einx.rearrange(
        "t v (nh ph) (nw pw) -> (nh nw) t v ph pw",
        arr, ph=patch, pw=patch,
    )
```

**Semantics: spatially ordered, not exchangeable.**
Adjacent patches share pixels if overlapping; reductions over `S` aggregate spatially (mosaicking, voting) rather than averaging realisations.

**Coord transform: `lat` and `lon` gain an `S` axis.**
Each patch occupies its own coordinate sub-rectangle, so `coords["lat"]` becomes shape `[S, H_p]` and `coords["lon"]` becomes shape `[S, W_p]`.
Patch-relative pixel offsets (`y_in_patch`, `x_in_patch`) are useful derived coords for translation-equivariant operators.
The patch's *origin* in the parent field — `(lat0[S], lon0[S])` — is the bookkeeping you need to reassemble outputs back into a single mosaic.

---

### 2.11 Why this matters for the dependency game

The dependency table in [Spatiotemporal operators §3](spatiotemporal_operators.md) treats `S` as the iid-samples axis, in the spirit of the sklearn contract.
That contract is exactly right for **ensemble** `S` — the members really are exchangeable.
It is **violated** by windows and patches, because adjacent slices share data.

This is the silent failure mode behind most EO-benchmark "good results that don't generalise":

- Train / test split drawn at random over *overlapping patches* → optimistic test F1 by 0.05–0.10.
- Hyperparameter selection on shuffled windows of a single river → false claims of model skill.
- EnKF run with too few ensemble members + reused windows → underestimated forecast variance.

When you reach for `S` in any operator, pause and ask: *which construction made this axis, and is the operator's reduction semantically valid for that construction?*
The construction is upstream of every other choice in the dependency game.

---

## 3. Coordinates — what travels alongside the data

Until now we have written `[T]` and `[X]` as bare integer-indexed axes.
In practice every axis carries a *coordinate array* of physical units — datetimes, latitudes, longitudes, elevations, viewing angles, member IDs.
Coordinates are not decoration; they are first-class data that operators consume.
A GP kernel takes positions, a transformer takes positional encodings, a variogram takes spatial lags, a harmonic regression takes time-as-real, a phenology classifier takes day-of-year features.
This section lays out the coord taxonomy and shows how it interlocks with the operator-side dependency game.

---

### 3.1 The four-quadrant taxonomy

Coordinates split along two independent axes — *raw vs feature* and *static vs dynamic*.

|              | **static** (fixed in time)             | **dynamic** (time-varying)                              |
|--------------|----------------------------------------|---------------------------------------------------------|
| **raw**      | DEM `(y[H], x[W])`, GHCN station lat/lon, mooring `(lat, lon)` | satellite swath `lat[T,H,W]`, drifter `(lat[S,T], lon[S,T])`, satellite ground track |
| **feature**  | elevation, distance-to-coast, biome, slope/aspect, climate zone | solar zenith angle, viewing geometry, ground speed, advected-source position |

*Raw* coords are how the data is *indexed*: they answer "where / when is this measurement?"
*Feature* coords are how operators *consume* coords: derived quantities (elevation, day-of-year sine, solar geometry) that enter the operator as additional inputs.
*Static* coords have shapes that do not include `T` or `S` — they broadcast across the data tensor.
*Dynamic* coords align with `T` (and possibly `S`) — they travel with the data.

The four quadrants are not exclusive; a single product usually has coords in several quadrants at once.
A Sentinel-2 ARD scene has raw-static `(lat, lon)` (the ARD grid) and raw-dynamic per-pixel `solar_zenith[T, H, W]` (every overpass differs).
A drifter has raw-dynamic `(lat[T], lon[T])` and feature-dynamic `(speed[T], heading[T])`.

---

### 3.2 Time coords — raw vs feature

Operators consume time differently, so coords have to support both views simultaneously.

**Raw time.**
A `datetime64` array, or a float array of seconds-since-epoch / days-since-reference.
This is what GP kernels (`Exp[-|t - t'| / L]`) and harmonic-regression basis matrices consume.

**Feature time.**
Cyclic encodings (sine and cosine of day-of-year, hour-of-day, day-of-week), Fourier-time features for periodic-aware models, transformer-style sinusoidal positional encodings, "time since last event" for spike / event data.

```python
def cyclic_time(
    t: Float[Array, "T"], period: float,
) -> Float[Array, "T 2"]:
    phase = 2 * jnp.pi * t / period
    return einx.rearrange(
        "t, t -> t two", jnp.cos(phase), jnp.sin(phase),
    )

# typical pack carried alongside the data tensor:
time_coords = {
    "time":            t,                                  # Float[Array, "T"]
    "doy_cyc":         cyclic_time(doy, period=365.25),    # Float[Array, "T 2"]
    "hour_cyc":        cyclic_time(hour, period=24),       # Float[Array, "T 2"]
    "year_progress":   year_fraction(t),                   # Float[Array, "T"]
}
```

The same operator family changes meaning depending on which time coord it consumes.
A per-pixel detrend with `t` as a column gives a slope-and-intercept (the fit-local / predict-local cell of the dependency game).
A per-pixel detrend with `(doy_cos, doy_sin)` columns added gives a slope-plus-seasonal-cycle removal — the basis of the BFAST / CCDC family of change-detection algorithms.

---

### 3.3 Space coords — static vs dynamic

This is where the taxonomy matters most for compute.

**Static** — coords fixed for the lifetime of the data.

```python
positions: Float[Array, "X 2"]                      # (lat, lon) per pixel
covariance = gp_kernel(positions, positions)        # compute ONCE — broadcast over T, S
```

Examples: a DEM grid, a fixed gauge network (GHCN, ICOADS), an ocean mooring, a single-orbit ARD product, a polar-stereographic ice grid, an unstructured but time-invariant mesh.
Operators that consume static coords amortise the cost: KDTrees, GP kernel matrices, mesh adjacency, distance-to-coast features — all built once.

**Dynamic** — coords that change with `T` (and possibly `S`).

```python
positions: Float[Array, "T X 2"]                            # swath / mesh moves with time
covariance_t = jax.vmap(gp_kernel)(positions, positions)    # rebuild per timestep — expensive
```

Examples: an Argo float trajectory `(lat[S, T], lon[S, T])`, a surface drifter `(lat[T], lon[T])`, GPS-tagged animal tracks, satellite L2 swaths (every TROPOMI / EMIT overpass has a different lat/lon footprint), pollution sources advected by wind, deformable-mesh PDE solver coordinates.

The static / dynamic distinction is *the* governing choice for the cost of any spatially-aware operator.
A static-coord kriging interpolator is `O(N²)` once.
A dynamic-coord kriging interpolator is `O(N²)` per timestep — and `T` is often 10⁴–10⁶.
This is why most large-scale operational pipelines reproject dynamic-coord swath data onto a static grid first; the cost is paid in resampling rather than in every downstream operator.

---

### 3.4 Structured vs unstructured

A second cross-cutting distinction: are the coords on a regular grid?

**Structured.**
Coords are implicit — `x = arange(W)`, `y = arange(H)` — and the data shape `[H, W]` already encodes neighborhood.
You can use `skimage`, FFT-based filters, and convolutions without ever materialising the coord arrays.

**Unstructured.**
Coords are explicit — every node carries its own `(lat, lon, elevation, …)` — and neighborhood must be looked up via a KDTree, mesh adjacency, or graph.
`skimage` does not apply; you need graph-based operators or mesh-aware ops.

This is the bridge to §4 (geometry): structured coords are *raster*; unstructured coords are *point set / mesh / vector*.
The static / dynamic distinction governs whether the neighborhood index can be precomputed once or must be rebuilt per timestep.

---

### 3.5 Real-world example matrix

| Product / source            | Time coords        | Space coords          | Coord quadrant                |
|-----------------------------|--------------------|-----------------------|-------------------------------|
| ERA5 surface reanalysis     | hourly datetime    | regular 0.25° grid    | static, structured            |
| Sentinel-2 ARD (cloud-opt.) | per-scene datetime | regular 10 m UTM grid | static, structured            |
| TROPOMI L2 swath            | per-pixel datetime | irregular `lat[T,X]`  | dynamic, unstructured         |
| EMIT L2 hyperspectral       | per-pixel datetime | irregular `lat[T,X]`  | dynamic, unstructured         |
| Argo float profile          | per-cycle datetime | `(lat[S,T], lon[S,T])`| dynamic, point set            |
| Surface drifter (GDP)       | per-fix datetime   | `(lat[T], lon[T])`    | dynamic, single point         |
| GHCN-D station network      | daily datetime     | fixed `(lat, lon)`    | static, point set             |
| SRTM 30 m DEM               | —                  | regular grid          | static, structured (no `T`)   |
| Eddy-covariance (FLUXNET)   | half-hourly        | fixed point           | static, point                 |
| GraphCast 0.25° forecast    | 6-hourly datetime  | regular grid          | static, structured            |
| ICOADS marine obs.          | per-report datetime| `(lat[N], lon[N])`    | dynamic, scattered point set  |
| Lagrangian particle ensemble| per-step datetime  | `(lat[S,T], lon[S,T])`| dynamic, point cloud per `t`  |

The patterns: optical / radar / reanalysis products on regular grids are *static-structured*; satellite L2 swaths and in-situ platforms are *dynamic-unstructured*; gauge networks are *static-unstructured-point-set*; particle ensembles are the most demanding — *dynamic-unstructured-and-S-axis*.

---

### 3.6 How coords interact with the dependency game

Walk through the four cells of the operator-side dependency game (see [Spatiotemporal operators §3](spatiotemporal_operators.md)) with the coord lens:

- **Fit-global / predict-global** (EOF, climatology) — usually *coord-free* in the algorithm itself.
  The parameter tensor is indexed in pixel-space; coords just label the output for the user.
  EOF doesn't care whether the underlying grid is structured or not, as long as you don't permute pixels between fit and predict.
- **Fit-global / predict-local** (KMeans, retrievals, foundation-model embeddings) — usually no spatial coord, but *time-of-year features* often enter as quantile-mapping conditioning, and *solar-geometry features* enter atmospheric-correction LUTs.
  Static-structured products are the easy case; dynamic-unstructured swaths force per-pixel coord-aware lookup.
- **Fit-local / predict-local** (per-pixel detrend, harmonic fit, BFAST/CCDC) — *raw time* is a column of the regression basis matrix.
  Spatial coords are usually ignored unless you regularise across pixels.
- **Fit-local / predict-global** (variograms, Moran's I, basin/national totals) — *raw spatial coords* are essential.
  The operator is *built on* lags; without coords there is no operator.
  Static coords let you precompute lag bins; dynamic coords force per-timestep recomputation.

The pattern is clear: as you move from *global-global* toward *local-global*, coords become more central, and the static / dynamic distinction becomes more decisive.

**Design implication for `geotoolz`.**
An operator that consumes coords cannot be wrapped as a function of `[S, T, V, X]` alone.
Its real signature is

```python
class Operator(eqx.Module):
    def __call__(
        self,
        data:   Float[Array, "S T V H W"],
        coords: dict[str, Array],
    ) -> tuple[Float[Array, "..."], dict[str, Array]]:
        ...
```

Both `data` and `coords` flow through every operator, and `Sequential` must thread both.
Operators that *only* consume `data` can ignore the coord pack (the framework passes it through unchanged); operators that *transform* the coord pack (a reprojection, a patch construction, a rolling-window construction) must return a new pack alongside the new `data`.

This is why the `S`-axis constructions in §2.8–§2.10 and the static / dynamic distinction in §3.3 belong in the same conceptual frame: they are *all coord-pack transforms*.
A coord-aware operator framework lets you compose them; a coord-blind one (sklearn, raw skimage) reduces them to ad-hoc reshape boilerplate that you have to reinvent for every pipeline.

---

## 4. Geometry — how the world delivers data, how the operator demands it

Coordinates are values on an axis.
Geometry is *what topology those values describe*.
Two arrays of `(lat, lon)` pairs of the same shape can be a regular grid, a scattered point set, the vertices of a polygon ring, or the fixes of a drifter trajectory — and the operator that consumes them must respect which one it is.

This section lays out the five geometric primitives, the four operations that bridge them, and why "raster" — the special case where geometry is a regular tessellation — is the case the operator-side tutorial focuses on.

---

### 4.1 The geometry hierarchy

Five primitives, ordered by "structure available for fast operators."

| Primitive       | Dim | Examples                                               | Why it matters                            |
|-----------------|-----|--------------------------------------------------------|-------------------------------------------|
| **Point**       | 0   | weather station, Argo float, eddy-covariance tower, plume source, gauge, buoy | how *observations* arrive                |
| **Line**        | 1   | ship cruise track, satellite ground track, ICESat-2 transect, river network, flight line, animal track | how *transects and trajectories* arrive |
| **Polygon**     | 2   | agricultural field, lake, fire perimeter, protected area, urban tract | how *parcels and AOIs* are defined      |
| **Multi-polygon** | 2 | country, watershed, EEZ, HydroSHEDS basin, climate zone (Köppen) | how the *administrative and relationship* world is encoded |
| **Raster**      | 2 (regular)  | every satellite scene, reanalysis grid, DEM, gridded climatology | **why fast operators exist** — regular tessellation makes neighborhood lookup O(1) |

```
   point         line          polygon            multi-polygon         raster
                                                                      
     •          •──•          •────•             •──•   •──•         ┌─┬─┬─┬─┐
                  \           │    │             │  │   │  │         ├─┼─┼─┼─┤
                   •──•       │    │             •──•   •──•         ├─┼─┼─┼─┤
                              •────•                                 ├─┼─┼─┼─┤
                                                                      └─┴─┴─┴─┘
```

The hierarchy reads bottom-up: a *raster* IS a *polygon* (its bounding box) discretised on a regular tessellation; a *polygon* is a closed *line*; a *line* is an ordered sequence of *points*; a *point* is the atomic primitive.
Multi-polygon is a *collection* of polygons (with optional holes), often with attributes attached (country code, watershed ID, parcel owner).

The progression also tracks computational cost: each step *down* the hierarchy adds explicit topology that must be carried alongside the data; each step *up* (toward raster) buys regularity that the operator can exploit.

---

### 4.2 Per-primitive deep dive

#### Point — how observations arrive

A 0-dimensional location with a value (or a time series of values) attached.

```python
data:   Float[Array, "N V"]                 # N points, V variables per point
coords = {"lat": Float[Array, "N"],
          "lon": Float[Array, "N"]}
# topology: none — points are atomic
```

**Examples.**
GHCN-D weather stations, FLUXNET eddy-covariance towers, Argo floats (per cycle), ocean moorings, surface drifters (per fix), tide gauges, methane plume source locations from TROPOMI, lightning strikes, earthquake epicentres.

**Operators.**
KDTree-based neighborhood lookup, kriging with explicit positions, GP regression, KNN / radius-neighbours interpolation, point-pattern statistics (Ripley's K, nearest-neighbor distance distribution).

**Coord plumbing.**
Coords are explicit arrays of shape `[N]` (or `[N, T]` if dynamic).
Adding a point means appending to *both* the data array and the coord arrays.

---

#### Line — how transects and trajectories arrive

An ordered sequence of points, often with associated times.
Lines are how *moving platforms* deliver data.

```python
data:    Float[Array, "T V"]                # values along the line at each fix
coords = {"lat":  Float[Array, "T"],
          "lon":  Float[Array, "T"],
          "time": Float[Array, "T"]}
topology = {"segment_ids": Int[Array, "T"]}  # which fixes belong to which segment
```

**Examples.**
Ship cruise tracks (CTD casts along a transect), satellite ground tracks (ICESat-2 ATL06 elevation profiles), airborne flight lines (AVIRIS hyperspectral surveys), animal tracks (GPS collars), river networks (linestrings between confluences), seismic acquisition lines.

**Operators.**
Path integration (cumulative distance / elevation gain), along-track filtering, transect interpolation (kriging-along-line), graph operators on river networks (flow-routing, headwater identification), trajectory clustering.

**Coord plumbing.**
Coords are dynamic (they change along the line), but the *connectivity* (which point follows which) is also part of the topology.
A line with no segment IDs is just a point set; the segment IDs are what make it a line.

---

#### Polygon — how parcels and AOIs are defined

A closed region of space, defined by a ring (or rings, with holes) of vertex coordinates.

```python
vertices = Float[Array, "Nv 2"]                  # ring of (lat, lon) pairs
topology = {
    "exterior_start":   0,
    "exterior_length":  Nv_ext,
    "hole_starts":      Int[Array, "Nh"],
    "hole_lengths":     Int[Array, "Nh"],
}
data:    Float[Array, "V"]                       # value(s) attached to the polygon
```

**Examples.**
Agricultural fields (CAP / LPIS parcels), lakes (HydroLAKES), fire perimeters (MTBS, Canadian NBAC), protected areas (WDPA), urban census tracts, building footprints (Microsoft / Google open buildings).

**Operators.**
Zonal statistics (mean / sum of a raster *inside* a polygon), spatial joins (which point falls inside which polygon?), buffering (polygon ± a distance), geometric set ops (union, intersection, difference), area / perimeter computations, point-in-polygon queries.

**Coord plumbing.**
Vertices are coordinates; topology is the ring-closure metadata.
A polygon's value lives in `[V]` (one value per polygon) or `[T, V]` (time-varying), separate from the vertex coords — *the data is on the polygon, not on its vertices*.

---

#### Multi-polygon — how the administrative and relationship world is encoded

A collection of polygons (each possibly with holes), tied together as a single conceptual unit, often with attributes.

```python
vertices = Float[Array, "Nv 2"]                  # all vertices of all polygons
topology = {
    "polygon_starts":   Int[Array, "Np"],        # offset of each polygon's exterior
    "polygon_lengths":  Int[Array, "Np"],
    "hole_starts":      Int[Array, "Nh"],
    "hole_lengths":     Int[Array, "Nh"],
    "polygon_in_unit":  Int[Array, "Np"],        # which conceptual unit each polygon belongs to
    "attribute":        Str[Array, "Nu"],        # e.g. country code, basin ID
}
data:    Float[Array, "Nu V"]                    # value(s) per unit (Nu units)
```

**Examples.**
Countries (Natural Earth, GADM), watersheds (HydroSHEDS, HUC), exclusive economic zones (EEZ, Marine Regions), administrative regions (NUTS, FIPS), climate zones (Köppen-Geiger), ecoregions (WWF), oil-and-gas basins (Permian, Sahara, North Sea), drainage basins (Greenland, Antarctica ice-sheet drainage).

**Operators.**
Hierarchical zonal statistics (national → continental aggregation), spatial joins between hierarchies (which parcel is in which municipality is in which country?), contiguity-based spatial regression, multi-resolution mosaicking, jurisdiction-aware operators (national greenhouse-gas inventories, basin-aggregated water budgets).

**Coord plumbing.**
The hierarchy is encoded in the topology fields — `polygon_in_unit` says which polygon belongs to which conceptual unit (Greece's many islands all belong to "Greece").
Attributes (`country_code`, `basin_name`) are how the multi-polygon connects to *non-spatial* data — and is what makes multi-polygons the natural geometry for the *relationship* world (every administrative join is a multi-polygon spatial join).

---

#### Raster — why fast operators exist

A 2-D regular tessellation of a rectangular domain, defined by an affine transform plus pixel values.

```python
data:    Float[Array, "V H W"]                   # value(s) at each cell
affine:  Float[Array, "6"]                       # (dx, 0, x0, 0, dy, y0)
# coords are IMPLICIT:
#   lon[w] = x0 + dx * w        for w in range(W)
#   lat[h] = y0 + dy * h        for h in range(H)
```

**Examples.**
Every satellite scene (Landsat, Sentinel-2, MODIS, VIIRS), every reanalysis grid (ERA5, MERRA-2, JRA-55), every DEM (SRTM, Copernicus DEM, ASTER GDEM), every gridded climatology (CHIRPS, WorldClim), every analysis-ready datacube product (ARD, Open Data Cube), every regular-grid PDE-model output (CMIP6, GraphCast).

**Operators.**
*Everything* — convolutions, FFTs, finite differences, separable filters, morphology, segmentation, U-Net, ConvLSTM, FNO, CNN classifiers, all of `skimage`, all of standard CNN-based ML.

**Coord plumbing.**
Coords are *not stored*; they are derived on demand from the six numbers of the affine transform.
This is the critical advantage: a 10,000 × 10,000 raster carries 6 numbers of geometry; a 10,000-point unstructured cloud at the same resolution carries 20,000 numbers of geometry (lat + lon per point).
For a `[T, V, H, W]` cube with `T = 10,000` timesteps, the saving is the difference between *six* numbers and *twenty billion* numbers of dynamic-coord storage.

---

### 4.3 The four operations between geometries

The bridges that make heterogeneous pipelines possible.
Every real EO / geo pipeline is a sequence of these four operations interleaved with operators.

| Operation     | Direction          | Real example                                                 |
|---------------|--------------------|--------------------------------------------------------------|
| **Rasterise** | vector → raster    | burn watershed polygons onto a 30 m grid for zonal masking    |
| **Vectorise** | raster → vector    | extract flood-extent polygons from a classified flood map     |
| **Sample**    | raster → points    | query MERRA-2 temperature at every GHCN station location      |
| **Aggregate** | raster → polygon scalar | mean NDVI per country, methane budget per oil-and-gas basin |

#### Rasterise

Convert a vector geometry (point / line / polygon / multi-polygon) into a raster mask, label image, or burned-value image.

```python
def rasterise(
    vertices: Float[Array, "Nv 2"],
    topology: dict,
    affine:   Float[Array, "6"],
    shape:    tuple[int, int],
) -> Int[Array, "H W"]:
    # production: rasterio.features.rasterize, gdal_rasterize, affine + ray casting
    ...
```

**When you reach for it.**
Building zonal masks (one mask per administrative unit), encoding training labels for semantic segmentation (polygon labels → pixel labels), discretising a continuous geometry for use in a PDE solver.

#### Vectorise

Extract vector geometries from a labelled raster — the inverse of rasterise.

```python
def vectorise(
    labels:   Int[Array, "H W"],
    affine:   Float[Array, "6"],
) -> tuple[Float[Array, "Nv 2"], dict]:
    # production: rasterio.features.shapes, OpenCV findContours, scikit-image marching_squares
    ...
```

**When you reach for it.**
Turning a CNN's segmentation output into polygons for parcel-level reporting, extracting flood polygons from a flood-classification raster, extracting cloud-shadow shapes from a cloud-mask raster.

#### Sample

Read raster values at point or line locations.

```python
def sample(
    raster:    Float[Array, "V H W"],
    affine:    Float[Array, "6"],
    points:    Float[Array, "N 2"],   # query (lat, lon)
) -> Float[Array, "N V"]:
    # using einx.get_at on the index computed from the affine:
    h_idx = ((points[:, 0] - affine[5]) / affine[4]).astype(int)
    w_idx = ((points[:, 1] - affine[2]) / affine[0]).astype(int)
    return einx.get_at("v [h w], n -> n v", raster, h_idx * raster.shape[2] + w_idx)
```

**When you reach for it.**
Querying ERA5 at every weather-station location for ML feature extraction, sampling a DEM along a flight track, getting per-pixel solar-zenith at scattered observation points.

#### Aggregate

Reduce raster values inside a polygon (or per-polygon for a multi-polygon) to a scalar (or per-band vector).

```python
def aggregate(
    raster:    Float[Array, "V H W"],
    affine:    Float[Array, "6"],
    masks:     Bool[Array, "Np H W"],   # one mask per polygon (from rasterise)
    op:        str = "mean",
) -> Float[Array, "Np V"]:
    return einx.reduce("v [h w], np [h w] -> np v", raster, masks, op=op)
```

**When you reach for it.**
Per-country emissions inventories (national methane / CO₂ budgets), per-basin water-storage trends from GRACE, per-municipality deforestation rates, per-watershed precipitation accumulations, parcel-level mean NDVI for crop-yield estimation.

This operation is the geometric form of the *fit-local / predict-global* cell of the dependency game (see [Spatiotemporal operators §3.4](spatiotemporal_operators.md)) — it is *the* operator that turns raster heterogeneity into administrative-world deliverables.

---

### 4.4 Why raster makes everything fast

A short callout for what regular tessellation actually buys you.

- **Neighborhood lookup is O(1) integer arithmetic.**
  No KDTree, no graph traversal — just `arr[h-1:h+2, w-1:w+2]`.
- **Convolutions, FFTs, separable filters, multigrid** all apply directly without coordinate-aware wrapping.
- **Coords don't need to be carried per pixel** — six numbers of affine replace `2·H·W` numbers of explicit lat/lon.
- **The neighborhood strategies of `KNN` vs `Radius`** (the central concern of the non-raster operator literature) collapse to fixed-stencil convolution — there is one canonical neighborhood per cell.
- **Storage and I/O are cheap** — Cloud-Optimised GeoTIFF, Zarr chunking, tile pyramids all assume regular tessellation.
- **Ecosystem support is enormous** — `xarray`, `rasterio`, `gdal`, `rioxarray`, `xarray-spatial`, `dask-image`, every CNN library; the alternative paths (graph / mesh / point) have orders of magnitude less tooling.

This is *why* production EO pipelines reproject swath, point, and polygon data onto a common raster grid as the very first step.
The cost is the resampling / rasterisation step; the benefit is that every downstream operator becomes a cheap raster op.

It is also why the companion [Spatiotemporal operators](spatiotemporal_operators.md) tutorial focuses on *raster* operators specifically — not because the others don't exist, but because the raster case is where the dependency-game framework has the cleanest shape and the operators have the cleanest cost model.
Graph and mesh operators are a future tutorial.

---

### 4.5 The two failure modes

The choice of *when* to convert between geometries is the single most consequential pipeline decision.
Both directions of error are common.

**Rasterise too early.**
A 3-metre methane plume on a 10-metre grid is gone — its peak concentration and its precise location are smeared across pixels.
A parcel boundary on a 30-metre grid is uncertain by ±15 m on every side, which is enough to misattribute crop yields between adjacent fields.
An extreme rainfall event recorded at a single gauge, rasterised to a 1° grid, becomes a watershed-mean that no longer represents an extreme at all.

This is the wrong call when *the geometry is the deliverable* — when the user wants per-parcel, per-station, or per-event answers.

**Rasterise too late.**
A pipeline that carries Argo floats as point geometry into a per-pixel GP kernel re-pays an `O(N²)` build at every step.
A pipeline that keeps satellite swaths in their native irregular grid through every downstream operator forces every reader, every resampler, every ML stage to handle dynamic-unstructured data.
The cumulative cost can be 10–100× the cost of an early reprojection.

This is the wrong call when *geometry is the substrate* — when the user wants gridded fields for ML, or when downstream operators are universally raster-centric.

The right rule of thumb is: **convert at the deliverable boundary, not before, not after**.
If your downstream is a CNN, rasterise immediately upstream of it.
If your downstream is a per-station report, keep points until the end.
If your downstream is per-watershed bookkeeping, aggregate into the multi-polygon at the last possible moment.

---

### 4.6 Geometry × dependency game

Walk the four cells of the operator-side dependency game (see [Spatiotemporal operators §3](spatiotemporal_operators.md)) through the geometric lens:

| Cell                       | Natural geometry                          | Why                                                     |
|----------------------------|-------------------------------------------|---------------------------------------------------------|
| Fit-global / predict-global | raster (or coord-free)                    | parameter tensor is index-space; geometry is decoration |
| Fit-global / predict-local  | raster (per pixel) or point (per station) | per-pixel retrieval is geometry-agnostic                |
| Fit-local / predict-local   | raster (per pixel) or point (per gauge)   | each unit gets its own model                            |
| Fit-local / predict-global  | **multi-polygon**                         | "one number per administrative region" is the natural shape — countries, basins, watersheds, EEZs |

Multi-polygon is the geometry that the *fit-local / predict-global* cell was waiting for.
National methane budgets, basin water-storage trends, ice-sheet mass balance per drainage basin, deforestation per municipality — all of these *are* fit-local-predict-global operators whose deliverable lives on a multi-polygon coverage.

This cross-tabulation also explains why most real EO pipelines look the same:

1. **Reproject** dynamic / unstructured input data onto a static raster grid.
2. **Run** raster-shaped operators (CNN / FNO / per-pixel ML / climatology) with one of the four dependency-game cells.
3. **Aggregate** results into a multi-polygon coverage to produce the deliverable (national / basin / parcel statistics).

Steps 1 and 3 are geometry transforms; step 2 is the operator.
Splitting them cleanly is what makes pipelines composable; conflating them is what makes them brittle.

---

### 4.7 Design implication for `geotoolz`

Promotes the coord-aware operator signature from §3.6 to a *geometry-aware* one:

```python
class Operator(eqx.Module):
    def __call__(
        self,
        data:     Array,
        coords:   dict[str, Array],
        topology: Geometry,           # Point | Line | Polygon | MultiPolygon | Raster
    ) -> tuple[Array, dict, Geometry]:
        ...
```

The four operations of §4.3 are exactly the operators that *change* the `topology` field:

- `Rasterise: Point | Line | Polygon | MultiPolygon → Raster`
- `Vectorise: Raster → Polygon | MultiPolygon`
- `Sample:    Raster → Point`
- `Aggregate: Raster + MultiPolygon → MultiPolygon-with-values`

Most operators *preserve* the topology field — it flows through unchanged.
The four bridging operations are the special class that transforms it.

This is also how `Sequential` becomes a *typed* pipeline.
An operator that produces a `Point` output can only feed an operator that consumes `Point` (or any operator that consumes "anything").
The same algebra as scientific units in xarray — type-checked composition rather than ad-hoc reshape boilerplate.

For now, [Spatiotemporal operators](spatiotemporal_operators.md) focuses on the case where the topology field is `Raster` throughout — which is most of operational EO and most of the operator-learning literature.
Future tutorials will lift the restriction.

---

## 5. Closing — how this connects

This tutorial covered three layers of *what arrives at an operator*:

- **Shape** (§2) — the canonical tensor `[S, T, V, X]` and the build-up from univariate time series; the three constructions of the `S` axis (ensembling, windowing, patching) and what they imply about reductions.
- **Coordinates** (§3) — the four-quadrant taxonomy (raw vs feature, static vs dynamic), the cost asymmetry of static vs dynamic, the bridge to the operator's coord-aware signature.
- **Geometry** (§4) — the five-primitive hierarchy (point / line / polygon / multi-polygon / raster), the four bridging operations, why raster is the fast case, and the geometry-aware operator signature.

Each layer adds structure that the operator can exploit — and adds constraints the operator must respect.
Picking the right shape, coords, and geometry is upstream of every operator-side decision in [Spatiotemporal operators](spatiotemporal_operators.md).

The companion tutorial assumes everything from this one as background.
Read it next if you want to see how the dependency game (fit-scope × predict-scope), the physics of locality, and the reshape contracts (sklearn / skimage / ConvLSTM / EOF) play out on raster data.
