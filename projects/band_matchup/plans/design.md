---
title: "band_matchup — cross-sensor band comparison and simulation"
short_title: "band_matchup design"
---

# `band_matchup` — cross-sensor band comparison and simulation

> *Given two satellite sensors (multispectral, hyperspectral, or
> geostationary), compare their spectral response functions (SRFs).
> Which bands of A correspond to which bands of B? Convolve a
> hyperspectral spectrum with an MSI sensor's SRFs to predict what
> that MSI sensor "would have seen" at the same pixel.*

**Status**: design only. No code in this PR.

The most underbuilt of the three apps — most cross-sensor SRF tooling
is buried inside individual mission Python packages (`spectral-tools`,
sensor-specific calval libraries) and there's no single user-facing
app that lets you compare any-to-any.

## Question answered

> "If I have a band/pixel from sensor A, what's the best-matching
> band on sensor B — and what would sensor B's pixel value be?"

Use cases:

- **Matchup studies** validating new sensors against established
  references.
- **Cross-calibration** for time-series that span multiple
  generations of an instrument family (S2A → S2B → S2C).
- **HSI → MSI simulation**: predict what S2 / Landsat / MODIS would
  see at an EMIT pixel, useful for synthetic matchups and for
  product-continuity studies.
- **Sensor selection**: "I need NIR around 850 nm — which sensors
  have a band there, and how do their SRFs compare?"

## Scope

### Sensors (SRF library to bundle)

| Family            | Sensors                                                   |
|-------------------|-----------------------------------------------------------|
| MSI polar         | Sentinel-2 A / B / C, Landsat 8 OLI, Landsat 9 OLI-2      |
| MSI moderate      | MODIS Terra / Aqua, VIIRS NPP / JPSS-1 / JPSS-2           |
| HSI               | EMIT, PRISMA, EnMAP, DESIS                                |
| Geostationary     | GOES-16 ABI, GOES-18 ABI, Himawari-9 AHI, MTG-I1 FCI,     |
|                   | Meteosat-11 SEVIRI (legacy)                               |

About 20 sensors covering everyone you'd practically compare.

### Wavelength range

VNIR + SWIR (0.4–2.5 μm). TIR is a separate beast (different
acquisition mode, different units, different cal); explicit out of
scope.

### Compute target

- v0/v1: instant — all in-memory SRF math on ~100 KB of data.
- v2 per pixel: milliseconds.
- v3 (coincident-scene matchup): same as `pixel_spectra` scale.

## Algorithm

### SRF data shape

Per `(sensor, band)`:

```
wavelength_nm : ndarray[float]   # nm grid, 1 nm step where possible
response      : ndarray[float]   # normalised so max = 1
provenance    : str              # e.g. "ESA S2-MSI 2024 release"
units         : str              # "relative" or "absolute"
```

Bundled under `data/srf/<sensor>/<band>.csv` with a top-level
`data/srf/manifest.json` carrying provenance + URL.

### v0 — SRF plot

```
For each (sensor, band) the user selects:
  Plot response vs wavelength on shared axes.
  Annotate band centre + FWHM.
```

### v1 — Similarity matrix

For sensors A (`n` bands) and B (`m` bands):

```
S[i,j] = (∫ SRF_Ai(λ) · SRF_Bj(λ) dλ)
        / sqrt(∫ SRF_Ai² · ∫ SRF_Bj²)
```

(Cosine on the sampled SRFs after resampling to a common λ grid.)
Output: `(n × m)` ndarray, plotted as a heatmap. Each row's argmax
is "best match for A's band i in sensor B."

### v2 — HSI → MSI simulation

For an HSI sensor (`n_λ` wavelengths) and an MSI sensor with band
`b` whose SRF is `r_b(λ)`:

```
B_b = ∫ r_b(λ) · R_HSI(λ) dλ / ∫ r_b(λ) dλ
```

Equivalent to a weighted average of the HSI reflectance values at
the SRF support. Per-pixel: one matrix multiply.

### v3 — Coincident-scene matchup

```
1. Find coincident scenes via satellite_viewer.search for two
       sensors at the AOI within a time window.
2. For each pixel:
     read HSI cube.
     read MSI bands.
     simulate MSI from HSI via v2.
     compare predicted vs observed (residual stats).
3. Report mean residual + scatter plot per band.
```

## Stages

| v   | What it does                                                              |
|-----|---------------------------------------------------------------------------|
| v0  | SRF browser — pick sensor(s) + band(s), plot overlay.                     |
| v1  | Similarity matrix between two sensors, heatmap of cosine overlap.         |
| v2  | HSI → MSI simulation: take a spectrum, predict band values for any        |
|     | target MSI sensor.                                                        |
| v3  | End-to-end coincident-scene matchup at an AOI: predicted vs observed.     |

## Architecture

```
projects/band_matchup/
├── pyproject.toml
├── README.md
├── data/
│   └── srf/
│       ├── manifest.json
│       ├── sentinel-2a/{B1.csv, ..., B12.csv}
│       ├── landsat-8/{B1.csv, ..., B7.csv}
│       ├── emit/full_spectrum.csv   # one wavelength axis + a flag
│       ├── modis-terra/{...}
│       ├── viirs-npp/{...}
│       ├── goes-16-abi/{...}
│       ├── himawari-9-ahi/{...}
│       └── mtg-i1-fci/{...}
├── src/band_matchup/
│   ├── __init__.py
│   ├── library.py        # load_srf(sensor, band), list_sensors()
│   ├── similarity.py     # cosine matrix, best-match search
│   ├── simulate.py       # v2 HSI→MSI convolution
│   └── matchup.py        # v3 coincident-scene runner
├── tests/
│   ├── test_library.py        # SRF JSON parses, shapes are right
│   ├── test_similarity.py     # similarity(S2A, S2A) == I (sanity)
│   └── test_simulate.py       # convolve uniform-reflectance HSI
└── apps/
    └── (see Stack options)
```

`georeader` only enters at v3. v0–v2 are SRF math alone — beautifully
self-contained.

## Output schema

For v1 — similarity matrix as a pandas DataFrame:

```
index   : sensor_A band ids (rows)
columns : sensor_B band ids (cols)
values  : float [0, 1]
attrs   : sensor_A, sensor_B, wavelength_grid_nm
```

For v2 — simulated MSI prediction:

```
DataFrame[
    band      : str
    centre_nm : float
    fwhm_nm   : float
    value     : float       # the simulated reflectance / radiance
]
```

For v3 — matchup statistics:

```
DataFrame[
    band, centre_nm, n_pixels,
    mean_predicted, mean_observed,
    bias, rmse, r2
]
```

## Stack options

### Option A — Pure Python + bundled CSV (recommended)

`pandas`/`numpy` for SRF loading, `scipy` for resampling, `altair`
or `matplotlib` for plots. ~500 lines total for v0–v2.

**Pro**: tiny dependency surface, easy to host, easy to test, easy
to embed in a notebook. All SRF math is just integration.
**Con**: you write the SRF library by hand (provenance + format
adapter per source). One-time cost; ~1 day of work for the 20-sensor
set.

### Option B — Wrap `pyrsr` / `spectral-tools`

Re-use existing Python libraries that carry SRF data:

- [`pyrsr`](https://github.com/GFZ/py_tools_ds) (GFZ) — has S2, L7-9,
  Sentinel-3, MODIS, RapidEye.
- `spectral` (SPy) — has utilities for SRF convolution.

**Pro**: don't re-implement loading; piggyback on existing
provenance.
**Con**: spotty geostationary coverage; geostationary SRFs need
hand-curating either way. Adds a heavy-ish dependency for what's
really just a CSV.

### Option C — Browser-first (D3 / Observable / plotly)

Ship the SRF library as static JSON, plot in-browser, no Python
server at all.

**Pro**: zero server runtime, embeds on a static-site (your MyST
docs), shareable as a URL.
**Con**: v2/v3 (simulation, matchup) need pixel reads → Python /
WebAssembly. Browser-only caps you at v0/v1.

### Option D — Earth Engine

EE has some sensor SRFs but the dataset is incomplete and the API
isn't designed for SRF comparison. Not a good fit; skip.

### Option E — Pre-compute everything to a static dataset

Pre-compute the full (sensor × band) × (sensor × band) similarity
tensor for every pair you care about, ship as a single Parquet, the
app is just a viewer.

**Pro**: tiny runtime, perfectly cacheable, instant heatmaps.
**Con**: less flexibility (e.g., the user can't change the λ grid
or add a sensor without rerunning the precompute).

**My recommendation**: A for the library + the apps; E as an
*output* of the pipeline so the heatmaps land in your MyST docs as
static images. v3 is where you reach for `satellite_viewer` +
`georeader`.

## UI integration

```
+--------------------+---------------------------------------+
| Sensor A : ▾       |  SRF overlay plot                     |
|   bands  : (multi) |  (wavelength on x, response on y,     |
| Sensor B : ▾       |   colour by sensor)                   |
|   bands  : (multi) |                                       |
| Common λ grid: 1nm |                                       |
+--------------------+---------------------------------------+
| Similarity matrix  |  Heatmap (A rows × B cols)            |
|  (cosine overlap)  |  Click → highlight band pair in plot  |
+--------------------+---------------------------------------+
| Simulator (v2)     |  upload spectrum (CSV) →              |
|                    |  show simulated bands per target      |
+--------------------+---------------------------------------+
| Matchup (v3)       |  AOI + date → coincident-scene runner |
+--------------------+---------------------------------------+
```

## Compute budget

- v0/v1: instant (~100 KB SRF data total).
- v2: ms per pixel.
- v3 per AOI: dominated by scene reads; comparable to `pixel_spectra`.

## Risks / open questions

- **SRF provenance is the hard part.** ESA + USGS publish official
  SRFs; geostationary missions (GOES, Himawari, MTG) sometimes only
  publish design SRFs not as-built. Each entry needs a citation +
  date; manifest.json captures this.
- **Normalisation conventions differ** — some sources publish
  "relative response" (peak = 1), others "transmittance" (peak < 1),
  some include out-of-band response (skirts), some don't. The loader
  must normalise to a single convention (relative, max=1) and record
  the original convention.
- **λ grid choice** — 1 nm is fine for VNIR; some SRFs (especially
  geostationary) are only published at 5–10 nm. Resampling
  introduces small errors; document the chosen interp scheme.
- **Geostationary vs. polar** subtlety: geostationary SRFs are full-
  disk-shared, no per-tile variation. Polar sensors with detectors
  may have small per-detector SRF differences (S2 has 12 detectors).
  v0 ships the published mission-mean SRFs and notes the caveat.
- **EMIT as a "sensor"** — EMIT's "SRF" is its instrument response
  per wavelength, ~7.4 nm FWHM Gaussians. Special-cased as a wide
  hyperspectral comb in the library.
- **TIR scope creep** — geostationary sensors all have TIR bands
  (GOES ABI's bands 7–16 are TIR). Document that the app handles
  only VNIR+SWIR (≤ 2.5 μm) and skip TIR bands in the manifest.

## Acceptance

- SRF library bundled for all 20 listed sensors, manifest carries
  provenance + URL for every file.
- Self-similarity check: `similarity(S2A, S2A) ≈ I` (within 0.99 on
  the diagonal, < 0.05 off-diagonal except for adjacent bands).
- HSI → MSI simulation against a coincident EMIT × S2 scene matches
  the observed S2 reflectance within 10% RMSE for clear-sky pixels.
- v0 / v1 UI loads + renders in < 1 s.

## Out of scope

- TIR bands (≥ 3 μm).
- Polarimetric channels.
- BRDF / view-angle effects.
- Atmospheric simulation — `band_matchup` does no RT; users supply
  L2A reflectance.
- Calibration adjustments (gains / offsets / dark current).
