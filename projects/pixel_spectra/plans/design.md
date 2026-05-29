---
title: "pixel_spectra — pixel-scale spectral inspector"
short_title: "pixel_spectra design"
---

# `pixel_spectra` — pixel-scale spectral inspector

> *Given a single pixel, a multi-point selection, or a small polygon
> AOI, pull the per-pixel reflectance spectrum from a hyperspectral
> scene, overlay a reference spectrum, and score similarity with a
> chosen distance metric. Built for **vetting candidate detections**
> (methane plumes, mineral identifications, snow / vegetation
> verification) before downstream analysis.*

**Status**: design only. No code in this PR.

## Question answered

> "Does the pixel I clicked on (or the patch I drew) actually look
> like the thing I think it is — and how close is it?"

This is upstream of the `methane_pod` workflow: before fitting a
point-process to a list of plume detections, you'd open each
candidate's AOI here, sanity-check that the spectrum matches a
methane-enhanced reference (Mag1c-style absorption signature, or a
HITRAN-derived synthetic), and reject false positives. The same tool
works for any "is this thing X?" question over a hyperspectral scene.

## Scope

- **Sensors (HSI first)**: EMIT (60 m, 285 bands). Add PRISMA / EnMAP /
  DESIS later via the same plug-in shape.
- **AOI**: `shapely.geometry` of any kind. The shape determines pixel
  selection:
  - `Point` → snap to nearest pixel.
  - `MultiPoint` → list of nearest pixels (deduped).
  - `Polygon` → all enclosed pixel centres.
  - `LineString` → pixels along the line (transect mode).
- **Reference spectra**: at minimum, three sources at v0:
  1. Bundled library (USGS splib07 minerals + ECOSTRESS for
     vegetation / soil / man-made).
  2. User-uploaded CSV (`wavelength_nm, reflectance`).
  3. Synthetic — radiative-transfer or HITRAN-derived for gases (the
     methane case).
- **Distance metrics**: SAM, SID, cosine similarity, Euclidean,
  matched-filter score (for narrow-band gases like CH4).
- **Compute target**: interactive (< 5 s per pixel-spectrum lookup).
  Bulk vetting (1k AOIs × 1 scene) is a separate batch mode.

## Algorithm

```
1. Discover scenes intersecting the AOI for the chosen sensor
       (delegate to satellite_viewer.search; the SENSORS registry
       already has emit-l2a-rfl).
2. For each scene:
     a. Sign asset hrefs (planetary-computer or earthaccess token).
     b. Read the full reflectance cube clipped to the AOI bbox via
        georeader → ndarray of shape (n_bands, h, w) plus the band
        wavelengths from the asset's metadata.
     c. Convert AOI geometry → pixel-coordinate list (snap/contains).
3. Build a pixel-spectra DataFrame: one row per (scene, pixel),
       columns = [lon, lat, datetime] + 285 reflectance values.
4. Reference spectrum: resample to scene's wavelength grid (linear
       interp, drop bands with no overlap).
5. Apply distance metric per pixel against the reference → one float
       per (scene, pixel).
6. (Optional) Mask non-usable bands: EMIT atmospheric water-vapor
       windows around 1.4 μm and 1.9 μm (~80 of the 285 bands).
       Mask choice is a knob exposed in the UI.
```

## Stages

| v   | What it does                                                              |
|-----|---------------------------------------------------------------------------|
| v0  | Point AOI → spectrum vs. reference, side-by-side plot. No metric.         |
| v1  | Add distance metric: single number per pixel, alongside the plot.         |
| v2  | Polygon / MultiPoint AOI → per-pixel metric histogram + mean spectrum     |
|     | with uncertainty (±σ) band; "good vs. bad" pixel scatter on the map.      |
| v3  | Bulk vetting: upload a GeoJSON / CSV of candidate detection AOIs, run     |
|     | the per-pixel metric over each, return a sorted table with score + the    |
|     | best-matching scene per candidate.                                        |

You ship after v2 for interactive use; v3 turns it into a vetting
pipeline.

## Architecture

```
projects/pixel_spectra/
├── pyproject.toml
├── README.md
├── src/pixel_spectra/
│   ├── __init__.py
│   ├── sensors.py        # HSI-only subset of the satellite_viewer registry
│   ├── reader.py         # AOI → pixel-spectra DataFrame (via georeader)
│   ├── references.py     # library loader (USGS splib, ECOSTRESS), interp helpers
│   ├── metrics.py        # SAM, SID, cosine, Euclidean, matched_filter
│   └── bulk.py           # the v3 vetting pipeline
├── tests/
│   ├── test_metrics.py   # numerical correctness vs. hand-computed cases
│   ├── test_reader.py    # AOI → pixel-coord conversion (offline)
│   └── test_references.py# library loader + resample
└── apps/                 # whichever stack we pick from "Stack options"
```

Repos reused: `satellite_viewer` (discovery + credentials),
`georeader` (windowed reads). Nothing from the `geotoolz` /
`geopatcher` / `geocatalog` stack is needed at this scale (single
scene, ≤ few hundred pixels).

## Output schema

A long-format pandas DataFrame:

```
scene_id      : string         # STAC item id
pixel_lon     : float64        # WGS84 lon of the pixel centre
pixel_lat     : float64        # WGS84 lat of the pixel centre
datetime      : datetime[UTC]  # acquisition time
n_bands       : int            # how many bands made it past masking
distance      : float64        # the chosen metric vs. reference
spectrum      : object         # list[float] of length n_bands (or
                               # path to a per-row Zarr for large dumps)
reference_id  : string         # which reference spectrum was used
metric        : string         # "SAM" / "SID" / "cosine" / ...
mask_id       : string         # which band-mask was applied
```

## Stack options

Different ways to build this, in increasing order of how much exists
already vs. how much you control.

### Option A — `satellite_viewer` + `georeader` (recommended)

Build on what's already in `research_notebook`:

- `satellite_viewer.search("emit-l2a-rfl", aoi, t0, t1)` for discovery.
- `satellite_viewer.credentials.earthdata()` for the URS token.
- `georeader.read_from_window(...)` for the windowed cube read.
- numpy / scipy for metrics. plotly / altair for the spectrum plot.

**Pro**: zero new dependencies, lights up the existing stack, plays
nicely with the planned `satellite_climatology` work.
**Con**: `georeader`'s HSI ergonomics on EMIT specifically still need
a micro-bench (it was designed against S2/L8 first).

### Option B — `rioxarray` + `pystac-client` directly

Skip `georeader`; open each EMIT asset as an xarray Dataset via
`rioxarray.open_rasterio(..., chunks={"band": 32})` and use xarray's
windowed indexing.

**Pro**: industry-standard pipeline, plays well with dask if it
grows. Every Python remote-sensing engineer can read it.
**Con**: more boilerplate; loses the `georeader` abstraction for
non-COG sensors you might add later (PRISMA HDF5).

### Option C — Earth Engine

`ee.ImageCollection("EMIT/L2A").filterBounds(aoi).getRegion(...)`
returns pixel values directly.

**Pro**: no asset signing, no scene reads, no infrastructure.
**Con**: EMIT may not be ingested in EE (check at start of M1); EE
also can't handle 285-band image returns cleanly — you'd hit element
limits. Probably a dead-end for HSI at this scale.

### Option D — Notebook recipe only

No app, just a Jupyter notebook + a `pixel_spectra.read_aoi(...)`
helper. Click in JupyterLab via `ipyleaflet` Draw control. Spectrum
plot inline.

**Pro**: shortest path to a working demo. Matches your geostack
notebook pattern.
**Con**: no bulk vetting UI; the v3 pipeline becomes a script not an
app.

**My recommendation**: A for the library + a *notebook* at v0/v1
(option D's shape) + a Panel/Streamlit subapp at v2/v3 (option A's
infra under a UI). Ship v0 as the notebook in week 1; promote to a
real app once the metric set stabilises.

## UI integration

Single-screen layout:

```
+---------------------------+--------------------------------------+
| Sensor   : emit-l2a-rfl   |  [ leafmap basemap ]                 |
| AOI      : draw or upload |  AOI overlay + selected pixels (dots)|
| Date     : range slider   |                                      |
| Reference: library | up.. |                                      |
| Metric   : SAM ▾          |                                      |
| Mask     : default ▾      |                                      |
| [ Inspect ]               |                                      |
+---------------------------+--------------------------------------+
| Spectrum panel  (selected pixel + reference, with mask shaded)   |
+------------------------------------------------------------------+
| Metric panel    (histogram for multi-pixel AOI, or scalar)       |
+------------------------------------------------------------------+
| Bulk table      (v3 only: sorted scores per candidate)           |
+------------------------------------------------------------------+
```

Click on the map → highlight the pixel → re-render the spectrum
panel against the same reference.

## Compute budget

- Single-pixel × single scene: 2 STAC search + 1 small windowed read
  + 285-element metric → < 3 s wall-clock.
- Polygon-AOI (~100 pixels) × single scene: one full-band read of the
  AOI bbox + 100 metric evaluations → 5–15 s.
- v3 bulk vet, 1k candidates × ~1 scene each: ~10–30 min on a laptop;
  parallelisable per-candidate.

Memory: one EMIT scene clipped to a 1 km AOI is ~30 MB float32. Trivial.

## Risks / open questions

- **EMIT atmospheric mask** — water vapour windows render ~80 bands
  unusable. Default mask must come from the L2A QA mask metadata,
  not be hand-coded.
- **Reference resampling** — USGS splib07 is at 1 nm; EMIT is at ~7.4
  nm. Linear interpolation is the simple answer but introduces error
  for narrow features; consider Gaussian convolution with EMIT's SRFs
  if the matched-filter metric matters (App 3 will help).
- **Methane reference** — no static library entry will work; needs
  RT model output (e.g., HITRAN convolution at the expected
  enhancement level). Document as a user-bring-your-own input for now.
- **Pixel snap semantics** for `Point` AOIs at sub-pixel offsets —
  pick nearest centre vs. interpolate. Stick with "nearest centre,"
  document it.
- **EMIT signing** — Planetary Computer + earthaccess have different
  signing flows; the credentials module already abstracts this.

## Acceptance

- Point AOI + single EMIT scene → spectrum + reference plot < 5 s.
- Polygon AOI → per-pixel metric histogram populated.
- SAM metric numerically matches reference implementation (spec one
  test case from Kruse et al. 1993).
- Bulk vetting: 100 candidate AOIs × 1 scene each completes in < 5 min.
- Spectrum plot correctly shades the water-vapour masked bands.

## Out of scope

- Atmospheric correction (we assume L2A reflectance).
- Plume strength quantification — that's Mag1c / Matched-Filter
  inversions; this tool flags candidates, doesn't quantify them.
- Multi-scene aggregation per AOI (not the use case).
- Radiative-transfer modelling — the reference spectra are inputs.
