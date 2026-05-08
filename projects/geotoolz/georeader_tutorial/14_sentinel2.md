---
title: Sentinel-2
subject: georeader tutorial
subtitle: Sentinel-2 SAFE products (L1C and L2A)
short_title: Ch. 14 — Sentinel-2
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: tutorial, georeader, sentinel2
---

> **Module:** `georeader/readers/S2_SAFE_reader.py` (1845 LOC — the largest file in the package)
> **Role:** read Sentinel-2 imagery in the official SAFE product format. Both Level-1C (top-of-atmosphere reflectance) and Level-2A (atmospherically-corrected surface reflectance) are supported, from local folders or Google Cloud's free public bucket.

---

## 1. Why this module is so large

A Sentinel-2 SAFE product is **not one file**. It's a folder hierarchy with:

- 13 separate JPEG2000 (`.jp2`) files — one per spectral band — at three different native resolutions (10 m, 20 m, 60 m).
- Two XML metadata files — product-level (`MTD_MSIL1C.xml`) and tile-level (`MTD_TL.xml`) — describing acquisition geometry, calibration coefficients, sun/viewing angles, cloud masks, and processing baseline.
- Per-band Quality Indicators (QI) at multiple resolutions: cloud masks, MSI defective pixel flags, cirrus mask, etc.
- A scene classification layer (SCL) for L2A products only.

The module's job is to hide all of that behind a single class that behaves like a `GeoData` (Chapter 2) — `s2.shape`, `s2.transform`, `s2.read_from_bounds(...)`, `s2.load()` all work as if S2 were one file.

The 1845 LOC accommodates: SAFE-folder discovery, XML parsing, granule resolution, per-band JP2 stacking via `RasterioReader`, DN→radiance conversion, SRF extraction, multi-resolution band alignment, and Google Cloud Storage path translation. All of that is invisible from the public API surface, which is essentially three classes (`S2Image`, `S2ImageL1C`, `S2ImageL2A`) and one factory (`s2loader`).

---

## 2. L1C vs L2A — the two product levels

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                 SENTINEL-2 PROCESSING LEVELS                             │
│                                                                          │
│   Level-1C (L1C)                      Level-2A (L2A)                     │
│   ─────────────────                   ─────────────────                  │
│                                                                          │
│   ☀️ Sun                               ☀️ Sun                             │
│    │                                   │                                 │
│    ▼                                   ▼                                 │
│   ┌─────────┐                        ┌─────────┐                        │
│   │Atmosphere│ ◄─ NOT corrected      │Atmosphere│ ◄─ CORRECTED          │
│   └────┬────┘                        └────┬────┘                        │
│        │                                  │                              │
│        ▼                                  ▼                              │
│   ┌─────────┐                        ┌─────────┐                        │
│   │ Surface │                        │ Surface │                        │
│   └─────────┘                        └─────────┘                        │
│        │                                  │                              │
│        ▼ 🛰️                              ▼ 🛰️                           │
│                                                                          │
│   TOA Reflectance                     BOA Reflectance                   │
│   - Includes atmospheric effects      - Surface reflectance             │
│   - Globally available                - Atmospheric correction applied  │
│   - Can convert to radiance           - Scene Classification (SCL)     │
│   - 13 bands (incl. B10 cirrus)       - 12 bands (no B10)              │
│                                                                          │
│   Use for:                            Use for:                          │
│   - Radiance-based analysis           - Land cover mapping              │
│   - Custom atmospheric correction     - Vegetation indices (NDVI)       │
│   - Cloud studies (B10)               - Change detection                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

The key technical differences:

- **L1C** — values are *top-of-atmosphere* reflectance scaled to int16 with a factor of `1 / 10000`. To convert to radiance, the module's `DN_to_radiance` applies per-band physical calibration constants from the metadata.
- **L2A** — values are *bottom-of-atmosphere* reflectance from ESA's Sen2Cor processor (or equivalent). Includes the SCL band — a per-pixel classification (cloud / snow / water / vegetation / bare ground / etc.). **B10 (cirrus, 1375 nm) is dropped** because Sen2Cor uses it in the correction process and doesn't produce a corrected surface value.

For ML pipelines you almost always want **L2A** — the atmospheric correction is consistent across scenes and removes much of the haze/illumination variability that confuses CNN training. For radiative-transfer studies that need to handle their own correction, you want L1C.

---

## 3. Spectral bands — the canonical S2 table

```text
Band │ Central λ │ Bandwidth │ Resolution │ L1C │ L2A │ Description
─────┼───────────┼───────────┼────────────┼─────┼─────┼─────────────────────
B01  │   443 nm  │   20 nm   │    60m     │  ✓  │  ✓  │ Coastal/Aerosol
B02  │   490 nm  │   65 nm   │    10m     │  ✓  │  ✓  │ Blue
B03  │   560 nm  │   35 nm   │    10m     │  ✓  │  ✓  │ Green
B04  │   665 nm  │   30 nm   │    10m     │  ✓  │  ✓  │ Red
B05  │   705 nm  │   15 nm   │    20m     │  ✓  │  ✓  │ Red Edge 1
B06  │   740 nm  │   15 nm   │    20m     │  ✓  │  ✓  │ Red Edge 2
B07  │   783 nm  │   20 nm   │    20m     │  ✓  │  ✓  │ Red Edge 3
B08  │   842 nm  │  115 nm   │    10m     │  ✓  │  ✓  │ NIR
B8A  │   865 nm  │   20 nm   │    20m     │  ✓  │  ✓  │ NIR Narrow
B09  │   945 nm  │   20 nm   │    60m     │  ✓  │  ✓  │ Water Vapour
B10  │  1375 nm  │   30 nm   │    60m     │  ✓  │  ✗  │ Cirrus (L1C only)
B11  │  1610 nm  │   90 nm   │    20m     │  ✓  │  ✓  │ SWIR 1
B12  │  2190 nm  │  180 nm   │    20m     │  ✓  │  ✓  │ SWIR 2
```

This table is **the** reference you'll come back to. Three things worth highlighting:

- **Three native resolutions: 10 m / 20 m / 60 m.** When you set `out_res=10`, the 20 m and 60 m bands are *upsampled* on read. When you set `out_res=20`, the 10 m bands are *downsampled* and the 60 m bands are *upsampled*. The `out_res=20` option is the practical sweet-spot when you don't strictly need 10 m resolution — you keep more bands at native resolution.
- **B08 is the broad NIR (842 nm), B8A is the narrow NIR (865 nm).** These look interchangeable but aren't — atmospheric scientists prefer B8A (narrower bandpass = cleaner SWIR-NIR spectroscopy); ML practitioners often use B08 (10 m native + matches Landsat-8 NIR more closely).
- **Bands 10/11/12 wavelength ordering looks weird.** B11 (1610 nm) is *between* B09 (945 nm) and B12 (2190 nm); B10 (1375 nm) is also between them. The naming follows ESA's historical band-ID scheme, not wavelength order. When constructing band lists for SRF binning or visualisation, use `BANDS_S2 = ["B01", "B02", ..., "B12"]` (the module-level constant) — that's the canonical order matching the array's band axis.

The module exports the canonical lists as `BANDS_S2` (full 13 for L1C), `BANDS_S2_L1C` (alias), and `BANDS_S2_L2A` (12 bands; no B10).

---

## 4. The class hierarchy

```python
S2Image                    # base class — do not instantiate directly
├── S2ImageL1C             # L1C-specific: DN→radiance, no SCL
└── S2ImageL2A             # L2A-specific: SCL band, no B10
```

`S2Image` lives at [S2_SAFE_reader.py:295](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/S2_SAFE_reader.py#L295). It implements the `GeoData` protocol (Chapter 2 §3) plus S2-specific machinery:

- Granule discovery: walking the SAFE folder to find per-band JP2 paths.
- Multi-resolution band alignment: each band has its native resolution; `out_res=` selects the output grid and bands at other native resolutions get resampled internally.
- XML metadata parsing for sun/view angles, processing baseline, and per-band offsets.
- Window focus delegation: `set_window` propagates to all underlying `RasterioReader`s for each JP2.

`S2ImageL1C` ([S2_SAFE_reader.py:1000](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/S2_SAFE_reader.py#L1000)) adds:

- The B10 cirrus band.
- DN-to-radiance conversion via `DN_to_radiance(s2obj, dn_data)` ([S2_SAFE_reader.py:1534](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/S2_SAFE_reader.py#L1534)) — uses the per-band `solar_irradiance` and `quantification_value` constants from `MTD_MSIL1C.xml`.

`S2ImageL2A` ([S2_SAFE_reader.py:918](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/S2_SAFE_reader.py#L918)) adds:

- The Scene Classification Layer (SCL) — a 20 m raster of integer class codes (0=NoData, 1=Saturated, 2=DarkArea, 3=CloudShadow, 4=Vegetation, 5=BareSoil, 6=Water, 7=Unclassified, 8=CloudMediumProb, 9=CloudHighProb, 10=ThinCirrus, 11=Snow). Useful for masking without running a separate cloud detector.

---

## 5. Constructor signature (common to both subclasses)

```python
S2ImageL2A(
    s2folder,                    # path to .SAFE folder (local or gs://)
    polygon=None,                # Shapely polygon (EPSG:4326) for AOI
    granules=None,               # dict[band → JP2 path]; auto-discovered if None
    out_res=10,                  # 10, 20, or 60 — output resolution in meters
    window_focus=None,           # rasterio Window for sub-region (in out_res grid)
    bands=None,                  # list[str] — band subset; defaults to all 12/13
    metadata_msi=None,           # explicit path to MTD_MSI*.xml; auto-located if None
)
```

A few non-obvious points:

- **`s2folder`** can be a `gs://` path. The module recognises `gs://gcp-public-data-sentinel-2/...` URIs and uses `fsspec` to walk the folder structure.
- **`polygon` is in `EPSG:4326`.** Almost always — most catalog queries return WGS84 polygons. The module reprojects internally to the tile's UTM zone.
- **`out_res=20` is often the best choice** for ML — keeps red-edge and SWIR at native resolution, avoids upsampling.
- **`bands=` lets you load a subset.** Memory-efficient for "just RGB" or "just NDVI" use cases. The default is all bands at the chosen `out_res`.
- **`window_focus`** — same semantics as `RasterioReader.set_window` (Chapter 3 §4). Restricts subsequent reads to a sub-region.

---

## 6. The factory: `s2loader`

```python
def s2loader(
    s2folder,
    polygon=None,
    out_res=10,
    window_focus=None,
    bands=None,
    metadata_msi=None,
)
```

Located at [S2_SAFE_reader.py:1603](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/S2_SAFE_reader.py#L1603). The factory you should use 95% of the time:

- Detects whether the folder is L1C or L2A from the SAFE name (the second path segment encodes `MSIL1C` vs `MSIL2A`).
- Returns the appropriate subclass.

Two related convenience functions for common catalog systems:

- **`s2_load_from_feature_element84(feature, ...)`** — load from an Element84 STAC feature dict.
- **`s2_load_from_feature_planetary_microsoft(feature, ...)`** — load from Microsoft Planetary Computer's STAC feature dict.

Both wrap `s2loader` after extracting the SAFE path from the feature's assets.

---

## 7. The Google Cloud public bucket

`gs://gcp-public-data-sentinel-2/` (constant: `FULL_PATH_PUBLIC_BUCKET_SENTINEL_2`) is a free, no-auth-required mirror of the entire Sentinel-2 archive. The `s2_public_bucket_path(...)` function ([S2_SAFE_reader.py:1739](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/S2_SAFE_reader.py#L1739)) constructs paths from `(tile_number_field, datetime, processing_baseline)` — useful when you have a SAFE name from a catalog query and need to turn it into a `gs://` URL.

The standard "load any S2 scene from anywhere" recipe:

```python
from georeader.readers.S2_SAFE_reader import s2loader

path = "gs://gcp-public-data-sentinel-2/tiles/29/S/ND/.../S2A_MSIL2A_20240615T...SAFE"
s2 = s2loader(path, out_res=10)

# s2 is an S2ImageL2A — behaves like a GeoData
gt = read.read_from_polygon(s2, my_aoi, crs_polygon="EPSG:4326")
```

The lazy access pattern matches what you'd get with a single-file `RasterioReader` — which is the design goal.

---

## 8. SRF reading

`read_srf(s2obj=None, mission=None, ...)` ([S2_SAFE_reader.py:1411](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/S2_SAFE_reader.py#L1411)) returns a `pd.DataFrame` with the **published S2 SRFs** — wavelength index, one column per band — exactly as ESA distributes them. Use this for the spectral-binning recipe in [Chapter 11 §8](11_reflectance.md):

```python
srf_df = read_srf(mission="S2A")            # or pass s2obj
e_per_band = integrated_irradiance(srf_df)  # band-integrated solar irradiance
toa_refl = radiance_to_reflectance(s2_radiance, e_per_band, ...)
```

The DataFrame uses the canonical band naming (`B01`, `B02`, ..., `B12`) so it lines up with the band axis of an S2 `GeoTensor` directly.

---

## 9. Function reference

**Classes**
- `S2Image` — base, do not instantiate directly
- `S2ImageL1C(s2folder, polygon=None, out_res=10, ...)` — L1C reader
- `S2ImageL2A(s2folder, polygon=None, out_res=10, ...)` — L2A reader

**Loaders**
- `s2loader(s2folder, ...)` — auto-detects L1C/L2A (the recommended entry point)
- `s2_load_from_feature_element84(feature, ...)` — Element84 STAC adapter
- `s2_load_from_feature_planetary_microsoft(feature, ...)` — MS Planetary Computer adapter

**Helpers**
- `s2_public_bucket_path(tile_number, datetime, processing_baseline) → str` — build `gs://` URL
- `s2_name_split(s2file) → tuple` — parse a SAFE filename into 7 fields
- `s2_old_format_name_split(...)` — legacy filename parser (pre-2017)
- `normalize_band_names(bands) → list[str]` — accept various spellings (`b1`, `B01`, `band1`, ...) and canonicalise

**Calibration**
- `DN_to_radiance(s2obj: S2ImageL1C, dn_data=None) → GeoTensor` — L1C DN → radiance
- `read_srf(s2obj=None, mission=None, ...) → pd.DataFrame` — published SRFs

**File / cloud**
- `islocalpath(path) → bool` — local vs cloud
- `get_filesystem(path, requester_pays=None)` — fsspec filesystem
- `get_file(remote_path, local_path)` — fetch a single file
- `read_xml(xml_file) → ET.Element` — XML parsing helper

**Constants**
- `BANDS_S2 = ["B01", ..., "B12"]` — full 13-band list (L1C order)
- `BANDS_S2_L1C` — alias for `BANDS_S2`
- `BANDS_S2_L2A` — 12-band list (L1C order minus B10)
- `PUBLIC_BUCKET_SENTINEL_2 = "gcp-public-data-sentinel-2"`
- `FULL_PATH_PUBLIC_BUCKET_SENTINEL_2 = "gs://gcp-public-data-sentinel-2/"`

---

## 10. Sharp edges

- **`out_res=10` upsamples 8 of 13 bands.** B01, B05–B07, B8A, B09, B10, B11, B12 are not natively at 10 m. Setting `out_res=10` makes them all `(10980, 10980)` via `Resampling.cubic_spline`; you don't get more information, just bigger files.
- **L2A drops B10.** Code that hardcodes `bands=BANDS_S2_L1C` will fail on L2A. Use `BANDS_S2_L2A` or filter dynamically based on `s2.bands`.
- **Old vs new SAFE names.** Pre-2017 products use a different naming convention (`s2_old_format_name_split`). The `s2loader` factory handles both; custom path parsing won't.
- **Processing-baseline drift.** ESA reprocesses the archive every few years. `S2A_*_N0500_*` (baseline 5.00) and `S2A_*_N0510_*` (5.10) have subtly different radiometric calibration. Match baselines in time-series analyses or you'll see fake change signals.
- **B10 in L1C is *only* useful for cloud detection.** It's centred on a strong water-vapour absorption — surface reflectance there is essentially zero, so any signal is from cirrus clouds in the upper atmosphere. Don't include B10 as a "regular" spectral feature.
- **SCL is at 20 m.** When using L2A's SCL band as a cloud mask at `out_res=10`, the mask is upsampled — single-pixel cloud features can become 4-pixel features at 10 m.
- **Requester-pays buckets.** The Microsoft / AWS S3 mirrors are requester-pays. Pass `requester_pays=True` to authenticate the read. The Google public bucket is free.
- **JP2 reads are not parallel-safe in the SAFE-reader sense.** Each band is a separate `RasterioReader`, so each is independently parallel-safe (Chapter 3), but reading multiple bands of one S2 scene from multiple processes is fine.

---

## 11. Connection to `geotoolz`

The whole `presets.s2` block in [`geotoolz.md` §1.2](../plans/geotoolz/geotoolz.md) sits on top of this module:

- **`presets.s2.S2_L2A_RGB(brightness=...)`** — `Sequential([s2.isel(["B04","B03","B02"]), ToFloat32, PercentileClip, Gamma])`. Loads via `s2loader`, picks RGB bands, normalises.
- **`presets.s2.S2_L2A_NDVI(...)`** — `Sequential([s2.load(), MaskClouds(scl_band), NDVI(red_idx=2, nir_idx=3)])`. Uses the SCL band for cloud masking — a free pass thanks to L2A.
- **`presets.s2.S2_L1C_TO_BOA_NDVI(...)`** — the harder pipeline: `Sequential([DN_to_radiance, TOAToBOA, MaskClouds, NDVI])`. Hits `DN_to_radiance` from this module before the radiometric work in [Chapter 11](11_reflectance.md).
- **`presets.s2.S2_QA60_CLOUD_MASK(...)`** — bit-flag extraction from the QA60 band on L1C.

Sentinel-2 is the most-used sensor in the package's ecosystem, and this module's design (single class subclassing `GeoData`, lazy granule access, multi-resolution alignment) is the template the other big sensor readers (`emit`, `prisma`, `enmap`) follow.

Next chapter: [15_hyperspectral.md](15_hyperspectral.md) — the curvilinear-sensor trio (EMIT, PRISMA, EnMAP) and their shared design pattern.
