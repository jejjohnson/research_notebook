# Ch. 15 — hyperspectral

> **Modules:**
> - `georeader/readers/emit.py` (1102 LOC)
> - `georeader/readers/prisma.py` (571 LOC)
> - `georeader/readers/enmap.py` (865 LOC)
> **Role:** read three hyperspectral satellite sensors — EMIT (NASA, ISS, 285 bands), PRISMA (ASI, 239 bands), EnMAP (DLR, 224 bands). All cover ~400–2500 nm with ~10 nm spectral sampling. The three readers share a design pattern but diverge on georeferencing — that's the interesting part.

---

## 1. The three sensors at a glance

| Sensor | Agency | Launched | Bands | Spectral range | GSD | Georeferencing |
|---|---|---|---|---|---|---|
| **EMIT** | NASA | 2022 (ISS) | 285 | 380–2500 nm | ~60 m | **GLT** (lookup table) |
| **PRISMA** | ASI | 2019 | 239 | 400–2500 nm | 30 m | **per-pixel lat/lon** (interpolation) |
| **EnMAP** | DLR | 2022 | 224 | 420–2450 nm | 30 m | **map-projected + RPCs** |

The trio matters because they're the workhorses of the spaceborne hyperspectral era. They map onto three distinct georeferencing strategies — the structural axis along which the readers differ.

---

## 2. EMIT — NASA's GLT-equipped spectrometer

### Data format

```text
Raw Data Structure (NetCDF file):
┌─────────────────────────────────────┐
│  radiance: (downtrack, crosstrack, bands)  │
│  └── Shape: (~1280, ~1242, 285)            │
│                                             │
│  location/glt_x: (rows, cols)              │
│  location/glt_y: (rows, cols)              │
│  └── Geographic Lookup Table (GLT)         │
└─────────────────────────────────────┘
```

EMIT ships data in **sensor coordinates** (raw pushbroom scan lines, not orthorectified), plus a Geographic Lookup Table that names — for each pixel of the *output* (orthorectified) grid — which sensor pixel to read from. This is the [`griddata.georreference` fast path from Chapter 7](07_griddata.md).

### GLT orthorectification (top-of-module diagram)

```text
Geographic Grid (Output)          Sensor Grid (Raw Data)
┌─────────────────────┐           ┌─────────────────────┐
│ (0,0)               │           │ radiance array      │
│   ┌───┬───┬───┐     │   GLT     │ ┌───────────────┐   │
│   │ a │ b │ c │     │ ──────→   │ │ (5,2) (5,3)   │   │
│   ├───┼───┼───┤     │ lookup    │ │ (6,1) (6,2)   │   │
│   │ d │ e │ f │     │           │ │ ...           │   │
│   └───┴───┴───┘     │           │ └───────────────┘   │
│               (H,W) │           │                     │
└─────────────────────┘           └─────────────────────┘

For pixel (row=1, col=2) in geographic grid:
    glt_x[1,2] = 5  →  raw_col = 5
    glt_y[1,2] = 2  →  raw_row = 2
    value = radiance[2, 5, :]  (all bands)

GLT values of 0 indicate invalid/no-data pixels
```

This approach allows:

1. Efficient storage (no wasted pixels from orthorectification padding).
2. Preservation of original radiometric values (no resampling).
3. Flexible reprojection to any target CRS.

### Class-level diagram (preserved verbatim)

```text
GLT Orthorectification:
┌────────────────────────────┐      ┌──────────────────────────┐
│    Geographic Grid         │      │   Sensor Grid (raw)      │
│  (orthorectified space)    │      │  (pushbroom scan)        │
│  ┌───┬───┬───┬───┐        │      │  ┌───┬───┬───┬───┐      │
│  │ · │ a │ b │ · │        │  GLT │  │ e │ a │ b │ · │      │
│  ├───┼───┼───┼───┤        │  ──→ │  ├───┼───┼───┼───┤      │
│  │ c │ d │ e │ f │        │      │  │ f │ c │ d │ · │      │
│  └───┴───┴───┴───┘        │      │  └───┴───┴───┴───┘      │
│  (pixels with data)        │      │  (original acquistion)   │
└────────────────────────────┘      └──────────────────────────┘

· = no data (GLT value = 0)

For geographic pixel (row, col):
    raw_x = glt_x[row, col]  
    raw_y = glt_y[row, col]
    value = radiance[raw_y, raw_x, :]
```

### Radiometric units & spectral

- L1B radiance: `μW / (cm²·sr·nm)` — note the unusual unit (not `W/m²` SI base; convert by factor 100).
- 285 bands, 380–2500 nm, FWHM ≈ 7–10 nm.
- ~60 m GSD (coarser than PRISMA/EnMAP because the ISS flies higher than typical Earth-observation orbits).

### Interface

`EMITImage(path, ...)` — main class. Methods include `load_radiance()`, `load_reflectance(...)`, `load_wavelengths([w1, w2, ...])`. Helpers `download_product()`, `get_radiance_link()` use NASA Earthdata credentials from `~/.georeader/auth_emit.json`.

---

## 3. PRISMA — ASI's interpolation-required spectrometer

### Data format

```text
PRISMA HDF5 File Structure:
┌─────────────────────────────────────────────────────────┐
│  /HDFEOS/SWATHS/PRS_L1_HCO/                             │
│  ├── Data Fields/                                        │
│  │   ├── VNIR_Cube: (bands, crosstrack, downtrack)      │
│  │   │   └── 400-1010 nm, ~66 bands                     │
│  │   └── SWIR_Cube: (bands, crosstrack, downtrack)      │
│  │       └── 920-2500 nm, ~173 bands                    │
│  ├── Geolocation Fields/                                 │
│  │   ├── Latitude_SWIR, Longitude_SWIR                  │
│  │   └── Latitude_VNIR, Longitude_VNIR                  │
│  └── Attributes (solar/view angles, timing, etc.)       │
│                                                          │
│  /KDP_AUX/                                               │
│  ├── Cw_Vnir_Matrix, Cw_Swir_Matrix (wavelengths)       │
│  └── Fwhm_Vnir_Matrix, Fwhm_Swir_Matrix                 │
└─────────────────────────────────────────────────────────┘
```

Unlike EMIT, **PRISMA L1 data is NOT orthorectified**. Instead, it ships per-pixel `Latitude_*` / `Longitude_*` arrays. Producing a regular grid from this requires **interpolation** — the slow path in [Chapter 7](07_griddata.md), using `read_to_crs(data, lons, lats, ...)` under the hood.

### Sensor grid → geographic grid

```text
Sensor Grid (raw)                  Geographic Grid (output)
┌─────────────────────┐            ┌─────────────────────┐
│ pushbroom scan      │            │ regular grid        │
│ ┌───┬───┬───┬───┐  │  gridding  │ ┌───┬───┬───┬───┐  │
│ │ a │ b │ c │ d │  │  ───────→  │ │ a'│ b'│ c'│ d'│  │
│ ├───┼───┼───┼───┤  │            │ ├───┼───┼───┼───┤  │
│ │ e │ f │ g │ h │  │            │ │ e'│ f'│ g'│ h'│  │
│ └───┴───┴───┴───┘  │            │ └───┴───┴───┴───┘  │
│ + lat/lon per pixel│            │ + affine transform  │
└─────────────────────┘            └─────────────────────┘
```

The `raw=True` / `raw=False` flag on PRISMA's load methods is the eject button: `raw=True` returns sensor coordinates (no interpolation, fast), `raw=False` runs `griddata` (slow, but produces a `GeoTensor` ready for downstream operators). For most ML workflows you want `raw=False`; for matched-filter retrievals that need exact radiometry, `raw=True` and handle gridding yourself if needed.

### Dual-sensor configuration

```text
VNIR Sensor                          SWIR Sensor
┌────────────────────┐               ┌────────────────────┐
│ 400 - 1010 nm      │               │ 920 - 2500 nm      │
│ ~66 bands          │               │ ~173 bands         │
│ ~10 nm sampling    │               │ ~10 nm sampling    │
│                    │               │                    │
│ Shared 30m GSD     │               │ Shared 30m GSD     │
└────────────────────┘               └────────────────────┘
          │                                    │
          └──────────── Overlap ───────────────┘
                     920-1010 nm
```

PRISMA uses two separate sensors. The reader takes care of selecting the right one when you ask for a wavelength — `prisma.load_wavelengths([850, 1600])` gives you 850 nm from VNIR and 1600 nm from SWIR. The 920–1010 nm overlap is mostly used for cross-calibration; for analysis pick one source per wavelength.

### Wavelength range diagram

```text
Wavelength Range:
├──────────────────────────────────────────────────────────────┤
400nm              1000nm                                 2500nm
├───────── VNIR ──────────┤
                  ├────────────────── SWIR ───────────────────┤
                  └─ overlap ─┘
                  920-1010nm
```

### Radiometric units

- L1 radiance: `mW / (m²·sr·nm)` — the SI-ish unit, factor of 1000 from `W/m²/sr/nm`.
- Calibration: DN → radiance via per-band scale + offset (applied automatically on load).

### Interface

`PRISMA(path)` — main class. Methods `load_wavelengths([w1, w2, ...], as_reflectance=False, raw=False)`, `load_rgb(as_reflectance=False, raw=False)`. The wavelength-list interface handles VNIR/SWIR routing transparently.

---

## 4. EnMAP — DLR's already-orthorectified spectrometer

### Data format

```text
EnMAP Product Structure:
┌─────────────────────────────────────────────────────────────────────┐
│  ENMAP01-____L1B-DT0000000000_20220501T101523Z_001_V010110_...     │
│  ├── *-METADATA.XML           ← Main metadata file (input)         │
│  ├── *-SPECTRAL_IMAGE_VNIR.TIF   420-1000 nm, ~88 bands            │
│  ├── *-SPECTRAL_IMAGE_SWIR.TIF   900-2450 nm, ~136 bands           │
│  ├── *-QL_QUALITY_CLOUD.TIF      Cloud mask                        │
│  ├── *-QL_QUALITY_CIRRUS.TIF     Cirrus mask                       │
│  ├── *-QL_QUALITY_SNOW.TIF       Snow mask                         │
│  ├── *-QL_QUALITY_HAZE.TIF       Haze mask                         │
│  └── *-QL_PIXELMASK_*.TIF        Per-sensor pixel masks            │
└─────────────────────────────────────────────────────────────────────┘
```

EnMAP differs from both: ships as **separate GeoTIFF files** with an XML metadata manifest. Importantly, **EnMAP L1B is already orthorectified** (map-projected) — no GLT, no per-pixel coords. Plus a set of RPCs (Rational Polynomial Coefficients) for refined geolocation if you need it.

### Class diagram

```text
File Structure:
┌────────────────────────────────────────────────────┐
│  METADATA.XML  ──→  wavelengths, FWHM, angles,    │
│                      gain/offset, RPCs             │
│                                                    │
│  SPECTRAL_IMAGE_VNIR.TIF  ──→  (88, H, W) bands   │
│  SPECTRAL_IMAGE_SWIR.TIF  ──→  (136, H, W) bands  │
│                                                    │
│  QL_QUALITY_*.TIF  ──→  quality masks              │
└────────────────────────────────────────────────────┘
```

The XML metadata file is the **input** — pass that path to the reader. The reader walks the sibling files for the actual band data and quality masks.

### Dual-sensor + radiometric calibration

```text
VNIR Detector                        SWIR Detector
┌────────────────────┐               ┌────────────────────┐
│ 420 - 1000 nm      │               │ 900 - 2450 nm      │
│ ~88 bands          │               │ ~136 bands         │
│ 6.5 nm sampling    │               │ 10 nm sampling     │
│ Si CCD             │               │ HgCdTe             │
└────────────────────┘               └────────────────────┘
          │                                    │
          └──────────── Overlap ───────────────┘
                     900-1000 nm
```

```text
Wavelength: 420nm ──── 1000nm ──── 2450nm
            ├── VNIR ────┤
                      ├──── SWIR ──────────┤
                      └ overlap┘
                      900-1000nm
```

VNIR has *finer* spectral sampling (6.5 nm vs 10 nm) than the other two sensors — useful for narrow-feature spectroscopy. The DN-to-radiance formula is the most peculiar:

```
L_λ = (GAIN × DN + OFFSET) × 1000   [mW/(m²·sr·nm)]

Note: DLR gains are multiplicative (not divisive as in some sensors)
```

The `× 1000` at the end is because DLR's spec gives the result in `W/(m²·sr·nm)` and the reader converts to `mW` units to match PRISMA / Thuillier conventions. **Don't double-multiply** — let the reader handle calibration.

### RPCs

```text
Pixel (col, row) ──→ RPC Transform ──→ Geographic (lon, lat)

RPCs model:
- Satellite orbit and attitude
- Sensor geometry  
- Terrain elevation effects (when height_off is set appropriately)
```

EnMAP includes Rational Polynomial Coefficients in metadata for fine geolocation. The reader can apply RPCs during loading for refined geolocation; default uses the simple affine transform from the GeoTIFF (which is already pretty good — RPCs add ~1 pixel of refinement).

### Interface

`EnMAPImage(metadata_xml_path, ...)` — the main class, takes the XML manifest path. Plus quality-mask accessors that pair with the per-pixel masks shipped alongside.

---

## 5. Side-by-side comparison

| | EMIT | PRISMA | EnMAP |
|---|---|---|---|
| **File format** | NetCDF (one) | HDF5 (one, `.he5`) | GeoTIFFs (many) + XML |
| **Georeferencing** | GLT lookup (fast) | per-pixel lat/lon (slow interp) | already map-projected (cheap) |
| **Radiance units** | `μW/(cm²·sr·nm)` | `mW/(m²·sr·nm)` | `W/(m²·sr·nm)` → `mW` after reader |
| **Bands** | 285 | 239 (66 VNIR + 173 SWIR) | 224 (88 VNIR + 136 SWIR) |
| **GSD** | ~60 m | 30 m | 30 m |
| **Sensor split** | single | VNIR + SWIR | VNIR + SWIR |
| **Module function used** | `griddata.georreference` | `griddata.read_to_crs` | regular `RasterioReader` |
| **Auth** | NASA Earthdata | manual download | manual download |
| **Reader cost (full scene)** | seconds | minutes (cubic interp) | seconds |

The cost asymmetry is the operational reality: PRISMA scenes are ~60× slower to ortho-load than EMIT. For workflows that read scenes repeatedly (parameter sweeps, hyperparameter searches), `load()` once into a `GeoTensor` and reuse — don't ortho-on-demand.

---

## 6. Common workflow patterns

### A — methane plume detection on EMIT

```python
from georeader.readers.emit import EMITImage
from georeader import reflectance, hyperspectral_index  # hypothetical

emit = EMITImage("EMIT_L1B_RAD_*.nc")
radiance = emit.load_radiance()         # uses GLT — fast
# (1, B=285, H, W) GeoTensor in μW/(cm²·sr·nm)
# Pass to a matched filter targeting ch4 absorption features...
```

### B — RGB visualization across sensors

```python
# EMIT
emit_rgb = emit.load_wavelengths([640, 550, 460], as_reflectance=True)

# PRISMA — raw=False forces gridding to a regular UTM grid
prisma_rgb = prisma.load_rgb(as_reflectance=True, raw=False)

# EnMAP
enmap_rgb = enmap.load_rgb(as_reflectance=True)
```

All three return a 3-band `GeoTensor` ready for `plot.show`. The interfaces deliberately converge despite the substrates diverging.

### C — Spectral binning to S2 bands

(Common to all three; used by `presets.enmap.ENMAP_TO_S2_BANDS` from the [`geotoolz` plan](../plans/geotoolz/geotoolz.md)):

```python
from georeader.readers.S2_SAFE_reader import read_srf
from georeader.reflectance import transform_to_srf, integrated_irradiance

s2_srf = read_srf(mission="S2A")                                  # S2 SRF DataFrame
s2_equiv = transform_to_srf(enmap_radiance, s2_srf, wavelengths=enmap.wavelengths)
# s2_equiv is now a 12-band cube with S2-shaped band responses
```

This recipe works identically for PRISMA and EMIT — the readers expose `wavelengths` and `fwhm` arrays as attributes.

---

## 7. Sharp edges

### EMIT
- **`μW/(cm²·sr·nm)` is unusual.** Convert by ÷100 to SI base or use `units="uW/cm^2/SR/nm"` to `radiance_to_reflectance` ([Chapter 11 §3](11_reflectance.md)). Wrong units silently produce wrong reflectance.
- **GLT requires the file's `location` group.** Some legacy EMIT products on third-party mirrors strip it. Check `glt_x` / `glt_y` exist before using `load_radiance()`.
- **NASA Earthdata auth.** Required for `download_product`. Stored in `~/.georeader/auth_emit.json` — make sure it's not world-readable (it's plaintext credentials).

### PRISMA
- **Cubic interpolation is the cost.** A full scene takes minutes. Use `raw=True` for radiometric work; only orthorectify when you need a regular grid for a downstream operator.
- **VNIR/SWIR overlap (920–1010 nm).** The reader doesn't deduplicate; you can ask for 950 nm and get a different value depending on which sensor. Be explicit about which you want when documenting analyses.
- **HDF5 path must end `.he5`.** Some catalog systems serve PRISMA as `.h5` — rename or symlink.

### EnMAP
- **Pass the `*-METADATA.XML` path, not the imagery path.** Easy to confuse since the SAFE-style folder contains many files.
- **`× 1000` in the radiance formula.** Don't apply it twice. The reader handles this; if you re-implement, factor it correctly.
- **RPCs are off by default.** Pass the appropriate flag to enable. Most analyses don't need them — the GeoTIFF affine is already accurate to ~10 m.

### Cross-sensor
- **Spectral sampling differs.** EMIT (~7–10 nm), PRISMA (~10 nm), EnMAP VNIR (6.5 nm) / SWIR (10 nm). Inter-sensor compatibility requires SRF binning (Chapter 11) — don't pixel-match across sensors directly.
- **Reflectance conversion needs solar geometry.** All three readers have helpers for sun angle / Earth-sun distance from acquisition timestamps; double-check this is set when calling `as_reflectance=True`.

---

## 8. Connection to `geotoolz`

The hyperspectral operators in [`geotoolz.md`](../plans/geotoolz/geotoolz.md) consume `GeoTensor`s produced by these readers:

- **`hyperspectral.MatchedFilter(target_spectrum, axis=0)`** — works on any of the three (post-orthorectification). Standard Reed-Yu detector.
- **`hyperspectral.ACEDetector` / `RXDetector` / `LinearUnmixing`** — same.
- **`presets.emit.EMIT_METHANE_MF`** — relies on EMIT's GLT speed: read once, MF many times.
- **`presets.enmap.ENMAP_TO_S2_BANDS`** — uses the SRF binning recipe (§6C above) for cross-sensor compatibility.

The split-object pattern in `geotoolz` ([Ch §4 of geotoolz.md](../plans/geotoolz/geotoolz.md)) is especially relevant here: compute the scene mean / covariance / endmember spectrum once (slow), then apply per-band detectors fast. The hyperspectral cube is the canonical case where state-as-artifact pays off.

Next chapter: [16_earth_engine.md](16_earth_engine.md) — Google Earth Engine integration (export tile splitting and parallel download).
