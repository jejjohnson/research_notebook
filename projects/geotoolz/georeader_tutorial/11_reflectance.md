---
title: reflectance
subject: georeader tutorial
subtitle: Radiometry, SRFs, and irradiance
short_title: Ch. 11 — Reflectance
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: tutorial, georeader, radiometry, srf
---

> **Module:** `georeader/reflectance.py` (971 LOC, 97 box-drawing characters — third densest in the package)
> **Role:** convert satellite imagery between physically meaningful radiometric quantities — radiance, top-of-atmosphere (ToA) reflectance, and band-integrated irradiance. Where the package crosses from "geospatial bookkeeping" into actual physics.

---

## 1. The job

Satellite imagery is delivered in any of three radiometric units depending on the sensor / processing level:

- **Digital numbers (DN)** — raw counts, sensor-specific, dimensionless. You almost never want to work with these.
- **Radiance (L)** — `W / m² / sr / nm` — the actual photon flux hitting the sensor, per unit area, per unit solid angle, per wavelength. Physical, directly comparable across sensors *if* you know the geometry.
- **Top-of-atmosphere reflectance (ρ)** — dimensionless, typically ∈ `[0, 1]` — the fraction of incoming solar radiation that the surface reflected back, **before** atmospheric correction. The starting point for most ML pipelines because it normalises out solar-illumination effects (sun angle, Earth-sun distance).

This module provides:

- The **conversion** between the three (`radiance_to_reflectance` / `reflectance_to_radiance`).
- The **geometric corrections** that go into them (Earth-sun distance, solar zenith angle).
- The **spectral integration** that maps hyperspectral radiance to multispectral bands via spectral response functions (SRFs).
- A **bundled solar spectrum** (Thuillier 2003) used as the default `E_sun(λ)`.

---

## 2. Unit conversion overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    RADIOMETRIC UNIT CONVERSION FLOW                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Raw DN ──────►  Radiance ──────────────────────────►  Reflectance     │
│   (counts)        (W/m²/sr/nm)                          (unitless 0-1)  │
│                                                                         │
│   Supported radiance units:                                             │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │  Unit              │  Factor to W/m²/sr/nm                     │    │
│   ├────────────────────┼───────────────────────────────────────────┤    │
│   │  W/m²/sr/nm        │  1.0         (no conversion)              │    │
│   │  mW/m²/sr/nm       │  ÷ 1000      (milli → base)               │    │
│   │  µW/cm²/sr/nm      │  ÷ 100       (micro/cm² → base/m²)        │    │
│   └────────────────────┴───────────────────────────────────────────┘    │
│                                                                         │
│   Solar Irradiance: W/m²/nm or mW/m²/nm (at TOA, perpendicular)         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The DN→radiance step is **sensor-specific** (linear scale + offset from per-band calibration coefficients) and lives inside the per-sensor reader, not this module. Once you have radiance, this module handles the rest.

The three supported radiance unit systems all ship in real-world products:

- `W/m²/sr/nm` — Sentinel-2 L1C after applying the calibration coefficients.
- `mW/m²/sr/nm` — common in legacy products, factor of 1000 larger.
- `µW/cm²/SR/nm` — the EMIT and PRISMA standard, factor of 100 larger than SI base.

The function takes a `units=` string and applies the conversion internally, so user code stays in whatever units the sensor delivered.

---

## 3. The reflectance equation

The ToA reflectance formula:

```text
ρ = (π × d² × L) / (E_sun × cos(θ_z))

where:
- L      = at-sensor radiance (W/m²/sr/nm)
- E_sun  = solar irradiance at TOA (W/m²/nm)
- d      = Earth-Sun distance in AU (varies ~3% annually)
- θ_z    = solar zenith angle (0° = Sun overhead)
```

Three corrections combined:

- **`π`** — convert from solid-angle-aware radiance to a Lambertian-equivalent reflectance.
- **`d²`** — Earth-sun distance correction. Earth's orbit is elliptical; in January (perihelion) we're closer to the Sun and get more incoming radiation than in July (aphelion).
- **`cos(θ_z)`** — solar zenith correction. When the Sun is low in the sky, the same surface element intercepts less radiation per unit horizontal area.

The module factors them into `obfactor = π × d² / cos(θ_z)` — `observation_date_correction_factor` — so the reflectance line collapses to `ρ = L × obfactor / E_sun`.

```text
┌─────────────────────────────────────────────────────────────────┐
│  UNIT CONVERSION FLOW                                            │
│                                                                   │
│  Input radiance        Normalized radiance      Output           │
│  (various units)   →   (W/m²/sr/nm)         →   reflectance     │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Input Unit       │ factor_div │ Conversion                  │ │
│  ├──────────────────┼────────────┼─────────────────────────────┤ │
│  │ W/m²/sr/nm       │ 1          │ No conversion               │ │
│  │ mW/m²/sr/nm      │ 1000       │ ×10⁻³ (milli → base)       │ │
│  │ µW/cm²/sr/nm     │ 100        │ ×10⁻⁶×10⁴ = ×10⁻²         │ │
│  └──────────────────┴────────────┴─────────────────────────────┘ │
│                                                                   │
│  Final calculation:                                              │
│                                                                   │
│  L [W/m²/sr/nm] × obfactor [sr⁻¹] / E_sun [W/m²/nm]            │
│  = dimensionless reflectance                                     │
│                                                                   │
│  Note: The steradian cancels with implicit assumptions about     │
│  the solar disk's solid angle as seen from Earth.               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Earth-sun distance

```text
┌────────────────────────────────────────────────────────────────────┐
│       Earth-Sun Distance Throughout the Year                       │
│                                                                    │
│  Distance   ▲                                                      │
│  (AU)       │     Aphelion (~July 4)                               │
│             │          ╭───────╮                                   │
│  1.017 ─────┼─────────╱         ╲─────────────────────────         │
│             │        ╱           ╲                                 │
│  1.000 ─────┼───────╱─────────────╲──────────────────────          │
│             │      ╱               ╲                               │
│  0.983 ─────┼─────╱─────────────────╲────────────────────          │
│             │    ╱    Perihelion     ╲                             │
│             │   ╱     (~Jan 3)        ╲                            │
│             └───┴──────────────────────┴────────────────► Day      │
│                 0    91   182   273   365                          │
│                Jan  Apr   Jul   Oct   Jan                          │
│                                                                    │
│  d = 1 - 0.01673 × cos(0.0172 × (day_of_year - 4))                 │
│                                                                    │
│  Impact: ~6.5% variation in irradiance (d² factor)                 │
└────────────────────────────────────────────────────────────────────┘
```

The annual ≈ ±1.7% variation in distance becomes ≈ ±3.4% in `d²` and therefore ≈ 6.5% peak-to-peak in irradiance over the year. **Ignoring this correction biases reflectance time series**: you'd see a fake annual cycle of ~3% amplitude with peaks in the Northern winter (smaller `d`, more apparent radiance, higher uncorrected reflectance).

The closed-form expression `d = 1 - 0.01673 × cos(0.0172 × (DOY − 4))` is what `earth_sun_distance_correction_factor(date_of_acquisition)` returns. It's accurate to ~0.001 AU — well below the precision needed for radiometric calibration.

---

## 5. Spectral response functions

```text
┌─────────────────────────────────────────────────────────────────┐
│  Spectral Response Function Convolution                          │
│                                                                   │
│  Hyperspectral               SRF for Band X             Result   │
│  Radiance L(λ)               R(λ)                       L_X     │
│                                                                   │
│  L(λ)│     ╱╲                R(λ)│   ╱╲                         │
│      │    ╱  ╲╱╲╱╲              │  ╱  ╲                         │
│      │   ╱        ╲             │ ╱    ╲                        │
│      │  ╱          ╲            │╱      ╲                       │
│      └──────────────── λ        └──────────── λ                 │
│            400-2500 nm              λ_center ± FWHM/2            │
│                                                                   │
│  Integration:  L_X = ∫ L(λ) × R(λ) dλ  /  ∫ R(λ) dλ             │
│                                                                   │
│  The SRF is typically Gaussian:                                  │
│  R(λ) = exp(-(λ - λ_center)² / (2σ²))                           │
│  where σ = FWHM / (2 × √(2 × ln(2))) ≈ FWHM / 2.355             │
└─────────────────────────────────────────────────────────────────┘
```

When you want to take a hyperspectral measurement (~285 narrow bands) and turn it into a multispectral measurement (~12 wide bands matching another sensor's bands), you **integrate** the hyperspectral spectrum against the target sensor's per-band SRF.

Two functions own this:

- **`srf(center_wavelengths, fwhm, wavelengths)`** — build a Gaussian SRF DataFrame from per-band centre/FWHM specs. Returns shape `(N_wavelengths, K_bands)` normalized so each column sums to 1.
- **`transform_to_srf(hyperspectral_data, srf, wavelengths)`** — apply the SRF: produce a `(K_bands, H, W)` multispectral cube from a `(N_wavelengths, H, W)` hyperspectral cube.

Why Gaussian? Most published SRFs are well-approximated by Gaussians, and Gaussian SRFs are uniquely specified by `(centre, FWHM)`. The exact published SRFs (S2, Landsat, etc.) tend to have ~1% non-Gaussian deviations; for most ML applications the difference doesn't matter, but for radiative-transfer studies you'd want to use the exact published curves rather than a Gaussian approximation.

The `2 × √(2 × ln(2)) ≈ 2.355` constant in the formula is the FWHM-to-σ ratio for a Gaussian — comes up again in any "spread parameter" discussion.

---

## 6. Band-integrated irradiance

```text
┌────────────────────────────────────────────────────────────────┐
│  Spectral Integration Process                                   │
│                                                                  │
│  E_sun(λ)          R(λ)               E_sun(λ) × R(λ)          │
│  (Solar)           (SRF)              (Product)                 │
│                                                                  │
│    │╲              │ ╱╲                 │  ╱╲                   │
│    │ ╲╲            │╱  ╲                │ ╱  ╲                  │
│    │  ╲╲╲          │    ╲               │╱    ╲                 │
│    │   ╲╲╲╲        │     ╲              │      ╲                │
│    └────────λ      └──────λ            └────────λ              │
│                                         ████████ ← Area = E_k  │
│                                                                  │
│  Solar spectrum   Band response    Weighted → integrate & norm  │
│  (~200-2500 nm)   (Gaussian)       gives band-effective E_sun   │
└────────────────────────────────────────────────────────────────┘
```

Reflectance computation needs `E_sun` **per band** — a single number that summarises the solar spectrum as seen by that band's SRF. The closed-form:

```
E_k = ∫ E_sun(λ) × R_k(λ) dλ  /  ∫ R_k(λ) dλ
```

`integrated_irradiance(srf, solar_irradiance=None, epsilon_srf=1e-4)` computes the integral per band. If `solar_irradiance=None`, it loads the **Thuillier (2003) reference spectrum** bundled with the package as `SolarIrradiance_Thuillier.csv` (Solar Physics 214). The Thuillier spectrum covers 200–2400 nm at ~1 nm resolution — fine enough for any current orbital sensor.

The `epsilon_srf` threshold zeroes out SRF values below a tiny floor — important because Gaussian SRFs technically extend forever, and the integration would otherwise pick up out-of-band noise in `E_sun(λ)`.

`load_thuillier_irradiance()` returns the cached DataFrame with columns `["Nanometer", "Radiance(mW/m2/nm)"]`. The `THUILLIER_RADIANCE = None` module-level cache means it's loaded lazily on first use.

---

## 7. The function reference

**Conversion**
- `radiance_to_reflectance(data, solar_irradiance, date_of_acquisition=None, center_coords=None, crs_coords=None, observation_date_corr_factor=None, units="W/m2/sr/nm")` → `GeoTensor`
- `reflectance_to_radiance(data, solar_irradiance, ..., units="W/m2/sr/nm")` → `GeoTensor` — exact inverse

**Geometric correction factors**
- `earth_sun_distance_correction_factor(date_of_acquisition: datetime) → float` (returns `d`, not `d²`)
- `compute_sza(center_coords, date_of_acquisition, crs_coords=None) → float` (degrees)
- `observation_date_correction_factor(center_coords, date_of_acquisition, crs_coords=None) → float` (combined `π × d² / cos(θ_z)`)

**Spectral integration**
- `srf(center_wavelengths, fwhm, wavelengths) → NDArray` — build Gaussian SRF
- `integrated_irradiance(srf, solar_irradiance=None, epsilon_srf=1e-4) → NDArray` — band-integrated E_sun
- `transform_to_srf(hyperspectral_data, srf, wavelengths) → GeoTensor / NDArray` — apply SRF to a hyperspectral cube

**Solar reference spectrum**
- `load_thuillier_irradiance() → pd.DataFrame` — load and cache Thuillier (2003)
- `THUILLIER_RADIANCE` — module-level cache (None until first load)

---

## 8. Two idiomatic uses

**A. Sentinel-2 DN → reflectance (already calibrated; needs only TOA correction):**

```python
# Suppose s2_radiance is a (B, H, W) GeoTensor in W/m²/sr/nm
import datetime as dt
toa_refl = reflectance.radiance_to_reflectance(
    data=s2_radiance,
    solar_irradiance=esun_per_band,            # (B,) array of W/m²/nm
    date_of_acquisition=dt.datetime(2024, 6, 15, 10, 30, tzinfo=dt.timezone.utc),
    units="W/m2/sr/nm",
)
# toa_refl is a (B, H, W) GeoTensor in [0, 1]
```

The `solar_irradiance` argument is what lets you specialise to *any* sensor — pass S2's per-band `E_sun` for S2, Landsat's for Landsat. The reader for each sensor typically ships these constants as a module-level array.

**B. Hyperspectral EMIT → S2-equivalent multispectral:**

```python
# emit_radiance: (285, H, W) hyperspectral cube; wavelengths: (285,) array
# Build S2 SRF (12 bands) at EMIT wavelengths
s2_centers = np.array([443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190])
s2_fwhm    = np.array([20,  65,  35,  30,  15,  15,  20,  115, 20,  20,  90,   180])
s2_srf = reflectance.srf(s2_centers, s2_fwhm, wavelengths=emit_wavelengths)

# Apply
s2_equivalent = reflectance.transform_to_srf(emit_radiance, s2_srf, wavelengths=emit_wavelengths)
# s2_equivalent: (12, H, W) — EMIT data convolved into S2-shaped bands

# Per-band irradiance for the new bands
s2_E = reflectance.integrated_irradiance(pd.DataFrame(s2_srf, index=emit_wavelengths))
```

This is the "spectral response binning" preset that the [`geotoolz.md` plan](../plans/geotoolz/geotoolz.md) mentions for `presets.enmap.ENMAP_TO_S2_BANDS` — same pattern, applied to EnMAP data.

---

## 9. Sharp edges

- **`solar_irradiance` units must be `W/m²/nm`, not `mW/m²/nm`.** Even when the input radiance is in mW units. The function normalises radiance internally; it does not normalise the solar input. Mismatched scaling here is the most common bug, off by a factor of 1000.
- **`center_coords` for solar zenith.** If `data` is a `GeoTensor`, the function can derive scene centre from `transform`. If it's a plain ndarray, you must pass `center_coords` explicitly. Forgetting this with a non-GeoTensor input gives a silent mis-correction.
- **`crs_coords` defaults to EPSG:4326.** If your `center_coords` are in UTM, pass `crs_coords="EPSG:32630"` (or whichever zone). Otherwise the SZA computation places your scene at lon=500000, lat=4500000 — somewhere in deep space.
- **`observation_date_corr_factor` shortcut.** Pass it pre-computed and the function skips date and centre coords. Useful for batched processing where you've calibrated `obfactor` once and want to apply it to N tiles cheaply.
- **Thuillier is in `mW/m²/nm`, not `W/m²/nm`.** Read the column name. If you pass it directly as `solar_irradiance` to `radiance_to_reflectance`, you'll be off by 1000.
- **`integrated_irradiance` returns same units as input solar spectrum.** Mixing Thuillier (`mW`) with a band-integrated value used as `W` argument is the dominant unit confusion. Convert explicitly.
- **`srf(center, fwhm, wavelengths)` doesn't normalise to unit area.** The `transform_to_srf` and `integrated_irradiance` functions do their own normalisation internally, but if you take the SRF DataFrame and use it elsewhere you may need to divide by `np.trapz(R, wavelengths)`.

---

## 10. Why this module is denser than its size

971 LOC, 97 box-drawing characters — the third densest in the package because **physics requires illustration**. The Earth-sun distance plot, the SRF convolution diagram, and the integration visual are doing real explanatory work that prose alone wouldn't. If you only keep one chapter's diagrams from this whole tutorial, this is one of the candidates — they're the ones that explain the *why* not just the *what*.

---

## 11. Connection to `geotoolz`

Three concrete operator-shapes from [`geotoolz.md`](../plans/geotoolz/geotoolz.md) wrap functions in this module:

- **`correction.TOAToBOA(sun_zenith=..., atmosphere=...)`** — wraps `radiance_to_reflectance` plus an atmospheric correction step. The radiance→ToA part is what this module already does.
- **`radiometry.SRFBin(target_centres, target_fwhm)`** — wraps `srf` + `transform_to_srf` to take a hyperspectral cube and bin it to a target sensor's bands. The basis for the `EnMAP → S2` and `EMIT → S2` preset operators.
- **`presets.s2.S2_L1C_TO_BOA_NDVI(...)`** — a `Sequential` that includes `radiance_to_reflectance` as one of its steps when starting from L1A radiance products.

The module is also implicitly used inside `readers.emit`, `readers.prisma`, and `readers.enmap` — those readers do DN → radiance internally (per their per-sensor calibration coefficients) and expose the radiance-units result. The reflectance step is then a one-line call to this module.

Next chapter: [12_save.md](12_save.md) — writing GeoTensors to disk; full Cloud-Optimized GeoTIFF (COG) anatomy.
