---
title: MODIS / curvilinear readers
subject: Sensor readers
subtitle: MODIS, VIIRS, AVHRR, Sentinel-3 OLCI / SLSTR
short_title: MODIS
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, readers, modis, viirs
---

> **Design Report** — design for MODIS, VIIRS, and related curvilinear-geolocation readers in [`spaceml-org/georeader`](https://github.com/spaceml-org/georeader).
> Companion to the geostationary readers design.

## Contents

- [User Story](#user-story)
- [Motivation](#motivation)
- [Mathematics: The Curvilinear Transformation](#mathematics-the-curvilinear-transformation)
- [Target API](#target-api) - [MODIS L1B reader](#modis-l1b-reader) - [VIIRS L1B reader](#viirs-l1b-reader) - [Variations within the family](#variations-within-the-family)
- [Example Use Cases](#example-use-cases)
- [Subtasks](#subtasks)
- [Open Design Question](#open-design-question)

## User Story

I'm continuing the migration from `rs_tools` into `georeader` with the next family of sensors: **MODIS** on Aqua/Terra, plus its close relatives **VIIRS** (S-NPP, NOAA-20/21), and eventually **AVHRR**, **Sentinel-3 OLCI/SLSTR**, and airborne **AVIRIS-NG**.
Unlike GOES/SEVIRI, none of these have a clean affine in any standard CRS — they're polar-orbiting (or airborne) curvilinear scanners whose native geometry is "this is the lat/lon of every single pixel." The PRISMA reader is the right precedent and `griddata.read_to_crs` is the right resampler, but the reader has to set things up carefully so the curvilinear → regular-grid step works correctly, especially around bowtie pixels, multi-resolution bands, and dateline crossings.

## Motivation

MODIS is the foundational dataset of the modern remote-sensing era — 25 years of daily global coverage, 36 spectral bands, three native resolutions (250 m / 500 m / 1 km), and an enormous downstream Level-2 product family (MOD09 surface reflectance, MOD11 LST, MOD13 vegetation indices, MOD14 fires, ...).
VIIRS is the operational successor (similar bands, similar geometry, NetCDF instead of HDF4).
Both ship as files where the *only* honest description of geometry is per-pixel `lons`/`lats`; any attempt to fit an affine to them is a lie at the swath edges.

| Sensor | Format | Geolocation | Native res. | Notes |
|---|---|---|---|---|
| MODIS L1B (Aqua/Terra) | HDF4 (HDF-EOS) | Separate `MOD03`/`MYD03` at 1 km; in-file at 500 m/250 m | 250 m / 500 m / 1 km | Bowtie at scan edges; 36 bands |
| VIIRS SDR (S-NPP, NOAA-20/21) | HDF5 / NetCDF | Companion `GIMGO` / `GMTCO` files | 375 m (I) / 750 m (M) | Bowtie aware; 22 bands |
| AVHRR (NOAA, MetOp) | L1B binary / NetCDF | In-file | ~1.1 km | Older format; defer |
| Sentinel-3 OLCI/SLSTR | NetCDF | In-file `geo_coordinates` | 300 m / 500 m / 1 km | Push-broom but still curvilinear; defer |
| AVIRIS-NG (airborne) | ENVI binary | Separate `IGM`/`GLT` files | sub-meter to ~10 m | Very long swaths; defer |

`rs_tools` solved download and patching for MODIS but not the read-arbitrary-AOI problem; `satpy` solves both at the cost of a heavy dependency tree.
The georeader-shaped equivalent is one reader per sensor following the PRISMA pattern, plus a small set of curvilinear utilities to handle quirks that PRISMA doesn't have.

The reader's job is, again, narrow: open the HDF/NetCDF files, parse calibration, link the geolocation file (when separate), expose `.lons`/`.lats` plus calibrated arrays and metadata, and let `griddata.read_to_crs` do the rest.
The new things this design has to address that the GEO design didn't:

1. **Multi-file products** (data file + separate geolocation file)
2. **The bowtie effect** (overlapping pixels at scan edges)
3. **Multi-resolution geolocation** (1 km comes from `MOD03`; 500 m / 250 m must be derived or upsampled)
4. **Antimeridian crossings** (lons jump from +180 to -180 mid-swath)
5. **The actual curvilinear → regular-grid resampling** in the middle

## Mathematics: The Curvilinear Transformation

The forward problem — pixel index $(i, j)$ → $(\varphi_{ij}, \lambda_{ij})$ — is given by the file's geolocation arrays.
The backward problem is the reprojection that `griddata.read_to_crs` does, and it's worth being explicit about what's happening, because the choice of resampling method interacts with bowtie pixels in ways that matter.

Let `V[i, j, k]` be the source array of shape `(H, W, C)`, with `lons[i, j]` and `lats[i, j]` giving the geographic location of pixel `(i, j)`.
For a target CRS and a target resolution `Δ`, we want a regular grid `V_dst[m, n, k]` at locations `(x_m, y_n)` in target-CRS meters.
The transformation has three steps:

```python
# Pseudocode for what griddata.read_to_crs is doing under the hood

# 1. Project source points: (φ, λ) → (x_src, y_src) in dst_crs meters
transformer = pyproj.Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
x_src, y_src = transformer.transform(lons, lats)            # (H, W) each

# 2. Build target grid (regular in dst_crs)
x_min, y_min, x_max, y_max = bounds_dst or auto_bounds(x_src, y_src)
x_grid = np.arange(x_min, x_max, resolution_dst)
y_grid = np.arange(y_max, y_min, -resolution_dst)
X_dst, Y_dst = np.meshgrid(x_grid, y_grid)                  # (H', W') each

# 3. Scattered → regular interpolation. Two common kernels:

#    (a) nearest neighbor via KD-tree
tree = scipy.spatial.cKDTree(np.column_stack([x_src.ravel(), y_src.ravel()]))
_, idx = tree.query(np.column_stack([X_dst.ravel(), Y_dst.ravel()]), k=1)
V_dst = V.reshape(-1, C)[idx].reshape(X_dst.shape + (C,))

#    (b) linear via Delaunay triangulation
V_dst = scipy.interpolate.griddata(
    points=np.column_stack([x_src.ravel(), y_src.ravel()]),
    values=V.reshape(-1, C),
    xi=(X_dst, Y_dst),
    method="linear",
)
```

Three numerical hazards show up here that the reader has to either fix or document:

> **Bowtie duplicates.** The source point cloud has multiple samples for the same ground location at scan edges.
> Nearest-neighbor handles this gracefully (one of the duplicates wins arbitrarily) but Delaunay-based linear interpolation can produce visible seams where degenerate triangles meet.

> **Antimeridian crossings.** A swath might have lons of +179.8° and -179.7° in adjacent pixels; if you project naively to a CRS that doesn't wrap (Web Mercator, UTM), the swath gets stretched halfway around the world.
> The reader handles this by detecting the discontinuity in `.lons` and either splitting the swath or unwrapping (e.g., `lons[lons < 0] += 360` when in the eastern-hemisphere tile).

> **Off-disk / fill values.** Sentinels like `-999` in `.lats`/`.lons` must be masked before the KD-tree is built, otherwise they pull nearest-neighbor queries into nonsense.

The reader does **not** implement the resampling — `griddata.read_to_crs` does — but it owns making sure the inputs to that function are clean: valid masks applied, antimeridian unwrapped or flagged, and bowtie status known.
As a small utility module:

```python
# georeader/readers/curvilinear/_swath.py — internal, sensor-agnostic

def crosses_antimeridian(lons: NDArray) -> bool:
    """True if the lon array contains a ±180° discontinuity."""
    return np.any(np.abs(np.diff(lons, axis=1)) > 180)


def unwrap_antimeridian(lons: NDArray) -> NDArray:
    """Add 360 to negative lons if the swath straddles the dateline.
    Returns lons with no discontinuity; downstream CRS must handle it."""
    if crosses_antimeridian(lons):
        return np.where(lons < 0, lons + 360, lons)
    return lons


def bowtie_mask(scan_size: int, n_scans: int, edge_rows: int = 2) -> NDArray:
    """True for pixels that are bowtie duplicates (scan edge rows).
    scan_size = lines per scan (10 for MODIS 1 km, 16 for VIIRS M-band)."""
    ...


def dedupe_bowtie(
    values: NDArray, lons: NDArray, lats: NDArray, scan_size: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Drop the bowtie duplicate rows. Returns reduced (values, lons, lats)."""
    ...
```

Users never call these directly.
Readers call them from `__init__` or expose them as opt-in methods.

## Target API

The MODIS L1B reader follows the PRISMA template precisely.
A separate geolocation file (`MOD03`/`MYD03`) is the norm at 1 km; for 500 m and 250 m the geolocation either lives inside the data file or must be derived by interpolation, and the reader handles both cases.

### MODIS L1B reader

```python
# georeader/readers/modis.py
from datetime import datetime
from numpy.typing import NDArray
from typing import Literal


class MODIS_L1B:
    """MODIS Level 1B calibrated radiances/reflectances.

    Handles MOD021KM/MYD021KM (1 km), MOD02HKM/MYD02HKM (500 m),
    and MOD02QKM/MYD02QKM (250 m). Geolocation from MOD03/MYD03.
    """

    # --- per-pixel geolocation (PRISMA convention) ---
    lons: NDArray                               # (H, W) degrees
    lats: NDArray                               # (H, W) degrees

    # --- viewing/solar geometry (free in MODIS, useful downstream) ---
    solar_zenith: NDArray                       # (H, W) degrees
    solar_azimuth: NDArray
    sensor_zenith: NDArray
    sensor_azimuth: NDArray

    # --- MODIS metadata ---
    bounds: tuple[float, float, float, float]   # in EPSG:4326 from lons/lats
    time: datetime                              # granule midpoint, tz-aware
    time_start: datetime
    time_end: datetime
    satellite: Literal["Terra", "Aqua"]
    sensor: str = "MODIS"
    resolution: Literal["1km", "500m", "250m"]
    bands: list[str]                            # e.g. ['B01', 'B02', 'B26', 'B31']
    wavelengths: dict[str, float]               # central wavelengths in µm
    units: str                                  # 'W m-2 sr-1 um-1' for radiance
    crosses_antimeridian: bool
    scan_size: int                              # 10 (1 km), 20 (500 m), 40 (250 m)

    def __init__(
        self,
        path: str,                              # MOD021KM (or HKM/QKM)
        geo_path: str | None = None,            # MOD03; if None, look inside path
        bands: list[str] | None = None,
        unwrap_antimeridian: bool = True,
    ): ...

    # --- load methods (PRISMA-style) ---
    def load_raw(self, *, calibrated: bool = True) -> NDArray:
        """(C, H, W) array; if calibrated, in self.units; else raw DN."""

    def load_band(self, band: str, *, calibrated: bool = True) -> NDArray: ...

    def to_reflectance(self) -> NDArray:
        """Solar bands → TOA reflectance, dividing by cos(solar_zenith)."""

    def to_brightness_temperature(self) -> NDArray:
        """Thermal bands (20–25, 27–36) → BT via Planck inversion."""

    # --- bowtie handling (MODIS-specific) ---
    def dedupe_bowtie(self) -> "MODIS_L1B":
        """Return a new reader with bowtie duplicate rows removed."""

    @property
    def bowtie_mask(self) -> NDArray: ...
```

Construction and use mirror PRISMA exactly:

```python
from georeader.readers import modis
from georeader import griddata
import numpy as np

obj = modis.MODIS_L1B(
    path="MYD021KM.A2024165.1855.061.2024166094021.hdf",
    geo_path="MYD03.A2024165.1855.061.2024166081542.hdf",
    bands=["B01", "B02", "B26", "B31"],     # red, NIR, cirrus, 11 µm
)
print(obj)
# MODIS_L1B: Aqua  res=1km  t=2024-06-13T18:55:00+00:00
#   bands: ['B01', 'B02', 'B26', 'B31']  shape: (4, 2030, 1354)
#   bounds: (-12.4, 35.6, 18.9, 58.3)  crosses_antimeridian: False
#   scan_size: 10 (bowtie aware)

raw = obj.load_raw(calibrated=True)         # (4, H, W) in W m-2 sr-1 um-1
geo = griddata.read_to_crs(
    np.moveaxis(raw, 0, 2),
    lons=obj.lons, lats=obj.lats,
    resolution_dst=1000, dst_crs="EPSG:3035",   # ETRS89 / LAEA Europe
    method="nearest",                            # bowtie-safe
)
# geo: GeoTensor (4, H', W') in EPSG:3035
```

### VIIRS L1B reader

VIIRS follows the same pattern with sensor-specific tweaks (the I-band/M-band split mirrors MODIS resolution groups; geolocation lives in `GMTCO`/`GIMGO` companion files):

```python
# georeader/readers/viirs.py
from typing import Literal


class VIIRS_L1B:
    """VIIRS SDR (Sensor Data Record) reader for I-bands (375 m) and M-bands (750 m)."""

    lons: NDArray                               # from GIMGO (I) or GMTCO (M)
    lats: NDArray
    solar_zenith: NDArray
    solar_azimuth: NDArray
    sensor_zenith: NDArray
    sensor_azimuth: NDArray

    satellite: Literal["S-NPP", "NOAA-20", "NOAA-21"]
    sensor: str = "VIIRS"
    resolution: Literal["I", "M"]               # 375 m or 750 m
    scan_size: int                              # 32 (I), 16 (M)
    bands: list[str]                            # e.g. ['I04', 'I05', 'M11']
    # ... rest same shape as MODIS_L1B
```

### Variations within the family

For Level-2 MODIS products (`MOD09`, `MOD11`, `MOD13`), the same reader pattern applies but the bands change and the calibration helpers (`to_reflectance` / `to_brightness_temperature`) become identity passes — the products are already in physical units.
Implement these as separate small classes (`MODIS_L2_SurfaceReflectance`, `MODIS_LST`) that share a base mixin if duplication appears.

For **Sentinel-3 OLCI/SLSTR** (NetCDF, push-broom but still curvilinear due to Earth rotation during the scan), the reader is structurally identical: open file, expose `.lons`/`.lats`, calibrate, route through `griddata.read_to_crs`.
Defer until MODIS and VIIRS land.

For **AVIRIS-NG** (airborne, ENVI format with separate IGM/GLT geolocation files), again structurally identical — the only difference is the file-parsing layer.

## Example Use Cases

### 1. MODIS true color over a methane site, reprojected to UTM

```python
from georeader.readers import modis
from georeader import griddata
import numpy as np

obj = modis.MODIS_L1B(
    "MYD021KM.A2024165.1855.061.....hdf",
    geo_path="MYD03.A2024165.1855.061.....hdf",
    bands=["B01", "B04", "B03"],            # red, green, blue
)
rgb = obj.to_reflectance()                  # (3, H, W)
out = griddata.read_to_crs(
    np.moveaxis(rgb, 0, 2),
    lons=obj.lons, lats=obj.lats,
    resolution_dst=1000, dst_crs="EPSG:32613",
    method="nearest",
)
```

### 2. Bowtie-aware processing for cloud detection

```python
obj = modis.MODIS_L1B(path, geo_path, bands=["B01", "B02", "B06", "B26", "B31"])
obj_clean = obj.dedupe_bowtie()              # rows reduced, no duplicates

# now linear interpolation in griddata.read_to_crs is safe
geo = griddata.read_to_crs(
    np.moveaxis(obj_clean.load_raw(), 0, 2),
    lons=obj_clean.lons, lats=obj_clean.lats,
    resolution_dst=1000, dst_crs="EPSG:3857",
    method="linear",
)
```

### 3. MODIS–Sentinel-2 collocation for fusion

```python
from georeader.readers import S2_SAFE_reader, modis
from georeader import griddata
import numpy as np

s2 = S2_SAFE_reader.s2loader(s2_path, out_res=60, bands=["B04", "B08", "B11"])
m  = modis.MODIS_L1B(modis_path, modis_geo_path, bands=["B01", "B02", "B06"])
m_on_s2 = griddata.read_to_crs(
    np.moveaxis(m.to_reflectance(), 0, 2),
    lons=m.lons, lats=m.lats,
    resolution_dst=s2.transform.a,           # match S2 grid
    dst_crs=s2.crs,
    bounds_dst=s2.bounds,
)
```

### 4. VIIRS at high latitudes with antimeridian crossing

The reader auto-unwraps `.lons` so downstream resampling doesn't fail:

```python
from georeader.readers import viirs
from georeader import griddata

v = viirs.VIIRS_L1B(path, geo_path, bands=["M15"], unwrap_antimeridian=True)
print(v.crosses_antimeridian)                # True
print(v.lons.max(), v.lons.min())            # 195.3, 174.2 (unwrapped)

geo = griddata.read_to_crs(
    v.load_raw(),
    lons=v.lons, lats=v.lats,
    resolution_dst=750, dst_crs="EPSG:3413",  # NSIDC north polar
)
```

## Subtasks

The work splits into a small curvilinear utility module, the per-sensor readers, and notebooks.

### Curvilinear utilities (~2 days)

`georeader/readers/curvilinear/_swath.py` with `crosses_antimeridian`, `unwrap_antimeridian`, `bowtie_mask`, `dedupe_bowtie`, plus tests.
These are sensor-agnostic and the only piece of new shared infrastructure.
Whether to expose any of these publicly is a design decision; start internal-only and promote if external users need them.

### MODIS L1B reader (~2 weeks)

The first concrete reader.
HDF4 parsing through `pyhdf` (the only mature HDF4 binding); calibration coefficients live in attributes per-band; geolocation comes from the linked `MOD03` file at 1 km, and from in-file SDS at 500 m / 250 m.
Three resolution variants share the class and dispatch on file name.
Tests against a small fixture granule in CI. The bowtie deduplication is the one MODIS-specific footgun and gets its own test.

### MODIS L2 readers (~1 week each)

`MODIS_L2_SurfaceReflectance` (`MOD09`), `MODIS_LST` (`MOD11`), `MODIS_VegetationIndex` (`MOD13`).
These are thinner because calibration is a no-op, but each has its own band list and quality-flag conventions.
Build only the ones I need first, generalize later.

### VIIRS L1B reader (~2 weeks)

HDF5 / NetCDF instead of HDF4 (cleaner).
I-band and M-band variants in one class with a `resolution` switch like MODIS. Geolocation from `GIMGO` / `GMTCO` companion files.
Tests against a small fixture.

### Sentinel-3 OLCI/SLSTR, AVHRR, AVIRIS-NG (~1–2 weeks each)

**Defer** until MODIS + VIIRS are landed.

### Notebooks (~1 week)

Three in the established style:

- `modis_quickstart.ipynb` mirroring `prisma_with_cloudsen12.ipynb` (open file, calibrate, run cloud detection, compare to S2)
- `viirs_quickstart.ipynb`
- `modis_s2_collocation.ipynb` showing the cross-sensor fusion workflow

### Staged release

- [ ] **PR 1.** Curvilinear utilities + MODIS L1B + MODIS quickstart notebook (cohesive, end-to-end)
- [ ] **PR 2.** VIIRS L1B + VIIRS quickstart notebook
- [ ] **PR 3.** MODIS L2 family (`MOD09`, `MOD11`, `MOD13`)
- [ ] **PR 4.** Sentinel-3 OLCI/SLSTR
- [ ] **PR 5.** AVHRR
- [ ] **PR 6.** AVIRIS-NG

## Open Design Question

The single biggest open design question, worth resolving before locking anything down:

> **Where does `dedupe_bowtie` live?** Three options:  1. As a method on the reader (`obj.dedupe_bowtie()` returns a new reader) 2. As a free function in `_swath.py` (users call it on raw arrays) 3. As a kwarg at construction time (`MODIS_L1B(..., bowtie_dedupe=True)`)

The PRISMA reader doesn't have analogous concerns so there's no precedent.
**Instinct: the method form** — leaves the default behavior untouched (all pixels exposed), but gives users a clean opt-in for resampling pipelines that need it.