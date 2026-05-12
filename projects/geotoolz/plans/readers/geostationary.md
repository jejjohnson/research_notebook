---
title: Geostationary readers
subject: Sensor readers
subtitle: GOES, MSG, MTG, Himawari readers
short_title: Geostationary
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, readers, goes, seviri
---

> **Design Report (v3)** — design for GOES-R ABI, MSG SEVIRI, MTG-FCI, and Himawari AHI readers in [`spaceml-org/georeader`](https://github.com/spaceml-org/georeader).

## Contents

- [User Story](#user-story)
- [Motivation](#motivation)
- [Mathematics](#mathematics)
- [Target API](#target-api)
  - [Track A — Clean `+proj=geos` affine](#track-a--clean-projgeos-affine-goes-abi-mtg-fci)
  - [Track B — Irregular file formats](#track-b--irregular-file-formats-seviri-ahi)
  - [Public bucket helpers](#public-bucket-helpers)
- [Example Use Cases](#example-use-cases)
- [Subtasks](#subtasks)

## User Story

I’m migrating geostationary readers from `rs_tools` into `georeader`, starting with **GOES-R ABI** and **MSG SEVIRI**, with a path to **MTG-FCI** and **Himawari AHI**.
The work is substantially smaller than first thought: `georeader` already owns CRS reprojection through `read.read_from_bounds` and irregular-geolocation reprojection through `griddata.read_to_crs`.
What’s missing is the **reader** — the per-sensor class that opens these files, parses sensor-specific metadata, exposes calibrated radiance, and conforms to whichever existing convention (S2-style `GeoData` or PRISMA-style raw-arrays-plus-`lons`/`lats`) fits the file format on disk.

## Motivation

Each geostationary sensor lives in its own file format with its own quirks:

|Sensor           |Format                          |Notes                                 |
|-----------------|--------------------------------|--------------------------------------|
|GOES-R ABI L1b/L2|NetCDF-CF                       |Clean `+proj=geos` affine, `sweep='x'`|
|MTG-FCI L1c      |NetCDF (chunked)                |Clean `+proj=geos` affine, `sweep='y'`|
|MSG SEVIRI       |`.nat` (Native) or xRIT segments|Custom binary; needs explicit parsing |
|Himawari AHI     |HSD segments                    |Custom binary                         |

A `satpy` install can read all of these but drags in xarray/dask/pyresample and is pipeline-shaped.
The `georeader`-shaped equivalent is much smaller in scope: a reader per sensor that produces either a `GeoData` (when the file’s affine is clean) or a PRISMA-like object exposing `.lons`, `.lats`, and raw arrays (when it isn’t), and lets `georeader`’s existing machinery do the rest.
The motivation is to keep `georeader` consistent: one mental model for the user (`georeader.readers.X`, then `read.read_from_bounds` or `griddata.read_to_crs`), regardless of sensor.

## Mathematics

The only math the reader itself owns is computing `.lons` and `.lats` from scan angles when the file format doesn’t already give them.
This is the standard `+proj=geos` forward projection, lifted directly from the [GOES-R PUG](https://www.goes-r.gov/users/docs/PUG-main-vol1.pdf) and MSG ICD. As a private utility:

```python
# georeader/readers/geostationary/_projection.py
import numpy as np
from numpy.typing import NDArray


def scan_to_geodetic(
    x: NDArray, y: NDArray,        # scan angles (rad), broadcastable
    lon_sub: float,                # sub-satellite longitude (rad)
    height_m: float = 35_786_023.0,
    r_eq: float = 6_378_137.0,
    r_pol: float = 6_356_752.31414,
) -> tuple[NDArray, NDArray, NDArray]:
    """(x, y) → (lat, lon, on_disk). Used to populate reader.lons/lats."""
    H = height_m + r_eq
    cx, sx, cy, sy = np.cos(x), np.sin(x), np.cos(y), np.sin(y)
    a   = cy**2 + (r_eq / r_pol)**2 * sy**2
    s_d = (H * cx * cy)**2 - a * (H**2 - r_eq**2)
    on_disk = s_d >= 0
    s_n = (H * cx * cy - np.sqrt(np.where(on_disk, s_d, 0.0))) / a
    s1, s2, s3 = H - s_n*cx*cy, -s_n*sx*cy, s_n*sy
    lat = np.arctan((r_eq / r_pol)**2 * s3 / np.sqrt(s1**2 + s2**2))
    lon = lon_sub + np.arctan2(s2, s1)
    return lat, lon, on_disk
```

That’s the entire math footprint inside the reader package.
Beyond it, anything users need (satellite/solar zenith, local resolution, disk mask) can live in a separate `geostationary_utils` module later — **out of scope** for this design.

## Target API

Two tracks fall out of the file formats.
The reader user-facing surface is identical in spirit; the *implementation* differs.

### Track A — Clean `+proj=geos` affine (GOES ABI, MTG-FCI)

Conforms to `GeoData` like the S2 reader.
The CRS is `+proj=geos` with the right sweep axis (`x` for GOES, `y` for FCI), the transform is linear in scan-angle-projected meters, and `read.read_from_bounds` works directly through rasterio:

```python
# georeader/readers/geostationary/abi.py
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor


class ABI_L1b(GeoData):
    """GOES-R ABI Level-1b radiance reader. NetCDF; one file per channel per scan."""

    # --- required by GeoData ---
    bounds: tuple[float, float, float, float]   # +proj=geos meters
    crs: CRS                                    # +proj=geos sweep=x lon_0=lon_sub
    transform: Affine
    shape: tuple[int, int, int]                 # (C, H, W)
    fill_value_default: float
    bands: list[str]                            # e.g. ['C02', 'C13']

    # --- ABI-specific metadata (attributes, not methods) ---
    time: datetime                              # scan midpoint, tz-aware
    time_start: datetime
    time_end: datetime
    satellite: str                              # 'G16' | 'G17' | 'G18' | 'G19'
    sub_satellite_lon: float                    # degrees
    sector: str                                 # 'RadF' | 'RadC' | 'RadM1' | 'RadM2'
    mode: int                                   # 3 | 4 | 6 (scan mode)
    nadir_resolution_m: dict[str, float]        # {'C02': 500, 'C13': 2000, ...}
    units: str                                  # 'W m-2 sr-1 um-1'

    def __init__(
        self,
        path: str | list[str],                  # one URI per channel, or a glob
        bands: list[str] | None = None,         # None → all available channels
        out_res: float | None = None,           # native if None, else common-grid m
    ): ...

    @classmethod
    def from_product_id(
        cls, product_id: str, bands=None, out_res=None,
        bucket: str = "noaa-goes16",
    ) -> "ABI_L1b":
        """Resolve a PRODUCT_ID to S3 URIs and instantiate."""

    # --- GeoData contract methods ---
    def load(self, *, boundless: bool = False) -> GeoTensor: ...
    def read_from_window(self, window: Window) -> "ABI_L1b": ...
    def isel(self, sel: dict) -> "ABI_L1b": ...

    # --- ABI-specific helpers ---
    def to_radiance(self) -> GeoTensor: ...     # apply scale+offset (default behavior)
    def to_reflectance(self) -> GeoTensor: ...  # for reflective channels (C01–C06)
    def to_brightness_temperature(self) -> GeoTensor: ...  # for thermal (C07–C16)
```

The `__repr__` matches the S2 / PRISMA conventions:

```bash
>>> obj = ABI_L1b.from_product_id(
...     "OR_ABI-L1b-RadF-M6_G16_s20240011200205", bands=["C02", "C13"], out_res=2000)
>>> obj
ABI_L1b: G16  sector=RadF  mode=6  t=2024-01-01T12:04:53+00:00
  bands: ['C02', 'C13']  shape: (2, 5424, 5424)  res: 2000 m
  sub_satellite_lon: -75.0°  sweep_axis: 'x'
  bounds (geos m): (-5434894, -5434894, 5434894, 5434894)
  CRS: +proj=geos +lon_0=-75 +h=35786023 +a=6378137 +b=6356752.31414 +sweep=x
```

Multi-resolution handling follows the S2 reader: bands keep their native rasterio handles internally, and `out_res` selects a common output resolution at load time:

```python
# Internal — sketch
class ABI_L1b(GeoData):
    def __init__(self, path, bands=None, out_res=None):
        self._datasets = {b: rasterio.open(p) for b, p in self._resolve_paths(path, bands)}
        self._native_res = {b: self._datasets[b].res[0] for b in self._datasets}
        self.out_res = out_res or max(self._native_res.values())
        self._calib = self._read_calibration_coeffs()    # scale, offset, kappa0, etc.
        self.crs, self.transform, self.shape, self.bounds = self._build_grid()

    def load(self, *, boundless=False) -> GeoTensor:
        arrays = []
        for band in self.bands:
            ds = self._datasets[band]
            arr = ds.read(1, out_shape=self._target_shape(band),
                          resampling=Resampling.bilinear)
            arr = self._calibrate(arr, band)             # counts → radiance
            arrays.append(arr)
        return GeoTensor(np.stack(arrays), transform=self.transform, crs=self.crs,
                         fill_value_default=self.fill_value_default)
```

### Track B — Irregular file formats (SEVIRI, AHI)

Mirrors the PRISMA reader exactly.
The reader does **not** conform to `GeoData`; instead it exposes raw arrays plus `.lons`/`.lats`, and users go through `griddata.read_to_crs`:

```python
# georeader/readers/geostationary/seviri.py
class SEVIRI_Native:
    """MSG SEVIRI Level-1.5 Native (.nat) reader. PRISMA-pattern."""

    # --- per-pixel geolocation (PRISMA convention) ---
    lons: NDArray                               # (H, W) degrees
    lats: NDArray                               # (H, W) degrees

    # --- SEVIRI metadata ---
    bounds: tuple[float, float, float, float]   # in EPSG:4326 from lons/lats
    time: datetime                              # nominal scan midpoint
    time_start: datetime
    time_end: datetime
    satellite: str                              # 'MSG1' | 'MSG2' | 'MSG3' | 'MSG4'
    sub_satellite_lon: float
    bands: list[str]                            # e.g. ['IR_108', 'WV_062', 'HRV']
    wavelengths: dict[str, float]               # central wavelengths in µm
    units: str
    calibration_coeffs: dict[str, tuple[float, float]]   # (slope, offset) per band

    def __init__(
        self,
        path: str,
        bands: list[str] | None = None,
        include_hrv: bool = False,              # HRV is 1 km, others 3 km — different grid
    ): ...

    def load_raw(self, *, calibrated: bool = True) -> NDArray:
        """Return (C, H, W) array; if calibrated, in self.units, else raw counts."""

    def load_band(self, band: str, *, calibrated: bool = True) -> NDArray: ...

    def to_reflectance(self) -> NDArray: ...    # solar channels
    def to_brightness_temperature(self) -> NDArray: ...   # thermal channels
```

User code follows the PRISMA notebook idiom verbatim:

```python
from georeader.readers.geostationary import seviri
from georeader import griddata
import numpy as np

obj = seviri.SEVIRI_Native("MSG3-SEVI-MSG15-...-NA.nat",
                           bands=["IR_108", "WV_062"])
print(obj)
# SEVIRI_Native: MSG3  t=2024-06-14T12:00:00+00:00
#   bands: ['IR_108', 'WV_062']  shape: (2, 3712, 3712)  res(nadir): 3000 m
#   sub_satellite_lon: 0.0°  sweep_axis: 'y'

raw = obj.load_raw()                                    # (2, H, W) in W m-2 sr-1 um-1
geo = griddata.read_to_crs(np.moveaxis(raw, 0, 2),
                           lons=obj.lons, lats=obj.lats,
                           resolution_dst=3000, dst_crs="EPSG:3857")
# geo: GeoTensor (2, H', W') in EPSG:3857
```

Internally, `_projection.scan_to_geodetic` populates `lons` and `lats` once during `__init__` from the file’s scan-angle metadata.
**Users never see the projection module.**

> **A note on the HRV channel.** SEVIRI’s HRV is 1 km native versus 3 km for the other 11 channels, on a smaller and offset grid.
> The reader handles this by treating HRV as a separate group with its own `lons`/`lats`, similar to S2’s 10/20/60 m groups.
> If `bands=['HRV']`, the reader’s grid is 1 km; if `bands=['IR_108', 'HRV']` is requested with `include_hrv=True`, the reader returns two parallel arrays and lets the user choose how to combine.
> Mixing resolutions silently is a known footgun and we don’t want it.

### Public bucket helpers

Mirroring `S2_SAFE_reader.s2_public_bucket_path`, one helper per sensor where applicable:

```python
# georeader/readers/geostationary/abi.py
def public_bucket_path(product_id: str, bucket: str = "noaa-goes16") -> str: ...


def find_files(
    start: datetime, end: datetime,
    satellite: Literal["G16", "G17", "G18", "G19"] = "G16",
    sector: Literal["RadF", "RadC", "RadM1", "RadM2"] = "RadF",
    channels: list[str] = ["C13"],
    product_level: Literal["L1b", "L2"] = "L1b",
) -> list[str]:
    """Returns S3 URIs (s3://noaa-goes16/ABI-L1b-RadF/YYYY/DOY/HH/...)."""
```

MSG and MTG are gated behind EUMETSAT Data Store (auth required); a separate auth-aware helper goes in the SEVIRI/FCI modules.

## Example Use Cases

### 1. ABI in a UTM box around an EMIT detection

Track A, vanilla `georeader` path:

```python
from georeader.readers.geostationary import abi
from georeader import read
from shapely.geometry import box

obj = abi.ABI_L1b.from_product_id(
    "OR_ABI-L1b-RadF-M6_G16_s20240011200205",
    bands=["C07", "C13"], out_res=2000,
)
aoi = box(-104.5, 31.9, -104.3, 32.1)
data = read.read_from_polygon(obj, aoi, crs_polygon="EPSG:4326",
                              dst_crs="EPSG:32613", resolution_dst=2000).load()
# data: GeoTensor (2, H, W) in UTM-13N
```

### 2. SEVIRI to Web Mercator over Europe

Track B, PRISMA path:

```python
from georeader.readers.geostationary import seviri
from georeader import griddata

obj = seviri.SEVIRI_Native("MSG3-SEVI-MSG15-...-NA.nat", bands=["IR_108"])
raw = obj.load_band("IR_108", calibrated=True)            # (H, W)
geo = griddata.read_to_crs(raw[..., None],                # add channel axis
                           lons=obj.lons, lats=obj.lats,
                           resolution_dst=3000, dst_crs="EPSG:3857",
                           bounds_dst=(-1.5e6, 4.5e6, 3.5e6, 8.5e6))
```

### 3. ABI–Sentinel-2 collocation

Both readers conform to `GeoData`; existing helper does the lift:

```python
from georeader.readers import S2_SAFE_reader
from georeader.readers.geostationary import abi
from georeader import read

s2 = S2_SAFE_reader.s2loader(s2_path, out_res=60, bands=["B12"])
goes = abi.ABI_L1b.from_product_id(closest_abi_id, bands=["C13"])
goes_on_s2_grid = read.read_reproject_like(goes, s2)      # already exists
```

### 4. Channel calibration

The reader exposes physically-meaningful methods rather than forcing users to dig out scale/offset themselves:

```python
obj = abi.ABI_L1b.from_product_id("...", bands=["C13"])
bt = obj.to_brightness_temperature().load()               # K, (1, H, W)
```

## Subtasks

The work splits into the projection utility, the two reader templates, the per-sensor implementations, and the bucket helpers.

### Projection utility (~1 day)

`_projection.py` with `scan_to_geodetic` (above) and tests against published corner coordinates:

```python
# tests/test_projection.py
def test_abi_corner_against_pug():
    x, y = -0.151844, 0.151844
    lat, lon, ok = scan_to_geodetic(x, y, lon_sub=np.deg2rad(-75.0))
    assert ok and np.isclose(np.rad2deg(lat), 33.846, atol=1e-3)
    assert np.isclose(np.rad2deg(lon), -84.690, atol=1e-3)


def test_seviri_corner_against_icd():
    ...
```

### ABI L1b reader (~1 week)

The first concrete reader.
NetCDF parsing through rasterio (`netcdf:path:Rad`); CRS construction from CF metadata or manually from sub-sat lon and sweep axis; calibration via `Rad`‘s `scale_factor`/`add_offset` plus `kappa0` for reflectance and Planck inversion for brightness temperature.
Bands at three native resolutions handled S2-style.
AWS S3 URIs work through fsspec/rasterio’s GDAL `/vsis3/` driver.
Tests against a small fixture file in CI.

### SEVIRI Native reader (~2 weeks)

This is the bigger lift because the `.nat` format needs custom binary parsing.
Two options:

1. Leverage `eumdac` / `satpy`’s SEVIRI native reader as an internal dependency for the parsing step only and rebuild calibration ourselves.
2. Write a from-scratch parser following the MSG Level 1.5 ICD (more work, fewer dependencies).

Lean toward an optional `satpy` dependency for the parsing only — cleaner and we can always replace it.
Calibration coefficients are in the file header; HRV handled as a separate grid.

### MTG-FCI reader (~1 week)

Follows ABI: NetCDF, CF metadata, similar calibration story.
Quirks are FCI’s chunked layout and the L1c grid.

### Himawari AHI HSD reader (~2 weeks)

Custom binary, similar in scope to SEVIRI Native.
**Defer** until ABI + SEVIRI are landed.

### SEVIRI HRIT reader (~2 weeks)

xRIT-compressed segments.
The messiest of the lot.
**Defer** until clear demand.

### Public bucket helpers (~1 day per sensor)

Small, self-contained.

### Documentation (~1 week)

Three notebooks in the same style as the S2 SAFE and PRISMA notebooks:

- `abi_quickstart.ipynb` mirroring `read_S2_SAFE_from_bucket.ipynb`
- `seviri_quickstart.ipynb` mirroring `prisma_with_cloudsen12.ipynb`
- `goes_s2_collocation.ipynb` showing the cross-sensor workflow

The notebooks are what users will actually read.

### Staged release

- [ ] **PR 1.** `_projection.py` + ABI L1b reader + bucket helper + ABI quickstart notebook (cohesive, end-to-end demonstrable)
- [ ] **PR 2.** SEVIRI Native reader + SEVIRI quickstart notebook
- [ ] **PR 3.** MTG-FCI reader
- [ ] **PR 4.** Himawari AHI HSD reader
- [ ] **PR 5.** SEVIRI HRIT reader

-----

If this lands, the next step is to pull the existing `S2_SAFE_reader.S2Image` and `prisma.PRISMA` source side by side and sketch `ABI_L1b` and `SEVIRI_Native` as concrete diffs against those templates — locking down the attribute names, calibration interfaces, and `__repr__` format before writing any sensor-specific parsing.