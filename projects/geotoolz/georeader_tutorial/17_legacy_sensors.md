# Ch. 17 — legacy sensors

> **Modules:**
> - `georeader/readers/spotvgt_image_operational.py` (389 LOC)
> - `georeader/readers/probav_image_operational.py` (701 LOC)
> **Role:** read SPOT VGT and Proba-V — two coarse-resolution operational vegetation-monitoring sensors that pre-date the Sentinel-2 era. **No ASCII diagrams in either file.** They're operational readers built before the docstring-illustration push.

---

## 1. Why these readers exist at all

SPOT VGT and Proba-V are vegetation-monitoring missions built around **daily global coverage** rather than high spatial resolution:

| Sensor | Operator | Years | GSD | Bands | Coverage |
|---|---|---|---|---|---|
| **SPOT VGT** | CNES + VITO | 1998–2014 | 1 km | Blue, Red, NIR, SWIR | Daily global |
| **Proba-V** | ESA + VITO | 2013–2020 | 100 m / 300 m / 1 km | Blue, Red, NIR, SWIR | Near-daily global |

Both are 1 km–100 m sensors with 4 bands (Blue, Red, NIR, SWIR) — **vastly less rich** than the Sentinel-2-era spectrometers. So why include them?

1. **Historical baseline.** Pre-2015 vegetation studies almost all use VGT. To build a multi-decadal time series you need to read VGT data the same way you read S2 data.
2. **Continuity.** Proba-V was the explicit gap-filler between VGT (2014 end-of-life) and Sentinel-3 (operational 2018). Some climate datasets stitch all three.
3. **Operational simplicity.** These products are tiled in WGS84 with simple GeoTIFFs — no SAFE complexity, no GLT, no per-pixel coords. They're useful as a "minimal sensor reader" reference.

The headers explicitly call out that these are **unofficial readers** — built by the package authors against published user manuals, not certified by the agencies. Behaviour matches the manual but isn't guaranteed against future product-format changes.

---

## 2. The shared design pattern

Both readers expose the same minimal `GeoData`-shaped surface:

```python
SpotVGT(path, ...)
ProbaV(path, ...)
ProbaVRadiometry(path, ...)   # subclass with band-level ToA reflectance
ProbaVSM(path, ...)            # subclass for the Status Map (QA mask)
```

Each class implements:
- **`transform`, `crs`, `shape`** — the standard `GeoData` triple.
- **`load(boundless=True)`** — return a `GeoTensor`.
- **`read_from_window(window, boundless=True)`** — window-restricted read.

So you can drop a `SpotVGT` or `ProbaV` instance into any `read.read_from_polygon(reader, ...)` call. The `GeoData` protocol is the abstraction that makes this just work.

---

## 3. Internal helpers — `read_band_toa`

Both modules have a `read_band_toa(dataset, band, slice_to_read)` function:

- Reads the digital-number band data via rasterio.
- Applies per-band ToA reflectance scale + offset from the band metadata.
- Returns a calibrated array.

Used internally by both readers — saves repeating the calibration boilerplate per call site.

---

## 4. SPOT VGT: `SpotVGT`

The single class is at [spotvgt_image_operational.py:65](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/spotvgt_image_operational.py#L65). Standard layout:

- Detects band files in the SPOT VGT product folder via filename regex.
- Parses acquisition date / sensor / processing version from the filename.
- Aligns the per-band GeoTIFFs onto a common WGS84 grid.
- Provides cloud-masking helpers via `sm_cloud_mask(sm, mask_undefined=False)` ([spotvgt_image_operational.py:365](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/spotvgt_image_operational.py#L365)) — extracts cloud / cloud-shadow / land-water flags from the SM (Status Map) bit-encoded raster.
- `mask_only_sm(sm)` is a more aggressive variant — drops anything not pristine clear sky.

### SM bit decoding

The SPOT VGT Status Map encodes 8 flags in one byte. The functions decode bits like:

| Bit | Meaning |
|---|---|
| 0 | Clear |
| 1 | Cloud shadow |
| 2 | Undefined |
| 3 | Cloud |
| 4 | Ice / snow |
| 5 | Water |
| 6 | Land |
| 7 | Mixed |

`sm_cloud_mask(sm, mask_undefined=True)` returns a boolean mask of "definitely cloudy" pixels — `cloud OR cloud_shadow [OR undefined]`.

---

## 5. Proba-V: `ProbaV`, `ProbaVRadiometry`, `ProbaVSM`

The Proba-V hierarchy is more elaborate ([probav_image_operational.py](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/probav_image_operational.py)):

- **`ProbaV`** ([line 64](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/probav_image_operational.py#L64)) — base reader for the 4-band imagery.
- **`ProbaVRadiometry`** ([line 507](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/probav_image_operational.py#L507)) — subclass that loads ToA reflectance directly (handles per-band gain/offset internally).
- **`ProbaVSM`** ([line 590](https://github.com/spaceml-org/georeader/blob/f0d92f0/georeader/readers/probav_image_operational.py#L590)) — subclass that exposes the Status Map for cloud / shadow / quality.

The Status Map decoder functions `sm_cloud_mask` and `mask_only_sm` are duplicated between the two modules — same names, same logic, different bit conventions. Proba-V's SM is a different format than VGT's despite both coming from VITO; the readers can't share decoder code.

The Proba-V file also has compression-availability checks (`is_compression_available`, `assert_compression_available`) — Proba-V uses LZW or DEFLATE per acquisition, and pre-2017 readers without recent libtiff would silently fail to decode some files. The check raises a clear error before reading.

---

## 6. Why the readers are similar but not identical

A code-deduplication argument would say "extract a shared `LegacyVITOReader` base." The package doesn't, for two pragmatic reasons:

1. **The user manuals describe them as separate products.** A user navigating from the VGT manual to the code wants a `SpotVGT` class, not a `LegacyVITOReader.create(sensor="vgt")`.
2. **Bit-flag conventions differ.** Even though both Status Maps look superficially similar (8 bits per pixel, one bit per quality flag), the bit→meaning mapping is different. Sharing a decoder would require a configuration map that obscures more than it saves.

This is a recurring pattern in sensor readers: the surface area looks like it should generalise, but the per-sensor calibration / bit / metadata details bake in enough divergence that the de-duplication isn't worth the indirection.

---

## 7. Function reference

### SPOT VGT (`spotvgt_image_operational.py`)
- `SpotVGT(path, ...)` — main reader class
- `read_band_toa(dataset, band, slice_to_read)` — internal calibration helper
- `sm_cloud_mask(sm, mask_undefined=False) → np.ndarray` — boolean cloud mask
- `mask_only_sm(sm) → np.ndarray` — strict clear-sky mask

### Proba-V (`probav_image_operational.py`)
- `ProbaV(path, ...)` — base reader
- `ProbaVRadiometry(path, ...)` — ToA reflectance subclass
- `ProbaVSM(path, ...)` — Status Map subclass
- `read_band_toa(dataset, band, slice_to_read)` — internal calibration helper
- `is_compression_available(dataset) → bool` — libtiff check
- `assert_compression_available(dataset)` — raise if compression unavailable
- `sm_cloud_mask(sm, mask_undefined=False) → np.ndarray` — Proba-V version
- `mask_only_sm(sm) → np.ndarray` — Proba-V version

---

## 8. Sharp edges

- **Filename-encoded metadata.** Both readers parse acquisition date / processing version from filenames via regex. Renamed files (or files served through path-rewriting CDNs) break the parser. Keep filenames intact.
- **WGS84-tiled.** Output is in geographic coordinates (degrees), not projected UTM. For pixel-area work you need to convert or accept that 1 km at the equator is not 1 km at high latitudes.
- **Small bands × big tiles.** The 1 km native resolution × global daily coverage means individual product files are large arrays of mostly-zeros (over oceans, deserts at night, etc.). Read with `boundless=False` if you want to detect coverage gaps.
- **Two `sm_cloud_mask` functions exist.** They have the same name in the two modules. Be explicit: `from georeader.readers.spotvgt_image_operational import sm_cloud_mask as vgt_cloud_mask` to avoid confusion.
- **Proba-V's compression check is silent until enabled.** If your libtiff is old, some files load partially zeros. Always call `assert_compression_available(dataset)` early in production code.
- **Status Map bit-flag conventions are sensor-specific.** Don't port code from VGT to Proba-V (or vice versa) by copy-paste — the bit meanings differ.
- **No GLT, no RPCs, no per-pixel coords.** These readers don't need [Chapter 7](07_griddata.md) — the products are already orthorectified to WGS84.

---

## 9. Why no diagrams

The two modules predate the documentation push that added ASCII diagrams to the rest of the package. They're stable, used by a handful of climate-baseline pipelines, and the maintainers haven't added illustrations because there's not much to illustrate — these are simple band-stack readers without the structural complexity of EMIT (GLT) or PRISMA (curvilinear) or S2 SAFE (multi-resolution + JP2 stacking).

Including them in this tutorial preserves the catalog completeness, but you can probably skip these if you're not specifically working with VGT or Proba-V. The readers are stable, well-named, and self-documenting in their docstrings.

---

## 10. Connection to `geotoolz`

Neither sensor has a dedicated preset in [`geotoolz.md`](../plans/geotoolz/geotoolz.md). Proba-V and SPOT VGT are unlikely to grow new operators because they're discontinued. They'd live in `geotoolz.presets.legacy` if ever — a `presets.legacy.PROBAV_NDVI` operator would be `Sequential([ProbaVRadiometry.load(), MaskClouds(via SM), NDVI(red_idx=1, nir_idx=2)])`. Trivial to implement, but rare enough that no-one's asked.

The general lesson: **operators are sensor-agnostic; readers are sensor-specific**. As long as a sensor's reader returns a `GeoData` with band semantics that match a downstream operator's `red_idx=` / `nir_idx=` arguments, the operator works without any sensor-specific code.

Next chapter: [18_query_and_download.md](18_query_and_download.md) — query / catalog / download helpers (`scihubcopernicus_query`, `download_pv_product`, `query_utils`, `tileserver`, `download_utils`).
