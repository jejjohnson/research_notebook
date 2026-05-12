---
title: Module catalog
subject: georeader tutorial
subtitle: What this tutorial covers, by source module
short_title: Catalog
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: tutorial, georeader
---

> **Source:** [`spaceml-org/georeader`](https://github.com/spaceml-org/georeader/tree/f0d92f0) @ branch `feature/geotensor_npapi`, commit [`f0d92f0`](https://github.com/spaceml-org/georeader/tree/f0d92f0) **Goal:** a module-by-module tutorial that captures the package's capabilities **and preserves all ASCII diagrams** before they're cleaned up downstream.

The package is ~20k LOC across 17 top-level files + 14 reader modules. ~1100 lines of box-drawing ASCII art are scattered across docstrings — this is the doc treasure we're rescuing.

---

## Recommended tutorial structure

Each chapter = one file under `georeader_tutorial/`.
We work through them one at a time.
Diagrams are copied verbatim into the tutorial; surrounding prose explains the concept and adds runnable examples.

### Part I — Core data model

| # | File | Source module | LOC | Diagrams | What you learn |
|---|------|---------------|-----|----------|----------------|
| 1 | `01_geotensor.md` | `geotensor.py` | 2532 | 86 | The numpy-subclass `GeoTensor` — dim conventions, `__array_ufunc__`, `__array_finalize__`, transform/CRS round-trip, time as a first-class dim. **The centerpiece of this branch.** |
| 2 | `02_abstract_reader.md` | `abstract_reader.py` | 257 | 46 | The `GeoData` / `GeoTensor` / Reader Protocol type hierarchy — what duck-types as a georeader source. |
| 3 | `03_rasterio_reader.md` | `rasterio_reader.py` | 1630 | 66 | Lazy file-backed reader: `RasterioReader`, multi-file stacks, windowed reads, overviews, cloud-native VSI paths. |

### Part II — Reading & windowing

| # | File | Source module | LOC | Diagrams | What you learn |
|---|------|---------------|-----|----------|----------------|
| 4 | `04_window_utils.md` | `window_utils.py` | 1471 | 102 | Window anatomy, pixel↔geo transforms, padding, alignment, intersection. The math of "what region am I reading." |
| 5 | `05_read.md` | `read.py` | 1967 | 123 | The high-level `read_from_*` family: bounds, polygon, center coords, bbox, with reprojection + resampling. **Most diagrams in the package.** |
| 6 | `06_slices.md` | `slices.py` | 404 | 60 | Tiling strategies: overlap vs stride vs chunked, generating windows for tiled inference. |

### Part III — Geometry & gridding

| # | File | Source module | LOC | Diagrams | What you learn |
|---|------|---------------|-----|----------|----------------|
| 7 | `07_griddata.md` | `griddata.py` | 617 | 96 | Irregular→regular gridding, GLT (geolocation lookup tables), `read_to_crs` for swath sensors. |
| 8 | `08_mosaic.md` | `mosaic.py` | 450 | 66 | Multi-raster mosaicking with reprojection + nodata fill — cloud-free composites. |
| 9 | `09_rasterize.md` | `rasterize.py` | 438 | 58 | Burning vector geometries (polygons, lines) into raster grids aligned to a `GeoTensor`. |
| 10 | `10_vectorize.md` | `vectorize.py` | 370 | 63 | Inverse: extracting polygons from binary masks (segmentation → GIS-ready vectors). |

### Part IV — Radiometry & I/O

| # | File | Source module | LOC | Diagrams | What you learn |
|---|------|---------------|-----|----------|----------------|
| 11 | `11_reflectance.md` | `reflectance.py` | 971 | 97 | Radiance↔ToA reflectance, spectral response functions (SRF), solar irradiance integrals (Thuillier). |
| 12 | `12_save.md` | `save.py` + `save_cog.py` | 586 | 58 | Writing GeoTensors to disk; full COG anatomy with the IFD/overview/tile diagram. |
| 13 | `13_misc_io.md` | `io.py`, `dataarray.py`, `plot.py` | 113+145+336 | 0 | Smaller utilities: fsspec wrappers, xarray bridge (`to_dataarray`/`from_dataarray`), `plot.show`. |

### Part V — Sensor readers

| # | File | Source module | LOC | Diagrams | What you learn |
|---|------|---------------|-----|----------|----------------|
| 14 | `14_sentinel2.md` | `readers/S2_SAFE_reader.py` | 1845 | 47 | S2 SAFE reader (L1C + L2A), band groups, GCS support, jp2 reads. |
| 15 | `15_hyperspectral.md` | `readers/{emit,prisma,enmap}.py` | 1102+571+865 | 27+38+33 | Hyperspectral trio — EMIT (ISS, 285 bands), PRISMA (ASI, 239 bands), EnMAP (DLR, 224 bands). All curvilinear; common pattern. |
| 16 | `16_earth_engine.md` | `readers/{ee_image,ee_query,ee_utils}.py` | 539+589+58 | 42 | GEE export with recursive tile splitting and parallel download. |
| 17 | `17_legacy_sensors.md` | `readers/{spotvgt,probav}_image_operational.py` | 389+701 | 0 | SPOT VGT and Proba-V — operational legacy readers. |
| 18 | `18_query_and_download.md` | `readers/{scihubcopernicus_query,download_pv_product,query_utils,tileserver,download_utils}.py` | small | 0 | Catalog/query helpers, downloaders, tile-server adapter. |

---

## Diagram inventory (count of box-drawing chars per file)

```
read.py                 123    ← reading workflow, AOI specification, reprojection
window_utils.py         102    ← window anatomy, pixel↔geo, alignment
reflectance.py           97    ← unit conventions, SRF, irradiance pipeline
griddata.py              96    ← irregular vs regular grids, GLT
geotensor.py             86    ← dim conventions, ufunc protocol, metadata flow
rasterio_reader.py       66    ← lazy I/O architecture
mosaic.py                66    ← spatial mosaic concept
vectorize.py             63    ← raster→vector pipeline
slices.py                60    ← tiling strategies
save.py                  58    ← COG structure
rasterize.py             58    ← vector→raster pipeline
S2_SAFE_reader.py        47    ← S2 product levels, band groups
abstract_reader.py       46    ← georeader type hierarchy
ee_image.py              42    ← GEE export architecture
prisma.py                38    ← PRISMA HDF5 layout
enmap.py                 33    ← EnMAP product structure
emit.py                  27    ← EMIT NetCDF layout
─────────────────────────────
TOTAL                  ~1148 box-drawing chars across 17 files
```

(Plus pipe-tables and `+--+`-style diagrams not counted above.
Per-chapter we'll grep these out exhaustively.)

---

## Working method

1. We pick a chapter from the table above.
2. I extract every ASCII diagram from that module's docstrings verbatim into the chapter file.
3. I add concise prose around each diagram explaining the concept.
4. I add 1–3 runnable code snippets per chapter using realistic geo data (no mock arrays).
5. We review and iterate before moving to the next chapter.

**Suggested starting order:** 1 (geotensor) → 2 (abstract_reader) → 3 (rasterio_reader) → 5 (read) → 4 (window_utils) → ... — front-loads the conceptual core, then radiates outward.

---

## Open questions before we start writing

- **Format:** `.md` per chapter (current proposal) or jupytext `.py` notebooks executed to `.ipynb` (matches the research_notebook convention)?
- **Code execution:** do you want runnable cells with real S2/EMIT downloads, or pseudocode-only (faster, no data deps)?
- **Scope of "Part V — Sensor readers":** all of them, or just the four flagship ones (S2, EMIT, PRISMA, EnMAP)?
