---
title: "spectral_indices — index catalog and transformation engine"
short_title: "spectral_indices design"
---

# `spectral_indices` — index catalog and transformation engine

> *Apply named spectral transformations (NDVI, NDWI, NBR, EVI, …) to
> scenes and time-series across any sensor whose bands you have.
> Browse a catalog of ~250 named indices; build composed formulas
> from the algebra; export results as derived bands.*

**Status**: design only. No code in this PR.

## Question answered

> "Given this AOI and this sensor, what does index X look like —
> as a map for one scene, and as a time-series across many scenes?"

This is the daylight-to-dusk applied tool: most users of remote
sensing data spend their time computing and comparing indices, and
right now there isn't a clean local app for it — they either reach
for Earth Engine or write per-index notebooks by hand.

## Scope

- **Sensors**: any sensor in the `satellite_viewer.SENSORS` registry
  whose `band → wavelength` mapping is known. v0 ships with Sentinel-2,
  Landsat 8/9, and MODIS; EMIT is treated as "every band available"
  and gets the full index catalog automatically.
- **Index catalog**: the
  [Awesome Spectral Indices](https://github.com/awesome-spectral-indices/awesome-spectral-indices)
  JSON registry — ~250 indices with formulas, band requirements,
  references, and citations. Ship it bundled (~150 KB JSON), refresh
  by pinned-version bump.
- **AOI**: any shapely geometry. v0 supports up to MGRS-tile-size
  AOIs (~110 km); larger goes through `geopatcher`.
- **Time scope**: single scene at v0, multi-scene time series at v2.
- **Compute target**: interactive for v0/v1, batch-on-demand for v2/v3.

## Algorithm

```
1. Index registry: load awesome-spectral-indices/spectral-indices.json
       into a dict keyed by short_name. Each entry has formula,
       bands, domain (vegetation/water/fire/...), and reference.
2. Sensor → band-to-wavelength mapping: maintained as a small per-
       sensor dict in our own sensors.py (overlays satellite_viewer's
       registry).
3. Resolve an index against a sensor: substitute the index's named
       band slots (B, G, R, RE1, ..., N, S1, S2) with the sensor's
       actual asset/file keys. If any required band is missing, raise.
4. Read the required bands at the AOI bbox via georeader (or
       odc-stac depending on stack choice).
5. Evaluate the index formula on the band arrays. The formula is a
       small arithmetic expression — safe to eval through `numexpr`
       or `asteval`, never raw eval().
6. Output: derived band raster at the union extent of the input bands,
       resampled to the coarsest input resolution.
7. (v2 only) repeat per scene in the catalog, stack along time dim.
```

## Stages

| v   | What it does                                                              |
|-----|---------------------------------------------------------------------------|
| v0  | Pick sensor + AOI + scene + index → derived band map.                     |
| v1  | Searchable index catalog: filter by domain (vegetation / water /          |
|     | fire / soil / snow / urban / generic), see formula + bands + citation.    |
| v2  | Time-series mode: catalog of N scenes (12-month window default) →         |
|     | derived band per scene → chart of the AOI-mean over time.                 |
| v3  | Formula builder: user types `(NDVI - NDWI) / (NDVI + NDWI)` and it        |
|     | gets parsed → safely evaluated → mapped, same as named indices.           |

You ship a useful tool after v1; v2/v3 are the depth.

## Architecture

```
projects/spectral_indices/
├── pyproject.toml
├── README.md
├── data/
│   └── spectral-indices.json   # pinned snapshot of the registry
├── src/spectral_indices/
│   ├── __init__.py
│   ├── registry.py     # load_indices(), get(name), filter_by_domain(...)
│   ├── sensors.py      # per-sensor band→wavelength + asset-key map
│   ├── resolve.py      # match an index's band slots to a sensor
│   ├── evaluate.py     # safe formula evaluation on ndarrays
│   ├── reader.py       # AOI + bands → ndarray (delegates to georeader)
│   └── series.py       # v2: catalog → time series
├── tests/
│   ├── test_registry.py
│   ├── test_resolve.py
│   ├── test_evaluate.py    # hand-check NDVI / NDWI on synthetic arrays
│   └── test_series.py
└── apps/
    └── (Panel / Streamlit / notebook subapp; see Stack options)
```

`geotoolz` is the natural place for `evaluate.py` — each named index
becomes a tiny `Operator` subclass and the formula builder becomes
`Sequential` / `Graph` composition. But that's a refactor decision,
not blocking.

## Output schema

For v0/v1 (single scene): an `xarray.DataArray`:

```
dims    : (y, x)
coords  : y, x   (in the sensor's UTM CRS)
attrs   : index, sensor, scene_id, formula, bands_used, units
```

For v2 (time series): `xarray.DataArray` with extra `time` dim, or a
DataFrame for the AOI-mean trace:

```
columns : [datetime, sensor, scene_id, index, value_mean, value_std,
           value_p10, value_p90, n_valid_pixels]
```

## Stack options

### Option A — `geotoolz` operators + `georeader` (recommended)

Each named index is an `Operator` subclass returning a single band:

```python
class NDVI(Operator):
    def _apply(self, gt):
        nir, red = gt.values[NIR_IDX], gt.values[RED_IDX]
        return gt.array_as_geotensor((nir - red) / (nir + red + 1e-10))
    def get_config(self):
        return {"nir_idx": NIR_IDX, "red_idx": RED_IDX}
```

Composed formulas are `Sequential` / `Graph`. Time series via
`geocatalog` + a per-scene `Operator` application loop. Reads via
`georeader`.

**Pro**: showcases the entire `geotoolz` + `geocatalog` + `georeader`
stack — this is the canonical use case. Reproducibility via
`Operator.get_config()`.
**Con**: writing all 250 indices as Operators by hand is silly.
Better: one generic `IndexOp` that takes a name and looks up the
formula from the registry, then evaluates it dynamically.

### Option B — Earth Engine

`ee.Image.expression("(N - R) / (N + R)", {"N": img.select("B8"), ...})`
evaluates band-math in EE natively. Combine with the
[`eemont`](https://github.com/davemlz/eemont) extension which has
the same awesome-spectral-indices catalog wired in:

```python
img = (
    ee.ImageCollection("COPERNICUS/S2_SR")
    .filterBounds(aoi)
    .filterDate(t0, t1)
    .map(lambda img: img.spectralIndices(["NDVI", "NDWI", "NBR"]))
)
```

**Pro**: zero infrastructure, every sensor EE has is pre-supported,
the index catalog is already wired. Implementation is ~50 lines.
**Con**: hosted, EE's terms-of-use, doesn't help build out your own
stack. Best as a comparison / sanity-check tool, not the primary app.

### Option C — `odc-stac` + `xarray` directly

Skip both `geotoolz` and `georeader`:

```python
items = stac_client.search(...).items()
ds = odc.stac.load(items, bands=["B04", "B08"], bbox=aoi.bounds)
ndvi = (ds.B08 - ds.B04) / (ds.B08 + ds.B04 + 1e-10)
```

**Pro**: industry-standard stack, plays well with dask + zarr, every
modern Python remote-sensing project uses this shape.
**Con**: less reproducibility metadata than `Operator.get_config()`;
doesn't showcase your stack.

### Option D — GDAL band-math via `gdal_calc.py`

Shell-tool / CLI approach. Each index becomes a saved GDAL `--calc`
expression.

**Pro**: zero Python at compute time; great for batch jobs that
just need to land COGs on disk.
**Con**: no interactivity, no time series, no UI.

**My recommendation**: A as the library, with C as the time-series
back-end if `geocatalog` scaling becomes a bottleneck. B as a
notebook sibling for "what does Earth Engine say?" cross-check.

## UI integration

Two natural shapes:

1. **Notebook** (always ship this): a `spectral_indices.compute(aoi,
   sensor, index, t0, t1)` helper that returns an `xarray.DataArray`,
   plus a `.plot()` recipe. Time-series via `.compute_series(...)`.
2. **Streamlit / Panel app**: sidebar = sensor + AOI + date + index
   picker (with searchable catalog) + formula editor; main = map +
   time-series chart + formula display + citation.

The catalog browser is the headline feature:

```
+---------------------+----------------------------------------+
| Domain  : water  ▾  |  Search: "modified"                    |
| Sensor  : sent.-2 ▾ |  Filter: sensors compatible only       |
+---------------------+----------------------------------------+
| MNDWI - Modified Normalized Difference Water Index           |
|   formula : (G - S1) / (G + S1)                              |
|   bands   : G=B3, S1=B11   (Sentinel-2)                      |
|   citation: Xu (2006)                                        |
|   [Apply to AOI]                                             |
+--------------------------------------------------------------+
| WI2015 - Water Index 2015                                    |
| ...                                                          |
+--------------------------------------------------------------+
```

## Compute budget

- Single index, single S2 scene, 10×10 km AOI: 2 STAC search + 2 band
  reads (~30 MB) + arithmetic → ~5 s.
- 12-month time series, single index, AOI: ~70 scenes × 5 s ÷ 8
  workers ≈ 1 min in parallel.
- Continental tile coverage: outside the per-AOI scope, would use
  `geopatcher`.

## Risks / open questions

- **awesome-spectral-indices schema drift** — pin a version and bump
  intentionally. Add a CI check that the bundled JSON parses against
  the loader.
- **Per-sensor band coverage** is uneven — EVI needs blue, NDSI needs
  SWIR, etc. Pre-compute a `(sensor, index) → applicable?` table and
  surface "incompatible" indices as greyed-out in the UI.
- **Reflectance vs DN** — most indices are defined on
  surface-reflectance L2A; L1C / L1T won't give meaningful values.
  Either enforce L2A or warn loudly.
- **Formula injection at v3** — never use raw `eval`. Use `numexpr`
  (fast, restricted), `asteval` (Python AST whitelisting), or a
  micro-parser. Document the safe subset.

## Acceptance

- 10 most-used indices (NDVI, NDWI, NBR, EVI, SAVI, MNDWI, GNDVI,
  NDBI, NDSI, RVI) compute correctly for both Sentinel-2 and
  Landsat 8 against a small synthetic test.
- The bundled awesome-spectral-indices JSON loads ≥ 200 indices.
- v2 time-series renders a 12-month chart for an AOI < 30 s.
- v3 formula builder rejects unsafe input (`__import__`, `;`, etc.)
  and produces correct output for `(NDVI - NDWI) / (NDVI + NDWI)`.
- `Operator.get_config()` round-trips every shipped index through
  YAML.

## Out of scope

- Atmospheric correction.
- Index calibration / empirical-line tuning.
- ML-derived "indices" (random forests, learned embeddings). Different
  app.
- Anomaly detection — could be a v3.5, deferred.
