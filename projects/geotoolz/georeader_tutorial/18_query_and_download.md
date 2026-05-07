# Chapter 18 — `readers/{scihubcopernicus_query, download_pv_product, query_utils, tileserver, download_utils}.py`: catalog query & download helpers

> **Modules** (all small):
> - `query_utils.py` (80 LOC) — generic spatial-overlap helpers
> - `scihubcopernicus_query.py` (111 LOC) — Copernicus SciHub query (Sentinel-1/2/3)
> - `download_utils.py` (61 LOC) — generic HTTP-with-auth download
> - `download_pv_product.py` (328 LOC) — Proba-V product download from VITO
> - `tileserver.py` (68 LOC) — XYZ tileserver consumer
>
> **Role:** the **discovery and acquisition** plumbing. Find scenes by AOI + time, download them, or read from a web tile server. **No ASCII diagrams in any of these.** They're all small utility modules.

---

## 1. The shape of the discovery layer

Most users don't have raw paths to satellite scenes — they have an AOI and a date range. The work flow is:

1. **Query a catalog** ("what scenes overlap my AOI in this date range?") → list of product metadata.
2. **Filter** ("only ones with < 20% cloud cover") → narrower list.
3. **Download** (or, increasingly, just construct cloud-bucket URLs and read lazily) → local files or paths.
4. **Read** with one of the chapters 14–17 readers.

The five small modules in this chapter cover steps 1–3. Step 4 is the rest of the package.

---

## 2. `query_utils.py` — generic spatial helpers

Three functions, all collection-agnostic. The shared infrastructure that the per-collection query modules build on.

| Function | What it does |
|---|---|
| `select_polygons_overlap(polygons, aoi)` | Return indices of polygons that maximally overlap an AOI. Used to rank candidate scenes by coverage. |
| `filter_products_overlap(area, ...)` | Filter a list of products by spatial overlap with an area. |
| `solar_datetime(area=None, ...)` | Compute local solar datetime for an AOI — useful for filtering scenes by daylight conditions or computing solar geometry. |

Used by the per-collection query modules (`scihubcopernicus_query`, `ee_query`, etc.) as common machinery.

---

## 3. `scihubcopernicus_query.py` — Copernicus SciHub queries

```python
get_api(api_url='https://scihub.copernicus.eu/dhus/') -> SentinelAPI
query(area, date_start, date_end, ...) -> gpd.GeoDataFrame
```

Wraps the [`sentinelsat`](https://sentinelsat.readthedocs.io/) library to issue spatial-temporal queries against the official Copernicus SciHub catalog. Returns a GeoDataFrame with one row per matched product, columns including `geometry`, `beginposition`, `cloudcoverpercentage`, `producttype`, `relativeorbitnumber`.

This is the **legacy** way to query Sentinel scenes. The Copernicus Data Space (`https://catalogue.dataspace.copernicus.eu/`) replaced SciHub in 2023; for new workflows, prefer:

- **GEE** ([Chapter 16](16_earth_engine.md)) — wider catalog, no auth complications.
- **STAC catalogs** — Element84, Microsoft Planetary Computer (Chapter 14 mentions both). The corresponding `s2_load_from_feature_*` adapters in `S2_SAFE_reader.py` consume STAC features directly.

The SciHub module remains for users with workflows that pre-date the 2023 transition. Authentication uses Copernicus credentials via `sentinelsat.SentinelAPI(user, password, api_url)`.

---

## 4. `download_utils.py` — generic HTTP download

Single function:

```python
download_product(link_down, filename=None, auth=None, ...) -> str
```

Stream-downloads a URL to disk with optional HTTP basic auth. The base building block for all the per-source downloaders. Uses `requests` (which is a hard dep of this module — `import` raises `ImportError` if not installed).

Notable: the `auth` argument is just `requests`-style — pass a `(user, password)` tuple or a `requests.auth.AuthBase` subclass. No package-specific auth abstraction.

---

## 5. `download_pv_product.py` — Proba-V VITO downloads

Proba-V data lives on [VITO's product distribution portal](https://www.vito-eodata.be/PDF/datapool/). Authentication uses VITO credentials stored in `~/.georeader/auth_vito.json`.

The module exposes a download workflow:

| Function | Purpose |
|---|---|
| `get_auth()` | Read VITO credentials from disk |
| `download_product(link_down, ...)` | Wraps `download_utils.download_product` with VITO auth |
| `fetch_products_date_region(date, bounding_box, ...)` | Find Proba-V products for a date + bbox |
| `download_L2A_date_region(date, bounding_box, dir_out, resolution="333M", ...)` | Find + download in one call |
| `download_L2A_product(year, month, day, ...)` | Download a single product by ID |
| `download_L2A_product_from_name(product_name, dir_out)` | Download by product name |
| `download_L2A_xml_from_name(product_name, dir_out)` | Download metadata XML only |
| `is_downloadable(url, auth=None)` | Check URL accessibility |
| `exists_product_name(product_name, dir_out)` | Local existence check |
| `extract_L2_file_naming_content(product_name)` | Parse fields from filename |
| `read_L2A_xml(product_name, dir_out)` | Parse the XML metadata |

The granularity is high because Proba-V's distribution is more rigid than modern STAC catalogs — there are specific URL patterns for `(year, month, day, camera, resolution)` tuples that the module encodes.

This pairs with [`probav_image_operational.py`](17_legacy_sensors.md) — the download module fetches the files, the operational reader reads them. Together they cover the full Proba-V pipeline.

---

## 6. `tileserver.py` — XYZ tile server consumer

```python
read_from_tileserver(tile_server, geometry, ...) -> GeoTensor
```

Reads from a slippy-map XYZ tile server (Mapbox, Google Maps, OSM-style basemaps). Given a polygon AOI:

1. Compute which `(x, y, z)` tiles cover the AOI via [`mercantile`](https://github.com/mapbox/mercantile).
2. HTTP-GET each tile from the tileserver URL pattern (e.g., `https://tile.openstreetmap.org/{z}/{x}/{y}.png`).
3. Decode the PNG/JPEG to a numpy array.
4. Mosaic them via `mosaic.spatial_mosaic` (Chapter 8) into a single `GeoTensor` in Web Mercator (EPSG:3857).

Useful for:

- **Basemaps as raster context.** Drop OSM tiles under a derived layer for visualisation.
- **Quick previews.** When you have an AOI but no archive access, a tileserver gives you something to work with.

The output is in **Web Mercator**, not the native CRS of the satellite imagery you're typically processing. If you need it in another CRS, follow with `read.read_to_crs` ([Chapter 5 §4](05_read.md)).

---

## 7. Function reference (quick)

| Module | Functions |
|---|---|
| `query_utils.py` | `select_polygons_overlap`, `filter_products_overlap`, `solar_datetime` |
| `scihubcopernicus_query.py` | `get_api`, `query` |
| `download_utils.py` | `download_product` |
| `download_pv_product.py` | 11 functions covering Proba-V's full discovery + download flow |
| `tileserver.py` | `read_from_tileserver` |

---

## 8. Sharp edges

- **SciHub is deprecated.** Don't build new code against `scihubcopernicus_query`. Use Microsoft Planetary Computer or Element84 STAC catalogs instead.
- **Credential files are plaintext JSON.** `~/.georeader/auth_emit.json`, `~/.georeader/auth_vito.json`. Make sure they're not in dotfile sync. The package doesn't encrypt them.
- **Tileserver outputs are in EPSG:3857.** Web Mercator is *not* the same as UTM zones — don't `same_extent` against an S2 scene without reprojecting first.
- **Tileserver licensing.** Pulling many tiles from OSM violates their tile-usage policy (rate limit, attribution). For production / research you'd want a self-hosted tileserver or a paid Mapbox account.
- **Proba-V download URLs encode many fields.** A typo in the camera ID or resolution silently returns 404. The `exists_product_name` helper exists for the same reason.
- **`requests` is a hard import for `download_utils`.** Other modules import lazily; this one fails on module load if `requests` isn't installed.

---

## 9. Connection to `geotoolz`

The discovery layer is **explicitly out of scope** for `geotoolz` per [`geotoolz.md` §1.3](../plans/geotoolz.md):

> **Not `georeader`.** No I/O, no CRS plumbing, no reader classes, no catalog construction. Those are `georeader`'s job.

But there's a clean handoff: `geotoolz.catalog_ops.CatalogPipeline(catalog, op).run()` (from [§1.2 of the plan](../plans/geotoolz.md)) consumes a `georeader.catalog.GeoCatalog` — which (per [`geocatalog.md`](../plans/geocatalog.md) and [`geoduckdb.md`](../plans/geoduckdb.md)) is a planned addition to georeader that would unify all the per-collection query modules in this chapter.

The future state, per the plans:

- All per-collection query modules return GeoDataFrames in the same schema.
- A `GeoCatalog` wraps any GeoDataFrame as a queryable, persistable spatial index.
- `geotoolz.catalog_ops.CatalogPipeline` takes a `GeoCatalog`, applies a `Sequential` of operators per row, writes outputs.

Today, you build that pipeline by hand — query, filter, loop, read, process, write. The chapters 14–18 give you the building blocks; the orchestration is your code.

---

## 10. Closing the tutorial

This is the final chapter. The full coverage:

- **Part I (Ch. 1–3):** core data model — `GeoTensor`, abstract reader protocols, `RasterioReader`.
- **Part II (Ch. 4–6):** reading & windowing — `window_utils`, `read.py`, `slices`.
- **Part III (Ch. 7–10):** geometry & gridding — `griddata`, `mosaic`, `rasterize`, `vectorize`.
- **Part IV (Ch. 11–13):** radiometry & I/O — `reflectance`, `save`, miscellaneous utilities.
- **Part V (Ch. 14–18):** sensor readers — Sentinel-2, hyperspectral trio, Earth Engine, legacy sensors, query/download.

Total preserved: **all ASCII diagrams across 17 modules** that had them, plus the implementation walkthroughs (the 8-step `read_reproject` annotation in [Chapter 5](05_read.md)) that were also at risk of being cleaned up. The tutorial mirrors the package structure so a reader can navigate from "I want to do X" → which module → which chapter → the diagrams + prose explaining it.

The closing observation: **georeader is a coherent stack, not a kitchen sink.** Every module builds on the layer below it. `GeoTensor` (Chapter 1) carries metadata; `window_utils` (Chapter 4) does pixel-geo math; `read.py` (Chapter 5) composes the two; `mosaic` / `rasterize` / `vectorize` (Chapters 8–10) build on `read.py`; the sensor readers (Chapters 14–17) plug into the `GeoData` protocol so all of the above just works on real-world products. That layering is what makes a `geotoolz` operator-composition library viable — it has a clean substrate to sit on.
