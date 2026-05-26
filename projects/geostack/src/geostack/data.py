"""Concrete real-data loaders used across the geostack notebooks.

Everything here returns either a ``GeoTensor`` (geotoolz / geopatcher
substrate), a plain numpy array, an xarray ``DataArray``, or a
``geopandas.GeoDataFrame`` — whatever the calling notebook needs.

All MPC reads are anonymous (the ``planetary_computer.sign_inplace``
modifier signs URLs with a short-lived SAS token transparently); no
API key is required.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import planetary_computer
import pystac_client
import rioxarray
import xarray as xr
from georeader.geotensor import GeoTensor
from rasterio.enums import Resampling


# ---------------------------------------------------------------------------
# Canonical AOIs reused across notebooks. Keep them tiny + well-known so
# the build is reproducible and the figures are visually rich.
# ---------------------------------------------------------------------------

#: Lake Tahoe main body, EPSG:4326.
LAKE_TAHOE_BBOX: tuple[float, float, float, float] = (-120.10, 38.92, -119.93, 39.27)
LAKE_TAHOE_TILE = "10SGJ"

#: Lisbon estuary, EPSG:4326 — Atlantic + urban, exercises water/land + admin polygons.
LISBON_BBOX: tuple[float, float, float, float] = (-9.30, 38.65, -9.05, 38.85)
LISBON_TILE = "29SMC"


# ---------------------------------------------------------------------------
# STAC plumbing.
# ---------------------------------------------------------------------------


def mpc_catalog() -> pystac_client.Client:
    """Open a signed Planetary Computer STAC client. Reused everywhere."""
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def load_stac_items(
    collection: str,
    bbox: tuple[float, float, float, float],
    date_range: str,
    *,
    max_cloud_cover: float = 5.0,
    tile: str | None = None,
    limit: int = 50,
) -> list[Any]:
    """Search MPC for items, sorted ascending by cloud cover.

    Args:
        collection: STAC collection id (e.g. ``"sentinel-2-l2a"``,
            ``"landsat-c2-l2"``).
        bbox: ``(west, south, east, north)`` in EPSG:4326.
        date_range: ISO ``"YYYY-MM-DD/YYYY-MM-DD"``.
        max_cloud_cover: ``eo:cloud_cover < max_cloud_cover`` filter.
            Pass ``100`` to disable.
        tile: Optional Sentinel-2 MGRS tile filter (e.g. ``"10SGJ"``).
            For non-Sentinel collections, leave ``None``.
        limit: Server-side row cap.
    """
    query: dict[str, Any] = {"eo:cloud_cover": {"lt": max_cloud_cover}}
    if tile is not None:
        query["s2:mgrs_tile"] = {"eq": tile}
    search = mpc_catalog().search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range,
        query=query,
        limit=limit,
    )
    items = list(search.items())
    return sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 0))


# ---------------------------------------------------------------------------
# Sentinel-2 single-chip loaders.
# ---------------------------------------------------------------------------


def _load_band(
    item: Any,
    asset_key: str,
    *,
    bbox: tuple[float, float, float, float],
    ref: xr.DataArray | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray:
    da = rioxarray.open_rasterio(item.assets[asset_key].href, masked=False)
    da = da.squeeze("band", drop=True).rio.clip_box(*bbox, crs="EPSG:4326")
    if ref is not None:
        da = da.rio.reproject_match(ref, resampling=resampling)
    return da


def load_s2_chip(
    *,
    bbox: tuple[float, float, float, float] = LAKE_TAHOE_BBOX,
    date_range: str = "2024-06-01/2024-07-15",
    tile: str | None = LAKE_TAHOE_TILE,
    bands: tuple[str, ...] = ("B02", "B03", "B04", "B08"),
    max_cloud_cover: float = 5.0,
    dtype: str = "uint16",
) -> GeoTensor:
    """Load one cloud-free Sentinel-2 L2A chip from MPC as a stacked ``GeoTensor``.

    Returns a ``(len(bands), H, W)`` carrier. Band order matches the order
    in ``bands`` — the default is **BGRN** (B02, B03, B04, B08), so:

    - blue  = idx 0
    - green = idx 1
    - red   = idx 2
    - nir   = idx 3

    Other bands resolve at runtime (e.g. ``"B11"`` SWIR-1, ``"B12"`` SWIR-2,
    ``"SCL"`` scene-classification). Non-10 m bands are bilinear-resampled
    to the B04 grid so all bands share one transform.
    """
    items = load_stac_items(
        "sentinel-2-l2a",
        bbox,
        date_range,
        max_cloud_cover=max_cloud_cover,
        tile=tile,
    )
    if not items:
        raise RuntimeError(
            f"no S2 L2A items found for bbox={bbox} date={date_range} "
            f"tile={tile} cloud<{max_cloud_cover}"
        )
    item = items[0]

    ref = _load_band(item, bands[0], bbox=bbox)
    stack = [ref.values]
    for band in bands[1:]:
        resampling = Resampling.nearest if band == "SCL" else Resampling.bilinear
        stack.append(
            _load_band(item, band, bbox=bbox, ref=ref, resampling=resampling).values
        )

    return GeoTensor(
        values=np.stack(stack, axis=0).astype(dtype),
        transform=ref.rio.transform(),
        crs=ref.rio.crs,
        fill_value_default=0,
    )


def load_s2_full_tile(
    *,
    date_range: str = "2024-06-01/2024-07-15",
    tile: str = LAKE_TAHOE_TILE,
    bands: tuple[str, ...] = ("B04", "B08"),
    max_cloud_cover: float = 5.0,
) -> GeoTensor:
    """Load **the full MGRS tile** (no bbox clip) for streaming demos.

    Sentinel-2 tiles are 109_800 x 109_800 m at 10 m -> ~10_980 x 10_980 px.
    Returns a ``(C, ~10980, ~10980)`` ``uint16`` carrier. Big enough to
    motivate ``SpatialOverlapAdd`` with the zarr-backed accumulator.
    """
    items = load_stac_items(
        "sentinel-2-l2a",
        (-180, -90, 180, 90),
        date_range,
        max_cloud_cover=max_cloud_cover,
        tile=tile,
    )
    if not items:
        raise RuntimeError(
            f"no S2 L2A items found for tile={tile} date={date_range} "
            f"cloud<{max_cloud_cover}"
        )
    item = items[0]

    ref = rioxarray.open_rasterio(item.assets[bands[0]].href, masked=False)
    ref = ref.squeeze("band", drop=True)
    stack = [ref.values]
    for band in bands[1:]:
        da = rioxarray.open_rasterio(item.assets[band].href, masked=False)
        da = da.squeeze("band", drop=True).rio.reproject_match(
            ref,
            resampling=Resampling.bilinear,
        )
        stack.append(da.values)

    return GeoTensor(
        values=np.stack(stack, axis=0).astype("uint16"),
        transform=ref.rio.transform(),
        crs=ref.rio.crs,
        fill_value_default=0,
    )


def load_s2_timestack(
    *,
    bbox: tuple[float, float, float, float] = LAKE_TAHOE_BBOX,
    date_range: str = "2024-04-01/2024-09-30",
    tile: str | None = LAKE_TAHOE_TILE,
    bands: tuple[str, ...] = ("B04", "B08"),
    max_items: int = 12,
    max_cloud_cover: float = 20.0,
) -> tuple[np.ndarray, list[str], xr.DataArray]:
    """Load a multi-date stack of Sentinel-2 chips.

    Returns ``(stack, dates, ref_da)`` where:

    - ``stack`` is ``(T, C, H, W)`` uint16
    - ``dates`` is a list of ISO date strings
    - ``ref_da`` is the reference ``xr.DataArray`` (carries the CRS /
      transform / spatial coords).
    """
    items = load_stac_items(
        "sentinel-2-l2a",
        bbox,
        date_range,
        max_cloud_cover=max_cloud_cover,
        tile=tile,
    )
    if not items:
        raise RuntimeError(
            f"no S2 L2A items found for bbox={bbox} date={date_range} tile={tile}"
        )
    # Sort by *date* (not cloud cover) for a temporal stack.
    items = sorted(items, key=lambda x: x.properties["datetime"])[:max_items]

    # Use the first scene's B04 as the reference grid for all dates.
    ref = _load_band(items[0], bands[0], bbox=bbox)
    chips, dates = [], []
    for item in items:
        per_band = [_load_band(item, bands[0], bbox=bbox, ref=ref).values]
        for band in bands[1:]:
            per_band.append(_load_band(item, band, bbox=bbox, ref=ref).values)
        chips.append(np.stack(per_band, axis=0))
        dates.append(item.properties["datetime"][:10])

    return np.stack(chips, axis=0).astype("uint16"), dates, ref


# ---------------------------------------------------------------------------
# Non-S2 helpers — Landsat / NAIP / xarray grids.
# ---------------------------------------------------------------------------


def load_landsat_chip(
    *,
    bbox: tuple[float, float, float, float] = LAKE_TAHOE_BBOX,
    date_range: str = "2023-06-01/2023-09-30",
    bands: tuple[str, ...] = ("blue", "green", "red", "nir08"),
    max_cloud_cover: float = 5.0,
) -> GeoTensor:
    """Load one Landsat 8/9 Collection-2 L2 chip (30 m surface reflectance)."""
    items = load_stac_items(
        "landsat-c2-l2",
        bbox,
        date_range,
        max_cloud_cover=max_cloud_cover,
    )
    if not items:
        raise RuntimeError(f"no Landsat items found for {bbox} {date_range}")
    item = items[0]

    ref = _load_band(item, bands[0], bbox=bbox)
    stack = [ref.values]
    for band in bands[1:]:
        stack.append(_load_band(item, band, bbox=bbox, ref=ref).values)

    return GeoTensor(
        values=np.stack(stack, axis=0).astype("uint16"),
        transform=ref.rio.transform(),
        crs=ref.rio.crs,
        fill_value_default=0,
    )


def load_naip_chip(
    *,
    bbox: tuple[float, float, float, float] = (-93.65, 41.99, -93.60, 42.03),
    date_range: str = "2021-01-01/2023-12-31",
) -> GeoTensor:
    """Load one NAIP RGB+NIR chip (0.6 m aerial) — defaults to Iowa farmland."""
    items = load_stac_items(
        "naip",
        bbox,
        date_range,
        max_cloud_cover=100,  # NAIP has no cloud %.
    )
    if not items:
        raise RuntimeError(f"no NAIP items found for {bbox} {date_range}")
    item = items[0]

    da = rioxarray.open_rasterio(item.assets["image"].href, masked=False)
    da = da.rio.clip_box(*bbox, crs="EPSG:4326")
    return GeoTensor(
        values=da.values.astype("uint8"),
        transform=da.rio.transform(),
        crs=da.rio.crs,
        fill_value_default=0,
    )


def load_era5_chip(
    *,
    bbox: tuple[float, float, float, float] = LAKE_TAHOE_BBOX,
    date_range: str = "2024-06-01/2024-06-30",
    variable: str = "air_temperature_at_2_metres",
    backend: Literal["era5-pds", "cop-dem-glo-30"] = "era5-pds",
) -> xr.DataArray:
    """Load a small xarray ``DataArray`` for the xarray-Field demos.

    Defaults to ERA5-PDS surface temperature — small enough to keep
    fetch latency down, real enough to exercise the xarray Field adapter
    end-to-end. The Copernicus DEM ``cop-dem-glo-30`` is also supported
    when you want elevation rather than weather.
    """
    if backend == "era5-pds":
        # ERA5 PDS is published as a zarr store at MPC — open the catalog,
        # find the asset, open as xarray.
        cat = mpc_catalog()
        coll = cat.get_collection("era5-pds")
        # Pick an asset that exposes the variable as a zarr group.
        # ERA5-PDS structure: yearly zarr per variable.
        year = date_range[:4]
        asset = coll.assets.get(f"{year}-{variable}", None)
        if asset is None:
            raise RuntimeError(
                f"era5-pds has no {variable} for year={year}. "
                "Try 2024 + air_temperature_at_2_metres."
            )
        store = xr.open_dataset(asset.href, engine="zarr", chunks={})
        var = store[variable]
        time_slice = slice(*date_range.split("/"))
        out = var.sel(
            time=time_slice,
            lon=slice(bbox[0] % 360, bbox[2] % 360),
            lat=slice(bbox[3], bbox[1]),
        ).load()
        return out
    if backend == "cop-dem-glo-30":
        items = load_stac_items(
            "cop-dem-glo-30",
            bbox,
            "1900-01-01/2099-12-31",
            max_cloud_cover=100,
        )
        if not items:
            raise RuntimeError(f"no DEM items found for {bbox}")
        da = rioxarray.open_rasterio(items[0].assets["data"].href, masked=False)
        return da.squeeze("band", drop=True).rio.clip_box(*bbox, crs="EPSG:4326")
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Vector helpers — GBIF points, Natural Earth admin polygons, Overture.
# ---------------------------------------------------------------------------


def load_gbif_points(
    species_key: int = 5285750,  # Quercus agrifolia (California live oak)
    *,
    bbox: tuple[float, float, float, float] = (-124, 32, -114, 42),
    limit: int = 500,
):
    """Pull GBIF occurrence points (vector ``geopandas.GeoDataFrame``).

    Default: California live oak (``Quercus agrifolia``, taxonKey 5285750)
    across the California bbox — gives a believable point cloud for
    KNN-graph / radius-graph geometry demos.
    """
    import geopandas as gpd
    import requests
    from shapely.geometry import Point

    params = {
        "taxonKey": species_key,
        "decimalLatitude": f"{bbox[1]},{bbox[3]}",
        "decimalLongitude": f"{bbox[0]},{bbox[2]}",
        "hasCoordinate": "true",
        "limit": limit,
    }
    r = requests.get(
        "https://api.gbif.org/v1/occurrence/search", params=params, timeout=60
    )
    r.raise_for_status()
    data = r.json()
    rows = [
        {
            "key": rec["key"],
            "scientificName": rec.get("scientificName"),
            "eventDate": rec.get("eventDate"),
            "geometry": Point(rec["decimalLongitude"], rec["decimalLatitude"]),
        }
        for rec in data.get("results", [])
        if rec.get("decimalLongitude") is not None
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def load_natural_earth_admin1():
    """Natural Earth admin-1 (states / provinces) — small global vector."""
    import geopandas as gpd

    url = (
        "https://naciscdn.org/naturalearth/110m/cultural/"
        "ne_110m_admin_1_states_provinces.zip"
    )
    return gpd.read_file(url)


def load_overture_buildings_url(*, release: str = "2024-08-20.0") -> str:
    """Return the public S3 URL of the Overture buildings GeoParquet release.

    Use with DuckDB's ``read_parquet(s3_url)`` for the
    ``catalog/04_duckdb`` walkthrough. The buildings theme is the
    largest Overture file (~1.5B rows at 2024-08-20), partitioned by
    ``country_code``. Use ``hive_partitioning=true`` to skip irrelevant
    partitions cheaply.
    """
    return (
        f"s3://overturemaps-us-west-2/release/{release}/theme=buildings/type=building/*"
    )
