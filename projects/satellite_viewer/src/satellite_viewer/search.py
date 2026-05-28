"""Sensor-agnostic discovery: AOI x date-range -> GeoDataFrame of hits.

One row per scene/granule. Columns:

| col            | dtype     | notes                                   |
|----------------|-----------|-----------------------------------------|
| `id`           | string    | STAC item id                            |
| `datetime`     | tz-aware  | acquisition timestamp                   |
| `geometry`     | geometry  | scene footprint                         |
| `sensor`       | string    | sensor name from the registry           |
| `cloud_cover`  | float     | percent 0-100, or NaN                   |
| `preview_url`  | string    | signed thumbnail URL, or None           |

Delegates to `pystac-client` and signs preview hrefs with
`planetary-computer` when the registry says so.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from satellite_viewer.sensors import SENSORS, SensorConfig


def search(
    sensor: str,
    aoi: BaseGeometry,
    start: datetime,
    end: datetime,
    *,
    max_items: int = 100,
    cloud_lt: float | None = None,
) -> gpd.GeoDataFrame:
    """Return scenes / granules of `sensor` intersecting `aoi` between dates."""

    if sensor not in SENSORS:
        raise KeyError(f"unknown sensor {sensor!r}; known: {sorted(SENSORS)}")
    cfg = SENSORS[sensor]
    return _stac_search(cfg, aoi, start, end, max_items, cloud_lt)


def _stac_search(
    cfg: SensorConfig,
    aoi: BaseGeometry,
    start: datetime,
    end: datetime,
    max_items: int,
    cloud_lt: float | None,
) -> gpd.GeoDataFrame:
    # Heavy imports kept lazy so `import satellite_viewer` stays cheap.
    import planetary_computer
    import pystac_client

    modifier = planetary_computer.sign_inplace if cfg.requires_pc_signing else None
    client = pystac_client.Client.open(cfg.stac_endpoint, modifier=modifier)

    query: dict[str, Any] | None = None
    if cloud_lt is not None and cfg.cloud_field is not None:
        query = {cfg.cloud_field: {"lt": cloud_lt}}

    stac_search = client.search(
        collections=[cfg.collection_id],
        intersects=aoi.__geo_interface__,
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        max_items=max_items,
        query=query,
    )

    rows: list[dict] = []
    for item in stac_search.items():
        preview_url: str | None = None
        if cfg.preview_asset and cfg.preview_asset in item.assets:
            preview_url = item.assets[cfg.preview_asset].href

        cloud: float | None = None
        if cfg.cloud_field is not None:
            cloud = item.properties.get(cfg.cloud_field)

        ts = pd.Timestamp(item.datetime)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

        rows.append(
            {
                "id": item.id,
                "datetime": ts,
                "geometry": shape(item.geometry),
                "sensor": cfg.name,
                "cloud_cover": cloud,
                "preview_url": preview_url,
            }
        )

    if not rows:
        return _empty_gdf()
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "id": pd.Series(dtype="string"),
            "datetime": pd.Series(dtype="datetime64[ns, UTC]"),
            "geometry": gpd.GeoSeries([], crs="EPSG:4326"),
            "sensor": pd.Series(dtype="string"),
            "cloud_cover": pd.Series(dtype="float64"),
            "preview_url": pd.Series(dtype="string"),
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
