"""Scan a STAC collection and grid it into a coverage Dataset.

Prototype pipeline (one function): STAC search → per-month footprint burn →
``xarray.Dataset`` on ``(time, lat, lon)``. Production swaps the scan for
``geocatalog.from_stac_search`` + a DuckDB grid GROUP BY and persists to
Zarr; the in-memory Dataset shape is identical so the app doesn't change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape

from satellite_climatology.grid import GridSpec
from satellite_climatology.sensors import SENSORS


CLOUD_FREE_LT = 20.0  # % — "clear" threshold for cloud_free_scene_count


def scan_and_grid(
    sensor_key: str,
    bbox: tuple[float, float, float, float] | None,
    start: str,
    end: str,
    grid: GridSpec,
    *,
    aoi=None,
    max_items: int = 5000,
) -> xr.Dataset:
    """Return monthly Available-coverage bands for one sensor over an area.

    Pass either ``bbox`` (lon/lat) or ``aoi`` (a shapely geometry — polygon
    or point — used as the STAC ``intersects`` filter). Bands:
    ``scenes_count`` and (if the sensor reports cloud) ``cloud_free_scene_count``
    on ``(time, lat, lon)`` with monthly bins.
    """
    import planetary_computer as pc
    import pystac_client
    from shapely.geometry import mapping

    cfg = SENSORS[sensor_key]
    client = pystac_client.Client.open(
        cfg.endpoint, modifier=pc.sign_inplace if cfg.requires_pc_signing else None
    )
    search_kwargs = dict(
        collections=[cfg.collection_id],
        datetime=f"{start}/{end}",
        max_items=max_items,
    )
    if aoi is not None:
        search_kwargs["intersects"] = mapping(aoi)
        extent = tuple(aoi.bounds)
    elif bbox is not None:
        search_kwargs["bbox"] = list(bbox)
        extent = tuple(bbox)
    else:
        raise ValueError("scan_and_grid needs either bbox or aoi")
    search = client.search(**search_kwargs)

    periods = pd.period_range(start=start, end=end, freq="M")
    scenes = {p: grid.empty() for p in periods}
    cloudfree = {p: grid.empty() for p in periods}

    n_items = 0
    for item in search.items():
        raw = item.datetime
        if raw is None:  # composites (MODIS) carry a start/end range instead
            raw = item.properties.get("start_datetime") or item.properties.get(
                "end_datetime"
            )
        ts = pd.Timestamp(raw)
        if pd.isna(ts):
            continue
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        period = ts.tz_localize(None).to_period("M")
        if period not in scenes:
            continue
        n_items += 1
        geom = shape(item.geometry)
        grid.burn(geom, scenes[period], 1.0)
        if cfg.cloud_field is not None:
            cloud = item.properties.get(cfg.cloud_field)
            if cloud is not None and cloud < CLOUD_FREE_LT:
                grid.burn(geom, cloudfree[period], 1.0)

    times = pd.to_datetime([p.to_timestamp() for p in periods])
    data = {
        "scenes_count": (("time", "lat", "lon"), np.stack([scenes[p] for p in periods]))
    }
    if cfg.cloud_field is not None:
        data["cloud_free_scene_count"] = (
            ("time", "lat", "lon"),
            np.stack([cloudfree[p] for p in periods]),
        )
    return xr.Dataset(
        data,
        coords={"time": times, "lat": grid.lats, "lon": grid.lons},
        attrs={
            "sensor": sensor_key,
            "n_items": n_items,
            "resolution": grid.resolution,
            "bbox": list(extent),
        },
    )


def select(
    ds: xr.Dataset,
    metric: str,
    *,
    agg: str = "rate",
    bbox: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Reduce the time axis to a single 2-D (lat, lon) layer.

    agg: rate=mean/month, total=sum, recent=last bin, worst=min (fewest).
    """
    da = ds[metric]
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        da = da.sel(lat=slice(maxy, miny), lon=slice(minx, maxx))  # lat descending
    if da.sizes["time"] <= 1:
        return da.isel(time=0)
    if agg == "total":
        return da.sum("time")
    if agg == "recent":
        return da.isel(time=-1)
    if agg == "worst":
        return da.min("time")
    return da.mean("time")  # rate
