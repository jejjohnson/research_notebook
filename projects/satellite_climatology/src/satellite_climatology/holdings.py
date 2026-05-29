"""Acquired layer — our holdings from an external PostGIS table.

Generic + open-source: this module knows nothing about any particular
project. Connection details and the table/column names come entirely from
environment variables (loaded from a gitignored ``.env``), so any private
schema lives only in your local ``.env`` — never in this repo. Any failure
(unconfigured, driver missing, DB unreachable) raises ``HoldingsUnavailable``
so the dashboard degrades to "Acquired: off" instead of crashing.

Configure via ``.env`` (see ``.env.example``): ``COVERAGE_DB_*`` for the
connection and ``COVERAGE_TILES_*`` to map the generic column names below to
your table's real columns.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box as shp_box

from satellite_climatology.grid import GridSpec


CLOUD_FREE_LT = 20.0


class HoldingsUnavailable(RuntimeError):
    """Raised when the holdings DB can't be read (unconfigured / driver / network)."""


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).with_name(".env"))  # package-local, gitignored
    load_dotenv()  # any .env up the cwd tree, without overriding


def _clean(value: str | None) -> str:
    return (value or "").strip().strip('"').strip("'")


def _engine():
    _load_env()
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL
    except ImportError as e:  # pragma: no cover
        raise HoldingsUnavailable(f"sqlalchemy/psycopg2 not installed: {e}") from e

    schema = _clean(os.environ.get("COVERAGE_DB_SCHEMA")) or "public"
    sslmode = _clean(os.environ.get("COVERAGE_DB_SSLMODE")) or "require"
    # Include public on the search_path so PostGIS functions (ST_Intersects,
    # ST_GeomFromText, …) resolve even when the data lives in another schema.
    connect_args = {"connect_timeout": 10, "options": f"-csearch_path={schema},public"}

    url_str = _clean(os.environ.get("COVERAGE_DB_URL"))
    if url_str:
        return create_engine(url_str, connect_args=connect_args), schema

    host = _clean(os.environ.get("COVERAGE_DB_HOST"))
    name = _clean(os.environ.get("COVERAGE_DB_NAME"))
    user = _clean(os.environ.get("COVERAGE_DB_USER"))
    pw = _clean(os.environ.get("COVERAGE_DB_PASSWORD"))
    port = _clean(os.environ.get("COVERAGE_DB_PORT")) or "5432"
    if not all([host, name, user, pw]):
        raise HoldingsUnavailable(
            "Holdings DB not configured. Set COVERAGE_DB_* in "
            "projects/satellite_climatology/src/satellite_climatology/.env "
            "(see .env.example)."
        )
    if ":" in host:  # tolerate a host that already includes the port
        host, _, maybe_port = host.partition(":")
        port = maybe_port or port
    connect_args["sslmode"] = sslmode
    url = URL.create(
        "postgresql+psycopg2",
        username=user,
        password=pw,
        host=host,
        port=int(port),
        database=name,
    )
    return create_engine(url, connect_args=connect_args), schema


def fetch_holdings(*, bbox=None, aoi=None, start: str, end: str):
    """Query the holdings table -> GeoDataFrame[datetime, sensor, cloud, geometry]
    for rows intersecting the area within the date range."""
    try:
        import geopandas as gpd
        from sqlalchemy import text
    except ImportError as e:  # pragma: no cover
        raise HoldingsUnavailable(f"geopandas not installed: {e}") from e

    engine, schema = _engine()
    table = _clean(os.environ.get("COVERAGE_TILES_TABLE")) or "tiles"
    g_col = _clean(os.environ.get("COVERAGE_TILES_GEOM")) or "geometry"
    d_col = _clean(os.environ.get("COVERAGE_TILES_DATE")) or "acquired_at"
    s_col = _clean(os.environ.get("COVERAGE_TILES_SENSOR")) or "sensor"
    c_col = _clean(os.environ.get("COVERAGE_TILES_CLOUD")) or "cloud_cover"

    geom = aoi if aoi is not None else (shp_box(*bbox) if bbox is not None else None)
    if geom is None:
        raise HoldingsUnavailable("fetch_holdings needs bbox or aoi")

    sql = text(
        f"SELECT {d_col} AS datetime, {s_col} AS sensor, {c_col} AS cloud, "
        f'{g_col} AS geometry FROM "{schema}".{table} '
        f"WHERE {d_col} >= :start AND {d_col} <= :end "
        f"AND ST_Intersects({g_col}, ST_GeomFromText(:wkt, 4326))"
    )
    params = {"start": str(start), "end": str(end), "wkt": geom.wkt}
    try:
        return gpd.read_postgis(sql, engine, geom_col="geometry", params=params)
    except Exception as e:  # connection / SQL / driver
        raise HoldingsUnavailable(f"{type(e).__name__}: {e}") from e


def holdings_stats(gdf, start: str, end: str, grid: GridSpec) -> xr.Dataset:
    """Grid holdings -> held_count, held_clear_count on (time, lat, lon)."""
    periods = pd.period_range(start=start, end=end, freq="M")
    held = {p: grid.empty() for p in periods}
    clear = {p: grid.empty() for p in periods}
    for _, row in gdf.iterrows():
        ts = pd.Timestamp(row["datetime"])
        if pd.isna(ts):
            continue
        period = (
            ts.tz_localize(None)
            if ts.tzinfo is None
            else ts.tz_convert("UTC").tz_localize(None)
        ).to_period("M")
        if period not in held:
            continue
        grid.burn(row["geometry"], held[period], 1.0)
        cloud = row.get("cloud")
        if cloud is not None and not pd.isna(cloud) and cloud < CLOUD_FREE_LT:
            grid.burn(row["geometry"], clear[period], 1.0)
    times = pd.to_datetime([p.to_timestamp() for p in periods])
    return xr.Dataset(
        {
            "held_count": (
                ("time", "lat", "lon"),
                np.stack([held[p] for p in periods]),
            ),
            "held_clear_count": (
                ("time", "lat", "lon"),
                np.stack([clear[p] for p in periods]),
            ),
        },
        coords={"time": times, "lat": grid.lats, "lon": grid.lons},
    )
