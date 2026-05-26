"""Shared helpers for the geostack walkthrough notebooks.

Every notebook under ``projects/geostack/notebooks/`` reaches for the
same handful of real-data loaders — Sentinel-2 chips from Microsoft
Planetary Computer, ERA5 / MODIS xarray grids, GBIF occurrence points,
Overture buildings GeoParquet, and so on. Centralising those loaders
here keeps the notebooks short and the MPC plumbing in one place.

Top-level imports surface the functions you use most:

    from geostack import load_s2_chip, load_s2_timestack

The full surface (incl. less-frequent helpers) lives in
:mod:`geostack.data`.
"""

from __future__ import annotations

from geostack.data import (
    LAKE_TAHOE_BBOX,
    LAKE_TAHOE_TILE,
    LISBON_BBOX,
    LISBON_TILE,
    load_gbif_points,
    load_natural_earth_admin1,
    load_overture_buildings_url,
    load_s2_chip,
    load_s2_full_tile,
    load_s2_timestack,
    load_stac_items,
    mpc_catalog,
)


__all__ = [
    "LAKE_TAHOE_BBOX",
    "LAKE_TAHOE_TILE",
    "LISBON_BBOX",
    "LISBON_TILE",
    "load_gbif_points",
    "load_natural_earth_admin1",
    "load_overture_buildings_url",
    "load_s2_chip",
    "load_s2_full_tile",
    "load_s2_timestack",
    "load_stac_items",
    "mpc_catalog",
]
