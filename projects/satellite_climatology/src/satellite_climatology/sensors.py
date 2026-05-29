"""Prototype sensor registry (Available layer only).

PC-hosted collections, anonymous read. The full v4 design
(``plans/satellite_climatology/v4_coverage_planner_api.md``) carries a
richer registry with multiple sources per sensor (PC / Earth Search / CDSE
/ CMR) and the Acquired (external holdings) + taskable fields; this prototype
keeps just what the global Available heatmap needs.
"""

from __future__ import annotations

from dataclasses import dataclass


_PC = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass(frozen=True)
class Sensor:
    key: str
    endpoint: str
    collection_id: str
    cloud_field: str | None
    requires_pc_signing: bool = True
    description: str = ""


SENSORS: dict[str, Sensor] = {
    "modis-09a1": Sensor(
        "modis-09a1",
        _PC,
        "modis-09A1-061",
        None,
        description="MODIS 8-day surface reflectance (500 m) — best global demo",
    ),
    "landsat-c2-l2": Sensor(
        "landsat-c2-l2",
        _PC,
        "landsat-c2-l2",
        "eo:cloud_cover",
        description="Landsat C2 L2 (30 m) — has cloud cover",
    ),
    "sentinel-2-l2a": Sensor(
        "sentinel-2-l2a",
        _PC,
        "sentinel-2-l2a",
        "eo:cloud_cover",
        description="Sentinel-2 L2A (10-60 m) — dense; use a regional bbox",
    ),
}
