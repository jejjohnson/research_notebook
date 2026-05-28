"""Sensor registry.

Each entry maps a short user-facing key (`"sentinel-2-l2a"`) to a
`SensorConfig` describing **how to discover scenes** for that sensor:
which STAC endpoint to hit, which collection ID to search, which
metadata fields hold the timestamp / cloud-cover, and which asset key
holds a renderable preview thumbnail.

Scope: polar-orbiting / tasked sensors only. Geostationary platforms
(GOES, Himawari, Meteosat) are deliberately out of scope here — their
~5-15 minute cadence makes them a poor fit for "is there imagery over
my AOI?" preview-style discovery.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorConfig:
    """Static configuration for one sensor."""

    name: str
    stac_endpoint: str
    collection_id: str
    # The STAC asset key that yields a renderable preview/thumbnail.
    preview_asset: str | None = None
    # The STAC item property that holds cloud cover percent (0-100).
    cloud_field: str | None = None
    # Whether the STAC endpoint requires `planetary-computer` URL signing.
    requires_pc_signing: bool = False
    # Human description (shown in the UI dropdown).
    description: str = ""


SENSORS: dict[str, SensorConfig] = {
    "sentinel-2-l2a": SensorConfig(
        name="sentinel-2-l2a",
        stac_endpoint="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection_id="sentinel-2-l2a",
        preview_asset="rendered_preview",
        cloud_field="eo:cloud_cover",
        requires_pc_signing=True,
        description="Sentinel-2 L2A surface reflectance (10-60 m, ~5 day revisit)",
    ),
    "sentinel-1-grd": SensorConfig(
        name="sentinel-1-grd",
        stac_endpoint="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection_id="sentinel-1-grd",
        preview_asset="rendered_preview",
        cloud_field=None,
        requires_pc_signing=True,
        description="Sentinel-1 C-band SAR GRD (10 m, all-weather)",
    ),
    "landsat-c2-l2": SensorConfig(
        name="landsat-c2-l2",
        stac_endpoint="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection_id="landsat-c2-l2",
        preview_asset="rendered_preview",
        cloud_field="eo:cloud_cover",
        requires_pc_signing=True,
        description=(
            "Landsat Collection-2 L2 surface reflectance (30 m, ~16 day revisit)"
        ),
    ),
    "modis-09a1": SensorConfig(
        name="modis-09a1",
        stac_endpoint="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection_id="modis-09A1-061",
        preview_asset="rendered_preview",
        cloud_field=None,
        requires_pc_signing=True,
        description="MODIS Terra/Aqua 8-day surface reflectance (500 m)",
    ),
    "emit-l2a-rfl": SensorConfig(
        name="emit-l2a-rfl",
        stac_endpoint="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection_id="emit-l2a-rfl",
        preview_asset="rendered_preview",
        cloud_field=None,
        requires_pc_signing=True,
        description="EMIT L2A imaging spectrometer reflectance (60 m, tasked)",
    ),
}
