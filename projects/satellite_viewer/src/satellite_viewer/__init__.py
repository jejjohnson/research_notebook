"""Preview intersecting satellite imagery before download.

Public surface:

- `SENSORS` — registry of every supported sensor.
- `search(...)` — sensor-agnostic AOI x time-range query returning a
  GeoDataFrame of hits (one row per scene/granule).
- `credentials` — per-service credential accessors (Earthdata, GEE,
  Planetary Computer). Each raises `CredentialsMissingError` with
  setup instructions when something is missing.
"""

from __future__ import annotations

from satellite_viewer import credentials
from satellite_viewer.credentials import CredentialsMissingError
from satellite_viewer.search import search
from satellite_viewer.sensors import SENSORS, SensorConfig


__all__ = [
    "SENSORS",
    "CredentialsMissingError",
    "SensorConfig",
    "credentials",
    "search",
]
