"""Preview intersecting satellite imagery before download.

Public surface:

- `SENSORS` — registry of every supported sensor.
- `search(...)` — sensor-agnostic AOI x time-range query returning a
  GeoDataFrame of hits (one row per scene/granule).
"""

from __future__ import annotations

from satellite_viewer.search import search
from satellite_viewer.sensors import SENSORS, SensorConfig


__all__ = ["SENSORS", "SensorConfig", "search"]
