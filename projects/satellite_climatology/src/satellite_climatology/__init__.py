"""satellite_climatology — v4 coverage planner (M4.0 prototype)."""

from satellite_climatology.coverage import scan_and_grid, select
from satellite_climatology.grid import GridSpec
from satellite_climatology.sensors import SENSORS


__all__ = ["SENSORS", "GridSpec", "scan_and_grid", "select"]
