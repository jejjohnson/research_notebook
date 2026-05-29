"""The shared global grid + footprint burning.

Prototype scope: a regular lat/lon grid in EPSG:4326 at a configurable
resolution (0.1° default -> 1800 x 3600 = 6.48M cells). Arrays are stored
**north-up** (row 0 = +90° lat) so they drop straight into a folium
ImageOverlay without flipping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class GridSpec:
    resolution: float = 0.1  # degrees

    @property
    def width(self) -> int:
        return round(360.0 / self.resolution)

    @property
    def height(self) -> int:
        return round(180.0 / self.resolution)

    @property
    def lats(self) -> np.ndarray:
        # Cell-centre latitudes, descending (north-up: row 0 = +90).
        r = self.resolution
        return np.linspace(90 - r / 2, -90 + r / 2, self.height)

    @property
    def lons(self) -> np.ndarray:
        r = self.resolution
        return np.linspace(-180 + r / 2, 180 - r / 2, self.width)

    def empty(self) -> np.ndarray:
        return np.zeros((self.height, self.width), dtype="float32")

    def burn(self, geom: BaseGeometry, arr: np.ndarray, value: float = 1.0) -> None:
        """Add ``value`` to every cell the footprint touches (in place).

        Rasterises only over the footprint's bounding window, then adds into
        the matching slice of ``arr`` — so a single MODIS tile or S2 scene is
        cheap even on the full global grid.
        """
        minx, miny, maxx, maxy = geom.bounds
        # Skip antimeridian-spanning footprints (their bounds smear a whole
        # row band); rare and not worth special-casing in a prototype.
        if maxx - minx > 180:
            return
        r = self.resolution
        col0 = max(int((minx + 180) / r), 0)
        col1 = min(int(np.ceil((maxx + 180) / r)), self.width)
        row0 = max(int((90 - maxy) / r), 0)  # north-up: small row = high lat
        row1 = min(int(np.ceil((90 - miny) / r)), self.height)
        if col1 <= col0 or row1 <= row0:
            return
        # North-up sub-window transform: origin at the window's NW corner.
        west = -180 + col0 * r
        north = 90 - row0 * r
        transform = from_origin(west, north, r, r)
        mask = rasterize(
            [(geom, 1)],
            out_shape=(row1 - row0, col1 - col0),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        arr[row0:row1, col0:col1] += mask.astype("float32") * value
