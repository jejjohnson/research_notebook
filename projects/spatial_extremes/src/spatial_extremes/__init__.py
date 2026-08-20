"""Spatial extremes: GEV + Gaussian-process modelling of climate extremes.

A step-by-step curriculum built on three packages:

* ``xrtoolz-reader`` — data layer (real CDS in-situ land stations over
  Iberia; import name ``xrreader``)
* ``xtremax`` — extreme-value layer (block maxima, GEV, return levels)
* ``pyrox-gp`` — Gaussian-process layer (kernels, latent fields, inference)

The :mod:`spatial_extremes.data` module is the shared bridge used by every
notebook; it transparently serves real CDS data when cached and a
schema-compatible synthetic series otherwise.
"""

from spatial_extremes.data import (
    IBERIA_BBOX,
    fetch_cds_insitu,
    load_annual_maxima,
    load_station_daily,
    synthetic_station_daily,
    to_station_time_dataarray,
)


__all__ = [
    "IBERIA_BBOX",
    "fetch_cds_insitu",
    "load_annual_maxima",
    "load_station_daily",
    "synthetic_station_daily",
    "to_station_time_dataarray",
]

__version__ = "0.1.0"
