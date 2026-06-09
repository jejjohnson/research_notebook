"""Human-readable labels for CDS in-situ stations.

The CDS surface-land archive identifies stations by GHCN/WMO codes (e.g.
``SP000008410``) with no place names attached. Two cheap, offline helpers make
the notebooks readable:

* :func:`country_from_id` decodes the 2-letter FIPS country prefix of the ID;
* :func:`nearest_city` maps a ``(lon, lat)`` to the closest major city from a
  small built-in gazetteer of the western Mediterranean.

Neither is authoritative (a station "near Córdoba" may sit some km outside it),
but they turn opaque codes into something a reader can place on a map.
"""

from __future__ import annotations

import numpy as np


# FIPS country prefixes that appear in the Iberian/W-Mediterranean bbox.
_COUNTRY = {
    "SP": "Spain",
    "PO": "Portugal",
    "GI": "Gibraltar",
    "MO": "Morocco",
    "AG": "Algeria",
    "FR": "France",
}

# Small gazetteer: (name, lon, lat). Covers Iberia + the N-African / island
# fringe that the bbox also catches.
CITIES: list[tuple[str, float, float]] = [
    ("Madrid", -3.70, 40.42),
    ("Barcelona", 2.17, 41.39),
    ("Valencia", -0.38, 39.47),
    ("Sevilla", -5.99, 37.39),
    ("Zaragoza", -0.89, 41.65),
    ("Málaga", -4.42, 36.72),
    ("Murcia", -1.13, 37.99),
    ("Alicante", -0.48, 38.35),
    ("Bilbao", -2.93, 43.26),
    ("Córdoba", -4.78, 37.89),
    ("Valladolid", -4.72, 41.65),
    ("Granada", -3.60, 37.18),
    ("Albacete", -1.86, 38.99),
    ("Badajoz", -6.97, 38.88),
    ("Salamanca", -5.66, 40.96),
    ("Cáceres", -6.37, 39.48),
    ("León", -5.57, 42.60),
    ("Burgos", -3.70, 42.34),
    ("Santander", -3.81, 43.46),
    ("A Coruña", -8.41, 43.36),
    ("Vigo", -8.72, 42.24),
    ("Lisbon", -9.14, 38.72),
    ("Porto", -8.61, 41.15),
    ("Faro", -7.93, 37.02),
    ("Coimbra", -8.43, 40.21),
    ("Gibraltar", -5.35, 36.14),
    ("Tangier", -5.80, 35.76),
    ("Melilla", -2.94, 35.29),
    ("Oran", -0.64, 35.70),
    ("Algiers", 3.06, 36.75),
]


def country_from_id(station_id: str) -> str:
    """Country name from a GHCN/WMO station ID's 2-letter FIPS prefix."""
    return _COUNTRY.get(str(station_id)[:2], "—")


def _haversine_km(lon1, lat1, lon2, lat2):
    radius = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def nearest_city(lon: float, lat: float) -> tuple[str, float]:
    """Return ``(city_name, distance_km)`` for the closest gazetteer city."""
    best_name, best_d = "—", np.inf
    for name, clon, clat in CITIES:
        d = float(_haversine_km(lon, lat, clon, clat))
        if d < best_d:
            best_name, best_d = name, d
    return best_name, best_d


def station_label(station_id: str, lon: float, lat: float) -> str:
    """A compact, readable label, e.g. ``'SP000008410 · near Córdoba, Spain'``."""
    city, dist = nearest_city(lon, lat)
    where = city if dist < 12 else f"near {city}"
    return f"{station_id} · {where}, {country_from_id(station_id)}"
