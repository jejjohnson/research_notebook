"""Derived spatial covariates for the station network.

The cached CDS archive carries only ``lon``/``lat`` per station. Physical
drivers of temperature extremes — **elevation** (lapse rate), **distance to the
coast** (maritime moderation) and local **terrain slope** — have to be derived
from external, public geodata:

* elevation & slope come from a public DEM via the OpenTopoData API
  (``srtm30m``), queried once and cached;
* distance-to-coast is computed offline from the Natural Earth coastline
  (shipped/downloaded by ``cartopy``) with a geodesic nearest-point distance.

Network access is needed **only** to build the cache. The notebooks call
:func:`load_station_features`, which reads the cached table — so a rendered
notebook never touches the network. Rebuild with ``scripts/build_features.py``
(or pass ``refresh=True``).
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import IBERIA_BBOX, default_cache_root


FEATURE_COLS = ["lon", "lat", "elevation", "dist_coast_km", "slope_deg"]
_DEM = "srtm30m"
_SLOPE_DELTA_DEG = 0.0045  # ~500 m finite-difference step for the terrain gradient


def features_path(root: Path | None = None) -> Path:
    """Location of the cached station-features table."""
    base = Path(root) if root is not None else default_cache_root()
    return base / "features" / "station_features.parquet"


# --------------------------------------------------------------------------- #
# Derivation primitives (network)                                             #
# --------------------------------------------------------------------------- #
def _opentopo_elevation(
    latlon: list[tuple[float, float]],
    dataset: str = _DEM,
    batch: int = 100,
    pause: float = 1.1,
    retries: int = 4,
) -> list[float | None]:
    """Elevation (m) for a list of ``(lat, lon)`` via OpenTopoData.

    Returns ``None`` for points the DEM has no data for (e.g. open sea).
    """
    out: list[float | None] = []
    for i in range(0, len(latlon), batch):
        chunk = latlon[i : i + batch]
        locs = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in chunk)
        url = f"https://api.opentopodata.org/v1/{dataset}?locations={locs}"
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    payload = json.load(resp)
                out.extend(d["elevation"] for d in payload["results"])
                break
            except Exception:
                time.sleep(pause * (attempt + 1))
        else:
            # All retries failed: abort rather than record None (which would be
            # cached as bogus 0 m elevation/slope). A real "no DEM data here"
            # (e.g. open sea) comes back as None *inside* a successful response.
            raise RuntimeError(
                f"OpenTopoData unreachable for a chunk after {retries} retries; "
                "aborting so a transient outage cannot poison the feature cache"
            )
        time.sleep(pause)
    return out


def _coast_geometry(bbox: tuple[float, float, float, float]):
    """Union of Natural Earth coastline segments near ``bbox`` (lon/lat)."""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box
    from shapely.ops import unary_union

    lon0, lon1, lat0, lat1 = bbox
    clip = box(lon0 - 1.5, lat0 - 1.5, lon1 + 1.5, lat1 + 1.5)
    fn = shpreader.natural_earth(
        resolution="10m", category="physical", name="coastline"
    )
    segs = [
        g.intersection(clip)
        for g in shpreader.Reader(fn).geometries()
        if g.intersects(clip)
    ]
    return unary_union(segs)


def _dist_coast_km(lonlat: np.ndarray, coast) -> np.ndarray:
    """Geodesic distance (km) from each ``(lon, lat)`` to the nearest coastline."""
    from pyproj import Geod
    from shapely.geometry import Point
    from shapely.ops import nearest_points

    geod = Geod(ellps="WGS84")
    out = np.empty(len(lonlat))
    for i, (lon, lat) in enumerate(lonlat):
        q = nearest_points(Point(lon, lat), coast)[1]
        _, _, dist_m = geod.inv(lon, lat, q.x, q.y)
        out[i] = dist_m / 1000.0
    return out


def _slope_deg(
    lonlat: np.ndarray, dataset: str = _DEM, delta: float = _SLOPE_DELTA_DEG
) -> np.ndarray:
    """Terrain slope (degrees) at each station from a 4-point DEM finite difference."""
    pts: list[tuple[float, float]] = []
    for lon, lat in lonlat:
        pts += [
            (lat, lon + delta),  # east
            (lat, lon - delta),  # west
            (lat + delta, lon),  # north
            (lat - delta, lon),  # south
        ]
    elev = _opentopo_elevation(pts, dataset=dataset)
    e = np.array([np.nan if v is None else v for v in elev], float).reshape(-1, 4)
    slope = np.empty(len(lonlat))
    for i, ((_lon, lat), (ee, ew, en, es)) in enumerate(zip(lonlat, e, strict=True)):
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(lat))
        dzdx = (ee - ew) / (2 * delta * m_per_deg_lon)
        dzdy = (en - es) / (2 * delta * m_per_deg_lat)
        grad = np.hypot(dzdx, dzdy)
        slope[i] = np.degrees(np.arctan(grad))
    return slope


def derive_station_features(stations: np.ndarray, dataset: str = _DEM) -> pd.DataFrame:
    """Derive ``FEATURE_COLS`` for ``stations`` (``(S, 2)`` lon/lat); hits the net."""
    lonlat = np.asarray(stations, float)
    lat_lon = [(float(lat), float(lon)) for lon, lat in lonlat]

    elev = _opentopo_elevation(lat_lon, dataset=dataset)
    elev = np.array([np.nan if v is None else float(v) for v in elev])
    # Coastal/offshore DEM gaps -> clip negatives to sea level, fill NaN with 0 m.
    elev = np.where(np.isnan(elev), 0.0, np.clip(elev, 0.0, None))

    coast = _coast_geometry(IBERIA_BBOX)
    dist = _dist_coast_km(lonlat, coast)
    slope = _slope_deg(lonlat, dataset=dataset)
    slope = np.where(np.isnan(slope), 0.0, slope)

    return pd.DataFrame(
        {
            "lon": lonlat[:, 0],
            "lat": lonlat[:, 1],
            "elevation": elev,
            "dist_coast_km": dist,
            "slope_deg": slope,
        }
    )


# --------------------------------------------------------------------------- #
# Cached loader (offline)                                                      #
# --------------------------------------------------------------------------- #
def _key(lon: float, lat: float) -> tuple[float, float]:
    return (round(float(lon), 4), round(float(lat), 4))


def synthetic_station_features(stations: np.ndarray) -> pd.DataFrame:
    """Deterministic stand-in features (**no network**) for the offline fallback.

    Mirrors the synthetic data generator's smooth, inland-peaking elevation
    field and derives a crude distance-to-coast and slope from station geometry,
    so the notebooks run fully offline when neither the real cache nor a network
    connection exists. The values are fake but finite and plausibly shaped.
    """
    lonlat = np.asarray(stations, float)
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    elev = 900.0 * np.exp(-(((lon + 4.0) / 4.0) ** 2 + ((lat - 41.0) / 3.0) ** 2))
    elev = np.clip(elev, 0.0, None)
    # distance to the nearest Iberia bbox edge, as a coast proxy (deg -> km)
    lon0, lon1, lat0, lat1 = IBERIA_BBOX
    edge_deg = np.minimum.reduce([lon - lon0, lon1 - lon, lat - lat0, lat1 - lat])
    dist = np.clip(edge_deg, 0.0, None) * 111.0
    # slope from the analytic gradient of the elevation field (degrees)
    dlon = elev * (-2.0 * (lon + 4.0) / 16.0)
    dlat = elev * (-2.0 * (lat - 41.0) / 9.0)
    slope = np.degrees(np.arctan(np.hypot(dlon, dlat) / 111_320.0))
    return pd.DataFrame(
        {
            "lon": lon,
            "lat": lat,
            "elevation": elev,
            "dist_coast_km": dist,
            "slope_deg": slope,
        }
    )


def load_station_features(
    stations: np.ndarray, *, root: Path | None = None
) -> pd.DataFrame:
    """Return spatial covariates for ``stations`` (``(S, 2)`` lon/lat), row-aligned.

    **Offline-safe**: returns the cached *real* features when the cache covers
    every requested station, otherwise a deterministic *synthetic* stand-in. It
    never touches the network, so the notebooks run with no cache and no
    connection (mirroring the data loader's synthetic fallback). Build the real
    cache with ``scripts/build_features.py`` / :func:`build_station_features`.
    """
    path = features_path(root)
    if path.exists():
        cache = pd.read_parquet(path)
        lookup = {_key(r.lon, r.lat): r for r in cache.itertuples()}
        if all(_key(lon, lat) in lookup for lon, lat in stations):
            rows = [lookup[_key(lon, lat)] for lon, lat in stations]
            return pd.DataFrame(
                {c: [getattr(r, c) for r in rows] for c in FEATURE_COLS}
            )
    return synthetic_station_features(stations)


def build_station_features(
    stations: np.ndarray,
    *,
    root: Path | None = None,
    refresh: bool = False,
    dataset: str = _DEM,
) -> pd.DataFrame:
    """Derive *real* features from public geodata and cache them. **Hits the net.**

    Used by ``scripts/build_features.py``; the notebooks call the offline-safe
    :func:`load_station_features` instead.
    """
    path = features_path(root)
    cache = None
    if path.exists() and not refresh:
        cache = pd.read_parquet(path)

    need = stations
    if cache is not None:
        have = {_key(r.lon, r.lat) for r in cache.itertuples()}
        missing = [s for s in stations if _key(s[0], s[1]) not in have]
        if not missing:
            lookup = {_key(r.lon, r.lat): r for r in cache.itertuples()}
            rows = [lookup[_key(lon, lat)] for lon, lat in stations]
            return pd.DataFrame(
                {c: [getattr(r, c) for r in rows] for c in FEATURE_COLS}
            )
        need = np.asarray(missing, float)

    fresh = derive_station_features(need, dataset=dataset)
    combined = fresh if cache is None else pd.concat([cache, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path)

    lookup = {_key(r.lon, r.lat): r for r in combined.itertuples()}
    rows = [lookup[_key(lon, lat)] for lon, lat in stations]
    return pd.DataFrame({c: [getattr(r, c) for r in rows] for c in FEATURE_COLS})
