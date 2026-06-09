"""Shared data layer for the spatial-extremes curriculum.

Two interchangeable sources, one schema:

* **Real** — daily near-surface ``air_temperature`` from CDS in-situ land
  stations over the Iberian peninsula, pulled and cached with
  :class:`xrtoolz.data.CDSInsituArchive`.
* **Synthetic** — a deterministic generator with the *same* tidy schema, so
  every notebook runs end-to-end with no network and no CDS credentials.

Swapping one for the other is a single flag (``prefer_real``). The real path
imports ``xrtoolz`` lazily, so this module imports cleanly in a minimal
environment (``numpy`` + ``pandas`` + ``xarray``) that has none of the
geospatial stack installed.

Schema returned by :func:`load_station_daily` (tidy, one row per obs):

==============  ==========================================
column          meaning
==============  ==========================================
``station_id``  station identifier (str)
``time``        observation day (``datetime64[ns]``)
``lon``         longitude (degrees east)
``lat``         latitude (degrees north)
``value``       daily ``air_temperature`` in ``degC``
==============  ==========================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# Iberia, matching the bbox used in xrtoolz's cds_insitu_land_demo.
# Order follows xrtoolz.types.BBox: (lon_min, lon_max, lat_min, lat_max).
# Spain incl. the Balearic Islands and Ceuta/Melilla (lat_min nudged to 35.0).
# The Canary Islands (~28°N) sit outside this box — widen to lat 27 / lon -18.5
# to include them, at the cost of a much larger, ocean-heavy download.
SPAIN_BBOX = (-10.0, 4.7, 35.0, 44.2)
IBERIA_BBOX = SPAIN_BBOX  # backward-compatible alias used by the viz helpers
PRESET = "cds_insitu_land"
TEMPERATURE_VARIABLE = "air_temperature"  # CDS alias in the land preset
# Keep only Spanish stations (GHCN/WMO country prefix "SP") — mainland, the
# Balearics and Ceuta/Melilla.
SPAIN_ID_PREFIX = "SP"

# Use the full record the dataset offers for these stations. Daily Spanish
# coverage is sparse before ~1950; 2024 is the last complete year.
DEFAULT_START = "1950-01-01"
DEFAULT_END = "2024-12-31"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def project_root() -> Path:
    """Return the ``projects/spatial_extremes`` directory."""
    # data.py -> spatial_extremes -> src -> spatial_extremes(project)
    return Path(__file__).resolve().parents[2]


def default_cache_root() -> Path:
    """Directory that holds the cached CDS archive (``<root>/<preset>/...``).

    Honours ``CDS_INSITU_SCRATCH_ROOT`` / ``XR_TOOLZ_CDS_ROOT`` if set, else
    falls back to ``projects/spatial_extremes/data``.
    """
    for var in ("CDS_INSITU_SCRATCH_ROOT", "XR_TOOLZ_CDS_ROOT"):
        override = os.environ.get(var)
        if override:
            return Path(override).expanduser()
    return project_root() / "data"


def _archive_data_path(root: Path) -> Path:
    return root / PRESET / "data.parquet"


# ---------------------------------------------------------------------------
# Real source — xrtoolz CDS in-situ land archive (lazy import)
# ---------------------------------------------------------------------------
def fetch_cds_insitu(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    *,
    bbox: tuple[float, float, float, float] = IBERIA_BBOX,
    root: Path | None = None,
    overwrite: bool = False,
    temperature_only: bool = True,
) -> Path:
    """Download + cache CDS in-situ land observations over ``bbox``.

    Requires ``xrtoolz[cds-insitu]`` and CDS credentials (``CDSAPI_URL`` /
    ``CDSAPI_KEY`` in the environment, a ``.env`` file, or ``~/.cdsapirc``).
    Returns the path to the cached GeoParquet archive. Re-running is cheap:
    the archive only fetches years not already present in its manifest.

    By default only ``air_temperature`` is requested (this curriculum models
    temperature extremes); set ``temperature_only=False`` to pull the whole
    land preset.
    """
    from xrtoolz.data import CDSInsituArchive, CDSSource
    from xrtoolz.types import AIR_TEMPERATURE, BBox

    variables = (AIR_TEMPERATURE,) if temperature_only else None
    root = Path(root) if root is not None else default_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    archive = CDSInsituArchive(
        root=root,
        preset=PRESET,
        source=CDSSource(),  # reads CDSAPI_URL / CDSAPI_KEY / ~/.cdsapirc
        time_aggregation="daily",
        variables=variables,
    )
    lon_min, lon_max, lat_min, lat_max = bbox
    archive.sync(
        start,
        end,
        bbox=BBox(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max),
        overwrite=overwrite,
    )
    return archive.data_path


def _load_real_daily(root: Path) -> pd.DataFrame | None:
    """Load the cached CDS archive into the tidy schema, or ``None``.

    Returns ``None`` (rather than raising) when the cache is absent or the
    parquet engine needed to read it is not installed, so callers can fall back
    to the synthetic generator.

    The cache is a GeoParquet, but we only need the scalar columns — so we read
    it with plain ``pandas`` + ``pyarrow`` (selecting non-geometry columns),
    which avoids a hard dependency on ``geopandas``/``xrtoolz`` just to *read*.
    """
    data_path = _archive_data_path(root)
    if not data_path.exists():
        return None
    cols = ["station_id", "time", "lon", "lat", "variable", "value", "quality_flag"]
    try:
        df = pd.read_parquet(data_path, columns=cols)  # pyarrow engine
    except Exception:
        # Fall back to the full geopandas/xrtoolz read if the column-subset
        # read is unavailable for some reason.
        try:
            from xrtoolz.data import CDSInsituArchive

            archive = CDSInsituArchive(
                root=root, preset=PRESET, time_aggregation="daily"
            )
            df = archive.load()
        except Exception:
            return None

    df = df[df["variable"] == TEMPERATURE_VARIABLE].copy()
    # Keep only QC-passed observations (flag 0). Non-zero flags carry corrupt
    # values — e.g. spurious ~455 K (182 degC) readings that would otherwise
    # dominate the daily/annual maxima.
    if "quality_flag" in df.columns:
        df = df[df["quality_flag"] == 0]
    # Spain + Spanish territories only (drop Algeria/Portugal/France/Morocco/...).
    df = df[df["station_id"].astype(str).str.startswith(SPAIN_ID_PREFIX)]
    if df.empty:
        return None
    out = pd.DataFrame(
        {
            "station_id": df["station_id"].astype(str).to_numpy(),
            "time": pd.to_datetime(df["time"], utc=True).dt.tz_localize(None),
            "lon": df["lon"].to_numpy(float),
            "lat": df["lat"].to_numpy(float),
            "value": df["value"].to_numpy(float),
        }
    )
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    # CDS serves air_temperature in Kelvin despite the 'degC' alias metadata;
    # convert to °C when the values are clearly absolute temperature.
    if out["value"].median() > 150.0:
        out["value"] = out["value"] - 273.15
    # Physical-range guard (AIR_TEMPERATURE valid_range is (-80, 60) degC);
    # drop anything outside a generous window as a final safety net.
    out = out[(out["value"] > -90.0) & (out["value"] < 60.0)]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic source — deterministic, schema-compatible, geo-stack-free
# ---------------------------------------------------------------------------
def synthetic_station_daily(
    n_stations: int = 40,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    *,
    bbox: tuple[float, float, float, float] = IBERIA_BBOX,
    seed: int = 2024,
) -> pd.DataFrame:
    """Deterministic stand-in for the real CDS daily station temperatures.

    Builds a physically plausible daily near-surface temperature series per
    station: a latitude/elevation-dependent climatological mean, an annual
    seasonal cycle, a slow warming trend, and correlated day-to-day noise.
    The spatial structure (warmer/peakier in the south-east, cooler at
    altitude) gives the downstream spatial GP genuine signal to recover.
    """
    rng = np.random.default_rng(seed)
    lon_min, lon_max, lat_min, lat_max = bbox

    # Station locations: jittered grid over land-ish Iberia + a synthetic
    # elevation field (higher inland / north).
    lon = rng.uniform(lon_min + 0.5, lon_max - 0.5, n_stations)
    lat = rng.uniform(lat_min + 0.5, lat_max - 0.5, n_stations)
    station_ids = [f"SYN{idx:03d}" for idx in range(n_stations)]
    # crude orography: peaks toward the centre-north of the peninsula
    elev = 900.0 * np.exp(
        -(((lon + 4.0) / 4.0) ** 2 + ((lat - 41.0) / 3.0) ** 2)
    ) + 150.0 * rng.standard_normal(n_stations)
    elev = np.clip(elev, 0.0, None)

    days = pd.date_range(start, end, freq="D")
    n_days = len(days)
    doy = days.dayofyear.to_numpy()
    year_frac = days.year.to_numpy() - days.year.min()

    # Climatological mean: warmer south + lower elevation.
    mean_temp = 16.5 - 0.40 * (lat - lat_min) - 0.0055 * elev  # degC
    # Seasonal amplitude: larger inland/continental (higher elevation, north).
    seas_amp = 8.5 + 0.012 * elev + 0.25 * (lat - lat_min)
    # Warming trend ~ 0.03 degC/yr, slightly stronger inland.
    trend_rate = 0.03 + 0.004 * (elev / 500.0)

    out_frames = []
    # peak of summer ~ day 205 (late July): cos peaks at +1 there
    seasonal_base = np.cos(2.0 * np.pi * (doy - 205) / 365.25)
    for sidx in range(n_stations):
        # AR(1) weather noise so block maxima are not trivially i.i.d.-Gaussian.
        eps = rng.standard_normal(n_days)
        noise = np.empty(n_days)
        noise[0] = eps[0]
        phi = 0.6
        for t in range(1, n_days):
            noise[t] = phi * noise[t - 1] + np.sqrt(1 - phi**2) * eps[t]
        noise *= 2.8  # degC

        temp = (
            mean_temp[sidx]
            + seas_amp[sidx] * seasonal_base
            + trend_rate[sidx] * year_frac
            + noise
            # mild heat-wave skew in summer -> heavier upper tail
            + 0.6
            * np.clip(seasonal_base, 0, None) ** 2
            * np.abs(rng.standard_normal(n_days))
        )
        out_frames.append(
            pd.DataFrame(
                {
                    "station_id": station_ids[sidx],
                    "time": days,
                    "lon": lon[sidx],
                    "lat": lat[sidx],
                    "value": temp,
                }
            )
        )
    return pd.concat(out_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def load_station_daily(
    *,
    prefer_real: bool = True,
    root: Path | None = None,
    n_stations: int = 40,
    start: str | None = None,
    end: str | None = None,
    seed: int = 2024,
) -> tuple[pd.DataFrame, bool]:
    """Load daily station temperatures, real if cached else synthetic.

    Returns ``(df, is_real)`` where ``df`` follows the tidy schema documented
    in the module docstring and ``is_real`` flags whether the data came from
    the CDS cache. Set ``prefer_real=False`` to force the synthetic generator
    (handy for reproducible teaching figures).

    ``start``/``end`` (``YYYY-MM-DD``) bound the returned record on **both**
    paths, so a given call is reproducible regardless of how many years the
    real cache happens to hold. The default ``None`` uses the *full* cached
    record for real data, and the ``DEFAULT_START``..``DEFAULT_END`` window for
    the synthetic fallback.
    """
    root = Path(root) if root is not None else default_cache_root()
    if prefer_real:
        real = _load_real_daily(root)
        if real is not None:
            if start is not None or end is not None:
                t = pd.to_datetime(real["time"])
                lo = pd.Timestamp(start) if start is not None else t.min()
                hi = pd.Timestamp(end) if end is not None else t.max()
                real = real[(t >= lo) & (t <= hi)].reset_index(drop=True)
            return real, True
    syn = synthetic_station_daily(
        n_stations=n_stations,
        start=start or DEFAULT_START,
        end=end or DEFAULT_END,
        seed=seed,
    )
    return syn, False


def to_station_time_dataarray(df_daily: pd.DataFrame, *, aggfunc: str = "max"):
    """Pivot the tidy daily frame to an xarray ``(station, time)`` DataArray.

    Carries ``lon``/``lat`` as non-dimension coordinates on ``station`` so
    downstream code can recover station geometry.

    The CDS daily land product reports several unlabeled statistics per
    station-day (daily min / mean / max). For *hot* extremes we collapse them
    with ``aggfunc="max"`` — the daily maximum temperature, the natural input to
    annual block maxima. (The synthetic source has one value per day, so the
    choice is a no-op there.)
    """
    import xarray as xr

    wide = df_daily.pivot_table(
        index="station_id", columns="time", values="value", aggfunc=aggfunc
    ).sort_index()
    coords = df_daily.groupby("station_id")[["lon", "lat"]].first().reindex(wide.index)
    da = xr.DataArray(
        wide.to_numpy(),
        dims=("station", "time"),
        coords={
            "station": wide.index.to_numpy(),
            "time": pd.to_datetime(wide.columns),
            "lon": ("station", coords["lon"].to_numpy()),
            "lat": ("station", coords["lat"].to_numpy()),
        },
        name=TEMPERATURE_VARIABLE,
    )
    return da


def load_annual_maxima(
    *,
    prefer_real: bool = True,
    root: Path | None = None,
    n_stations: int = 40,
    start: str | None = None,
    end: str | None = None,
    seed: int = 2024,
    min_periods: int = 300,
    min_years: int | None = None,
):
    """Convenience used by the capstone notebooks.

    Returns ``(maxima, stations, years, is_real)``:

    * ``maxima`` — ``float`` array ``(S, T)`` of annual maximum temperature
    * ``stations`` — ``float`` array ``(S, 2)`` of ``[lon, lat]``
    * ``years`` — ``int`` array ``(T,)``
    * ``is_real`` — whether the source was the real CDS cache

    Annual maxima are extracted with
    :func:`xtremax.extraction.temporal_block_maxima`, the same routine the
    foundations notebooks introduce step by step.

    Station selection across the ``(station, year)`` grid:

    * ``min_years=None`` (default) — keep only **fully-covered** stations (a
      year for every column), so ``maxima`` has no gaps. Back-compatible.
    * ``min_years=k`` — keep every station observed in **at least ``k``** years.
      Missing station-years are returned as ``NaN``; downstream models must mask
      them (``mask = ~np.isnan(maxima)``). This admits far more stations at the
      cost of ragged records — the regime where pooling earns its keep.
    """
    from xtremax.extraction import temporal_block_maxima

    df, is_real = load_station_daily(
        prefer_real=prefer_real,
        root=root,
        n_stations=n_stations,
        start=start,
        end=end,
        seed=seed,
    )
    da = to_station_time_dataarray(df)
    bm = temporal_block_maxima(da, freq="YE", time_dim="time", min_periods=min_periods)
    bm = bm.dropna("time", how="all")
    if min_years is None:
        # stations with full coverage only -> a clean (S, T) matrix
        bm = bm.dropna("station", how="any")
    else:
        # stations seen in >= min_years -> a ragged (S, T) matrix with NaN gaps
        keep = bm.notnull().sum("time") >= int(min_years)
        bm = bm.isel(station=keep.to_numpy())

    maxima = bm.transpose("station", "time").to_numpy().astype(float)
    stations = np.stack(
        [bm["lon"].to_numpy().astype(float), bm["lat"].to_numpy().astype(float)],
        axis=1,
    )
    years = pd.to_datetime(bm["time"].to_numpy()).year.to_numpy()
    return maxima, stations, years, is_real
