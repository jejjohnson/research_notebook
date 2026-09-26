"""Station inventory that spans the live network *and* the archive.

AEMET retires and renumbers stations, so ``sync_stations()`` — which
returns whatever the API advertises *today* — is a moving target. On
2026-08-19 it returned 921 stations while the monthly archive already
held 949, a 28-station shortfall.

That matters because :meth:`AemetArchive.sync` fetches exactly the
stations it is handed. Resuming a historical scrape against the live
inventory alone silently drops every retired station from every window
still to come: the pre-1955 rows keep their 949 stations, the post-1955
rows only ever get 921, and the archive ends up with a discontinuity at
the resume point rather than a full-network record.

:func:`merged_inventory` closes that gap by unioning the live inventory
with the stations already present in the cached GeoParquet. Retired
stations are reconstructed from their archived coordinates and flagged
``active=False`` so downstream consumers can tell them apart.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from xrreader import AemetArchive, Station, StationCollection


def archived_station_coords(path: Path) -> dict[str, tuple[float, float]]:
    """Map ``station_id -> (lon, lat)`` for every station in the archive.

    Returns an empty mapping when the archive does not exist yet, so a
    first run degrades to "live inventory only" without special-casing.
    """
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["station_id", "lon", "lat"])
    if df.empty:
        return {}
    first = df.groupby("station_id")[["lon", "lat"]].first()
    return {str(sid): (float(r.lon), float(r.lat)) for sid, r in first.iterrows()}


def merged_inventory(archive: AemetArchive, preset: str) -> StationCollection:
    """Refresh the live inventory and re-add stations only the archive knows.

    The live inventory wins on metadata for any station in both — it is
    the fresher record. Stations that have dropped out of the API are
    appended with their archived coordinates, a placeholder name, and
    ``active=False``.
    """
    live = archive.sync_stations()
    live_ids = {str(i) for i in live.ids()}

    archived = archived_station_coords(archive.root / f"{preset}.parquet")
    retired_ids = sorted(set(archived) - live_ids)

    if not retired_ids:
        logger.info(
            f"inventory: {len(live_ids)} stations (no retired stations to re-add)"
        )
        return live

    retired = [
        Station(
            id=sid,
            name=f"retired:{sid}",
            lon=archived[sid][0],
            lat=archived[sid][1],
            active=False,
            attrs={},
        )
        for sid in retired_ids
    ]
    merged = StationCollection.from_iter([*live.stations, *retired])
    logger.info(
        f"inventory: {len(merged.ids())} stations "
        f"({len(live_ids)} live + {len(retired)} retired re-added from the archive)"
    )
    return merged
