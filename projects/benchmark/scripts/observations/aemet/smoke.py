"""AEMET smoke test — validates the full pipeline (~5-10 min).

What it does:

1. Sets up the paced archive (60 req/min, single worker).
2. Refreshes the station inventory.
3. Fetches twenty years of monthly data (2005-2024) for about two
   stations per autonomous community (~37-40 stations).
4. Reads the archive back as a GeoDataFrame and prints a brief summary.

The scope is deliberately wide enough to exercise the pacing gate,
chunk-stitching, archive append, and GeoParquet round-trip under
sustained load — not just a one-shot validation. Run this first after any
credential change; if it comes back clean you are safe to launch the long
monthly / daily scrapes.

Everything writes to ``<scratch>/smoke/`` and logs to ``logs/aemet_smoke.log``.
Safe to interrupt and re-run.

Run:
    uv run python scripts/observations/aemet/smoke.py
"""

from __future__ import annotations

import time

from benchmark.observations.aemet import build_archive, setup_logging
from loguru import logger
from xrreader import StationCollection


def main() -> None:
    setup_logging("aemet_smoke")
    archive = build_archive("smoke")

    t0 = time.monotonic()
    logger.info("refreshing station inventory")
    inventory = archive.sync_stations()
    logger.info(
        f"inventory: {len(inventory)} stations across "
        f"{len(inventory.communities())} communities"
    )

    # Pick ~2 well-instrumented reference stations per community so the
    # smoke covers every autonomous community and exercises the pacing /
    # retry / merge paths under sustained load.
    per_community = 2
    picks: list = []
    for community in inventory.communities():
        pool = inventory.filter(community=community, has_wmo=True)
        if len(pool) < per_community:
            pool = inventory.filter(community=community)
        picks.extend(list(pool)[:per_community])
    sample = StationCollection.from_iter(picks)
    logger.info(
        f"sampled {len(sample)} stations across {len(sample.communities())} communities"
    )

    logger.info("fetching monthly 2005-2024 for the sample (~5-10 min expected)")
    ds = archive.sync(
        "aemet_monthly",
        stations=sample,
        since="2005-01-01",
        until="2024-12-31",
    )
    logger.info(
        f"fetched slice: stations={ds.sizes['station']}, "
        f"months={ds.sizes['time']}, variables={len(ds.data_vars)}"
    )

    logger.info("reading archive back as GeoParquet")
    gdf = archive.load("aemet_monthly")
    logger.info(f"archive rows: {len(gdf):,}")
    logger.info(f"archive CRS:  EPSG:{gdf.crs.to_epsg()}")
    non_null = gdf["air_temperature_daily_mean"].notna().sum()
    logger.info(
        f"air_temperature_daily_mean non-null: {non_null}/{len(gdf)} "
        f"({non_null / len(gdf):.0%})"
    )

    logger.info(f"done in {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
