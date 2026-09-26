"""Full-network AEMET daily scrape — multi-period, resumable.

Daily is the big one. AEMET's daily climatological endpoint caps each
request at 180 days, so a single station's decade takes ~20 chunks —
roughly 6x the monthly budget. Expect on the order of 60 hours of wall
time at 60 req/min pacing for ~947 stations across 1920-2025.

Most stations have no data before ~1950, so ``--start 1950`` is usually
the right call if quota or wall time is a concern.

Run:
    uv run python scripts/observations/aemet/daily.py                    # full
    uv run python scripts/observations/aemet/daily.py --start 1950
    uv run python scripts/observations/aemet/daily.py --start 2015 --end 2025
"""

from __future__ import annotations

import argparse
import time

from benchmark.observations.aemet import (
    build_archive,
    build_periods,
    select_periods,
    setup_logging,
)
from benchmark.observations.aemet.inventory import merged_inventory
from benchmark.observations.aemet.periods import FIRST_YEAR, LAST_YEAR
from loguru import logger


# Two-year windows — same granularity as monthly, but each one is far
# heavier here because of the 180-day request cap.
PERIODS = build_periods(step=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=FIRST_YEAR, help="first year to scrape")
    p.add_argument("--end", type=int, default=LAST_YEAR, help="last year to scrape")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("aemet_daily")
    archive = build_archive("daily")

    logger.info("refreshing station inventory")
    inventory = merged_inventory(archive, "aemet_daily")

    todo = select_periods(PERIODS, args.start, args.end)
    logger.info(f"{len(todo)} periods to scrape ({todo[0][1]}-{todo[-1][2]})")

    t0 = time.monotonic()
    for i, y1, y2 in todo:
        logger.info(f"period {i}/{len(PERIODS)}: {y1}-{y2}")
        period_t0 = time.monotonic()
        try:
            ds = archive.sync(
                "aemet_daily",
                stations=inventory,
                since=f"{y1}-01-01",
                until=f"{y2}-12-31",
            )
        except KeyboardInterrupt:
            logger.warning(f"interrupted in period {y1}-{y2}; archive is safe")
            raise
        logger.info(
            f"  period {y1}-{y2}: stations={ds.sizes['station']}, "
            f"days={ds.sizes['time']}, "
            f"elapsed={time.monotonic() - period_t0:.1f}s"
        )

    logger.info(f"all periods done in {time.monotonic() - t0:.1f}s")
    logger.info("archive at: {}", archive.root)


if __name__ == "__main__":
    main()
