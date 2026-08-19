"""Full-network AEMET monthly scrape — multi-period, resumable.

Walks AEMET's monthly climatological endpoint from 1920 to the present
for every station in the network, in short period windows so progress is
checkpointed to GeoParquet at regular intervals. The archive is
idempotent — interrupt with Ctrl-C or kill the process and re-run.

Pacing (see ``build_archive``) targets 60 req/min, half of AEMET's
~150 req/min rolling cap, single-worker. With ~947 stations that is
roughly 50 minutes per two-year window.

**Resuming.** ``AemetArchive.sync`` has no per-station skip — re-running
a window refetches every station in it. So resume at the first window
that did not complete, not the one that did. Check current coverage with
``python scripts/observations/aemet/coverage.py`` first.

Run:
    uv run python scripts/observations/aemet/monthly.py                # all
    uv run python scripts/observations/aemet/monthly.py --start 1955   # resume
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
from benchmark.observations.aemet.periods import FIRST_YEAR, LAST_YEAR
from loguru import logger


# Two-year windows: ~50 min each, so an interrupted run loses under an
# hour rather than the ~2 h the original five-year windows cost.
PERIODS = build_periods(step=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=FIRST_YEAR, help="first year to scrape")
    p.add_argument("--end", type=int, default=LAST_YEAR, help="last year to scrape")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("aemet_monthly")
    archive = build_archive("monthly")

    logger.info("refreshing station inventory")
    inventory = archive.sync_stations()
    logger.info(f"inventory: {len(inventory)} stations")

    todo = select_periods(PERIODS, args.start, args.end)
    logger.info(f"{len(todo)} periods to scrape ({args.start}-{args.end})")

    t0 = time.monotonic()
    for i, y1, y2 in todo:
        logger.info(f"period {i}/{len(PERIODS)}: {y1}-{y2}")
        period_t0 = time.monotonic()
        try:
            ds = archive.sync(
                "aemet_monthly",
                stations=inventory,
                since=f"{y1}-01-01",
                until=f"{y2}-12-31",
            )
        except KeyboardInterrupt:
            logger.warning(f"interrupted in period {y1}-{y2}; archive is safe")
            raise
        logger.info(
            f"  period {y1}-{y2}: stations={ds.sizes['station']}, "
            f"months={ds.sizes['time']}, "
            f"elapsed={time.monotonic() - period_t0:.1f}s"
        )

    logger.info(f"all periods done in {time.monotonic() - t0:.1f}s")
    logger.info("archive at: {}", archive.root)


if __name__ == "__main__":
    main()
