"""Report what the AEMET archive already holds — and where to resume.

Reads the cached GeoParquet directly (no network, no credentials) and
prints per-year row counts plus non-null density for a reference
variable. The last line is the ``--start`` value to hand the scraper.

This exists because ``AemetArchive.sync`` has no per-station resume: it
refetches every station in a window. Knowing the exact first incomplete
year is the difference between redoing 50 minutes and redoing a day.

Run:
    uv run python scripts/observations/aemet/coverage.py
    uv run python scripts/observations/aemet/coverage.py --preset aemet_daily
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from benchmark.observations.aemet.paths import scratch_root
from benchmark.observations.aemet.periods import LAST_YEAR


# Archive subdirectory per preset — mirrors the build_archive() calls in
# the scrape scripts.
SUBDIR = {"aemet_monthly": "monthly", "aemet_daily": "daily"}

REFERENCE_VARIABLE = "air_temperature_daily_mean"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        default="aemet_monthly",
        choices=sorted(SUBDIR),
        help="which archive to report on",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = scratch_root() / SUBDIR[args.preset] / f"{args.preset}.parquet"

    if not path.exists():
        print(f"no archive at {path}")
        print("nothing scraped yet — start from the beginning")
        return 1

    df = pd.read_parquet(path, columns=["station_id", "time", REFERENCE_VARIABLE])
    df["year"] = pd.to_datetime(df["time"]).dt.year

    by_year = df.groupby("year").agg(
        rows=("station_id", "size"),
        stations=("station_id", "nunique"),
        non_null=(REFERENCE_VARIABLE, lambda s: s.notna().sum()),
    )
    by_year["density"] = (by_year["non_null"] / by_year["rows"] * 100).round(1)

    print(f"archive: {path}")
    print(f"rows: {len(df):,}   stations: {df.station_id.nunique()}")
    print(f"years: {by_year.index.min()}-{by_year.index.max()}\n")
    print(by_year.to_string())

    # A year is "held" if it has any rows at all. The first year with no
    # rows is where the scrape stopped; gaps in the middle mean an
    # interrupted window that needs re-running, so report those loudly.
    held = set(by_year.index)
    span = range(int(by_year.index.min()), LAST_YEAR + 1)
    missing = [y for y in span if y not in held]

    print()
    if not missing:
        print(f"complete through {LAST_YEAR} — nothing to resume")
        return 0

    gaps = [y for y in missing if y < max(held)]
    if gaps:
        print(f"WARNING: interior gaps at {gaps} — re-run those windows too")
    print(f"resume with: --start {missing[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
