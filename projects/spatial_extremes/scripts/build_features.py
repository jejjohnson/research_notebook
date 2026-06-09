"""Derive and cache the station spatial-feature table.

Covariates: elevation, distance-to-coast, terrain slope. One-time and
network-bound; the GP-primer notebook loads the cached table offline.

Usage (from the project root, in the project venv):
    .venv/bin/python scripts/build_features.py
    .venv/bin/python scripts/build_features.py --refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spatial_extremes import (
    data,
    features as feat,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-years", type=int, default=20, help="station-selection threshold"
    )
    ap.add_argument("--refresh", action="store_true", help="re-derive even if cached")
    args = ap.parse_args()

    logger.info("loading station network (min_years={})", args.min_years)
    _, stations, _, is_real = data.load_annual_maxima(min_years=args.min_years)
    src = "REAL" if is_real else "SYNTHETIC"
    logger.info("source={} stations={}", src, len(stations))

    path = feat.features_path()
    if path.exists() and not args.refresh:
        logger.info("cache at {} — aligning (use --refresh to rebuild)", path)
    else:
        logger.info("deriving features (OpenTopoData DEM + Natural Earth coast)")

    df = feat.build_station_features(stations, refresh=args.refresh)
    logger.info("features ready: {} rows -> {}", len(df), feat.features_path())
    desc = df[["elevation", "dist_coast_km", "slope_deg"]].describe().round(1)
    logger.info("summary:\n{}", desc.to_string())


if __name__ == "__main__":
    main()
