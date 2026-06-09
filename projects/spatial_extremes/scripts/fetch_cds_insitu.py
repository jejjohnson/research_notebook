#!/usr/bin/env python
"""Download + cache CDS in-situ land-station data over Iberia.

The one-shot real-data step for the spatial-extremes curriculum. It drives
``xrreader.CDSInsituArchive`` one year at a time and caches a GeoParquet
archive under ``projects/spatial_extremes/data/cds_insitu_land/``. Once the
cache exists, every notebook's ``load_station_daily(prefer_real=True)`` picks it
up automatically; without it the notebooks fall back to a synthetic series.

Progress is logged with **loguru** to both the console and a rotating file under
``projects/spatial_extremes/logs/`` so a long historical pull can be watched
(e.g. inside ``tmux``) and audited afterwards. The fetch is **resumable**: years
already in the archive manifest are skipped, so re-running only pulls what is
missing, and every year is committed to disk as it lands.

Credentials (Climate Data Store account + accepted dataset licence):
    CDSAPI_URL=https://cds.climate.copernicus.eu/api
    CDSAPI_KEY=<your-key>
exported in the environment, in a ``.env`` at the repo root, or in ``~/.cdsapirc``.

Usage
-----
    python scripts/fetch_cds_insitu.py                      # 1800-2026, Iberia
    python scripts/fetch_cds_insitu.py --start 1950-01-01   # narrower window
    python scripts/fetch_cds_insitu.py --overwrite          # re-pull every year
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


# Make ``spatial_extremes`` importable whether or not the package is installed.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from loguru import logger
from spatial_extremes import data


def _setup_logging(log_file: Path) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}"
        ),
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level="DEBUG",
        rotation="20 MB",
        retention=8,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
    )
    logger.info("logging to {}", log_file)


def _completed(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    try:
        return set(json.loads(manifest_path.read_text()).get("completed_chunks", []))
    except Exception:
        return set()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="1800-01-01", help="inclusive start YYYY-MM-DD")
    p.add_argument("--end", default="2026-12-31", help="inclusive end YYYY-MM-DD")
    p.add_argument(
        "--root", default=None, help="cache root (default: the project's data/ dir)"
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="re-download every year, ignoring the manifest",
    )
    p.add_argument("--log-file", default=None, help="override the log-file path")
    p.add_argument(
        "--ascending",
        action="store_true",
        help="fetch oldest year first (default: newest first)",
    )
    p.add_argument(
        "--stop-after-invalid",
        type=int,
        default=3,
        help="newest-first: stop after N consecutive out-of-range "
        "(HTTP 400) years, i.e. once past the dataset's start",
    )
    args = p.parse_args()

    # Default to the same cache root the loaders read from, so the fetch and the
    # notebooks always agree on where the cache lives (both honour
    # CDS_INSITU_SCRATCH_ROOT / XR_TOOLZ_CDS_ROOT); pass --root to override.
    root = Path(args.root).expanduser() if args.root else data.default_cache_root()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = (
        Path(args.log_file)
        if args.log_file
        else data.project_root() / "logs" / f"fetch_cds_{ts}.log"
    )
    _setup_logging(log_file)

    start_yr, end_yr = int(str(args.start)[:4]), int(str(args.end)[:4])
    years = list(range(start_yr, end_yr + 1))
    if not args.ascending:
        years = years[::-1]
    order = "oldest-first" if args.ascending else "newest-first"
    manifest_path = root / data.PRESET / "manifest.json"

    logger.info("CDS in-situ land fetch · Iberia bbox {}", data.IBERIA_BBOX)
    logger.info(
        "years {}-{} ({} total, {}) · root {}",
        start_yr,
        end_yr,
        len(years),
        order,
        root,
    )

    try:
        from xrreader import CDSInsituArchive, CDSSource
        from xrreader._src.cds.archive import _scope_fingerprint
        from xrreader.types import AIR_TEMPERATURE, BBox
    except Exception as exc:  # pragma: no cover - environment guard
        logger.exception("xrreader[cds-insitu] is not importable: {}", exc)
        raise SystemExit(1) from exc

    root.mkdir(parents=True, exist_ok=True)
    lon_min, lon_max, lat_min, lat_max = data.IBERIA_BBOX
    bbox = BBox(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)
    archive = CDSInsituArchive(
        root=root,
        preset=data.PRESET,
        source=CDSSource(),
        time_aggregation="daily",
        variables=(AIR_TEMPERATURE,),
    )

    # Refuse to append into a cache built with a different bbox/variables — that
    # would silently leave the archive partial (xrreader raises on this too; we
    # check first for a clean message).
    scope = _scope_fingerprint(bbox, (AIR_TEMPERATURE,))
    stored = (
        json.loads(manifest_path.read_text()).get("scope")
        if manifest_path.exists()
        else None
    )
    if stored and stored != scope:
        logger.error(
            "scope mismatch — existing cache is {!r} but this request is {!r}. "
            "Appending would corrupt it; point --root at a fresh directory.",
            stored,
            scope,
        )
        raise SystemExit(2)
    logger.info(
        "scope {} · {} year(s) already cached", scope, len(_completed(manifest_path))
    )

    t_all = time.time()
    n_fetch = n_skip = n_empty = n_err = n_real_err = 0
    total_rows = 0
    consec_invalid = 0
    seen_data = False  # gate the newest-first auto-stop to the *old* end only
    for yr in years:
        if not args.overwrite and str(yr) in _completed(manifest_path):
            logger.debug("{} · already cached — skip", yr)
            n_skip += 1
            seen_data = True
            consec_invalid = 0
            continue
        t0 = time.time()
        try:
            df = archive.sync(
                f"{yr}-01-01", f"{yr}-12-31", bbox=bbox, overwrite=args.overwrite
            )
        except KeyboardInterrupt:
            logger.warning("interrupted at {} — everything fetched so far is saved", yr)
            break
        except Exception as exc:
            n_err += 1
            msg = str(exc)
            out_of_range = ("InvalidRequest" in msg) or ("400" in msg)
            if not out_of_range:
                n_real_err += 1  # bad creds / network / API error — a real failure
            logger.error(
                "{} · {}: {}",
                yr,
                "out of range" if out_of_range else "FAILED",
                msg[:90],
            )
            # Newest-first: a run of out-of-range years means we have walked off
            # the bottom of the dataset — stop instead of erroring to 1800.
            if out_of_range and not args.ascending and seen_data:
                consec_invalid += 1
                if consec_invalid >= args.stop_after_invalid:
                    logger.info(
                        "{} consecutive out-of-range years — reached the dataset's "
                        "earliest year (~{}); stopping.",
                        consec_invalid,
                        yr,
                    )
                    break
            else:
                consec_invalid = 0
            continue
        consec_invalid = 0
        seen_data = True
        dt = time.time() - t0
        if df is None or df.empty:
            n_empty += 1
            logger.warning("{} · no data returned ({:.1f}s)", yr, dt)
        else:
            n_fetch += 1
            total_rows += len(df)
            n_st = df["station_id"].astype(str).nunique()
            logger.success(
                "{} · +{:,} rows, {} stations ({:.1f}s) · cumulative {:,}",
                yr,
                len(df),
                n_st,
                dt,
                total_rows,
            )

    logger.info(
        "FINISHED in {:.0f}s · fetched {} · skipped {} · empty {} · "
        "errors {} · +{:,} rows",
        time.time() - t_all,
        n_fetch,
        n_skip,
        n_empty,
        n_err,
        total_rows,
    )

    # Summarise what the notebooks will now load (Spanish, QC-passed temperature).
    try:
        df_all = data._load_real_daily(root)
        if df_all is not None and not df_all.empty:
            tt = data.pd.to_datetime(df_all["time"])
            logger.success(
                "cache now spans {}..{} · {:,} Spanish rows · {} stations",
                tt.min().date(),
                tt.max().date(),
                len(df_all),
                df_all["station_id"].nunique(),
            )
    except Exception as exc:
        logger.warning("could not summarise final cache: {}", exc)

    if n_real_err:
        logger.error(
            "{} year(s) failed with real errors (bad credentials / network / "
            "API), not just out-of-range — the cache may be incomplete; exiting "
            "non-zero so callers and CI notice instead of assuming success.",
            n_real_err,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
