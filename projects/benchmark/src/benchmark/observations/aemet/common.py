"""Shared setup for the AEMET scrape entry points.

Keeps loguru + archive wiring out of the individual scripts so each one
stays a short, readable list of period windows.

Ported from the original ``xr_toolz/scripts/_aemet_common.py``. The only
API change is the import: the reader moved into the xrtoolz monorepo as
the ``xrtoolz-reader`` distribution, but the import namespace is still
``xrreader``, so ``AemetArchive`` / ``AemetSource`` come straight off the
top-level package instead of ``xrtoolz.data``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from xrreader import AemetArchive, AemetSource

from benchmark.observations.aemet.paths import LOG_ROOT, scratch_root


def setup_logging(name: str) -> Path:
    """Configure loguru: stderr + per-script file with default formatting.

    Returns the log-file path for reference (e.g. so tmux users can
    ``tail -f`` it independently of the running process).
    """
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, level="INFO", rotation="20 MB", retention=5)
    return log_path


def build_archive(
    subdir: str,
    *,
    min_interval_s: float = 1.0,
    max_workers: int = 1,
    max_retries: int = 6,
    timeout_s: float = 30.0,
) -> AemetArchive:
    """Build a paced :class:`~xrreader.AemetArchive` under ``<root>/<subdir>``.

    Defaults tuned for long-running scrapes that survive AEMET's
    rate-limit window: **60 req/min** (``min_interval_s=1.0``) with a
    **single worker**. The two-worker / 120 req/min setting we originally
    shipped tripped 429s because the minute bucket never actually
    drained — while one worker was backing off, the other kept the bucket
    hot. ``AemetSource`` now also globally pauses all workers on any 429
    (see its ``_trip_rate_limit``) but the safer default is still
    single-worker at 1 req/s.
    """
    root = scratch_root() / subdir
    source = AemetSource(
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_workers=max_workers,
        min_interval_s=min_interval_s,
    )
    archive = AemetArchive(root=root, source=source)
    logger.info(f"archive root: {root}")
    logger.info(
        f"source: max_workers={max_workers}, "
        f"min_interval_s={min_interval_s}, timeout_s={timeout_s}"
    )
    return archive
