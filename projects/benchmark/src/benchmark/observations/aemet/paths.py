"""Filesystem locations for the AEMET scrapes.

Deliberately stdlib-only: the offline tooling (``coverage.py``) needs the
archive path without importing the reader stack, so a resume point can be
computed on a checkout that has not installed the ``[aemet]`` extra.
"""

from __future__ import annotations

import os
from pathlib import Path


# Project root — ``src/benchmark/observations/aemet/paths.py`` is four
# levels below ``projects/benchmark/``.
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Where logs go — under the project, git-ignored.
LOG_ROOT = PROJECT_ROOT / "logs"


def scratch_root() -> Path:
    """Where the scrape archives live.

    Resolution order (first match wins):

    1. ``AEMET_SCRATCH_ROOT`` environment variable — set this to point at
       any directory, e.g. ``AEMET_SCRATCH_ROOT=~/aemet`` or
       ``/mnt/data/aemet``. This is the one the existing archive uses.
    2. ``BENCHMARK_AEMET_ROOT`` environment variable (alias).
    3. ``<project>/data/aemet`` — the portable default that works on any
       checkout, and is git-ignored.
    """
    for var in ("AEMET_SCRATCH_ROOT", "BENCHMARK_AEMET_ROOT"):
        override = os.environ.get(var)
        if override:
            return Path(override).expanduser()
    return PROJECT_ROOT / "data" / "aemet"
