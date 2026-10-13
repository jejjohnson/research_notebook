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


def _env_file_value(name: str) -> str | None:
    """Read ``name`` from the project ``.env``, if that file defines it.

    Deliberately a hand-rolled parser rather than a python-dotenv import:
    this module is the one piece of the package that stays stdlib-only so
    ``coverage.py`` can compute a resume point on a checkout without the
    reader stack installed. Handles ``KEY=value``, surrounding quotes, and
    ``#`` comments — which is the whole of what ``.env.example`` uses.
    """
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return None
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() != name:
            continue
        return value.split(" #")[0].strip().strip("\"'") or None
    return None


def scratch_root() -> Path:
    """Where the scrape archives live.

    Resolution order (first match wins):

    1. ``AEMET_SCRATCH_ROOT`` environment variable — set this to point at
       any directory, e.g. ``AEMET_SCRATCH_ROOT=~/aemet`` or
       ``/mnt/data/aemet``. This is the one the existing archive uses.
    2. ``BENCHMARK_AEMET_ROOT`` environment variable (alias).
    3. Either name as defined in the project ``.env``. The exported
       environment wins over ``.env`` so a one-off ``AEMET_SCRATCH_ROOT=...``
       in front of a command still overrides the committed default.
    4. ``<project>/data/aemet`` — the portable default that works on any
       checkout, and is git-ignored.
    """
    names = ("AEMET_SCRATCH_ROOT", "BENCHMARK_AEMET_ROOT")
    for var in names:
        override = os.environ.get(var)
        if override:
            return Path(override).expanduser()
    for var in names:
        override = _env_file_value(var)
        if override:
            return Path(override).expanduser()
    return PROJECT_ROOT / "data" / "aemet"
