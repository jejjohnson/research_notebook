"""AEMET OpenData scrape wiring — paced archive, logging, period schedules.

``build_archive`` and ``setup_logging`` are resolved lazily (PEP 562) so
that importing this package for its paths or period schedules does not
pull in the reader stack. That keeps the offline tooling — ``coverage.py``
computing a resume point from the cached GeoParquet — usable on a checkout
without the ``xrtoolz-reader[aemet]`` extra installed.
"""

from __future__ import annotations

from typing import Any

from benchmark.observations.aemet.paths import LOG_ROOT, PROJECT_ROOT, scratch_root
from benchmark.observations.aemet.periods import build_periods, select_periods


__all__ = [
    "LOG_ROOT",
    "PROJECT_ROOT",
    "build_archive",
    "build_periods",
    "scratch_root",
    "select_periods",
    "setup_logging",
]

_LAZY = {"build_archive", "setup_logging"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from benchmark.observations.aemet import common

        return getattr(common, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
