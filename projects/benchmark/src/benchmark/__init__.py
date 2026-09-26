"""Reference observational and reanalysis archives for benchmarking.

Each source lives under a category subpackage — currently
:mod:`benchmark.observations` for station networks. A source module owns
its archive wiring and period schedule; the runnable entry points are thin
CLIs under ``scripts/``.
"""

from __future__ import annotations


__all__: list[str] = []
