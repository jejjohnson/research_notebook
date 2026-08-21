"""Period schedules for the multi-period AEMET scrapes.

The scrape walks history in fixed-width year windows so progress is
checkpointed to GeoParquet at regular intervals. Window width is a pure
checkpoint-granularity knob — it has no effect on what gets fetched, only
on how much work an interrupted run has to redo.

Why the windows are short: the original monthly scrape used five-year
windows (~2 h each) and was repeatedly killed mid-window by Azure ML's
idle-stop agent, losing the whole window every time. Two-year windows cut
that blast radius to ~50 min. See ``scripts/observations/aemet/resume.sh``
for the CPU-keepalive sidecar that addresses the kill itself.
"""

from __future__ import annotations


# AEMET's climatological baseline — no station records precede this.
FIRST_YEAR = 1920

# Last complete calendar year to scrape by default. Bump as years close.
LAST_YEAR = 2025


def build_periods(
    step: int = 2,
    first: int = FIRST_YEAR,
    last: int = LAST_YEAR,
) -> list[tuple[int, int]]:
    """Inclusive ``(start_year, end_year)`` windows of ``step`` years.

    The final window is clipped to ``last``, so it may be shorter.

    >>> build_periods(step=2, first=1920, last=1924)
    [(1920, 1921), (1922, 1923), (1924, 1924)]
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if last < first:
        raise ValueError(f"last ({last}) precedes first ({first})")
    return [(y, min(y + step - 1, last)) for y in range(first, last + 1, step)]


def select_periods(
    periods: list[tuple[int, int]],
    start: int,
    end: int,
) -> list[tuple[int, int, int]]:
    """Clip ``periods`` to the ``[start, end]`` year range.

    Windows fully outside the range are dropped; a window straddling a
    boundary is trimmed to it. Positions are preserved as 1-based indices
    into the *original* schedule so log lines like ``period 18/53`` stay
    meaningful when resuming partway through.

    Returns ``(index, start_year, end_year)`` triples.

    Raises ``ValueError`` rather than returning an empty list when the
    request cannot select anything. A silent empty selection is the worst
    outcome here: the scrape logs the range it was asked for, fetches
    nothing, and then reports that every period completed — so a typo like
    ``--start 2026`` against a schedule ending in 2025 looks like success.

    >>> select_periods([(1920, 1921), (1922, 1923)], 1921, 1922)
    [(1, 1921, 1921), (2, 1922, 1922)]
    >>> select_periods([(1920, 1921)], 2026, 2026)
    Traceback (most recent call last):
        ...
    ValueError: requested 2026-2026 but the schedule only covers 1920-1921
    """
    if not periods:
        raise ValueError("period schedule is empty")
    if start > end:
        raise ValueError(f"start ({start}) is after end ({end})")

    first_year, last_year = periods[0][0], periods[-1][1]
    if end < first_year or start > last_year:
        raise ValueError(
            f"requested {start}-{end} but the schedule only covers "
            f"{first_year}-{last_year}"
        )

    selected: list[tuple[int, int, int]] = []
    for i, (y1, y2) in enumerate(periods, 1):
        if y2 < start or y1 > end:
            continue
        selected.append((i, max(y1, start), min(y2, end)))
    return selected
