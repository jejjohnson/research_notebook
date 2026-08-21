---
title: Benchmark Archives
short_title: Benchmark
authors:
  - name: Juan Emmanuel Johnson
date: 2026-08-19
---

# Benchmark Archives

Reference observational archives for benchmarking, pulled from their
source APIs and cached as GeoParquet. Each source is a paced, resumable
scrape: slow on purpose, safe to interrupt, and idempotent on re-run.

The data layer is [`xrtoolz-reader`](https://github.com/jejjohnson/xr_toolz)
(the reader package of the xrtoolz monorepo). Note the distribution is named
`xrtoolz-reader` but the **import namespace is still `xrreader`** —
`from xrreader import AemetArchive` is unchanged from the standalone package.

## Layout

```
projects/benchmark/
├── src/benchmark/observations/aemet/   # archive wiring, period schedules
└── scripts/observations/aemet/         # thin CLI entry points
```

New sources slot in as siblings — `observations/ghcn`, `reanalysis/era5` —
reusing the same pacing and logging conventions.

## observations/aemet — Spanish national network

AEMET OpenData's climatological endpoints for all ~947 stations, from
1920 to the present.

| Script | What it does |
|---|---|
| `smoke.py` | ~10 min end-to-end validation over ~40 stations. Run first. |
| `coverage.py` | Offline report of what the archive holds + the resume year. |
| `monthly.py` | Full-network monthly scrape, two-year windows. |
| `daily.py` | Full-network daily scrape. Much heavier — see below. |
| `resume.sh` | tmux wrapper with a CPU keepalive sidecar. **Use this** for long runs. |

And one notebook:

| Notebook | What it covers |
|---|---|
| [`notebooks/01_aemet_archive_overview`](notebooks/01_aemet_archive_overview.ipynb) | Tour of the archive — network, coverage, trends, spatial and spatio-temporal structure, and the pitfalls. |

### Setup

The repo standard is Pixi, and the benchmark is registered as its own
environment + task set, so a clean checkout needs no second toolchain:

```bash
cp projects/benchmark/.env.example projects/benchmark/.env   # add AEMET_API_KEY
pixi run -e benchmark aemet-smoke                            # ~15 min live check
```

| Task | What it runs |
|---|---|
| `pixi run -e benchmark aemet-coverage` | Offline coverage + resume year |
| `pixi run -e benchmark aemet-smoke` | ~15 min live validation |
| `pixi run -e benchmark aemet-monthly` | Full monthly scrape |
| `pixi run -e benchmark aemet-daily` | Full daily scrape |
| `pixi run -e benchmark aemet-resume monthly --start 2020` | tmux + keepalive wrapper |
| `pixi run -e benchmark execute-benchmark` | Re-run the overview notebook against the cache |

Where the archive lands is resolved by `scratch_root()`: exported
`AEMET_SCRATCH_ROOT`, then `BENCHMARK_AEMET_ROOT`, then either name in
`projects/benchmark/.env`, then `<project>/data/aemet`. Point it off the
repo disk so the data survives VM rebuilds:

```bash
export AEMET_SCRATCH_ROOT=$HOME/scratch/aemet   # or set it in .env
```

Working inside `projects/benchmark/` directly, `uv` also resolves the
same project — `uv sync && uv run python scripts/observations/aemet/smoke.py`.
Both paths install `xrtoolz-reader[aemet]` from the same pinned commit.

### Current state of the monthly scrape

The archive holds **1920–2019 complete** — 1,116,120 rows. The 2026-08-20
run carried it from 1954 to 2019 in 33 two-year windows (~23 h) and was
interrupted partway through 2020–2021, so **three windows remain**
(2020–2021, 2022–2023, 2024–2025).

Coverage is thin early and dense late, which is the network, not the
scrape: non-null temperature runs ~2% in 1920, ~7% by 1954, ~12% by 1980,
and 86% by 2019.

Station counts differ by era for a real reason. Rows before 1955 carry 947
stations; rows from 1955 on carry the 921 the API advertised when that
window ran. AEMET retires and renumbers stations, so the live inventory is
a moving target — `merged_inventory()` now unions it with the stations
already in the archive (currently 921 live + 28 retired = 949) so later
windows extend the retired stations instead of dropping them.

To resume, confirm the restart year and go:

```bash
pixi run -e benchmark aemet-coverage        # prints: resume with --start 2020
pixi run -e benchmark aemet-resume monthly --start 2020
```

### Pacing — why it is slow

`build_archive` defaults to 60 req/min, single worker. That is half of
AEMET's ~150 req/min rolling cap, and it is deliberate: the original
two-worker / 120 req/min config tripped 429s, because while one worker
backed off the other kept the minute bucket hot. `AemetSource` now also
pauses *all* workers globally on any 429, but single-worker at 1 req/s
remains the setting that finishes.

Measured over the 33 windows of the 2026-08-20 run: **34 min** per
two-year window early on, rising to **55 min** for the recent, denser
years — the request count per window grows with the number of reporting
stations. Budget ~1 h per remaining window, and considerably more for
daily, whose endpoint caps each request at 180 days so a station-decade
costs ~20 chunks.

### Two failure modes to recognise

**Process vanishes, no traceback.** Two different causes, worth telling
apart before you reach for a fix.

*Idle-stop.* Azure ML kills compute it judges idle, and a rate-limited
scrape sits at near-zero CPU. `resume.sh` runs a ~0.5% duty-cycle CPU
sidecar to keep the sampler seeing activity. Launch long runs through it,
not directly.

*Control-plane stop.* The 2026-08-19 run died six minutes in, and it was
not idle-stop: `idleMinutesBeforeStop` was **120** and the CPU had been
busy throughout, while the kernel logged `hv_utils: Shutdown request
received — graceful shutdown initiated`. That is an external stop —
scheduled shutdown policy, a portal stop, or platform maintenance — and no
userspace keepalive prevents it. Check the instance's schedule before
committing to a multi-hour run. (Which of the two ended the 2026-05-05 run
was never established; the logs for it are gone.)

**`AemetAuthError` after weeks of working.** AEMET API keys are JWTs and
they expire. Re-request one before debugging anything else. (Not the cause
of the 2026-05-05 stop — the key still authenticated on 2026-08-19.)

### Resume semantics

`AemetArchive.sync` is idempotent, not additive, and has **no per-station
resume** — re-running a window refetches every station in it. Interrupting
is always safe for the data, but partial windows are not credited, so
restart at the first *incomplete* year. `coverage.py` computes it, and
flags interior gaps left by an interrupted window.

The overview notebook works through what this coverage pattern means in
practice — in particular why averaging every station reporting in a given
year understates the warming trend by about a third.

Year bounds are validated rather than silently clipped: a request that
selects no window at all — `--start 2026` against a schedule ending in
2025 — raises instead of logging the requested range, fetching nothing,
and reporting that every period completed.
