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

### Setup

```bash
cd projects/benchmark
uv sync                                    # installs xrtoolz-reader[aemet]
cp .env.example .env                       # add your AEMET_API_KEY
export AEMET_SCRATCH_ROOT=$HOME/scratch/aemet   # keep data off the repo disk
uv run python scripts/observations/aemet/smoke.py
```

### Current state of the monthly scrape

The archive holds **1920–1954 complete** for all 947 stations (397,740
rows). The run stopped on 2026-05-05 partway through 1955–1959 and was
never restarted.

Note the inventory has since shrunk to **921 stations** (2026-08-19) — AEMET
retires and renumbers stations, so the live network no longer matches the 947
frozen into the archive. Rows for retired stations stay in the archive; new
windows simply will not extend them.

Worth knowing before you judge that progress: the years already held are
the sparse ones. Non-null temperature runs ~2% in 1920 rising to ~7% by
1954 — the network only densifies after 1960, so essentially all the
usable data is still ahead.

To resume, confirm the restart year and go:

```bash
uv run python scripts/observations/aemet/coverage.py     # prints: resume with --start 1955
scripts/observations/aemet/resume.sh monthly --start 1955
```

### Pacing — why it is slow

`build_archive` defaults to 60 req/min, single worker. That is half of
AEMET's ~150 req/min rolling cap, and it is deliberate: the original
two-worker / 120 req/min config tripped 429s, because while one worker
backed off the other kept the minute bucket hot. `AemetSource` now also
pauses *all* workers globally on any 429, but single-worker at 1 req/s
remains the setting that finishes.

Budget roughly **50 minutes per two-year monthly window** (~36 windows
left, so ~30 hours), and considerably more for daily — its endpoint caps
each request at 180 days, so a station-decade costs ~20 chunks.

### Two failure modes to recognise

**Process vanishes, no traceback.** Azure ML's idle-stop agent kills
compute it judges idle, and a rate-limited scrape sits at near-zero CPU.
This is what ended the 2026-05-05 run. `resume.sh` exists to prevent it —
it runs a ~0.5% duty-cycle CPU sidecar alongside the scrape. Launch long
runs through it, not directly.

**`AemetAuthError` after weeks of working.** AEMET API keys are JWTs and
they expire. Re-request one before debugging anything else. (Not the cause
of the 2026-05-05 stop — the key still authenticated on 2026-08-19.)

### Resume semantics

`AemetArchive.sync` is idempotent, not additive, and has **no per-station
resume** — re-running a window refetches every station in it. Interrupting
is always safe for the data, but partial windows are not credited, so
restart at the first *incomplete* year. `coverage.py` computes it, and
flags interior gaps left by an interrupted window.
