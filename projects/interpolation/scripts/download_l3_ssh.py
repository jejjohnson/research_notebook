"""Download raw L3 along-track SSH (sea level anomaly) for the standalone SSH map.

This is the *raw* altimetry product, not the gridded L4 DUACS analysis. Each
satellite measures sea level only along its 1 Hz nadir ground track (`PT1S`), so
a single day sees a handful of thin tracks crossing the box and ~99% of the grid
is unobserved — the canonical altimetry mapping problem the L4 product solves by
optimal interpolation. We grab the un-interpolated tracks and map them ourselves.

CMEMS publishes L3 *per mission* (there is no merged L3 — merging is what makes
L4), so we pull every altimeter flying over the Gulf Stream box in 2019 and merge
them in the loader. The mission codes below are the ones whose reprocessed (`my`)
record actually covers all of 2019 — note `alg`/`h2ag` (drifting-orbit phases),
not the early `al`/`h2a` segments that end in 2015-16:

  j3    Jason-3                10-day repeat reference track
  s3a   Sentinel-3A            27-day sun-synchronous
  s3b   Sentinel-3B            27-day sun-synchronous (offset from S3A)
  alg   SARAL/AltiKa drifting  Ka-band, drifting orbit (sharp mesoscale sampling)
  c2    Cryosat-2              369-day drifting track (dense long-term coverage)
  h2ag  HY-2A drifting         drifting-orbit phase

Variables: `sla_filtered` (the editing-and-filtering SLA we map), `sla_unfiltered`
(same SLA before the along-track low-pass, for noise studies), and `mdt` (mean
dynamic topography, so ADT = SLA + MDT is available). Same Gulf Stream box and
year as the SST / CHL / SSS downloads so every experiment lines up.

Run:  pixi run python projects/interpolation/scripts/download_l3_ssh.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import copernicusmarine as cm


OUT_DIR = Path(__file__).resolve().parents[1] / "data"
REGION = dict(
    minimum_longitude=-72.0,
    maximum_longitude=-60.0,
    minimum_latitude=34.0,
    maximum_latitude=42.0,
)
TIME = dict(start_datetime="2019-01-01", end_datetime="2019-12-31")

MISSIONS = ["j3", "s3a", "s3b", "alg", "c2", "h2ag"]
VARIABLES = ["sla_filtered", "sla_unfiltered", "mdt"]

JOBS = [
    dict(
        dataset_id=f"cmems_obs-sl_glo_phy-ssh_my_{m}-l3-duacs_PT1S",
        variables=VARIABLES,
        output_filename=f"l3_ssh_{m}_gulfstream_2019.nc",
    )
    for m in MISSIONS
]


def main() -> None:
    """Fetch each mission's along-track box subset and flatten to one file.

    copernicusmarine ignores `output_filename` for these along-track datasets
    and writes the mission's native name into a sub-directory, so we stage each
    download in its own folder and move the single produced .nc to the flat
    target name the loaders expect.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for job in JOBS:
        out = OUT_DIR / job["output_filename"]
        if out.exists():
            print(f"already present: {out}")
            continue
        stage = OUT_DIR / f"{out.stem}_staging"
        if stage.exists():
            shutil.rmtree(stage)
        cm.subset(
            dataset_id=job["dataset_id"],
            variables=job["variables"],
            output_directory=str(stage),
            output_filename=out.name,  # .nc extension forces NetCDF (else CSV)
            **REGION,
            **TIME,
        )
        produced = sorted(stage.rglob("*.nc"))
        if len(produced) != 1:
            raise RuntimeError(f"{job['dataset_id']}: expected 1 .nc, got {produced}")
        produced[0].replace(out)
        shutil.rmtree(stage)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
