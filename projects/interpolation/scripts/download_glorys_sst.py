"""Download a gap-free Mercator/GLORYS SST subset for the DINEOF example.

GLORYS12V1 (`cmems_mod_glo_phy_my_0.083deg_P1D-m`) is the Mercator Global Ocean
Physics Reanalysis: daily-mean, 1/12 deg, gap-free model output. We pull surface
potential temperature (`thetao`) over a Gulf Stream box for one year. Because the
field is gap-free, it serves as ground truth: the notebook punches synthetic
cloud-like holes and scores the reconstruction against the held-out pixels.

Requires Copernicus Marine credentials (already present at
~/.copernicusmarine/.copernicusmarine-credentials).

Run:  pixi run python projects/interpolation/scripts/download_glorys_sst.py
"""

from __future__ import annotations

from pathlib import Path

import copernicusmarine as cm


# Gulf Stream box — strong, low-rank SST variability, good EOF testbed.
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_FILE = "glorys_sst_gulfstream_2019.nc"

REGION = dict(
    minimum_longitude=-72.0,
    maximum_longitude=-60.0,
    minimum_latitude=34.0,
    maximum_latitude=42.0,
)
TIME = dict(start_datetime="2019-01-01T00:00:00", end_datetime="2019-12-31T00:00:00")
SURFACE = dict(minimum_depth=0.0, maximum_depth=1.0)  # ~0.49 m level -> SST proxy


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / OUT_FILE
    if out.exists():
        print(f"already present: {out}")
        return
    cm.subset(
        dataset_id=DATASET_ID,
        variables=["thetao"],
        output_directory=str(OUT_DIR),
        output_filename=OUT_FILE,
        **REGION,
        **TIME,
        **SURFACE,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
