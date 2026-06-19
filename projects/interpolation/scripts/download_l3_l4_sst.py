"""Download real gappy L3S SST + gap-free L4 OSTIA SST for the DINEOF test.

L3S (`cmems_obs-sst_glo_phy_my_l3s_P1D-m`) is multi-sensor satellite SST with
*real* cloud gaps (~80% missing per day in winter). L4
(`METOFFICE-GLO-SST-L4-REP-OBS-SST`, OSTIA reprocessed) is the gap-free optimal-
interpolation analysis we verify the DINEOF fill against. Same Gulf Stream box
and year as the GLORYS download so the three experiments line up.

Run:  pixi run python projects/interpolation/scripts/download_l3_l4_sst.py
"""

from __future__ import annotations

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

JOBS = [
    dict(
        dataset_id="cmems_obs-sst_glo_phy_my_l3s_P1D-m",
        # raw + bias-adjusted SST, the QC flag, and the per-pixel obs-error std
        # (sses_standard_deviation) used as R in the noise-aware 3D-Var.
        variables=[
            "sea_surface_temperature",
            "adjusted_sea_surface_temperature",
            "quality_level",
            "sses_standard_deviation",
        ],
        output_filename="l3s_sst_gulfstream_2019.nc",
    ),
    dict(
        dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        variables=["analysed_sst"],
        output_filename="l4_ostia_gulfstream_2019.nc",
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for job in JOBS:
        out = OUT_DIR / job["output_filename"]
        if out.exists():
            print(f"already present: {out}")
            continue
        cm.subset(
            dataset_id=job["dataset_id"],
            variables=job["variables"],
            output_directory=str(OUT_DIR),
            output_filename=job["output_filename"],
            **REGION,
            **TIME,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
