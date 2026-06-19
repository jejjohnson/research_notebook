"""Download L3 ocean-colour chlorophyll + L3 SSS for the multivariate DINEOF.

CHL (`cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D`) is multi-sensor ocean
colour: cloud-gapped like SST, log-normal, strongly correlated with SST fronts.
SSS (`cmems_obs-mob_glo_phy-sss_mynrt_smos-asc_P1D`) is SMOS microwave salinity:
swath gaps but *cloud-penetrating*, so it carries information into the cloud
voids where SST and CHL are both blocked. Same Gulf Stream box and year.

Run:  pixi run python projects/interpolation/scripts/download_chl_sss.py
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
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D",
        variables=["CHL"],
        output_filename="l3_chl_gulfstream_2019.nc",
    ),
    dict(
        dataset_id="cmems_obs-mob_glo_phy-sss_mynrt_smos-asc_P1D",
        variables=["Sea_Surface_Salinity_Rain_Corrected"],
        output_filename="l3_sss_gulfstream_2019.nc",
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
