"""Does multivariate DINEOF (SST + CHL + SSS) improve the SST gap-fill?

Joint DINEOF over three co-located variables, all deseasonalised and
standardised, so shared temporal modes couple them. The interesting physics:
CHL shares SST's cloud gaps (correlated structure, little new coverage), while
SMOS SSS is microwave and *cloud-penetrating*, so it can carry information into
the very voids where SST is blocked.

Scored on the SST block only, with the same realistic cloud-shaped hold-out as
notebook 07: Check A vs held-out real L3 SST, Check B vs gap-free L4.
"""

from __future__ import annotations

from pathlib import Path

import dineof_core as dc
import numpy as np
import pandas as pd
import xarray as xr


DATA = Path(__file__).resolve().parents[1] / "data"
KELVIN = 273.15


def load_aligned():
    """All variables regridded onto the SST L3 grid; returns flat (T, HW) arrays."""
    l3 = xr.open_dataset(DATA / "l3s_sst_gulfstream_2019.nc")
    lat, lon = l3.latitude, l3.longitude
    sst = l3["sea_surface_temperature"] - KELVIN
    doy = l3["time"].dt.dayofyear.values.astype(float)

    l4 = xr.open_dataset(DATA / "l4_ostia_gulfstream_2019.nc")["analysed_sst"]
    l4 = l4.interp(latitude=lat, longitude=lon) - KELVIN

    chl = xr.open_dataset(DATA / "l3_chl_gulfstream_2019.nc")["CHL"]
    logchl = np.log10(chl.where(chl > 0)).interp(latitude=lat, longitude=lon)

    sss = xr.open_dataset(DATA / "l3_sss_gulfstream_2019.nc")[
        "Sea_Surface_Salinity_Rain_Corrected"
    ]
    # align SSS time axis to SST days, then regrid in space.
    sss = sss.reindex(time=l3.time, method="nearest", tolerance=np.timedelta64(1, "D"))
    sss = sss.interp(latitude=lat, longitude=lon)

    T = l3.sizes["time"]

    def flat(da):
        return da.values.reshape(T, -1)

    return dict(
        T=T,
        hw=(l3.sizes["latitude"], l3.sizes["longitude"]),
        doy=doy,
        sst=flat(sst),
        l4=flat(l4),
        chl=flat(logchl),
        sss=flat(sss),
    )


def deseason_std(Y, M, doy):
    Ya, clim = dc.deseasonalize(np.where(M, Y, 0.0), M, doy)
    std = Ya[M].std() if M.any() else 1.0
    return Ya / std, clim, std


def main():
    d = load_aligned()
    doy = d["doy"]
    sst, l4, chl, sss = d["sst"], d["l4"], d["chl"], d["sss"]
    domain = np.isfinite(sst).any(0) & np.isfinite(l4).any(0)
    sst, l4 = sst[:, domain], l4[:, domain]
    chl, sss = chl[:, domain], sss[:, domain]

    M_sst = np.isfinite(sst)
    M_chl = np.isfinite(chl)
    M_sss = np.isfinite(sss)
    print(
        f"domain N={domain.sum()}  coverage: SST={M_sst.mean():.3f} "
        f"CHL={M_chl.mean():.3f} SSS={M_sss.mean():.3f}"
    )

    held = dc.crossval_cloud_mask(M_sst, np.random.default_rng(0))
    M_fit = M_sst & ~held

    Yas, clim_s, std_s = deseason_std(sst, M_fit, doy)
    Yac, _, _ = deseason_std(chl, M_chl, doy)
    Yax, _, _ = deseason_std(sss, M_sss, doy)

    see = held & M_sss  # held pixels where SSS sees through the cloud

    def score(fill_anom, tag):
        fill = fill_anom * std_s + clim_s
        a = np.sqrt(np.mean((fill[held] - sst[held]) ** 2))
        gapL4 = (~M_sst) & np.isfinite(l4)
        b = np.sqrt(np.mean((fill[gapL4] - l4[gapL4]) ** 2))
        a_see = np.sqrt(np.mean((fill[see] - sst[see]) ** 2))
        print(f"{tag:<14} checkA={a:.3f}  checkB={b:.3f}  A@SSS-seen={a_see:.3f}")
        return dict(
            config=tag,
            checkA=round(a, 3),
            checkB=round(b, 3),
            checkA_SSS_seen=round(a_see, 3),
        )

    rows = []
    single, _ = dc.dineof_iterative(
        np.where(M_fit, Yas, 0.0), M_fit, K=20, n_iter=80, temporal_filter=3.0
    )
    rows.append(score(single, "SST only"))
    for label, blocks in [
        ("SST+CHL", [(Yas, M_fit), (Yac, M_chl)]),
        ("SST+SSS", [(Yas, M_fit), (Yax, M_sss)]),
        ("SST+CHL+SSS", [(Yas, M_fit), (Yac, M_chl), (Yax, M_sss)]),
    ]:
        fills = dc.multivariate_dineof(blocks, K=20, temporal_filter=3.0)
        rows.append(score(fills[0], label))
    pd.DataFrame(rows).to_csv(DATA / "ablation_multivariate.csv", index=False)

    # diagnostic: cross-variable anomaly correlations explain the (non-)gain.
    # Deseasonalise on the FULL observed mask here (not the sparse CV M_fit, whose
    # ~12% per-pixel coverage makes the harmonic fit unreliable) so the reported
    # correlation is the true physical one.
    Yas_full, _, _ = deseason_std(sst, M_sst, doy)

    def corr(A, Ma, B, Mb):
        m = Ma & Mb
        return round(float(np.corrcoef(A[m], B[m])[0, 1]), 3)

    cor = dict(
        sst_chl=corr(Yas_full, M_sst, Yac, M_chl),
        sst_sss=corr(Yas_full, M_sst, Yax, M_sss),
        chl_sss=corr(Yac, M_chl, Yax, M_sss),
        cov_sst=round(float(M_sst.mean()), 3),
        cov_chl=round(float(M_chl.mean()), 3),
        cov_sss=round(float(M_sss.mean()), 3),
    )
    pd.DataFrame([cor]).to_csv(DATA / "multivariate_correlations.csv", index=False)
    print("correlations:", cor)


if __name__ == "__main__":
    main()
