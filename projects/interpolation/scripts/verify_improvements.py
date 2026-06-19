"""Ablation: which of improvements 1-5 actually beat the nb06 DINEOF baseline?

Each improvement is switched on one at a time. Metrics, both at fixed K=20:
  Check A — RMSE vs held-out REAL L3 using realistic cloud-shaped hold-out
            (improvement 3); the honest extrapolation skill.
  Check B — RMSE vs gap-free L4 at the real cloud gaps.

The smoothness weight is swept on top of the best config, because a global
Laplacian both helps in data voids and hurts at the sharp SST front.
"""

from __future__ import annotations

from pathlib import Path

import dineof_core as dc
import numpy as np
import pandas as pd
import xarray as xr


DATA = Path(__file__).resolve().parents[1] / "data"
KELVIN = 273.15
K = 20


def load():
    l3 = xr.open_dataset(DATA / "l3s_sst_gulfstream_2019.nc")
    l4 = xr.open_dataset(DATA / "l4_ostia_gulfstream_2019.nc")["analysed_sst"]
    l4 = l4.interp(latitude=l3.latitude, longitude=l3.longitude)
    return (
        l3["sea_surface_temperature"].values - KELVIN,
        l3["quality_level"].values,
        l3["sses_standard_deviation"].values,
        l4.values - KELVIN,
        l3["time"].dt.dayofyear.values.astype(float),
        l3.sizes,
    )


def build(sst, ql, sses, l4cube, doy, sizes):
    T = sst.shape[0]
    H, W = sizes["latitude"], sizes["longitude"]
    flat, flat4 = sst.reshape(T, -1), l4cube.reshape(T, -1)
    domain = np.isfinite(flat).any(0) & np.isfinite(flat4).any(0)
    return dict(
        T=T,
        hw=(H, W),
        domain=domain,
        Y=flat[:, domain],
        QL=ql.reshape(T, -1)[:, domain],
        S=sses.reshape(T, -1)[:, domain],
        L4=flat4[:, domain],
        doy=doy,
        L=dc.build_grid_laplacian(domain, (H, W)),
    )


def dineof_fill(d, held, *, deseason, qc, temp_filter, obs_r):
    """Return (anomaly fill, climatology, M_fit) for one configuration."""
    Y = d["Y"]
    M = np.isfinite(Y)
    if qc:
        M = M & (d["QL"] >= 5)
    M_fit = M & ~held
    if deseason:
        Ya, clim = dc.deseasonalize(np.where(M_fit, Y, 0.0), M_fit, d["doy"])
    else:
        cnt = M_fit.sum(0)
        mu = np.where(M_fit, Y, 0.0).sum(0) / np.maximum(cnt, 1)
        clim = np.broadcast_to(mu, Y.shape)
        Ya = Y - clim
    r = np.maximum(d["S"], 0.1) ** 2
    oe = float(np.sqrt(np.median(r[M_fit]))) if obs_r else None
    fill_anom, _ = dc.dineof_iterative(
        np.where(M_fit, Ya, 0.0),
        M_fit,
        K=K,
        n_iter=60,
        temporal_filter=3.0 if temp_filter else None,
        obs_err=oe,
    )
    return Ya, fill_anom, clim, M_fit, r


def score(d, fill, held):
    Y, L4 = d["Y"], d["L4"]
    M = np.isfinite(Y)
    a = np.sqrt(np.mean((fill[held] - Y[held]) ** 2))
    gapL4 = (~M) & np.isfinite(L4)
    b = np.sqrt(np.mean((fill[gapL4] - L4[gapL4]) ** 2))
    return a, b


def main():
    d = build(*load())
    M = np.isfinite(d["Y"])
    both = M & np.isfinite(d["L4"])
    base = np.sqrt(np.mean((d["Y"][both] - d["L4"][both]) ** 2))
    held = dc.crossval_cloud_mask(M, np.random.default_rng(0))
    print(
        f"domain N={d['domain'].sum()}  L3 coverage {M.mean():.3f}  "
        f"baseline RMSE(L3 vs L4)={base:.3f} degC  "
        f"held-out cloud pixels={held.mean():.3f}\n"
    )

    configs = [
        (
            "baseline (=nb06)",
            dict(deseason=False, qc=False, temp_filter=False, obs_r=False),
        ),
        (
            "+ deseason (1)",
            dict(deseason=True, qc=False, temp_filter=False, obs_r=False),
        ),
        (
            "+ temporal filter (5)",
            dict(deseason=True, qc=False, temp_filter=True, obs_r=False),
        ),
        (
            "+ QC ql>=5 (2a)",
            dict(deseason=True, qc=True, temp_filter=True, obs_r=False),
        ),
        (
            "+ obs-R, no QC (2b)",
            dict(deseason=True, qc=False, temp_filter=True, obs_r=True),
        ),
    ]
    rows = []
    good_base = None
    for name, cfg in configs:
        Ya, fill_anom, clim, M_fit, r = dineof_fill(d, held, **cfg)
        a, b = score(d, fill_anom + clim, held)
        rows.append(dict(config=name, checkA=round(a, 3), checkB=round(b, 3)))
        print(f"{name:<24} checkA={a:.3f}  checkB={b:.3f}")
        if name == "+ temporal filter (5)":  # the best base for the smooth sweep
            good_base = (Ya, fill_anom, clim, M_fit, r)

    # smoothness sweep on the GOOD base (deseason + temporal filter, no QC).
    print("\nsmoothness sweep on best base (deseason+tempfilter, improvement 4):")
    Ya, fill_anom, clim, M_fit, r = good_base
    for lam in (0.05, 0.1, 0.3, 1.0):
        out = dc.smooth_to_background(Ya, M_fit, fill_anom, r, d["L"], lam, beta=1.0)
        a, b = score(d, out + clim, held)
        rows.append(
            dict(
                config=f"+ smooth lam={lam} (4)", checkA=round(a, 3), checkB=round(b, 3)
            )
        )
        print(f"  lam={lam:<5} checkA={a:.3f}  checkB={b:.3f}")

    pd.DataFrame(rows).to_csv(DATA / "ablation_singlevar.csv", index=False)


if __name__ == "__main__":
    main()
