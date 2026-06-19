"""Figures for notebook 08: multivariate DINEOF diagnosis (SST + CHL + SSS)."""

from __future__ import annotations

from pathlib import Path

import dineof_core as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from verify_multivariate import deseason_std, load_aligned


ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "notebooks" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    d = load_aligned()
    doy = d["doy"]
    sst, chl, sss = d["sst"], d["chl"], d["sss"]
    H, W = d["hw"]
    dom = np.isfinite(sst).any(0)
    sst, chl, sss = sst[:, dom], chl[:, dom], sss[:, dom]
    Ms, Mc, Mx = np.isfinite(sst), np.isfinite(chl), np.isfinite(sss)
    Yas, clim_s, std_s = deseason_std(sst, Ms, doy)
    Yac, clim_c, std_c = deseason_std(chl, Mc, doy)
    Yax, clim_x, std_x = deseason_std(sss, Mx, doy)

    def to_grid(vec):
        full = np.full(dom.size, np.nan)
        full[dom] = vec
        return full.reshape(H, W)

    # ---- Figure 1: coverage on a representative day (complementary gaps) -----
    t = int(np.argsort(Ms.mean(1))[len(Ms) // 2])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    cov_panels = [
        (Ms, "SST (IR/cloud)"),
        (Mc, "CHL (cloud)"),
        (Mx, "SSS (microwave swath)"),
    ]
    for a, (m, name) in zip(ax, cov_panels, strict=False):
        a.imshow(
            to_grid(m[t].astype(float)), origin="lower", cmap="Greens", vmin=0, vmax=1
        )
        a.set_title(f"{name}\n{m[t].mean():.0%} observed")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Observed pixels on one day — green = data")
    fig.savefig(FIGS / "mv_coverage.png", dpi=110)
    plt.close(fig)

    # ---- Figure 2: anomaly correlations (why joint SVD doesn't help) ---------
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    corr_panels = [(Yac, Mc, "log-CHL"), (Yax, Mx, "SSS")]
    for a, (B, MB, name) in zip(ax, corr_panels, strict=False):
        m = Ms & MB
        x, y = Yas[m], B[m]
        a.hist2d(x, y, bins=80, range=[[-4, 4], [-4, 4]], cmap="magma", cmin=1)
        r = np.corrcoef(x, y)[0, 1]
        a.set_title(f"SST vs {name} anomaly\nr = {r:.2f}")
        a.set_xlabel("SST anomaly (std)")
        a.set_ylabel(f"{name} anomaly (std)")
    fig.savefig(FIGS / "mv_correlations.png", dpi=110)
    plt.close(fig)

    # ---- Figure 3: single vs multivariate Check A ----------------------------
    df = pd.read_csv(ROOT / "data" / "ablation_multivariate.csv")
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["checkA"], 0.4, label="Check A (all held)")
    ax.bar(x + 0.2, df["checkA_SSS_seen"], 0.4, label="Check A @ SSS-seen pixels")
    ax.axhline(df["checkA"].iloc[0], color="k", ls="--", lw=0.8, label="SST-only")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("SST RMSE (degC)")
    ax.set_ylim(1.3, 1.55)
    ax.set_title("Multivariate does not improve the SST fill here")
    ax.legend(fontsize=8)
    fig.savefig(FIGS / "mv_ablation.png", dpi=110)
    plt.close(fig)

    # ---- Figure 4: joint reconstruction of ALL variables, one day -----------
    # Run the full multivariate fill (all observations), then map each variable's
    # gappy input next to its joint DINEOF reconstruction in physical units.
    fills = dc.multivariate_dineof(
        [(Yas, Ms), (Yac, Mc), (Yax, Mx)], K=20, temporal_filter=3.0
    )
    sst_fill = fills[0] * std_s + clim_s
    chl_fill = 10.0 ** (fills[1] * std_c + clim_c)  # log10 -> mg/m3
    sss_fill = fills[2] * std_x + clim_x

    rows = [
        (
            "SST [°C]",
            np.where(Ms, sst, np.nan),
            sst_fill,
            "turbo",
            dict(vmin=10, vmax=26),
        ),
        (
            "CHL [mg/m³]",
            np.where(Mc, 10.0**chl, np.nan),
            chl_fill,
            "viridis",
            dict(norm=LogNorm(vmin=0.1, vmax=5)),
        ),
        (
            "SSS [PSU]",
            np.where(Mx, sss, np.nan),
            sss_fill,
            "cividis",
            dict(vmin=31, vmax=37),
        ),
    ]
    fig, ax = plt.subplots(3, 2, figsize=(8, 9), constrained_layout=True)
    for r, (name, obs, fill, cmap, kw) in enumerate(rows):
        for c, (field, col) in enumerate(
            [(obs[t], "observed (gappy)"), (fill[t], "joint DINEOF fill")]
        ):
            im = ax[r, c].imshow(to_grid(field), origin="lower", cmap=cmap, **kw)
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(col)
            if c == 0:
                ax[r, c].set_ylabel(name, fontsize=11)
        fig.colorbar(im, ax=ax[r, :], shrink=0.7)
    fig.suptitle(f"Multivariate DINEOF: joint reconstruction (day {t})")
    fig.savefig(FIGS / "mv_reconstruction.png", dpi=110)
    plt.close(fig)
    print("wrote mv_coverage, mv_correlations, mv_ablation, mv_reconstruction; day", t)


if __name__ == "__main__":
    main()
