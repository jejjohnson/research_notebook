"""Figures for notebook 07: improved single-variable SST DINEOF."""

from __future__ import annotations

from pathlib import Path

import dineof_core as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from verify_improvements import K, build, load


ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "notebooks" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
LAM = 1.0  # best smoothness weight from the ablation sweep


def best_fill(d, rng):
    """Best config from the ablation: deseason (1) + temporal filter (5)
    + smoothness anchored to the DINEOF fill (4). QC and obs-R are excluded
    (they were harmful / neutral). Uses all observations."""
    Y = d["Y"]
    M = np.isfinite(Y)  # no QC: coverage matters more than marginal quality
    Ya, clim = dc.deseasonalize(np.where(M, Y, 0.0), M, d["doy"])
    fill_anom, _ = dc.dineof_iterative(
        np.where(M, Ya, 0.0), M, K=K, n_iter=80, temporal_filter=3.0
    )
    r = np.maximum(d["S"], 0.1) ** 2
    out = dc.smooth_to_background(Ya, M, fill_anom, r, d["L"], lam=LAM, beta=1.0)
    return out + clim, M


def main():
    d = build(*load())
    H, W = d["hw"]
    domain = d["domain"]
    fill, M = best_fill(d, np.random.default_rng(0))
    Y, L4 = d["Y"], d["L4"]

    def to_grid(vec):
        full = np.full(domain.size, np.nan)
        full[domain] = vec
        return full.reshape(H, W)

    cov = M.mean(1)
    t = int(np.argsort(cov)[len(cov) // 2])
    panels = [
        (
            to_grid(np.where(M[t], Y[t], np.nan)),
            f"real L3 ({cov[t]:.0%} obs)",
            "turbo",
            None,
        ),
        (to_grid(fill[t]), "improved DINEOF fill", "turbo", None),
        (to_grid(L4[t]), "L4 OSTIA", "turbo", None),
        (to_grid(fill[t] - L4[t]), "fill - L4", "RdBu_r", (-2, 2)),
    ]
    v = (np.nanmin(to_grid(L4[t])), np.nanmax(to_grid(L4[t])))
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)
    for a, (field, title, cmap, lim) in zip(ax, panels, strict=False):
        lo, hi = lim if lim else v
        im = a.imshow(field, origin="lower", vmin=lo, vmax=hi, cmap=cmap)
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])
        fig.colorbar(im, ax=a, shrink=0.8, label="degC")
    fig.savefig(FIGS / "dineof_improved_scene.png", dpi=110)
    plt.close(fig)

    # ablation bar chart from the saved csv.
    df = pd.read_csv(ROOT / "data" / "ablation_singlevar.csv")
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["checkA"], 0.4, label="Check A (vs held-out L3)")
    ax.bar(x + 0.2, df["checkB"], 0.4, label="Check B (vs L4 @ gaps)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (degC)")
    ax.set_title("Single-variable SST: cumulative improvements")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(FIGS / "dineof_improved_ablation.png", dpi=110)
    plt.close(fig)
    print("wrote improved scene + ablation figures for day", t)


if __name__ == "__main__":
    main()
