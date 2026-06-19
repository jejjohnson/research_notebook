"""Figures for the real-L3 / L4 DINEOF comparison."""

from __future__ import annotations

from pathlib import Path

import dineof_core as dc
import matplotlib.pyplot as plt
import numpy as np
from verify_l3_l4 import load


ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "notebooks" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

K = 20  # EOFs for the displayed fill


def main() -> None:
    l3cube, l4cube, hw = load()
    H, W = hw
    T = l3cube.shape[0]
    flat3, flat4 = l3cube.reshape(T, -1), l4cube.reshape(T, -1)
    domain = np.isfinite(flat3).any(0) & np.isfinite(flat4).any(0)
    Y, L4 = flat3[:, domain], flat4[:, domain]
    M = np.isfinite(Y)
    Yfit = np.where(M, Y, 0.0)

    filled, _ = dc.dineof_iterative(Yfit, M, K=K, n_iter=100)

    def to_grid(vec_over_domain):
        full = np.full(domain.size, np.nan)
        full[domain] = vec_over_domain
        return full.reshape(H, W)

    # pick a representative cloudy day (~median coverage).
    cov = M.mean(1)
    t = int(np.argsort(cov)[len(cov) // 2])

    l3_g = to_grid(np.where(M[t], Y[t], np.nan))
    fill_g = to_grid(filled[t])
    l4_g = to_grid(L4[t])
    diff_g = fill_g - l4_g

    vmin = np.nanmin(l4_g)
    vmax = np.nanmax(l4_g)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)
    panels = [
        (
            l3_g,
            f"real L3 ({cov[t]:.0%} observed)",
            dict(vmin=vmin, vmax=vmax, cmap="turbo"),
        ),
        (
            fill_g,
            f"DINEOF fill (K={K}, L3 only)",
            dict(vmin=vmin, vmax=vmax, cmap="turbo"),
        ),
        (l4_g, "L4 OSTIA (gap-free)", dict(vmin=vmin, vmax=vmax, cmap="turbo")),
        (diff_g, "fill - L4", dict(vmin=-2, vmax=2, cmap="RdBu_r")),
    ]
    for a, field, title, kw in zip(ax, *zip(*panels, strict=False), strict=False):
        im = a.imshow(field, origin="lower", **kw)
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])
        fig.colorbar(im, ax=a, shrink=0.8, label="degC")
    fig.savefig(FIGS / "dineof_l3l4_scene.png", dpi=110)
    plt.close(fig)
    print("wrote", FIGS / "dineof_l3l4_scene.png", "for day", t)


if __name__ == "__main__":
    main()
