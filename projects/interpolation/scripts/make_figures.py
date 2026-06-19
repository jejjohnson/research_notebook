"""Generate the figures for the DINEOF notebook from verified results."""

from __future__ import annotations

from pathlib import Path

import dineof_core as dc
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "glorys_sst_gulfstream_2019.nc"
FIGS = ROOT / "notebooks" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ds = xr.open_dataset(DATA)
    cube = ds["thetao"].squeeze("depth").values
    _T, H, W = cube.shape
    X, ocean = dc.fields_to_matrix(cube)
    rng = np.random.default_rng(0)

    M = dc.punch_gaps(X, frac=0.40, rng=rng)
    Yobs = np.where(M, X, 0.0)
    basis = dc.fit_eofs(X, K=20)
    recon = dc.reduced_3dvar_batch(Yobs, M, basis, R=1e-4)

    # --- Figure 1: one scene, truth / gappy / reconstruction / error ----------
    t = 200
    truth_g = dc.matrix_to_field(X[t : t + 1], ocean, (H, W))[0]
    gap_g = dc.matrix_to_field(np.where(M[t], X[t], np.nan)[None], ocean, (H, W))[0]
    rec_g = dc.matrix_to_field(recon[t : t + 1], ocean, (H, W))[0]
    err_g = rec_g - truth_g

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)
    vmin, vmax = np.nanmin(truth_g), np.nanmax(truth_g)
    for a, field, title, kw in [
        (ax[0], truth_g, "GLORYS truth", dict(vmin=vmin, vmax=vmax, cmap="turbo")),
        (ax[1], gap_g, "40% gaps (observed)", dict(vmin=vmin, vmax=vmax, cmap="turbo")),
        (
            ax[2],
            rec_g,
            "3D-Var reconstruction (K=20)",
            dict(vmin=vmin, vmax=vmax, cmap="turbo"),
        ),
        (ax[3], err_g, "reconstruction error", dict(vmin=-2, vmax=2, cmap="RdBu_r")),
    ]:
        im = a.imshow(field, origin="lower", **kw)
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])
        fig.colorbar(im, ax=a, shrink=0.8, label="degC")
    fig.savefig(FIGS / "dineof_scene.png", dpi=110)
    plt.close(fig)

    # --- Figure 2: K sweep + equivalence -------------------------------------
    Ks = [2, 5, 10, 15, 20, 30, 40, 60]
    rmse_var, rmse_din = [], []
    idx = [0, 100, 200, 300]
    for K in Ks:
        b = dc.fit_eofs(X, K=K)
        # score both methods on the SAME scenes so the equivalence is visible.
        rmse_var.append(
            np.mean(
                [
                    dc.rmse_on_heldout(
                        X[s], dc.reduced_3dvar(Yobs[s], M[s], b, 1e-4), M[s]
                    )
                    for s in idx
                ]
            )
        )
        rmse_din.append(
            np.mean(
                [
                    dc.rmse_on_heldout(
                        X[s], dc.dineof_classic(Yobs[s], M[s], b, 60), M[s]
                    )
                    for s in idx
                ]
            )
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(Ks, rmse_var, "o-", label="reduced-order 3D-Var")
    ax.plot(Ks, rmse_din, "x--", label="classic DINEOF")
    ax.set_xlabel("number of EOFs K")
    ax.set_ylabel("held-out RMSE (degC)")
    ax.set_title("DINEOF == reduced-order 3D-Var; RMSE vs K")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIGS / "dineof_ksweep.png", dpi=110)
    plt.close(fig)

    print("wrote", FIGS / "dineof_scene.png", "and", FIGS / "dineof_ksweep.png")


if __name__ == "__main__":
    main()
