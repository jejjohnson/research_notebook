"""End-to-end verification harness for the DINEOF-as-3D-Var example.

Loads the GLORYS SST cube, punches synthetic gaps, reconstructs with the
reduced-order 3D-Var and with classic DINEOF, and scores both against the
held-out truth. Prints a table and a K-sweep so we know the math is real before
authoring the notebook.
"""

from __future__ import annotations

from pathlib import Path

import dineof_core as dc
import numpy as np
import xarray as xr


DATA = Path(__file__).resolve().parents[1] / "data" / "glorys_sst_gulfstream_2019.nc"


def main() -> None:
    ds = xr.open_dataset(DATA)
    cube = ds["thetao"].squeeze("depth").values  # (T, H, W)
    _T, _H, _W = cube.shape
    X, _ocean = dc.fields_to_matrix(cube)  # (T, N) over ocean pixels
    print(f"cube {cube.shape}  ocean pixels N={X.shape[1]}  (land dropped)")

    rng = np.random.default_rng(0)

    # --- i.i.d. gaps, 40% removed ---------------------------------------------
    M = dc.punch_gaps(X, frac=0.40, rng=rng)
    Yobs = np.where(M, X, 0.0)  # 0 where unobserved (mean-centred later)
    print(f"\nobserved fraction (i.i.d.): {M.mean():.3f}")

    # Basis fit on the FULL gap-free truth = the ideal background covariance B.
    for K in (5, 10, 20, 40):
        basis = dc.fit_eofs(X, K=K)
        recon_var = dc.reduced_3dvar_batch(Yobs, M, basis, R=1e-4)
        rmse_var = dc.rmse_on_heldout(X, recon_var, M)
        # classic DINEOF on a few scenes (projector is N x N -> heavier).
        idx = [0, 100, 200, 300]
        rmse_din = np.mean(
            [
                dc.rmse_on_heldout(
                    X[t], dc.dineof_classic(Yobs[t], M[t], basis, 60), M[t]
                )
                for t in idx
            ]
        )
        # agreement between the two methods on held-out pixels (R->0 limit).
        agree = np.mean(
            [
                np.sqrt(
                    np.mean(
                        (
                            recon_var[t][~M[t]]
                            - dc.dineof_classic(Yobs[t], M[t], basis, 60)[~M[t]]
                        )
                        ** 2
                    )
                )
                for t in idx
            ]
        )
        print(
            f"K={K:>3}  3D-Var RMSE={rmse_var:.4f} degC   "
            f"DINEOF RMSE={rmse_din:.4f}   |3DVar-DINEOF|={agree:.4f}"
        )

    # --- R sensitivity at fixed K ---------------------------------------------
    print("\nR sweep (K=20, i.i.d. 40% gaps):")
    basis = dc.fit_eofs(X, K=20)
    for R in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
        recon = dc.reduced_3dvar_batch(Yobs, M, basis, R=R)
        print(f"  R={R:<7} RMSE={dc.rmse_on_heldout(X, recon, M):.4f} degC")

    # --- cloud blobs (the regime DINEOF is built for) -------------------------
    Mb = dc.punch_gaps(X, frac=0.30, rng=rng, blob=200)
    Yb = np.where(Mb, X, 0.0)
    basis = dc.fit_eofs(X, K=20)
    recon = dc.reduced_3dvar_batch(Yb, Mb, basis, R=1e-4)
    print(
        f"\ncloud-blob gaps: observed={Mb.mean():.3f}  "
        f"3D-Var RMSE={dc.rmse_on_heldout(X, recon, Mb):.4f} degC"
    )

    # field-scale reference: stdev of the SST anomalies we are reconstructing.
    print(f"\nSST anomaly std (scale reference): {X.std():.4f} degC")


if __name__ == "__main__":
    main()
