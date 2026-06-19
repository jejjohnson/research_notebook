"""Verify DINEOF gap-filling of REAL L3 satellite SST against gap-free L4.

The fill uses ONLY the gappy L3S data (iterative DINEOF estimates the EOF basis
from the gaps themselves). L4 OSTIA is held out purely for verification. Two
checks:

  A. Cross-validation on real L3 observations — hide a random subset of pixels
     L3 actually saw, reconstruct, score against those held-out *measurements*.
     This is verification against real data; no L4 involved.

  B. Compare the fill to L4 at the real cloud-gap pixels — where L3 is missing
     but L4 has a value. This is the operational-interpolation comparison.

A baseline RMSE(L3 vs L4) where both exist tells us the L3/L4 disagreement floor
(skin-vs-foundation + obs noise), so we know what a "good" check-B number is.
"""

from __future__ import annotations

from pathlib import Path

import dineof_core as dc
import numpy as np
import xarray as xr


DATA = Path(__file__).resolve().parents[1] / "data"
KELVIN = 273.15


def load() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Return (l3 cube degC with NaN gaps, l4 cube degC regridded, (H, W))."""
    l3 = xr.open_dataset(DATA / "l3s_sst_gulfstream_2019.nc")["sea_surface_temperature"]
    l4 = xr.open_dataset(DATA / "l4_ostia_gulfstream_2019.nc")["analysed_sst"]
    # regrid L4 (0.05 deg) onto the coarser L3S grid (0.10 deg) for pixel match.
    l4 = l4.interp(latitude=l3.latitude, longitude=l3.longitude)
    l3c = l3.values - KELVIN  # (T, H, W), NaN where cloudy
    l4c = l4.values - KELVIN
    return l3c, l4c, l3c.shape[1:]


def main() -> None:
    l3cube, l4cube, _hw = load()
    T = l3cube.shape[0]
    obs_frac = np.mean(np.isfinite(l3cube))
    print(f"L3 cube {l3cube.shape}  mean daily observed fraction: {obs_frac:.3f}")

    # matrix over pixels L3 saw at least once (the reconstructable domain).
    flat3 = l3cube.reshape(T, -1)
    flat4 = l4cube.reshape(T, -1)
    domain = np.isfinite(flat3).any(0) & np.isfinite(flat4).any(0)
    Y = flat3[:, domain]  # (T, N) real L3, NaN at gaps
    L4 = flat4[:, domain]
    M = np.isfinite(Y)  # True where L3 observed
    N = Y.shape[1]
    print(f"reconstructable domain N={N} pixels; overall L3 coverage {M.mean():.3f}")

    # L3/L4 disagreement floor where both present.
    both = M & np.isfinite(L4)
    print(
        f"baseline RMSE(L3 obs vs L4) where both exist: "
        f"{np.sqrt(np.mean((Y[both] - L4[both]) ** 2)):.3f} degC\n"
    )

    rng = np.random.default_rng(0)
    # hold out 20% of real L3 observations for check A.
    held = M & (rng.random(M.shape) < 0.20)
    M_fit = M & ~held
    Yfit = np.where(M_fit, Y, 0.0)

    print("K   checkA RMSE(vs held-out L3)   checkB RMSE(fill vs L4 @ gaps)")
    best = None
    for K in (5, 10, 20, 30, 50):
        filled, _ = dc.dineof_iterative(Yfit, M_fit, K=K, n_iter=100)
        a = np.sqrt(np.mean((filled[held] - Y[held]) ** 2))  # vs real measurements
        gapL4 = (~M) & np.isfinite(L4)  # real clouds, L4 known
        b = np.sqrt(np.mean((filled[gapL4] - L4[gapL4]) ** 2))
        print(f"{K:<3} {a:>20.3f}        {b:>20.3f}")
        if best is None or a < best[1]:
            best = (K, a, b)

    K, a, b = best
    print(f"\nbest K={K}: held-out-L3 RMSE={a:.3f} degC, fill-vs-L4 RMSE={b:.3f} degC")


if __name__ == "__main__":
    main()
