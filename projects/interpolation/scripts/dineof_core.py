"""Core math for DINEOF-as-reduced-order-3D-Var on a gappy SST field.

Everything here is plain NumPy: a truncated SVD for the EOF basis, a per-scene
ridge least-squares for the reduced-order 3D-Var, and the classic DINEOF
alternating projection as a cross-check. No JAX needed for v1 — the coefficient
problem is K-dimensional (K ~ 20), so it is trivial to solve in closed form.

Notation matches projects/interpolation/notebooks/dineof.md:
  x = x_b + Phi w,   w ~ N(0, Lambda),   B = Phi Lambda Phi^T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Float = NDArray[np.floating]
Bool = NDArray[np.bool_]


# ---------------------------------------------------------------------------
# Field <-> matrix plumbing
# ---------------------------------------------------------------------------
def fields_to_matrix(cube: Float) -> tuple[Float, Bool]:
    """(T, H, W) field stack -> (T, N) matrix over the *ocean* pixels only.

    Land/permanently-missing pixels (NaN in every time step) are dropped so the
    EOF basis is built on real water. Returns the matrix and the (H*W,) boolean
    ocean mask so reconstructions can be scattered back to the grid.
    """
    T = cube.shape[0]
    flat = cube.reshape(T, -1)
    ocean = ~np.all(np.isnan(flat), axis=0)  # pixel wet in at least one step
    return flat[:, ocean], ocean


def matrix_to_field(mat: Float, ocean: Bool, hw: tuple[int, int]) -> Float:
    """(T, N) ocean matrix -> (T, H, W) grid, NaN on land."""
    T = mat.shape[0]
    out = np.full((T, ocean.size), np.nan, dtype=mat.dtype)
    out[:, ocean] = mat
    return out.reshape(T, *hw)


# ---------------------------------------------------------------------------
# EOF basis (truncated SVD == PPCA principal subspace)
# ---------------------------------------------------------------------------
@dataclass
class EOFBasis:
    mean: Float  # (N,)  ensemble/background mean x_b
    Phi: Float  # (N, K) orthonormal spatial EOFs  -> synthesis dictionary
    Lambda: Float  # (K,)  coefficient prior variances (eigenvalues)

    @property
    def K(self) -> int:
        return self.Phi.shape[1]

    def synth(self, w: Float) -> Float:
        """x = x_b + Phi w   (w: (..., K) -> x: (..., N))."""
        return self.mean + w @ self.Phi.T


def fit_eofs(X: Float, K: int) -> EOFBasis:
    """Truncated SVD of the centred ensemble -> top-K EOFs and variances.

    X is (T, N). Columns of Phi are the leading right singular vectors (spatial
    patterns); Lambda = s^2/(T-1) are the temporal variances of the PCs, which is
    exactly the coefficient prior in the reduced-order 3D-Var.
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    # economy SVD; Vt rows are spatial EOFs.
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Phi = Vt[:K].T
    Lambda = (s[:K] ** 2) / (X.shape[0] - 1)
    return EOFBasis(mean=mean, Phi=Phi, Lambda=Lambda)


# ---------------------------------------------------------------------------
# Reduced-order 3D-Var: ridge least-squares in EOF-coefficient space
# ---------------------------------------------------------------------------
def reduced_3dvar(y: Float, m: Bool, basis: EOFBasis, R: float) -> Float:
    """One scene. Minimise  0.5||w||^2_{Lambda^-1} + 0.5/R ||m(y - x_b - Phi w)||^2.

    Closed form:  (A^T A / R + diag(1/Lambda)) w = A^T d / R,  with
    A = Phi[obs], d = (y - x_b)[obs]. Returns the full (N,) reconstruction.
    As R -> 0 this is the DINEOF projection; with finite R it is noise-aware.
    """
    A = basis.Phi[m]  # (n_obs, K)
    d = (y - basis.mean)[m]  # (n_obs,)
    lhs = (A.T @ A) / R + np.diag(1.0 / basis.Lambda)
    rhs = (A.T @ d) / R
    w = np.linalg.solve(lhs, rhs)
    return basis.synth(w)


def reduced_3dvar_batch(Y: Float, M: Bool, basis: EOFBasis, R: float) -> Float:
    """Apply :func:`reduced_3dvar` to a stack of scenes -> (T, N)."""
    return np.stack([reduced_3dvar(Y[t], M[t], basis, R) for t in range(Y.shape[0])])


# ---------------------------------------------------------------------------
# Classic DINEOF alternating projection (fixed basis, R -> 0 limit)
# ---------------------------------------------------------------------------
def dineof_classic(y: Float, m: Bool, basis: EOFBasis, n_iter: int = 50) -> Float:
    """Alternating projection with a *fixed* EOF basis.

    x <- x_b + Phi Phi^T (x - x_b);  x <- m*y + (1-m)*x.  Converges to the
    R -> 0 fixed point of :func:`reduced_3dvar`: matches data where observed,
    lives in the EOF span elsewhere. (Real DINEOF re-estimates the basis from the
    gappy data each sweep; here the basis is given, isolating the projection.)
    """
    x = np.where(m, y, basis.mean)
    P = basis.Phi @ basis.Phi.T  # (N, N) projector onto EOF span
    for _ in range(n_iter):
        x = basis.mean + (x - basis.mean) @ P.T
        x = np.where(m, y, x)
    return x


# ---------------------------------------------------------------------------
# Real DINEOF: estimate the EOF basis *from the gappy data itself*
# ---------------------------------------------------------------------------
def dineof_iterative(
    Y: Float,
    M: Bool,
    K: int,
    n_iter: int = 100,
    tol: float = 1e-5,
    temporal_filter: float | None = None,
    obs_err: Float | None = None,
) -> tuple[Float, Float]:
    """Beckers-Rixen DINEOF: fill -> truncated SVD -> refill, basis from the data.

    Unlike :func:`dineof_classic` (which takes a basis fitted on clean data), this
    estimates the rank-K subspace from the *gappy* matrix by iteration — the real
    algorithm, using only the observed values. Operates on the per-pixel temporal
    anomaly (each column's mean over its observed times is removed), which strips
    the spatial-mean / seasonal field so the EOFs model variability, not climate.

    ``temporal_filter`` (Alvera-Azcarate et al. 2009): if set, the temporal
    singular vectors are Gaussian-smoothed (sigma in time steps) each iteration,
    enforcing temporal coherence and damping noise in the reconstruction.

    Y: (T, N) data (value at missing entries is ignored). M: (T, N) bool observed.
    Returns (filled (T, N) in original units, per-pixel mean (N,)).
    """
    # per-pixel temporal mean over observed times (pixels are wet at least once).
    counts = M.sum(axis=0)
    mu = np.where(
        counts > 0, np.where(M, Y, 0.0).sum(axis=0) / np.maximum(counts, 1), 0.0
    )
    A = np.where(M, Y - mu, 0.0)  # anomaly, zero-filled at gaps
    prev_fill = np.zeros_like(A)
    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        Uk = U[:, :K]
        if temporal_filter:
            from scipy.ndimage import gaussian_filter1d

            Uk = gaussian_filter1d(Uk, temporal_filter, axis=0, mode="nearest")
        low = (Uk * s[:K]) @ Vt[:K]  # rank-K reconstruction
        if obs_err is None:
            A = np.where(M, Y - mu, low)  # hard replacement (trust obs exactly)
        else:
            # noise-aware shrinkage (improvement 2): blend obs toward the
            # low-rank estimate by the per-pixel signal-to-noise ratio, so
            # noisy observations are not interpolated as if exact.
            sig = float(np.var(low))
            alpha = sig / (sig + obs_err**2)
            A = np.where(M, alpha * (Y - mu) + (1 - alpha) * low, low)
        delta = np.sqrt(np.mean((low[~M] - prev_fill[~M]) ** 2))
        prev_fill = low
        if delta < tol:
            break
    return A + mu, mu


# ---------------------------------------------------------------------------
# Improvement 6: multivariate DINEOF (joint SVD across correlated variables)
# ---------------------------------------------------------------------------
def multivariate_dineof(
    blocks: list[tuple[Float, Bool]],
    K: int,
    n_iter: int = 80,
    temporal_filter: float | None = None,
) -> list[Float]:
    """Joint DINEOF over several variables so cross-correlations fill each gap.

    ``blocks`` is one (anomaly, observed-mask) pair per variable, each (T, N_v),
    already deseasonalised AND standardised to a common scale (divide by the
    variable's std) so the joint SVD is not dominated by the loudest variable.
    The blocks are concatenated along the spatial axis and a single DINEOF is run
    on the stack (Alvera-Azcarate et al. 2007); the shared temporal modes couple
    the variables, so an observed covariate (e.g. cloud-penetrating SSS) pulls
    information into another variable's voids. Returns the filled (T, N_v) list.
    """
    sizes = [ya.shape[1] for ya, _ in blocks]
    Y = np.concatenate([np.where(m, ya, 0.0) for ya, m in blocks], axis=1)
    M = np.concatenate([m for _, m in blocks], axis=1)
    filled, _ = dineof_iterative(
        Y, M, K=K, n_iter=n_iter, temporal_filter=temporal_filter
    )
    out, i = [], 0
    for n in sizes:
        out.append(filled[:, i : i + n])
        i += n
    return out


# ---------------------------------------------------------------------------
# Improvement 1: deseasonalising harmonic climatology
# ---------------------------------------------------------------------------
def deseasonalize(
    Y: Float, M: Bool, doy: Float, n_harmonics: int = 3, ridge: float = 1e-3
) -> tuple[Float, Float]:
    """Remove a smooth per-pixel seasonal cycle fitted on the observed entries.

    Fits ``mean + sum_h a_h cos + b_h sin`` of the annual frequency (and its
    harmonics) to each pixel's observed time series, vectorised as N tiny
    least-squares problems. SST anomalies about this climatology are far lower
    rank than the raw field, so the EOF fill needs fewer modes and extrapolates
    less wildly into voids.

    doy: (T,) day-of-year. Returns (anomaly (T, N), climatology (T, N)).
    """
    t = 2.0 * np.pi * doy[:, None] / 365.25
    cols = [np.ones_like(doy)]
    for h in range(1, n_harmonics + 1):
        cols += [np.cos(h * t[:, 0]), np.sin(h * t[:, 0])]
    P = np.stack(cols, axis=1)  # (T, p)
    Y0 = np.where(M, Y, 0.0)
    # per-pixel normal equations G c = rhs, batched over pixels.
    G = np.einsum("tn,tp,tq->npq", M.astype(float), P, P)
    rhs = np.einsum("tn,tp->np", M * Y0, P)
    G += ridge * np.eye(P.shape[1])[None]  # stabilise sparse-coverage pixels
    coeffs = np.linalg.solve(G, rhs[..., None])[..., 0]  # (N, p)
    clim = np.einsum("tp,np->tn", P, coeffs)  # (T, N)
    return Y - clim, clim


# ---------------------------------------------------------------------------
# Improvement 4: spatial smoothness prior (graph Laplacian over the grid)
# ---------------------------------------------------------------------------
def build_grid_laplacian(domain: Bool, hw: tuple[int, int]):
    """4-neighbour graph Laplacian (sparse, N x N) over the domain pixels.

    ``domain`` is the (H*W,) boolean of reconstructable pixels; the Laplacian
    connects each pixel to its observed-domain grid neighbours so a penalty
    ``lam * ||L x||^2`` makes the fill smooth across cloud voids instead of
    letting high-order EOFs extrapolate freely.
    """
    from scipy import sparse

    H, W = hw
    idx = -np.ones(H * W, dtype=int)
    idx[domain] = np.arange(domain.sum())
    grid = idx.reshape(H, W)
    rows, cols = [], []
    for shift_axis in (0, 1):
        a = grid
        b = np.roll(grid, -1, axis=shift_axis)
        valid = (a >= 0) & (b >= 0)  # both endpoints in domain
        if shift_axis == 0:
            valid[-1, :] = False
        else:
            valid[:, -1] = False
        rows += [a[valid], b[valid]]
        cols += [b[valid], a[valid]]
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    n = int(domain.sum())
    adj = sparse.coo_matrix((np.ones_like(r, dtype=float), (r, c)), shape=(n, n))
    adj = adj.tocsr()
    deg = sparse.diags(np.asarray(adj.sum(axis=1)).ravel())
    return (deg - adj).tocsr()


def smooth_solve_prep(basis: EOFBasis, L, lam: float):
    """Precompute the basis-dependent pieces of the smooth 3D-Var (once)."""
    LPhi = L @ basis.Phi  # (N, K)
    Lxb = L @ basis.mean  # (N,)
    return dict(PhiLLPhi=lam * (LPhi.T @ LPhi), LPhiLxb=lam * (LPhi.T @ Lxb))


def reduced_3dvar_smooth(
    y: Float,
    m: Bool,
    basis: EOFBasis,
    r_vec: Float,
    prep: dict | None = None,
) -> Float:
    """3D-Var fill with per-pixel obs error (improvement 2) + smoothness (4).

    Minimises  0.5||w||^2_{Lambda^-1} + 0.5 sum_obs (y-xb-Phi w)^2 / r
               + 0.5 lam ||L (xb + Phi w)||^2 .
    ``r_vec`` is the per-pixel observation-error variance (e.g. sses^2); ``prep``
    is :func:`smooth_solve_prep` output (omit for no smoothness term).
    """
    A = basis.Phi[m]  # (n_obs, K)
    d = (y - basis.mean)[m]
    rinv = 1.0 / r_vec[m]
    lhs = (A.T * rinv) @ A + np.diag(1.0 / basis.Lambda)
    rhs = A.T @ (rinv * d)
    if prep is not None:
        lhs = lhs + prep["PhiLLPhi"]
        rhs = rhs - prep["LPhiLxb"]
    w = np.linalg.solve(lhs, rhs)
    return basis.synth(w)


def smooth_to_background(
    Y: Float,
    M: Bool,
    xb: Float,
    r_vec: Float,
    L,
    lam: float,
    beta: float,
) -> Float:
    """Per-scene smoothing anchored to a background (e.g. the DINEOF fill).

    For each scene minimises, over the full field x,
        sum_obs (x - y)^2 / r  +  lam ||L x||^2  +  beta ||x - xb||^2 .
    This is the right place for obs-error R (improvement 2) and spatial
    smoothness (4) on a *contiguous-cloud* problem: the background xb carries the
    temporally-pooled DINEOF estimate into the data voids, so observations only
    refine and denoise it — they never replace the temporal information, which a
    per-scene basis projection would discard. Solves a sparse SPD system per
    scene. Y, M, xb, r_vec are (T, N); returns (T, N).
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    LtL = (L.T @ L).tocsc()
    out = np.empty_like(Y)
    for t in range(Y.shape[0]):
        w = np.where(M[t], 1.0 / r_vec[t], 0.0)
        A = (diags(w + beta) + lam * LtL).tocsc()
        rhs = w * np.where(M[t], Y[t], 0.0) + beta * xb[t]
        out[t] = spsolve(A, rhs)
    return out


# ---------------------------------------------------------------------------
# Synthetic gaps + scoring
# ---------------------------------------------------------------------------
def punch_gaps(X: Float, frac: float, rng: np.random.Generator, blob: int = 0) -> Bool:
    """Boolean observation mask (True = observed) removing ~`frac` of pixels.

    If blob > 0, removes contiguous square cloud-like blobs of side `blob`
    instead of i.i.d. pixels, which is the regime DINEOF is actually built for.
    """
    T, N = X.shape
    M = rng.random((T, N)) > frac
    if blob > 0:
        # crude cloud blobs: knock out random windows in flattened index space.
        n_blobs = int(frac * N / (blob * blob)) + 1
        for t in range(T):
            for _ in range(n_blobs):
                start = rng.integers(0, max(1, N - blob))
                M[t, start : start + blob] = False
    return M


def rmse_on_heldout(truth: Float, recon: Float, observed: Bool) -> float:
    """RMSE over pixels that were hidden (observed == False), vs known truth."""
    hidden = ~observed
    return float(np.sqrt(np.mean((truth[hidden] - recon[hidden]) ** 2)))


def crossval_cloud_mask(M: Bool, rng: np.random.Generator) -> Bool:
    """Improvement 3: realistic cloud-shaped validation hold-out.

    For each day, borrow the cloud pattern of a randomly chosen *other* day and
    hold out the pixels that are observed today but cloudy on the donor day. This
    validates on contiguous cloud-shaped gaps — the hard extrapolation regime —
    rather than the too-easy i.i.d. pixels, so K is tuned for the case that
    actually matters (Beckers & Rixen 2003; Alvera-Azcarate et al.).
    """
    donor = rng.permutation(M.shape[0])
    return M & ~M[donor]
