"""Householder QR decomposition of an orthogonal matrix.

Given ``Q ∈ O(d)``, return ``d`` reflector vectors so that, when fed
into :class:`gaussianization.gauss_keras.bijectors.Householder`, the
layer computes ``y = x @ Q`` on batched rows.

The standard Householder QR algorithm produces ``H̃_0, …, H̃_{d-1}``
with ``H̃_{d-1} … H̃_0 Q = R`` upper-triangular. For an orthogonal
``Q`` the resulting ``R`` is a diagonal of ``±1``. With the sign
convention ``α = +‖x‖`` (reflect to the positive axis), ``R = I`` is
achievable *iff* ``det(Q) = (-1)^d``. If the input ``Q`` does not have
the right determinant parity we flip one column (any orthonormal basis
is equivalent up to column signs for the uses IG init puts it to), so
the decomposition is always exact up to floating point.

The layer applies reflectors on row vectors as
``y = y @ H_0 @ H_1 @ … @ H_{K-1}``. Since ``Q = H̃_0 H̃_1 … H̃_{d-1}``
(Householders are self-inverse), setting ``layer.v[j] = v_qr[j]``
in the same forward QR order makes the layer compute ``x @ Q`` — no
reversal is needed. The function returns ``v_qr`` accordingly.
"""

from __future__ import annotations

import numpy as np


def _sign(x):
    return 1.0 if x >= 0 else -1.0


def householder_decompose(q, expected_num_reflectors=None):
    """Decompose orthogonal ``q`` into the reflector matrix expected by our layer.

    Args:
        q: a ``(d, d)`` array; must be orthogonal to within ``1e-5``.
        expected_num_reflectors: if given, raise ``ValueError`` when
            ``expected_num_reflectors != d`` (IG init needs a full-rank
            Householder layer).

    Returns:
        ``v`` of shape ``(d, d)``: each row is one reflector. Assign to
        ``layer.v`` (after casting to ``float32``).
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError(f"q must be a square 2-D matrix; got shape {q.shape}")
    d = q.shape[0]
    if expected_num_reflectors is not None and expected_num_reflectors != d:
        raise ValueError(
            "Householder IG init requires num_reflectors == d = "
            f"{d}; got {expected_num_reflectors}. Rebuild the layer "
            "with num_reflectors=d or use FixedOrtho rotations."
        )
    if not np.allclose(q @ q.T, np.eye(d), atol=1e-5):
        raise ValueError("q is not orthogonal to 1e-5 tolerance")

    # Pre-sign q so det(q) = (-1)^d, which guarantees R = I after QR.
    det = float(np.linalg.det(q))
    if _sign(det) != (-1.0) ** d:
        q = q.copy()
        q[:, 0] *= -1.0

    r = q.copy()
    vs_qr = []  # H̃_0, H̃_1, …, H̃_{d-1}
    for k in range(d):
        x = r[k:, k].copy()
        n = float(np.linalg.norm(x))
        v_full = np.zeros(d, dtype=np.float64)
        if n > 1e-12:
            e1 = np.zeros_like(x)
            e1[0] = 1.0
            v_sub = x - n * e1  # reflect x to +n · e_1
            v_norm_sq = float(v_sub @ v_sub)
            if v_norm_sq > 1e-14:
                v_full[k:] = v_sub
                # Apply H̃_k to the remaining submatrix.
                sub = r[k:, :]
                r[k:, :] = sub - 2.0 * np.outer(v_sub, v_sub @ sub) / v_norm_sq
            # Else: x already aligned with +e_1; v_full stays zero (identity).
        vs_qr.append(v_full)

    # Sanity: after the loop, r should be ≈ I. Tolerate 1e-6.
    if not np.allclose(r, np.eye(d), atol=1e-6):
        raise RuntimeError(
            f"Householder decomposition failed: residual ‖R - I‖ = "
            f"{float(np.max(np.abs(r - np.eye(d)))):.2e}"
        )

    # The layer applies reflectors as ``y = y @ H_0 @ H_1 … @ H_{K-1}`` on
    # batched rows. With ``Q = H̃_0 H̃_1 … H̃_{d-1}`` from QR, we can set
    # ``layer.v[j] = v_qr[j]`` in the same order (no reversal).
    return np.array(vs_qr, dtype=np.float64)


def apply_reflectors(v, x):
    """Numpy-side replica of the ``Householder`` layer's forward pass.

    Used by IG init to propagate ``Y`` through a Householder block
    without routing through Keras.

    Args:
        v: ``(K, d)`` reflector stack (layer.v).
        x: ``(n, d)`` batched inputs.

    Returns:
        ``(n, d)`` output of ``H_{K-1} … H_0 x``.
    """
    v = np.asarray(v, dtype=np.float64)
    y = np.asarray(x, dtype=np.float64).copy()
    for j in range(v.shape[0]):
        v_j = v[j]
        norm_sq = float(v_j @ v_j) + 1e-12
        proj = y @ v_j
        y = y - (2.0 * proj[:, None] / norm_sq) * v_j[None, :]
    return y
