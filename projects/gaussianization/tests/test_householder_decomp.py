"""Tests for the Householder QR decomposition used by IG init."""

from __future__ import annotations

import numpy as np
import pytest

from gaussianization.gauss_keras.bijectors._householder_decomp import (
    apply_reflectors,
    householder_decompose,
)


def _haar_random(d, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    return q


def test_decompose_then_apply_reconstructs_q_det_plus():
    for d in (2, 4, 5, 7):
        q = _haar_random(d, seed=d)
        # Ensure det = (-1)^d.
        if np.sign(np.linalg.det(q)) != (-1.0) ** d:
            q[:, 0] *= -1.0
        v = householder_decompose(q)
        # Apply reflectors to an identity matrix -> should reproduce q.
        q_reconstructed = apply_reflectors(v, np.eye(d)).T
        # apply_reflectors acts row-wise: for input row x, output = Q x.
        # So applying to rows of I gives columns of Q^T => rows are Q^T rows.
        # The reconstructed matrix above has each row = Q x => = (Q I)^T row-wise.
        # Simpler check: apply on a batch of random vectors and compare to Q @ x.
        rng = np.random.default_rng(100 + d)
        x = rng.standard_normal((16, d))
        y_layer = apply_reflectors(v, x)
        y_target = x @ q  # layer applies right-multiplication semantics
        np.testing.assert_allclose(y_layer, y_target, atol=1e-6)


def test_decompose_pre_flips_wrong_det():
    # Build q with det = -(-1)^d; decompose should still succeed (internally flips).
    d = 4
    q = _haar_random(d, seed=0)
    if np.sign(np.linalg.det(q)) == (-1.0) ** d:
        q[:, 0] *= -1.0
    # Now det(q) has the wrong sign; decompose must still produce a valid result
    # (not equal to q, but orthogonal and constructed from the sign-corrected q).
    v = householder_decompose(q)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, d))
    y_layer = apply_reflectors(v, x)
    # The applied Q will differ from q by one column sign flip. Check orthogonality
    # via a roundtrip on the norm preserved.
    np.testing.assert_allclose(
        np.linalg.norm(y_layer, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-6
    )


def test_decompose_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        householder_decompose(np.zeros((3, 4)))


def test_decompose_rejects_non_orthogonal():
    with pytest.raises(ValueError, match="orthogonal"):
        householder_decompose(np.eye(3) * 2.0)


def test_decompose_rejects_wrong_expected_count():
    d = 4
    q = _haar_random(d, seed=2)
    with pytest.raises(ValueError, match="num_reflectors == d"):
        householder_decompose(q, expected_num_reflectors=d - 1)
