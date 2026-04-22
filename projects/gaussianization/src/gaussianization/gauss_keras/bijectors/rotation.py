"""Rotation bijectors: Householder and fixed orthogonal.

Both have log |det J| = 0, so their ``add_loss`` contribution is zero
and they are effectively free during training — their job is to
redistribute non-Gaussianity across dimensions between marginal
Gaussianization layers.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import ops

from gaussianization.gauss_keras.bijectors.base import Bijector


class Householder(Bijector):
    """Orthogonal transform parameterised as a product of Householder reflectors.

    Let ``v_1, …, v_K ∈ R^d``. The transform is

        Q = H_K … H_1,  H_j = I - 2 v_j v_j^T / ‖v_j‖²

    which is orthogonal by construction (det = ±1, so log |det| = 0).
    The inverse is ``Q^T = H_1 … H_K`` — i.e. apply the reflectors in
    reverse order. No matrix is materialised; each reflector is a
    rank-1 update of cost ``O(batch · d)``.

    Args:
        num_reflectors: number of Householder vectors ``K``. ``K = d``
            covers the full orthogonal group; smaller ``K`` gives a
            restricted-rank family (cheaper, expressive enough for toy
            data).
    """

    def __init__(self, num_reflectors, **kwargs):
        super().__init__(**kwargs)
        self.num_reflectors = int(num_reflectors)

    def build(self, input_shape):
        d = int(input_shape[-1])
        self._d = d
        self.v = self.add_weight(
            name="v",
            shape=(self.num_reflectors, d),
            initializer=keras.initializers.RandomNormal(stddev=1.0),
            trainable=True,
        )
        super().build(input_shape)

    def _apply_reflectors(self, x, reverse):
        """Apply the stack of reflectors to ``x`` (shape ``(batch, d)``)."""
        order = range(self.num_reflectors - 1, -1, -1) if reverse else range(self.num_reflectors)
        y = x
        for j in order:
            v_j = self.v[j]
            norm_sq = ops.sum(v_j * v_j) + 1e-12
            proj = ops.sum(y * v_j, axis=-1, keepdims=True)
            y = y - (2.0 * proj / norm_sq) * v_j
        return y

    def forward_and_log_det(self, x):
        self._ensure_built(x)
        z = self._apply_reflectors(x, reverse=False)
        ldj = ops.zeros(ops.shape(x)[0], dtype=x.dtype)
        return z, ldj

    def inverse_and_log_det(self, z):
        self._ensure_built(z)
        x = self._apply_reflectors(z, reverse=True)
        ldj = ops.zeros(ops.shape(z)[0], dtype=z.dtype)
        return x, ldj


class FixedOrtho(Bijector):
    """Frozen orthogonal transform.

    Stores a ``(d, d)`` orthogonal matrix as a non-trainable weight.
    Two factory methods are provided:

    - :meth:`random_haar`  — Haar-distributed sample of O(d).
    - :meth:`from_pca`     — eigenvectors of the sample covariance of
      a training batch (useful as a first-layer decorrelator).
    """

    def __init__(self, q, **kwargs):
        super().__init__(**kwargs)
        q = np.asarray(q)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError(f"q must be a square 2-D matrix; got shape {q.shape}")
        self._q_init = q.astype(np.float32)

    def build(self, input_shape):
        d = int(input_shape[-1])
        if self._q_init.shape[0] != d:
            raise ValueError(
                f"FixedOrtho built for d={d} but q has shape {self._q_init.shape}"
            )
        self.q = self.add_weight(
            name="q",
            shape=(d, d),
            initializer=keras.initializers.Constant(self._q_init),
            trainable=False,
        )
        super().build(input_shape)

    def forward_and_log_det(self, x):
        self._ensure_built(x)
        z = ops.matmul(x, self.q)
        ldj = ops.zeros(ops.shape(x)[0], dtype=x.dtype)
        return z, ldj

    def inverse_and_log_det(self, z):
        self._ensure_built(z)
        x = ops.matmul(z, ops.transpose(self.q))
        ldj = ops.zeros(ops.shape(z)[0], dtype=z.dtype)
        return x, ldj

    @classmethod
    def random_haar(cls, dim, seed=None, **kwargs):
        """Sample a Haar-random orthogonal matrix via QR of a Gaussian.

        Args:
            dim: dimensionality ``d``.
            seed: integer seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((dim, dim))
        q, r = np.linalg.qr(a)
        # Fix sign convention so the sample is Haar-distributed.
        q = q * np.sign(np.diag(r))
        return cls(q, **kwargs)

    @classmethod
    def from_pca(cls, x, **kwargs):
        """Build from the eigenvectors of the sample covariance of ``x``.

        Args:
            x: training data of shape ``(n, d)``.
        """
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (n, d); got shape {x.shape}")
        xc = x - x.mean(axis=0, keepdims=True)
        cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh returns ascending eigvals; reverse for PCA convention.
        q = eigvecs[:, ::-1]
        return cls(q, **kwargs)
