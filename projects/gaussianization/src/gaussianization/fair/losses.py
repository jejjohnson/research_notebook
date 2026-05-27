"""Fairness losses computed in Gaussianized space.

Both losses take ``(y_true=q, y_pred=z)`` to match the convention used
by ``fairkl.models.FairModelWrapper`` — see
``fair_wrapper.FairModelWrapper.call``.
"""

from __future__ import annotations

import keras
from keras import ops


class GaussianizedXCovLoss(keras.losses.Loss):
    r"""Frobenius norm of cross-covariance in Gaussianized space ("G-XCOV").

    Given two pretrained flows :math:`T_z, T_q` and a batch
    :math:`\{(z_i, q_i)\}_{i=1}^n`,

    .. math::

        \mathcal{L} \;=\; \bigl\lVert
            \widehat{\mathrm{Cov}}\bigl(T_z(z), T_q(q)\bigr)
        \bigr\rVert_F^2.

    Because :math:`T_z, T_q` are diffeomorphisms,
    :math:`T_z(z) \perp T_q(q) \iff z \perp q`. Because the marginals
    of :math:`T_z(z), T_q(q)` are (approximately) standard Normal, the
    cross-covariance Frobenius norm is a tight low-order surrogate for
    full non-linear dependence in data space.

    The flows are expected to be **frozen** (call
    :func:`gaussianization.fair.freeze.freeze_flow` before passing them
    here). The loss is bounded below by zero and goes to zero when
    :math:`z` and :math:`q` are independent.

    Args:
        flow_z: Pretrained Gaussianization flow for the predictor output
            space (``log_prob`` not required; only ``__call__`` is used).
        flow_q: Pretrained Gaussianization flow for the sensitive
            attribute space.
        normalize: If True, divide by the product of marginal variances
            so the loss is scale-invariant and roughly comparable across
            problems. Default ``True``.
        eps: Small constant guarding the division.
    """

    def __init__(
        self,
        flow_z: keras.Model,
        flow_q: keras.Model,
        normalize: bool = True,
        eps: float = 1e-12,
        name: str = "g_xcov",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.flow_z = flow_z
        self.flow_q = flow_q
        self.normalize = bool(normalize)
        self.eps = float(eps)

    def call(self, q_true, z_pred):
        z_pred = ops.cast(z_pred, "float32")
        if len(ops.shape(z_pred)) == 1:
            z_pred = ops.expand_dims(z_pred, axis=-1)
        q = ops.cast(q_true, z_pred.dtype)
        if len(ops.shape(q)) == 1:
            q = ops.expand_dims(q, axis=-1)

        zg = self.flow_z(z_pred)
        qg = self.flow_q(q)

        zg = zg - ops.mean(zg, axis=0, keepdims=True)
        qg = qg - ops.mean(qg, axis=0, keepdims=True)
        n = ops.cast(ops.shape(zg)[0], zg.dtype)
        denom = ops.maximum(n - 1.0, 1.0)
        cross = ops.matmul(ops.transpose(zg), qg) / denom
        loss = ops.sum(cross * cross)

        if self.normalize:
            sz = ops.sum(zg * zg) / denom
            sq = ops.sum(qg * qg) / denom
            loss = loss / (sz * sq + self.eps)
        return loss


class GaussianizedHSICLoss(keras.losses.Loss):
    r"""HSIC with linear kernels applied to Gaussianized features ("G-HSIC").

    .. math::

        \mathcal{L} \;=\; \tfrac{1}{(n-1)^2}\,
            \mathrm{tr}\bigl( H K_z H K_q \bigr)

    where :math:`K_z = T_z(z) T_z(z)^\top`,
    :math:`K_q = T_q(q) T_q(q)^\top`, and :math:`H = I - \tfrac{1}{n} 1 1^\top`
    is the centring matrix.

    This is mathematically equivalent to the squared Frobenius norm of
    the (un-normalised) cross-covariance — i.e. exactly
    :class:`GaussianizedXCovLoss` with ``normalize=False`` — but computed
    through the kernel formulation that mirrors
    ``fairkl.metrics.hsic.HSICLoss``. We keep both because the kernel
    form is the cleanest drop-in for users coming from ``fairkl``.

    Args:
        flow_z, flow_q: As in :class:`GaussianizedXCovLoss`.
    """

    def __init__(
        self,
        flow_z: keras.Model,
        flow_q: keras.Model,
        name: str = "g_hsic",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.flow_z = flow_z
        self.flow_q = flow_q

    def call(self, q_true, z_pred):
        z_pred = ops.cast(z_pred, "float32")
        if len(ops.shape(z_pred)) == 1:
            z_pred = ops.expand_dims(z_pred, axis=-1)
        q = ops.cast(q_true, z_pred.dtype)
        if len(ops.shape(q)) == 1:
            q = ops.expand_dims(q, axis=-1)

        zg = self.flow_z(z_pred)
        qg = self.flow_q(q)

        kz = ops.matmul(zg, ops.transpose(zg))
        kq = ops.matmul(qg, ops.transpose(qg))
        n = ops.cast(ops.shape(zg)[0], zg.dtype)
        h = (
            ops.eye(ops.shape(zg)[0], dtype=zg.dtype)
            - ops.ones((ops.shape(zg)[0], ops.shape(zg)[0]), dtype=zg.dtype) / n
        )
        khk = ops.matmul(ops.matmul(h, kz), ops.matmul(h, kq))
        denom = ops.maximum((n - 1.0) ** 2, 1.0)
        return ops.trace(khk) / denom
