"""Fairness losses computed in Gaussianized space.

All losses take ``(y_true=q, y_pred=z)`` to match the convention used
by :class:`fairkl.models.FairModelWrapper` -- see
``fair_wrapper.FairModelWrapper.call``.

Three penalties are provided:

- :class:`GaussianizedXCovLoss` (G-XCOV): the **second-moment** dependence
  in Gaussianised space -- linear-CKA-normalised squared cross-covariance
  Frobenius norm.  Smooth, bounded in ``[0, 1]``.  Equivalent to HSIC with
  linear kernels on the Gaussianised features.
- :class:`GaussianizedMutualInfoLoss` (G-MI): the **closed-form Gaussian
  mutual-information** estimate on Gaussianised features,
  :math:`-\\tfrac{1}{2} \\log\\det(I - C C^\\top)`.  Sharper near perfect
  dependence (gradient diverges) so it can push the last bit of
  dependence out of a predictor that G-XCOV would leave behind.
- :class:`GaussianizedTotalCorrelationLoss` (G-TC): the negative
  log-likelihood under :math:`\\mathcal{N}(0, I)` of a frozen **joint**
  flow trained on the product-of-marginals distribution.  Captures
  higher-order dependence that the Gaussian-MI assumption misses; the
  most flexible (and the most expensive) of the three.
"""

from __future__ import annotations

import keras
from keras import ops


def _pack_batch(z_pred, q_true):
    """Cast and broadcast (z, q) to (n, d_z), (n, d_q) float32 batches."""
    z_pred = ops.cast(z_pred, "float32")
    if len(ops.shape(z_pred)) == 1:
        z_pred = ops.expand_dims(z_pred, axis=-1)
    q = ops.cast(q_true, z_pred.dtype)
    if len(ops.shape(q)) == 1:
        q = ops.expand_dims(q, axis=-1)
    return z_pred, q


def _gaussianised_cross_cov(zg, qg):
    """Return centred ``zg, qg`` and their sample cross-covariance ``C``.

    Shapes: ``zg : (n, d_z)``, ``qg : (n, d_q)``, ``C : (d_z, d_q)``.
    """
    zg = zg - ops.mean(zg, axis=0, keepdims=True)
    qg = qg - ops.mean(qg, axis=0, keepdims=True)
    n = ops.cast(ops.shape(zg)[0], zg.dtype)
    denom = ops.maximum(n - 1.0, 1.0)
    C = ops.matmul(ops.transpose(zg), qg) / denom
    return zg, qg, C, denom


class GaussianizedXCovLoss(keras.losses.Loss):
    r"""Linear-CKA-normalised cross-covariance in Gaussianised space (**G-XCOV**).

    Given pretrained flows :math:`T_z, T_q` and a batch
    :math:`\{(z_i, q_i)\}_{i=1}^n`, write
    :math:`Z = T_z(z)`, :math:`Q = T_q(q)`. Let
    :math:`C = \widehat{\mathrm{Cov}}(Z, Q) \in \mathbb{R}^{d_z \times d_q}`,
    :math:`S_z = \widehat{\mathrm{Cov}}(Z, Z)`,
    :math:`S_q = \widehat{\mathrm{Cov}}(Q, Q)`.

    .. math::

        \mathcal{L}_{\text{G-XCOV}}
          \;=\;
          \begin{cases}
            \dfrac{\lVert C \rVert_F^2}
                  {\lVert S_z \rVert_F \, \lVert S_q \rVert_F + \varepsilon}
              & \text{if ``normalize=True``  (default)},\\[6pt]
            \lVert C \rVert_F^2 & \text{otherwise.}
          \end{cases}

    The normalised form is **exact linear CKA applied to the Gaussianised
    features** (Cortes et al. 2012); in :math:`d_z = d_q = 1` it collapses
    to :math:`\rho^2` where :math:`\rho` is the Gaussianised cross-correlation.

    Because :math:`T_z, T_q` are diffeomorphisms,
    :math:`T_z(z) \perp T_q(q) \iff z \perp q`. Because the marginals of
    :math:`Z, Q` are (approximately) :math:`\mathcal{N}(0, 1)`, the
    cross-covariance norm is a tight low-order surrogate for full
    non-linear dependence in data space.

    .. note::

       The un-normalised form (``normalize=False``) is **identical to HSIC
       with linear kernels** on the Gaussianised features. There is no
       separate ``GaussianizedHSICLoss`` class -- just instantiate this
       loss with ``normalize=False`` if you want the kernel formulation.

    The flows are expected to be **frozen** (call
    :func:`gaussianization.fair.freeze.freeze_flow` before passing them
    here). The loss is bounded below by zero and goes to zero when
    :math:`z, q` are independent.

    Args:
        flow_z: Pretrained Gaussianization flow for the predictor output
            space (``log_prob`` not required; only ``__call__`` is used).
        flow_q: Pretrained Gaussianization flow for the sensitive
            attribute space.
        normalize: If ``True`` (default), use the linear-CKA Frobenius-norm
            denominator. If ``False``, return the raw squared cross-cov
            Frobenius norm (= HSIC with linear kernels).
        eps: Small constant added to the denominator for numerical stability.
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
        z_pred, q = _pack_batch(z_pred, q_true)
        zg = self.flow_z(z_pred)
        qg = self.flow_q(q)
        zg, qg, C, denom = _gaussianised_cross_cov(zg, qg)

        loss = ops.sum(C * C)
        if self.normalize:
            S_z = ops.matmul(ops.transpose(zg), zg) / denom
            S_q = ops.matmul(ops.transpose(qg), qg) / denom
            fz = ops.sqrt(ops.sum(S_z * S_z))
            fq = ops.sqrt(ops.sum(S_q * S_q))
            loss = loss / (fz * fq + self.eps)
        return loss


class GaussianizedMutualInfoLoss(keras.losses.Loss):
    r"""Closed-form Gaussian-MI on Gaussianised features (**G-MI**).

    Assuming :math:`(Z, Q) = (T_z(z), T_q(q))` is *jointly* approximately
    Gaussian with standardised marginals -- a strictly weaker assumption
    than assuming the raw :math:`(z, q)` is Gaussian -- mutual information
    has the Gel'fand--Yaglom closed form:

    .. math::

       I_{\text{G-MI}}(z; q) \;=\;
         -\tfrac{1}{2}\,\log\det\!\bigl(I_{d_z} - C\, S_q^{-1} C^\top\, S_z^{-1}\bigr)

    where :math:`C = \widehat{\mathrm{Cov}}(Z, Q)`,
    :math:`S_z = \widehat{\mathrm{Cov}}(Z, Z)`,
    :math:`S_q = \widehat{\mathrm{Cov}}(Q, Q)`.

    After Gaussianisation, :math:`S_z \approx I_{d_z}` and
    :math:`S_q \approx I_{d_q}` by construction, so to leading order this
    is :math:`-\tfrac{1}{2}\log\det(I - C C^\top)`. We **standardise**
    :math:`Z, Q` explicitly along axis-0 before computing :math:`C` so the
    formula holds even when the flow leaves residual marginal variance.

    In :math:`d_z = d_q = 1` this collapses to the familiar
    :math:`-\tfrac{1}{2}\log(1 - \rho^2)` of the Gaussianised
    cross-correlation. We use that scalar fast path when both dimensions
    are 1, otherwise we go through a symmetric eigendecomposition.

    Gradient behaviour: the loss is :math:`+\infty` at perfect dependence
    (:math:`CC^\top \to I`), so the gradient is **unbounded**. That is
    why this loss can drive the last bit of dependence out of a predictor
    that G-XCOV plateaus on -- but it also means you need a smaller
    :math:`\mu` and/or gradient clipping to keep training stable.

    The eigenvalues of :math:`I - C C^\top` are clipped at ``eps`` to
    keep both the log and the gradient finite.

    Args:
        flow_z: Pretrained, frozen Gaussianization flow for :math:`z`.
        flow_q: Pretrained, frozen Gaussianization flow for :math:`q`.
        eps: Numerical floor for the clipped eigenvalues. Default ``1e-4``.
            Caps the maximum loss at :math:`-\tfrac{1}{2}\log\varepsilon`
            -- e.g. ``4.6`` for ``eps=1e-4`` (in 1-D).
    """

    def __init__(
        self,
        flow_z: keras.Model,
        flow_q: keras.Model,
        eps: float = 1e-4,
        name: str = "g_mi",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.flow_z = flow_z
        self.flow_q = flow_q
        self.eps = float(eps)

    def call(self, q_true, z_pred):
        z_pred, q = _pack_batch(z_pred, q_true)
        zg = self.flow_z(z_pred)
        qg = self.flow_q(q)

        # Explicitly standardise so the closed-form holds even if the
        # flow leaves residual marginal scale.
        zg = zg - ops.mean(zg, axis=0, keepdims=True)
        qg = qg - ops.mean(qg, axis=0, keepdims=True)
        n = ops.cast(ops.shape(zg)[0], zg.dtype)
        denom = ops.maximum(n - 1.0, 1.0)
        sz = ops.sqrt(ops.mean(zg * zg, axis=0, keepdims=True) + self.eps)
        sq = ops.sqrt(ops.mean(qg * qg, axis=0, keepdims=True) + self.eps)
        zg = zg / sz
        qg = qg / sq

        d_z = ops.shape(zg)[1]
        d_q = ops.shape(qg)[1]

        if int(d_z) == 1 and int(d_q) == 1:
            # Scalar fast path: rho^2, then -0.5 log(1 - rho^2).
            rho = ops.sum(zg * qg) / denom
            one_minus = ops.maximum(1.0 - rho * rho, self.eps)
            return -0.5 * ops.log(one_minus)

        # General path: M = I - C C^T (symmetric PSD when standardised).
        C = ops.matmul(ops.transpose(zg), qg) / denom
        M = ops.eye(d_z, dtype=zg.dtype) - ops.matmul(C, ops.transpose(C))
        # Symmetrise to kill float noise before eigh.
        M = 0.5 * (M + ops.transpose(M))
        eigvals = ops.linalg.eigh(M)[0]
        eigvals = ops.maximum(eigvals, self.eps)
        return -0.5 * ops.sum(ops.log(eigvals))


class GaussianizedTotalCorrelationLoss(keras.losses.Loss):
    r"""Total correlation under a frozen joint flow (**G-TC**).

    Train a 2-D (or :math:`(d_z + d_q)`-D) joint Gaussianization flow
    :math:`T_{zq}` on the **product distribution** of the baseline:
    pairs :math:`(z^{(0)}, q^{(\pi)})` where :math:`q^{(\pi)}` is a random
    permutation of :math:`q`. Under this distribution :math:`z, q` are
    independent by construction, so a well-fit :math:`T_{zq}` Gaussianises
    independent draws to :math:`\mathcal{N}(0, I)`. Freeze :math:`T_{zq}`.

    At downstream training time, evaluate the joint flow on the
    **actual** :math:`(z, q) = (f_\theta(X), q)` and compute the negative
    log-density under :math:`\mathcal{N}(0, I)`:

    .. math::

        \mathcal{L}_{\text{G-TC}}
          \;=\; -\tfrac{1}{n}\sum_{i=1}^{n}
                \log p_{\mathcal{N}(0, I)}\!\bigl(T_{zq}(z_i, q_i)\bigr).

    When :math:`(z, q)` is independent like the baseline product
    distribution, :math:`T_{zq}(z, q) \sim \mathcal{N}(0, I)` and the
    loss is the entropy of :math:`\mathcal{N}(0, I)` (a constant).
    When :math:`(z, q)` carries dependence, :math:`T_{zq}` no longer
    Gaussianises the joint and the NLL is strictly larger.

    This is essentially :math:`D_{\mathrm{KL}}\!\bigl(p(z, q) \,\|\, p(z)\,p(q)\bigr)
    \;=\; I(z; q)`, i.e. **mutual information**, but computed through a
    learned joint flow rather than under the closed-form Gaussian-joint
    assumption of :class:`GaussianizedMutualInfoLoss`. The flow can capture
    higher-order joint structure (XOR-style dependence, multi-modal
    dependence) that the Gaussian-MI estimator misses.

    Args:
        joint_flow: Pretrained, frozen joint Gaussianization flow over
            the concatenated input :math:`(z, q)`.  Must expose ``__call__``
            for forward and ``log_prob(x)`` for the per-sample
            log-density.  Train it on shuffled (independent) pairs --
            see :func:`gaussianization.fair.pretrain.fit_and_freeze_joint`.
    """

    def __init__(
        self,
        joint_flow: keras.Model,
        name: str = "g_tc",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.joint_flow = joint_flow

    def call(self, q_true, z_pred):
        z_pred, q = _pack_batch(z_pred, q_true)
        zq = ops.concatenate([z_pred, q], axis=-1)
        log_p = self.joint_flow.log_prob(zq)
        return -ops.mean(log_p)
