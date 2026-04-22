"""Real-NVP style coupling layer with a mixture-CDF Gaussianization transform.

Given a boolean mask ``m`` of length ``d`` the layer splits ``x`` into an
"identity" half (``x_a`` where ``m`` is True) and a "transformed" half
(``x_b`` where ``m`` is False). A user-supplied ``conditioner`` network
maps ``x_a`` to per-batch mixture parameters, and the forward transform
applies the conditional mixture-CDF Gaussianization on ``x_b`` only:

    z_a = x_a
    (π, μ, log σ) = conditioner(x_a)
    z_b = Φ⁻¹(F(x_b; π, μ, σ))
    log |det J| = Σ_{i ∈ b}  log f(x_{b,i}; π_i, μ_i, σ_i) - log φ(z_{b,i})

The Jacobian is lower-triangular, so the log-det is the per-b-dim
diagonal sum. The inverse uses the *same* conditioner (applied to
``z_a = x_a``, which is preserved by design) and bisects
``F(x_b; θ(z_a)) = Φ(z_b)``.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import ops

from gaussianization.gauss_keras._math import norm_cdf, norm_icdf, norm_log_pdf
from gaussianization.gauss_keras.bijectors.base import Bijector
from gaussianization.gauss_keras.mixtures import MixtureOfGaussians


# --------------------------------------------------------------------- #
# Standard log-scale clampers — keep σ bounded so the conditioner can't
# emit 1e10 or 1e-10 scales during early training.                       #
# --------------------------------------------------------------------- #


def tanh_log_scale_clamp(bound=3.0):
    """``log σ = bound · tanh(raw)`` → σ ∈ [exp(-bound), exp(bound)].

    Default ``bound=3`` gives σ ∈ [≈0.05, ≈20]. The returned callable
    carries ``.kind = "tanh"`` and ``.bound`` so that IG init can invert
    the clamp to recover the raw pre-clamp bias value.
    """
    bound = float(bound)

    def clamp(raw):
        return bound * ops.tanh(raw)

    clamp.kind = "tanh"
    clamp.bound = bound
    return clamp


def sigmoid_log_scale_clamp(min_log_scale=-3.0, max_log_scale=3.0):
    """``log σ = lo + (hi - lo) · sigmoid(raw)``, strictly bounded to [lo, hi]."""
    lo = float(min_log_scale)
    hi = float(max_log_scale)

    def clamp(raw):
        return lo + (hi - lo) * ops.sigmoid(raw)

    clamp.kind = "sigmoid"
    clamp.lo = lo
    clamp.hi = hi
    return clamp


# --------------------------------------------------------------------- #
# Mask helper                                                             #
# --------------------------------------------------------------------- #


def default_half_mask(d):
    """Return a length-``d`` bool mask with the first ``d // 2`` dims True."""
    d = int(d)
    mask = np.zeros(d, dtype=bool)
    mask[: d // 2] = True
    return mask


# --------------------------------------------------------------------- #
# Standard conditioner builders                                           #
# --------------------------------------------------------------------- #


class _RepeatAcrossDims(keras.layers.Layer):
    """(batch, 3*K) → (batch, 3*d_b*K) by tiling the per-param block d_b times."""

    def __init__(self, d_b, num_components, **kwargs):
        super().__init__(**kwargs)
        self.d_b = int(d_b)
        self.num_components = int(num_components)

    def call(self, x):
        k = self.num_components
        # (batch, 3*K) -> (batch, 3, 1, K) -> (batch, 3, d_b, K) -> (batch, 3*d_b*K)
        params = ops.reshape(x, (-1, 3, 1, k))
        params = ops.tile(params, (1, 1, self.d_b, 1))
        return ops.reshape(params, (-1, 3 * self.d_b * k))


def make_mlp_conditioner(
    d_b,
    num_components,
    hidden=(64, 64),
    activation="relu",
):
    """Per-dim conditioner MLP: ``(batch, d_a) → (batch, 3 · d_b · K)``.

    The final Dense is zero-initialised so the coupling layer starts as
    an identity transform (Glow convention): all mixture components
    collapse to ``𝒩(0, 1)``, giving ``F(x) = Φ(x)`` and ``z = x``.
    """
    layers = []
    for h in hidden:
        layers.append(keras.layers.Dense(int(h), activation=activation))
    layers.append(
        keras.layers.Dense(
            3 * int(d_b) * int(num_components),
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )
    )
    return keras.Sequential(layers)


def make_shared_mlp_conditioner(
    d_b,
    num_components,
    hidden=(64, 64),
    activation="relu",
):
    """Shared-mixture conditioner: one mixture reused across all b-dims.

    Equivalent to :func:`make_mlp_conditioner` followed by a copy / tile
    layer that broadcasts a single ``(3 · K)`` parameter block to all
    ``d_b`` dimensions. Cheaper than per-dim but less expressive.
    """
    layers = []
    for h in hidden:
        layers.append(keras.layers.Dense(int(h), activation=activation))
    layers.append(
        keras.layers.Dense(
            3 * int(num_components),
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )
    )
    layers.append(_RepeatAcrossDims(d_b=d_b, num_components=num_components))
    return keras.Sequential(layers)


# --------------------------------------------------------------------- #
# The coupling bijector                                                   #
# --------------------------------------------------------------------- #


class MixtureCDFCoupling(Bijector):
    """Coupling bijector with a conditional mixture-CDF transform on ``x_b``.

    Args:
        mask: length-``d`` bool array; ``True`` entries are the identity
            half (passed unchanged, fed to the conditioner).
        conditioner: a ``keras.Layer`` / ``keras.Model`` mapping
            ``(batch, d_a) → (batch, 3 · d_b · K)``. The output tensor
            is reshaped to ``(batch, 3, d_b, K)`` and split into
            ``(logits, means, log_scales)`` along axis 1. Use
            :func:`make_mlp_conditioner` or
            :func:`make_shared_mlp_conditioner` for standard builds.
        num_components: mixture components ``K`` per b-dim.
        bisect_steps / bisect_range / eps: inverse-bisection settings.
        log_scale_clamp: callable mapping raw log-scale to clamped log-scale.
            Default is :func:`tanh_log_scale_clamp` with ``bound=3``.

    At build time the layer runs one probe through the conditioner and
    raises a descriptive error if the output width does not match
    ``3 · d_b · K``.
    """

    def __init__(
        self,
        mask,
        conditioner,
        num_components=8,
        bisect_steps=40,
        bisect_range=50.0,
        eps=1e-6,
        log_scale_clamp=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.size == 0:
            raise ValueError("mask must be non-empty")
        self._mask_np = mask
        self._d = int(mask.size)
        self._a_idx = np.where(mask)[0].astype(np.int32)
        self._b_idx = np.where(~mask)[0].astype(np.int32)
        self.d_a = int(self._a_idx.size)
        self.d_b = int(self._b_idx.size)
        if self.d_a == 0 or self.d_b == 0:
            raise ValueError(
                f"mask must contain both True and False entries; got "
                f"d_a={self.d_a}, d_b={self.d_b}"
            )
        self._perm = np.argsort(
            np.concatenate([self._a_idx, self._b_idx])
        ).astype(np.int32)

        self.conditioner = conditioner
        self.num_components = int(num_components)
        self.bisect_steps = int(bisect_steps)
        self.bisect_range = float(bisect_range)
        self.eps = float(eps)
        self.log_scale_clamp = (
            log_scale_clamp if log_scale_clamp is not None else tanh_log_scale_clamp()
        )

    def build(self, input_shape):
        d = int(input_shape[-1])
        if d != self._d:
            raise ValueError(
                f"MixtureCDFCoupling: mask length {self._d} != input dim {d}"
            )
        if not self.conditioner.built:
            self.conditioner.build((None, self.d_a))
        # Probe: verify the conditioner outputs 3 * d_b * K floats.
        probe = ops.zeros((1, self.d_a))
        try:
            out = self.conditioner(probe, training=False)
        except TypeError:
            out = self.conditioner(probe)
        expected = 3 * self.d_b * self.num_components
        got = int(out.shape[-1])
        if got != expected:
            raise ValueError(
                f"MixtureCDFCoupling: conditioner output width {got} != "
                f"3 * d_b * K = 3 * {self.d_b} * {self.num_components} = {expected}. "
                f"Use make_mlp_conditioner(d_b={self.d_b}, "
                f"num_components={self.num_components}) or "
                f"make_shared_mlp_conditioner(d_b={self.d_b}, "
                f"num_components={self.num_components})."
            )
        super().build(input_shape)

    def _params_from_conditioner(self, x_a):
        flat = self.conditioner(x_a)
        k = self.num_components
        params = ops.reshape(flat, (-1, 3, self.d_b, k))
        logits = params[:, 0]
        means = params[:, 1]
        log_scales = self.log_scale_clamp(params[:, 2])
        return MixtureOfGaussians(
            logits=logits, means=means, log_scales=log_scales
        )

    def _split(self, x):
        x_a = ops.take(x, self._a_idx, axis=-1)
        x_b = ops.take(x, self._b_idx, axis=-1)
        return x_a, x_b

    def _join(self, y_a, y_b):
        cat = ops.concatenate([y_a, y_b], axis=-1)
        return ops.take(cat, self._perm, axis=-1)

    def forward_and_log_det(self, x):
        self._ensure_built(x)
        x_a, x_b = self._split(x)
        mog = self._params_from_conditioner(x_a)
        u = ops.clip(mog.cdf(x_b), self.eps, 1.0 - self.eps)
        z_b = norm_icdf(u)
        ldj = ops.sum(mog.log_pdf(x_b) - norm_log_pdf(z_b), axis=-1)
        return self._join(x_a, z_b), ldj

    def inverse_and_log_det(self, z):
        self._ensure_built(z)
        z_a, z_b = self._split(z)
        mog = self._params_from_conditioner(z_a)
        u = ops.clip(norm_cdf(z_b), self.eps, 1.0 - self.eps)
        lo = ops.ones_like(z_b) * (-self.bisect_range)
        hi = ops.ones_like(z_b) * self.bisect_range
        for _ in range(self.bisect_steps):
            mid = 0.5 * (lo + hi)
            go_right = mog.cdf(mid) < u
            lo = ops.where(go_right, mid, lo)
            hi = ops.where(go_right, hi, mid)
        x_b = 0.5 * (lo + hi)
        ldj = ops.sum(norm_log_pdf(z_b) - mog.log_pdf(x_b), axis=-1)
        return self._join(z_a, x_b), ldj
