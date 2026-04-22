"""Per-dimension marginal Gaussianization via a mixture-of-Gaussians CDF.

For each input dimension ``i`` with trainable mixture parameters, the
forward map is

    u_i = F_i(x_i),     z_i = Φ⁻¹(u_i)

and the inverse is obtained by monotone bisection on ``F_i(x_i) = Φ(z_i)``.
The per-sample log-det Jacobian sums

    log |dz_i / dx_i| = log f_i(x_i) - log φ(z_i)

across dimensions, where ``f_i = F_i'`` is the mixture pdf.
"""

from __future__ import annotations

import math

import keras
from keras import ops

from gaussianization.gauss_keras._math import norm_cdf, norm_icdf, norm_log_pdf
from gaussianization.gauss_keras.bijectors.base import Bijector
from gaussianization.gauss_keras.mixtures import MixtureOfGaussians


class MixtureCDFGaussianization(Bijector):
    """Per-dim mixture-CDF Gaussianization layer.

    Args:
        num_components: number of Gaussian mixture components per dim.
        bisect_steps: bisection iterations for the inverse; 40 gives
            ``(hi - lo) / 2^{40} ≈ 1e-10`` relative precision.
        bisect_range: half-width of the symmetric initial bracket used
            for the inverse; inputs outside ``[-bisect_range, bisect_range]``
            will be clipped.
        eps: clamp applied to ``F(x) ∈ [eps, 1 - eps]`` before ``Φ⁻¹`` to
            keep the quantile bounded.
        init_mean_range: uniform range for initialising component means.
        init_log_scale: constant initialiser for all component log-scales.
    """

    def __init__(
        self,
        num_components,
        bisect_steps=40,
        bisect_range=50.0,
        eps=1e-6,
        init_mean_range=3.0,
        init_log_scale=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_components = int(num_components)
        self.bisect_steps = int(bisect_steps)
        self.bisect_range = float(bisect_range)
        self.eps = float(eps)
        self.init_mean_range = float(init_mean_range)
        self.init_log_scale = float(init_log_scale)

    def build(self, input_shape):
        d = int(input_shape[-1])
        k = self.num_components
        self._d = d
        self.logits = self.add_weight(
            name="logits",
            shape=(d, k),
            initializer="zeros",
            trainable=True,
        )
        self.means = self.add_weight(
            name="means",
            shape=(d, k),
            initializer=keras.initializers.RandomUniform(
                -self.init_mean_range, self.init_mean_range
            ),
            trainable=True,
        )
        self.log_scales = self.add_weight(
            name="log_scales",
            shape=(d, k),
            initializer=keras.initializers.Constant(self.init_log_scale),
            trainable=True,
        )
        super().build(input_shape)

    def _mog(self):
        return MixtureOfGaussians(
            logits=self.logits, means=self.means, log_scales=self.log_scales
        )

    def _cdf(self, x):
        return self._mog().cdf(x)

    def _log_pdf(self, x):
        return self._mog().log_pdf(x)

    def forward_and_log_det(self, x):
        self._ensure_built(x)
        u = self._cdf(x)
        u = ops.clip(u, self.eps, 1.0 - self.eps)
        z = norm_icdf(u)
        ldj = ops.sum(self._log_pdf(x) - norm_log_pdf(z), axis=-1)
        return z, ldj

    def inverse_and_log_det(self, z):
        self._ensure_built(z)
        u = ops.clip(norm_cdf(z), self.eps, 1.0 - self.eps)
        lo = ops.ones_like(z) * (-self.bisect_range)
        hi = ops.ones_like(z) * self.bisect_range
        for _ in range(self.bisect_steps):
            mid = 0.5 * (lo + hi)
            go_right = self._cdf(mid) < u
            lo = ops.where(go_right, mid, lo)
            hi = ops.where(go_right, hi, mid)
        x = 0.5 * (lo + hi)
        ldj = ops.sum(norm_log_pdf(z) - self._log_pdf(x), axis=-1)
        return x, ldj

    def adapt_means_from_quantiles(self, x):
        """Data-dependent initialisation of component means.

        Places component means at evenly-spaced quantiles of ``x`` per
        dimension so the first forward pass does not collapse to a
        single component. ``x`` is a ``(n, d)`` numpy-like array.
        """
        import numpy as np

        x = np.asarray(x)
        d = x.shape[-1]
        k = self.num_components
        if not self.built:
            self.build(input_shape=(None, d))
        qs = np.linspace(0.5 / k, 1.0 - 0.5 / k, k)
        means = np.stack(
            [np.quantile(x[:, i], qs) for i in range(d)], axis=0
        ).astype("float32")
        self.means.assign(means)
        # Set log-scales to a reasonable bandwidth based on inter-quantile spacing.
        spacing = float(np.mean(np.diff(qs) * (x.std(axis=0).mean())))
        self.log_scales.assign(
            self.log_scales.numpy() * 0 + math.log(max(spacing, 0.1))
        )
