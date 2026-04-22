"""Per-dimension mixture of Gaussians used inside the marginal layer.

The Gaussianization flow's marginal-CDF step needs, for each input
dimension, a flexible monotone CDF. A mixture of ``K`` Gaussians with
trainable weights, means, and scales is the standard choice
(Meng, Song & Ermon 2020). This module provides the CDF / pdf / log-pdf
primitives so that both the layer (PR B) and the test suite can compute
the mixture quantities from explicit tensor parameters.

Parameter convention: for ``d`` input dimensions and ``K`` mixture
components, all parameters are shape ``(d, K)``. The mixture operates
independently per dimension.
"""

from __future__ import annotations

from keras import ops

from gaussianization.gauss_keras._math import norm_cdf, norm_log_pdf


class MixtureOfGaussians:
    """Per-dim mixture of ``K`` Gaussians, vectorised over ``d`` dims.

    Args:
        logits:     mixture-weight logits, shape ``(d, K)``.
        means:      component means, shape ``(d, K)``.
        log_scales: component log-scales, shape ``(d, K)``; the scale
                    is ``σ = exp(log_scales)`` so it is positive by
                    construction.

    Inputs to ``cdf`` / ``pdf`` / ``log_pdf`` have shape ``(batch, d)``
    and outputs have shape ``(batch, d)``.
    """

    def __init__(self, logits, means, log_scales):
        self.logits = logits
        self.means = means
        self.log_scales = log_scales

    def _broadcast(self, x):
        # (batch, d) -> (batch, d, 1) so it broadcasts against (d, K).
        return ops.expand_dims(x, axis=-1)

    def _log_weights(self):
        # (d, K) — numerically stable mixture-weight log-probs.
        return ops.log_softmax(self.logits, axis=-1)

    def cdf(self, x):
        """Per-dim mixture CDF ``F(x) = Σ_k π_k Φ((x - μ_k)/σ_k)``."""
        weights = ops.softmax(self.logits, axis=-1)
        scales = ops.exp(self.log_scales)
        z = (self._broadcast(x) - self.means) / scales
        comp_cdf = norm_cdf(z)
        return ops.sum(weights * comp_cdf, axis=-1)

    def pdf(self, x):
        """Per-dim mixture pdf (derivative of :meth:`cdf`)."""
        return ops.exp(self.log_pdf(x))

    def log_pdf(self, x):
        """Numerically stable per-dim mixture log-pdf."""
        log_w = self._log_weights()
        scales = ops.exp(self.log_scales)
        z = (self._broadcast(x) - self.means) / scales
        log_comp = norm_log_pdf(z) - self.log_scales
        return ops.logsumexp(log_w + log_comp, axis=-1)
