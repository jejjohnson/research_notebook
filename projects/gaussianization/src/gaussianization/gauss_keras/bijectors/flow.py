"""GaussianizationFlow — stacked bijectors with density + sampling APIs."""

from __future__ import annotations

import keras
from keras import ops

from gaussianization.gauss_keras._math import norm_log_pdf


class GaussianizationFlow(keras.Model):
    """Composition of :class:`Bijector` layers with a standard normal base.

    Training path: ``call(x) -> z``. Each sub-bijector contributes
    ``-mean(log_det)`` via ``self.add_loss``; the main (non-loss) model
    output is the latent ``z``. A compatible training loss is
    ``-mean(sum(norm_log_pdf(z), axis=-1))`` (see
    :func:`gaussianization.gauss_keras.training.base_nll_loss`).

    Density-evaluation path: :meth:`log_prob` calls
    ``forward_and_log_det`` on each bijector directly, threading the
    per-sample ldj sum without touching ``add_loss``.

    Sampling path: :meth:`sample` draws ``z ~ N(0, I)`` and inverts the
    composition.
    """

    def __init__(self, bijectors, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.bijector_layers = list(bijectors)
        self._d = int(input_dim)
        # Force registration of children so Keras tracks their weights
        # even before the first call.
        for i, b in enumerate(self.bijector_layers):
            setattr(self, f"_bijector_{i}", b)

    @property
    def input_dim(self):
        return self._d

    def call(self, x):
        for b in self.bijector_layers:
            x = b(x)
        return x

    def invert(self, z):
        """Inverse pass ``x = f⁻¹(z)`` applied in reverse block order."""
        x = z
        for b in reversed(self.bijector_layers):
            x = b(x, inverse=True)
        return x

    def log_prob(self, x):
        """Per-sample ``log p_X(x)`` of shape ``(batch,)``."""
        ldj_total = ops.zeros(ops.shape(x)[0], dtype=x.dtype)
        y = x
        for b in self.bijector_layers:
            y, ldj = b.forward_and_log_det(y)
            ldj_total = ldj_total + ldj
        base = ops.sum(norm_log_pdf(y), axis=-1)
        return base + ldj_total

    def forward_with_intermediates(self, x):
        """Return ``[x, f_1(x), (f_2 ∘ f_1)(x), …]`` for visualisation."""
        states = [x]
        y = x
        for b in self.bijector_layers:
            y, _ = b.forward_and_log_det(y)
            states.append(y)
        return states

    def sample(self, num_samples, seed=None):
        """Draw ``num_samples`` points from the flow by inverting ``N(0, I)``."""
        seed_gen = keras.random.SeedGenerator(seed) if seed is not None else None
        z = keras.random.normal(
            shape=(int(num_samples), self._d), seed=seed_gen
        )
        return self.invert(z)
