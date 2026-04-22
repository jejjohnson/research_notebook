"""Bijector abstract layer.

Each concrete bijector implements the two explicit methods

    forward_and_log_det(x)  -> (z, log_det_jacobian)
    inverse_and_log_det(z)  -> (x, log_det_jacobian)

and inherits ``call(x, inverse=False)`` which dispatches to one of them
and contributes ``-mean(log_det_jacobian)`` to the layer loss via
``self.add_loss``. Only the forward direction adds to the loss — inverse
is used at sampling time and must not accumulate a training signal.

Per-sample log-densities needed by ``GaussianizationFlow.log_prob`` call
``forward_and_log_det`` / ``inverse_and_log_det`` directly and thread
the log-det sum explicitly; this keeps the density-evaluation path
independent of the ``add_loss`` side-channel.
"""

from __future__ import annotations

import keras
from keras import ops


class Bijector(keras.layers.Layer):
    """Abstract bijective layer.

    Concrete subclasses must override ``forward_and_log_det`` and
    ``inverse_and_log_det``. Both return a ``(y, ldj)`` tuple where
    ``ldj`` is the per-sample log |det J| of the transform actually
    applied (forward log-det for forward, inverse log-det for inverse).
    """

    def _ensure_built(self, x):
        """Build the layer on first use if ``call`` has not yet run."""
        if not self.built:
            self.build(x.shape)

    def forward_and_log_det(self, x):
        """Forward map ``z = f(x)`` plus per-sample log |det J_f(x)|.

        Subclasses must call ``self._ensure_built(x)`` at the top of
        their override so direct use from ``GaussianizationFlow.log_prob``
        (which bypasses ``__call__``) still triggers weight creation.
        """
        raise NotImplementedError

    def inverse_and_log_det(self, z):
        """Inverse map ``x = f⁻¹(z)`` plus per-sample log |det J_{f⁻¹}(z)|."""
        raise NotImplementedError

    def call(self, x, inverse=False):
        """Dispatch to forward/inverse and add -mean(ldj) on the forward pass."""
        if inverse:
            y, _ldj = self.inverse_and_log_det(x)
            return y
        y, ldj = self.forward_and_log_det(x)
        self.add_loss(-ops.mean(ldj))
        return y
