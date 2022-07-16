from typing import List, Tuple
import treex as tx
import jax
import jax.numpy as jnp
from flowjax._src.utils.types import Array


class Transform(tx.Module):
    """Base class for transform"""

    def forward_and_log_det(self, x):
        raise NotImplementedError()

    def inverse_and_log_det(self, x):
        raise NotImplementedError()

    def forward(self, x: Array) -> Array:
        """Computes f(x)"""
        z, _ = self.forward_and_log_det(x)

        return z

    def inverse(self, z: Array) -> Array:
        """Computes x = f^{-1}(z)"""
        x, _ = self.inverse_log_det_jacobian(z)

        return x

    def forward_log_det_jacobian(self, x):
        """Computes log|det J(f)(x)|"""

        _, logdetj = self.forward_and_log_det(x)

        return logdetj

    def inverse_log_det_jacobian(self, z):
        """Computes log|det J(f^{-1})(z)|"""

        _, logdetj = self.inverse_and_log_det(z)

        return logdetj

    @property
    def name(self) -> str:
        """Name of the bijector"""
        return self.__class__.__name__


class Composite(tx.Sequential):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        return self.forward(x)

    def forward(self, x):
        # loop through layers
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, x):
        # loop through layers
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def forward_and_log_det(self, x):

        # transform first layer
        x, ldj = self.layers[0].forward_and_log_det(x)

        # loop through remaining layers
        for ibijector in self.layers[1:]:
            x, ildj = ibijector.forward_and_log_det(x)
            ldj += ildj

        return x, ldj

    def inverse_and_log_det(self, x):

        # transform last layer
        x, ldj = self.layers[-1].inverse_and_log_det(x)

        # loop through remaining (reversed) layers
        for ibijector in reversed(self.layers[:-1]):
            x, ildj = ibijector.inverse_and_log_det(x)
            ldj += ildj

        return x, ldj
