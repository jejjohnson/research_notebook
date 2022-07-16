from typing import Tuple
import jax.numpy as jnp
import treex as tx
import tensorflow_probability.substrates.jax as tfp
from distrax._src.bijectors.sigmoid import Sigmoid
from distrax._src.distributions.normal import Normal

tfd = tfp.distributions


class InverseGaussCDF(tx.Module):
    eps: float

    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        return self.forward(x)

    def forward(self, x):

        dist = tfd.Normal(loc=0, scale=1)

        x = jnp.clip(x, self.eps, 1 - self.eps)

        return dist.quantile(x)

    def inverse(self, x):

        dist = tfd.Normal(loc=0, scale=1)

        return dist.cdf(x)

    def forward_and_log_det(self, x):
        dist = tfd.Normal(loc=0, scale=1)
        x = jnp.clip(x, self.eps, 1 - self.eps)
        # x = _clamp_preserve_gradients(x, self.eps, 1 - self.eps)

        # forward transform
        z = dist.quantile(x)

        # ldj
        ldj = -dist.log_prob(z)

        return z, ldj

    def inverse_and_log_det(self, x):

        raise NotImplementedError()


class Logit(tx.Module):
    eps: float

    def __init__(self, eps: float = 1e-5):

        self.eps = eps
        # self.base_dist = tfd.Normal(loc=0, scale=1)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.clip(x, self.eps, 1 - self.eps)

        return Sigmoid().inverse(x)

    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        return Sigmoid().forward(x)

    def forward_and_log_det(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return Sigmoid().inverse_and_log_det(x)

    def inverse_and_log_det(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return Sigmoid().forward_and_log_det(x)
