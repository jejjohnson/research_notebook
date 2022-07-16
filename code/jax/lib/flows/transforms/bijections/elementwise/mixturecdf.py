import treex as tx
import jax.numpy as jnp
from flowjax._src.ops.mixtures.init_params import init_GMM_marginal
from flowjax._src.ops.mixtures.gaussian import gaussian_mixture_transform
from flowjax._src.ops.mixtures.logistic import logistic_mixture_transform
import numpy as np


class GaussianMixtureCDF(tx.Module):
    logit_weights: jnp.ndarray = tx.Parameter.node()
    means: jnp.ndarray = tx.Parameter.node()
    log_scales: jnp.ndarray = tx.Parameter.node()
    num_mixtures: int
    eps: float
    max_iters: int

    def __init__(self, num_mixtures: int = 5, eps: float = 1e-5, max_iters: int = 100):

        self.num_mixtures = num_mixtures
        self.eps = eps
        self.max_iters = max_iters

    def __call__(self, x):
        if self.initializing():
            # self.logit_weights = jnp.log(jnp.ones(x.shape[0], self.num_mixtures) / self.num_mixtures)
            # self.means = jnp.ones(self.num_features, self.num_mixtures)
            # self.log_scales = jnp.log(0.1 * jnp.ones((self.num_features, self.num_mixtures)))
            # data-dependent initialization
            logit_weights, means, log_scales = init_GMM_marginal(
                np.asarray(x),
                n_components=self.num_mixtures,
            )
            self.logit_weights = jnp.array(logit_weights)
            self.means = jnp.array(means)
            self.log_scales = jnp.array(log_scales)

        return self.forward(x)

    def forward(self, x):

        z, _ = self.forward_and_log_det(x)

        return z

    def inverse(self, x):

        z, _ = self.inverse_and_log_det(x)

        return z

    def forward_and_log_det(self, x):
        z, ldj = gaussian_mixture_transform(
            x,
            logit_weights=self.logit_weights,
            means=self.means,
            log_scales=self.log_scales,
            eps=self.eps,
            max_iters=self.max_iters,
            inverse=False,
        )
        return z, ldj

    def inverse_and_log_det(self, x):
        z, ldj = gaussian_mixture_transform(
            x,
            logit_weights=self.logit_weights,
            means=self.means,
            log_scales=self.log_scales,
            eps=self.eps,
            max_iters=self.max_iters,
            inverse=True,
        )
        return z, ldj


class LogisticMixtureCDF(GaussianMixtureCDF):
    def forward_and_log_det(self, x):
        z, ldj = logistic_mixture_transform(
            x,
            logit_weights=self.logit_weights,
            means=self.means,
            log_scales=self.log_scales,
            eps=self.eps,
            max_iters=self.max_iters,
            inverse=False,
        )
        return z, ldj

    def inverse_and_log_det(self, x):
        z, ldj = logistic_mixture_transform(
            x,
            logit_weights=self.logit_weights,
            means=self.means,
            log_scales=self.log_scales,
            eps=self.eps,
            max_iters=self.max_iters,
            inverse=True,
        )
        return z, ldj
