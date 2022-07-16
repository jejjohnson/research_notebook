import math
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, cho_factor, cho_solve
from filterjax._src.utils import sum_except_batch

LOG2PI = math.log(2 * math.pi)
INV2PI = (2 * math.pi) ** -1


def sample_likelihood(data, mu, std: float = 0.01):
    """
    Log-Likelihood of one sample.

    Parameters
    ----------
    data : np.ndarray, shape=(n_dims,)
    mu:
    """
    return -((data - mu) ** 2) / (2 * std**2) - jnp.log(std) - jnp.log(2 * jnp.pi) / 2


def likelihood(preds, data, mask):
    """
    Compute log-likelihood of data w/ mask under current predictions.
    """
    # get likelihood for samples
    ell = sample_likelihood(data[None] * mask, preds * mask)

    # sum all except batch
    ll = sum_except_batch(ell)

    # divide by masked values
    ll /= jnp.sum(mask)

    return ll


def masked_mean_squared_error(preds, data, mask):
    """
    Return mean squared error w/ mask.
    """
    mse = jnp.square(data[None] * mask - preds * mask)

    mse = sum_except_batch(mse)

    mse /= jnp.sum(mask)

    return mse


def kl_div(mean, std, mean_prior: float = 0.0, std_prior: float = 1.0):
    """
    Analytically compute KL between z0 distribution and prior.
    """

    var_ratio = (std / std_prior) ** 2
    t1 = ((mean - mean_prior) / std_prior) ** 2
    return jnp.mean(0.5 * (var_ratio + t1 - 1 - jnp.log(var_ratio)))


def get_kl_coef(epoch):
    """
    Tuning schedule for KL coefficient. (annealing)
    """
    return max(0.0, 1 - 0.99 ** (epoch - 10))


def mvn_logpdf(x, mean, cov, mask=None):
    """
    evaluate a multivariate Gaussian (log) pdf

    Parameters
    ----------
    x : np.ndarray, shape=(n_features)
        the input
    mean : np.ndarray, shape=(n_features)
    cov : np.ndarray, shape=(n_features, n_features)
    mask : np.ndarray (Optional), shape=(n_features)
    """
    x = x.reshape(-1, 1)
    mean = mean.reshape(-1, 1)
    if mask is not None:
        # build a mask for computing the log likelihood of a partially observed multivariate Gaussian
        maskv = mask.reshape(-1, 1)
        x = jnp.where(maskv, 0.0, x)
        mean = jnp.where(maskv, 0.0, mean)
        cov_masked = jnp.where(
            maskv + maskv.T, 0.0, cov
        )  # ensure masked entries are independent
        cov = jnp.where(
            jnp.diag(mask), INV2PI, cov_masked
        )  # ensure masked entries return log like of 0

    n = mean.shape[0]
    cho, low = cho_factor(cov, lower=True)
    log_det = 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(cho))))
    diff = x - mean
    scaled_diff = cho_solve((cho, low), diff)
    distance = diff.T @ scaled_diff
    return jnp.squeeze(-0.5 * (distance + n * LOG2PI + log_det))
