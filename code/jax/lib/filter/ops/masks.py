import math
import jax.numpy as jnp
from einops import repeat
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
INV2PI = (2 * math.pi) ** -1


def mask_observation_operator(operator, mask):
    """
    Parameters
    ----------
    operator : jnp.ndarray, shape=(obs_dim, state_dim)
        the linear operator
    mask : jnp.ndarray, shape=(obs_dim,)
        the mask where the observed values are zeros

    Returns
    -------
    masked_operator : jnp.ndarray, shape=(obs_dim, state_dim)
        the same operator with zeros where the mask is located.
    """

    # to enable broadcasting
    mask = repeat(mask, "... -> ... 1")

    # mask all unobserved dims
    operator = jnp.where(mask == 1.0, 0.0, operator)

    return operator


def mask_observation_noise(noise, mask):
    """
    Parameters
    ----------
    noise : array, shape=(obs_dim, obs_dim)
        the linear operator
    mask : array, shape=(obs_dim)
        the mask where the observed values are zeros

    Returns
    -------
    noise : array, shape=(obs_dim, obs_dim)
        the same operator with zeros where the mask is located.
    """
    n_dims = mask.shape[0]

    mask = 1.0 - mask

    # create cov for mask
    maskv = repeat(mask, "... -> ... 1")
    mask_cov = 0.5 * (maskv + maskv.T)

    # mask all non-covariate entries
    noise = jnp.where(mask_cov == 1.0, noise, 0.0)

    # create base identity matrix
    identities = jnp.eye(n_dims, n_dims)
    # print(identities)

    # remove identity entries for masked values
    identities = jnp.where(jnp.diag(mask), 0, identities)
    # print(jnp.diag(mask))
    # print(identities)

    # replace the remaining identity values
    noise = jnp.where(identities, 1.0, noise)

    return noise


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

    Returns
    -------
    score : np.ndarray, shape=()
        the log likelihood for the outputs
    """
    if mask is not None:
        x, mean, cov = mask_mean_cov(x, mean, cov, mask)

    return tfd.MultivariateNormalFullCovariance(mean, cov).log_prob(x)


def mask_mean_cov(x, mean, cov, mask):
    """
    mask the observations, mean and cov such that it will
    be negligable with the negative log-likelihood.

    Parameters
    ----------
    x : np.ndarray, shape=(n_features)
    mean : np.ndarray, shape=(n_features)
    cov : np.ndarray, shape=(n_features, n_features)
    mask : np.ndarray (Optional), shape=(n_features)

    Returns
    -------
    x : np.ndarray, shape=(n_features)
    mean : np.ndarray, shape=(n_features)
    cov : np.ndarray, shape=(n_features, n_features)
    """

    # build a mask for computing the log likelihood of a partially observed multivariate Gaussian
    x = jnp.where(mask, 0.0, x)
    mean = jnp.where(mask, 0.0, mean)

    # covariance masking
    maskv = mask.reshape(-1, 1)
    cov_masked = jnp.where(
        maskv + maskv.T, 0.0, cov
    )  # ensure masked entries are independent
    cov = jnp.where(
        jnp.diag(mask), INV2PI, cov_masked
    )  # ensure masked entries return log like of 0
    # print(x)
    # print(mean)
    # print(cov)
    return x, mean, cov
