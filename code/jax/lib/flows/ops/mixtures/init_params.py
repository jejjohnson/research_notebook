from sklearn.mixture import GaussianMixture
import numpy as np
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp


def init_GMM_marginal(
    X: np.ndarray, n_components: int, covariance_type: str = "diag", **kwargs
):
    """Initialize means with K-Means

    Parameters
    ----------
    X : np.ndarray
        (n_samples, n_features)
    n_components : int
        the number of clusters for the K-Means

    Returns
    -------
    clusters : np.ndarray
        (n_features, n_components)"""

    weights, means, covariances = [], [], []

    for iX in X.T:
        clf = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            **kwargs,
        ).fit(iX[:, None])
        weights.append(clf.weights_)
        means.append(clf.means_.T)
        covariances.append(clf.covariances_.T)

    weights = np.vstack(weights)
    means = np.vstack(means)
    covariances = np.vstack(covariances)

    # do inverse param transformations
    log_scales = tfp.math.softplus_inverse(jnp.sqrt(covariances))
    prior_logits = jnp.log(weights)

    return prior_logits, means, log_scales
