import pytest
import jax
import jax.numpy as jnp
import numpy as np
from filterjax._src.losses import mvn_logpdf

JITTER = 1e-5


@pytest.mark.parametrize("use_mask", [True, False])
def test_mvn_logpdf_univariate(use_mask):

    n_features = 100
    x = np.random.randn(n_features)
    if use_mask:
        mask = np.random.randint(0, 1, n_features)
    else:
        mask = None
    mean = np.random.randn(n_features)
    cov = np.random.randn(n_features, n_features)

    loss = mvn_logpdf(x, mean, cov, mask=mask)

    assert loss.shape == ()


# def test_mvn_logpdf_univariate_masked_zero():

#     n_features = 100
#     x = np.random.randn(n_features)
#     mask = np.zeros_like(x)

#     mean = np.zeros(n_features)
#     cov = JITTER * np.eye(n_features)

#     loss = mvn_logpdf(x, mean, cov, mask)

#     assert loss.sum() == 0.0


@pytest.mark.parametrize("use_mask", [True, False])
def test_mvn_logpdf_multivariate(use_mask):

    n_features = 100
    n_samples = 10
    x = np.random.randn(n_samples, n_features)
    if use_mask:
        mask = np.random.randint(0, 1, (n_samples, n_features))
    else:
        mask = None
    mean = np.random.randn(n_samples, n_features)
    cov = np.random.randn(n_samples, n_features, n_features)

    if use_mask:
        loss = jax.vmap(mvn_logpdf, in_axes=(0, 0, 0, 0))(x, mean, cov, mask)
    else:
        loss = jax.vmap(mvn_logpdf, in_axes=(0, 0, 0))(x, mean, cov)

    assert loss.shape == (n_samples,)
