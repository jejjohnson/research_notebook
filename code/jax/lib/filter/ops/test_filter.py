import pytest
import jax
import jax.numpy as jnp
import numpy as np
from einops import repeat

from filterjax._src.ops.filter import (
    filter_step_sequential,
    forward_filter,
    forward_filter_samples,
    kalman_step,
    predict_step_obs,
    predict_step_state,
)

JITTER = 1e-5


def test_predict_step_state_shape():

    state_dim = 10

    F = np.random.randn(state_dim, state_dim)
    Q = np.random.randn(state_dim, state_dim)

    mu = np.random.randn(state_dim)
    Sigma = np.random.randn(state_dim, state_dim)

    mu_, Sigma_ = predict_step_state(mu, Sigma, F, Q)

    assert mu_.shape == mu.shape
    assert Sigma_.shape == Sigma.shape


def test_predict_step_state_shape_vectorized():

    state_dim = 10
    n_samples = 10

    F = np.random.randn(state_dim, state_dim)
    Q = np.random.randn(state_dim, state_dim)

    mu = np.random.randn(n_samples, state_dim)
    Sigma = np.random.randn(n_samples, state_dim, state_dim)

    mu_, Sigma_ = jax.vmap(predict_step_state, in_axes=(0, 0, None, None))(
        mu, Sigma, F, Q
    )

    assert mu_.shape == mu.shape
    assert Sigma_.shape == Sigma.shape


def test_predict_step_state_shape_vectorized_continuous():

    state_dim = 10
    n_samples = 10

    F = np.random.randn(n_samples, state_dim, state_dim)
    Q = np.random.randn(n_samples, state_dim, state_dim)

    mu = np.random.randn(n_samples, state_dim)
    Sigma = np.random.randn(n_samples, state_dim, state_dim)

    mu_, Sigma_ = jax.vmap(predict_step_state, in_axes=(0, 0, 0, 0))(mu, Sigma, F, Q)

    assert mu_.shape == mu.shape
    assert Sigma_.shape == Sigma.shape


def test_predict_step_state_shape_vectorized_noises():

    state_dim = 10
    n_samples = 10

    F = np.random.randn(state_dim, state_dim)
    Q = np.random.randn(n_samples, state_dim, state_dim)

    mu = np.random.randn(n_samples, state_dim)
    Sigma = np.random.randn(n_samples, state_dim, state_dim)

    mu_, Sigma_ = jax.vmap(predict_step_state, in_axes=(0, 0, None, 0))(mu, Sigma, F, Q)

    assert mu_.shape == mu.shape
    assert Sigma_.shape == Sigma.shape


def test_predict_step_obs_shape():

    obs_dim = 5
    state_dim = 10

    H = np.random.randn(obs_dim, state_dim)
    R = np.random.randn(obs_dim, obs_dim)

    mu = np.random.randn(state_dim)
    Sigma = np.random.randn(state_dim, state_dim)

    mu_, Sigma_ = predict_step_obs(mu, Sigma, H, R)

    assert mu_.shape == (obs_dim,)
    assert Sigma_.shape == (obs_dim, obs_dim)


def test_predict_step_obs_shape_vectorized():

    state_dim = 10
    obs_dim = 5
    n_samples = 10

    H = np.random.randn(obs_dim, state_dim)
    R = np.random.randn(obs_dim, obs_dim)

    mu = np.random.randn(n_samples, state_dim)
    Sigma = np.random.randn(n_samples, state_dim, state_dim)

    mu_, Sigma_ = jax.vmap(predict_step_state, in_axes=(0, 0, None, None))(
        mu, Sigma, H, R
    )

    assert mu_.shape == (n_samples, obs_dim)
    assert Sigma_.shape == (n_samples, obs_dim, obs_dim)


def test_predict_step_obs_shape_vectorized_continuous():

    state_dim = 10
    obs_dim = 5
    n_samples = 10

    H = np.random.randn(n_samples, obs_dim, state_dim)
    R = np.random.randn(n_samples, obs_dim, obs_dim)

    mu = np.random.randn(n_samples, state_dim)
    Sigma = np.random.randn(n_samples, state_dim, state_dim)

    mu_, Sigma_ = jax.vmap(predict_step_state, in_axes=(0, 0, 0, 0))(mu, Sigma, H, R)

    assert mu_.shape == (n_samples, obs_dim)
    assert Sigma_.shape == (n_samples, obs_dim, obs_dim)


def test_predict_step_obs_shape_vectorized_noises():

    state_dim = 10
    obs_dim = 5
    n_samples = 10

    H = np.random.randn(obs_dim, state_dim)
    R = np.random.randn(n_samples, obs_dim, obs_dim)

    mu = np.random.randn(n_samples, state_dim)
    Sigma = np.random.randn(n_samples, state_dim, state_dim)

    mu_, Sigma_ = jax.vmap(predict_step_state, in_axes=(0, 0, None, 0))(mu, Sigma, H, R)

    assert mu_.shape == (n_samples, obs_dim)
    assert Sigma_.shape == (n_samples, obs_dim, obs_dim)


@pytest.mark.parametrize("return_likelihood", [False, True])
@pytest.mark.parametrize("mask", [None, True])
def test_kalman_step(return_likelihood, mask):

    state_dim = 10
    obs_dim = 5

    # observations
    obs = np.random.randn(obs_dim)

    # masked observations
    if mask:
        mask = np.zeros_like(obs)
        mask[::2] = 1.0

    # state
    mu = np.zeros(state_dim)
    Sigma = JITTER * np.eye(state_dim)

    # transition
    F = np.random.randn(state_dim, state_dim)
    Q = JITTER * np.eye(state_dim)

    # emission
    H = np.random.randn(obs_dim, state_dim)
    R = JITTER * np.eye(obs_dim)

    results = kalman_step(
        obs=obs,
        mu=mu,
        Sigma=Sigma,
        F=F,
        Q=Q,
        H=H,
        R=R,
        return_likelihood=return_likelihood,
        mask=mask,
    )

    assert results[0].shape == (state_dim,)
    assert results[1].shape == (state_dim, state_dim)
    assert results[2].shape == (state_dim,)
    assert results[3].shape == (state_dim, state_dim)
    if return_likelihood:
        assert results[4].shape == ()


@pytest.mark.parametrize("return_predict", [False, True])
@pytest.mark.parametrize("masks", [None, True])
def test_filter_step_sequential(return_predict, masks):

    n_time = 20
    state_dim = 10
    obs_dim = 5

    # observations
    obs = np.random.randn(n_time, obs_dim)

    # masked observations
    if masks:
        masks = np.zeros_like(obs)
        masks[:, ::2] = 1.0

    # initial state
    mu0 = np.zeros(state_dim)
    Sigma0 = JITTER * np.eye(state_dim)

    # transition
    F = np.random.randn(state_dim, state_dim)
    Q = JITTER * np.eye(state_dim)

    # emission
    H = np.random.randn(obs_dim, state_dim)
    R = JITTER * np.eye(obs_dim)

    results = filter_step_sequential(
        obs=obs,
        mu0=mu0,
        Sigma0=Sigma0,
        F=F,
        Q=Q,
        H=H,
        R=R,
        return_predict=return_predict,
        masks=masks,
    )

    assert results[0].shape == (n_time, state_dim)
    assert results[1].shape == (n_time, state_dim, state_dim)
    assert results[2].shape == (n_time,)


@pytest.mark.parametrize("return_predict", [False, True])
@pytest.mark.parametrize("masks", [None, True])
def test_forward_filter(return_predict, masks):

    n_samples = 30
    n_time = 20
    state_dim = 10
    obs_dim = 5

    # observations
    obs = np.random.randn(n_samples, n_time, obs_dim)

    # masked observations
    if masks:
        masks = np.zeros_like(obs)
        masks[:, ::2] = 1.0

    # initial state
    mu0 = np.zeros(state_dim)
    Sigma0 = JITTER * np.eye(state_dim)

    # transition
    F = np.random.randn(state_dim, state_dim)
    Q = JITTER * np.eye(state_dim)

    # emission
    H = np.random.randn(obs_dim, state_dim)
    R = JITTER * np.eye(obs_dim)

    results = forward_filter(
        obs=obs,
        mu0=mu0,
        Sigma0=Sigma0,
        F=F,
        Q=Q,
        H=H,
        R=R,
        return_predict=return_predict,
        masks=masks,
    )

    assert results[0].shape == (n_samples, n_time, state_dim)
    assert results[1].shape == (n_samples, n_time, state_dim, state_dim)
    assert results[2].shape == (
        n_samples,
        n_time,
    )


@pytest.mark.parametrize("return_predict", [False, True])
@pytest.mark.parametrize("masks", [None, True])
def test_forward_filter_samples(return_predict, masks):

    n_samples = 30
    n_time = 20
    state_dim = 10
    obs_dim = 5

    # observations
    obs = np.random.randn(n_samples, n_time, obs_dim)

    # masked observations
    if masks:
        masks = np.zeros_like(obs)
        masks[:, ::2] = 1.0

    # initial state
    mu0s = repeat(np.zeros(state_dim), "... -> B ...", B=n_samples)
    Sigma0s = repeat(JITTER * np.eye(state_dim), "... -> B ...", B=n_samples)

    # transition
    F = np.random.randn(state_dim, state_dim)
    Qs = repeat(JITTER * np.eye(state_dim), "... -> B ...", B=n_samples)

    # emission
    H = np.random.randn(obs_dim, state_dim)
    Rs = repeat(JITTER * np.eye(obs_dim), "... -> B ...", B=n_samples)

    results = forward_filter_samples(
        obs=obs,
        mu0s=mu0s,
        Sigma0s=Sigma0s,
        F=F,
        Qs=Qs,
        H=H,
        Rs=Rs,
        return_predict=return_predict,
        masks=masks,
    )

    assert results[0].shape == (n_samples, n_time, state_dim)
    assert results[1].shape == (n_samples, n_time, state_dim, state_dim)
    assert results[2].shape == (
        n_samples,
        n_time,
    )
