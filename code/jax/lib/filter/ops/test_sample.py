import pytest
import numpy as np
import jax
import jax.numpy as jnp

from filterjax._src.ops.sample import (
    sample_sequential,
    sample_step,
    sample_step_sequential,
)

JITTER = 1e-5


@pytest.mark.parametrize("jitted", [(False), (True,)])
def test_sample_step(jitted):

    state_dim = 10
    obs_dim = 5

    # initial condition
    mu = np.zeros(state_dim)

    # transition
    F = np.random.randn(state_dim, state_dim)
    H = np.random.randn(obs_dim, state_dim)

    # noise epsilon
    state_noise = JITTER * np.ones(state_dim)
    obs_noise = JITTER * np.ones(obs_dim)
    if jitted:
        mu_t, obs_t = jax.jit(sample_step)(
            mu=mu, F=F, state_noise=state_noise, H=H, obs_noise=obs_noise
        )
    else:
        mu_t, obs_t = sample_step(
            mu=mu, F=F, state_noise=state_noise, H=H, obs_noise=obs_noise
        )

    assert mu_t.shape == (state_dim,)
    assert obs_t.shape == (obs_dim,)


@pytest.mark.parametrize("jitted", [(False), (True,)])
def test_sample_sequential(jitted):

    n_time = 20
    state_dim = 10
    obs_dim = 5

    # initial condition
    mu = np.zeros(state_dim)

    # transition
    F = np.random.randn(state_dim, state_dim)
    H = np.random.randn(obs_dim, state_dim)

    # noise epsilon
    state_noises = JITTER * np.ones((n_time, state_dim))
    obs_noises = JITTER * np.ones((n_time, obs_dim))

    if jitted:
        mu_t, obs_t = jax.jit(sample_step_sequential)(
            mu=mu, F=F, state_noises=state_noises, H=H, obs_noises=obs_noises
        )
    else:
        mu_t, obs_t = sample_step_sequential(
            mu=mu, F=F, state_noises=state_noises, H=H, obs_noises=obs_noises
        )

    assert mu_t.shape == (
        n_time,
        state_dim,
    )
    assert obs_t.shape == (
        n_time,
        obs_dim,
    )


@pytest.mark.parametrize("batch_prior", [False, True])
@pytest.mark.parametrize("jitted", [False, True])
def test_sample_sequential_vectorized(batch_prior, jitted):

    n_samples = 30
    n_time = 20
    state_dim = 10
    obs_dim = 5

    # initial condition
    if batch_prior:
        mu = np.zeros((n_samples, state_dim))
    else:
        mu = np.zeros((state_dim,))

    # transition
    F = np.random.randn(state_dim, state_dim)
    H = np.random.randn(obs_dim, state_dim)

    # noise epsilon
    state_noises = JITTER * np.ones((n_samples, n_time, state_dim))
    obs_noises = JITTER * np.ones((n_samples, n_time, obs_dim))

    if jitted:
        mu_t, obs_t = jax.jit(sample_sequential)(
            mu=mu, F=F, state_noises=state_noises, H=H, obs_noises=obs_noises
        )
    else:
        mu_t, obs_t = sample_sequential(
            mu=mu, F=F, state_noises=state_noises, H=H, obs_noises=obs_noises
        )

    assert mu_t.shape == (n_samples, n_time, state_dim)
    assert obs_t.shape == (n_samples, n_time, obs_dim)
