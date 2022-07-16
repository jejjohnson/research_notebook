import pytest
import jax
import jax.numpy as jnp
import numpy as np
from einops import repeat
from filterjax._src.constants import JITTER
from filterjax._src.ops.filter import (
    filter_step_sequential,
    forward_filter,
    forward_filter_samples,
    kalman_step,
)
from filterjax._src.ops.smooth import (
    rauch_tung_striebel_smoother,
    rauch_tung_striebel_smoother_samples,
    smoother_step,
    smoother_step_sequential,
)


@pytest.mark.parametrize("return_likelihood", [False, True])
@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize("return_full", [False, True])
def test_smoother_step(return_likelihood, mask, return_full):

    state_dim = 10
    obs_dim = 5

    # observations
    obs = np.random.randn(obs_dim)

    # masked observations
    if mask:
        mask = np.zeros_like(obs)
        mask[::2] = 1.0
    else:
        mask = None

    # state
    mu = np.zeros(state_dim)
    Sigma = JITTER * np.eye(state_dim)

    # transition
    F = np.random.randn(state_dim, state_dim)
    Q = JITTER * np.eye(state_dim)

    # emission
    H = np.random.randn(obs_dim, state_dim)
    R = JITTER * np.eye(obs_dim)

    # Forward
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

    # BACKWARDS
    results_inv = smoother_step(
        filter_mu=results[0],
        filter_Sigma=results[1],
        smooth_mu=results[0],
        smooth_Sigma=results[1],
        F=F,
        Q=Q,
        H=H,
        R=R,
        return_full=return_full,
    )

    assert results_inv[0].shape == results[0].shape
    assert results_inv[1].shape == results[1].shape
    if not return_full:
        assert results_inv[2].shape == (obs_dim,)
        assert results_inv[3].shape == (obs_dim, obs_dim)
    else:
        assert results_inv[2].shape == (state_dim, state_dim)


@pytest.mark.parametrize("return_predict", [False, True])
@pytest.mark.parametrize("masks", [None, True])
@pytest.mark.parametrize("return_full", [False, True])
def test_smoother_step_sequential(return_predict, masks, return_full):

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

    # BACKWARDS
    results_inv = smoother_step_sequential(
        filter_mus=results[0],
        filter_Sigmas=results[1],
        F=F,
        Q=Q,
        H=H,
        R=R,
        return_full=return_full,
    )

    assert results_inv[0].shape == (n_time, state_dim)
    assert results_inv[1].shape == (n_time, state_dim, state_dim)
    assert results_inv[2].shape == (n_time, state_dim, state_dim)


@pytest.mark.parametrize("return_predict", [False, True])
@pytest.mark.parametrize("masks", [None, True])
@pytest.mark.parametrize("return_full", [False, True])
def test_rts_smoother(return_predict, masks, return_full):

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

    # FORWARD
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
    assert results[2].shape == (n_samples, n_time)

    # BACKWARDS
    results_inv = rauch_tung_striebel_smoother(
        filter_mus=results[0],
        filter_Sigmas=results[1],
        F=F,
        Q=Q,
        H=H,
        R=R,
        return_full=return_full,
    )

    assert results_inv[0].shape == (n_samples, n_time, state_dim)
    assert results_inv[1].shape == (n_samples, n_time, state_dim, state_dim)
    assert results_inv[2].shape == (n_samples, n_time, state_dim, state_dim)


@pytest.mark.parametrize("return_predict", [False, True])
@pytest.mark.parametrize("masks", [None, True])
@pytest.mark.parametrize("return_full", [False, True])
def test_rts_smoother_samples(return_predict, masks, return_full):

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

    # BACKWARDS
    results_inv = rauch_tung_striebel_smoother_samples(
        filter_mus=results[0],
        filter_Sigmas=results[1],
        F=F,
        Qs=Qs,
        H=H,
        Rs=Rs,
        return_full=return_full,
    )

    assert results_inv[0].shape == (n_samples, n_time, state_dim)
    assert results_inv[1].shape == (n_samples, n_time, state_dim, state_dim)
    assert results_inv[2].shape == (n_samples, n_time, state_dim, state_dim)
