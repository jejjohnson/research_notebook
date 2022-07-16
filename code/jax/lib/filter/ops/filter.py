from typing import Optional, Sequence, Tuple, Union
import collections.abc
import operator
import functools
import jax
import jax.numpy as jnp
import numpy as np


# from filterjax._src.losses import mvn_logpdf
from filterjax._src.ops.linalg import solve
from filterjax._src.ops.masks import (
    mask_observation_noise,
    mask_observation_operator,
    mvn_logpdf,
)

IntLike = Union[int, np.int16, np.int32, np.int64]


def kalman_step(
    obs: jnp.ndarray,
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    F: jnp.ndarray,
    Q: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    return_likelihood: bool = False,
    mask: Optional[jnp.ndarray] = None,
):
    """Computes the kalman step

    Parameters
    ----------
    obs : np.ndarray, shape=(obs_dims)
        the observations
    mu : np.ndarray, shape=(state_dims)
        the mean state
    sigma : np.ndarray, shape=(state_dims, state_dims)
        the sigma state
    F : np.ndarray, shape=(state_dims, state_dims)
        the transition matrix for the transition function
    Q : np.ndarray, shape=(state_dims, state_dims)
        the transition noise for the transition function
    H : np.ndarray, shape=(obs_dim, state_dims)
        the emission matrix for the emission function
    R : np.ndarray, shape=(obs_dim, obs_dim)
        the emission noise for the emission function
    return_likelihood : bool, default=False
        option to return the log likelihood for the update step
    mask : np.ndarray (Optional), default=None, shape=(obs_dims)
        an optional mask for the observations (needed for calculating
        the log-likelihood function)

    Returns
    -------
    mu_t : np.ndarray, shape=(state_dims)
        the posterior state mean given the observations
    Sigma_t : np.ndarray, shape=(state_dims, state_dims)
        the posterior state cov given the observations
    mu_tt_1 : np.ndarray, shape=(state_dims)
        the predicted state mean
    Sigma_tt_1 : np.ndarray, shape=(state_dims, state_dims)
        the predicted state cov
    ell : float, shape=()
        the likelihood value for the update step
    """

    # get dims
    state_dim = F.shape[-1]
    I = jnp.eye(state_dim)

    # ==============
    # PREDICT STEP
    # ==============

    # state - predictive mean, cov, t|t-1
    mu_z_tt_1 = F @ mu
    Sigma_z_tt_1 = F @ Sigma @ F.T + Q

    # print("State:", mu_z_tt_1.shape, Sigma_z_tt_1.shape)

    # observations - predictive mean, cov, t
    mu_x_t = H @ mu_z_tt_1
    HP = H @ Sigma_z_tt_1
    Sigma_x_t = HP @ H.T + R

    # print("Obs:", mu_x_t.shape, Sigma_x_t.shape)

    # ============
    # UPDATE STEP
    # ============

    if mask is not None:
        # print(H.shape, R.shape, mask.shape)
        H = mask_observation_operator(H, mask)
        R = mask_observation_noise(R, mask)
        # print(H.shape, R.shape)

    # innovation
    r_t = obs - mu_x_t

    # Kalman gain
    K_t = solve(Sigma_x_t, H @ Sigma_z_tt_1).T

    # correction
    mu_z_t = mu_z_tt_1 + K_t @ r_t
    Sigma_z_t = (I - K_t @ H) @ Sigma_z_tt_1

    # print("State (Update):", mu_z_t.shape, Sigma_z_t.shape)

    if return_likelihood:
        # # updated prediction
        # mu_x_t = H @ mu_z_t
        # Sigma_x_t = F @ Sigma_z_t @ F.T + Q
        ell = mvn_logpdf(obs, mu_x_t, Sigma_x_t, mask)

        return mu_z_t, Sigma_z_t, mu_x_t, Sigma_x_t, ell

    else:
        return mu_z_t, Sigma_z_t, mu_x_t, Sigma_x_t


def filter_step_sequential(
    obs: jnp.ndarray,
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    F: jnp.ndarray,
    Q: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    return_predict: int = False,
    masks: Optional[jnp.ndarray] = None,
):
    """Computes the kalman step

    Parameters
    ----------
    obs : np.ndarray, shape=(n_time, obs_dims)
        the observations
    mu0 : np.ndarray, shape=(state_dims)
        the mean state
    sigma0 : np.ndarray, shape=(state_dims, state_dims)
        the sigma state
    F : np.ndarray, shape=(state_dims, state_dims)
        the transition matrix for the transition function
    Q : np.ndarray, shape=(state_dims, state_dims)
        the transition noise for the transition function
    H : np.ndarray, shape=(obs_dim, state_dims)
        the emission matrix for the emission function
    R : np.ndarray, shape=(obs_dim, obs_dim)
        the emission noise for the emission function
    masks : np.ndarray (Optional), default=None, shape=(n_time, obs_dims)
        an optional mask for the observations (needed for calculating
        the log-likelihood function)
    return_predict : bool, default=False
        whether to return the transition for the states or the
        updated/corrected predictions given the observations

    Returns
    -------
    mu_t : np.ndarray, shape=(n_time, state_dims)
        the posterior state mean given the observations if return_predict
        is False. Else the filtering predictions.
    Sigma_t : np.ndarray, shape=(n_time, state_dims, state_dims)
        the posterior state cov given the observations if return_predict
        is False. Else the filtering predictions.
    ell : float, shape=(n_time,)
        the likelihood value for each of the update steps
    """

    # if masks is None:
    #     masks = jnp.ones_like(obs)

    # define ad-hoc body for kalman step
    def body(carry, inputs):
        # unroll inputs
        obs, mask = inputs
        mu, Sigma = carry

        # do Kalman Step
        mu_z_t, Sigma_z_t, mu_x_t, Sigma_x_t, ell = kalman_step(
            obs, mu, Sigma, F, Q, H, R, return_likelihood=True, mask=mask
        )
        return (mu_z_t, Sigma_z_t), (mu_z_t, Sigma_z_t, mu_x_t, Sigma_x_t, ell)

    # loop through samples
    _, (mu_zs, Sigma_zs, mu_xs, Sigma_xs, logliks) = jax.lax.scan(
        body, init=(mu0, Sigma0), xs=(obs, masks)
    )

    return mu_zs, Sigma_zs, mu_xs, Sigma_xs, logliks


def forward_filter(
    obs: jnp.ndarray,
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    F: jnp.ndarray,
    Q: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    masks: Optional[jnp.ndarray] = None,
    return_predict: bool = False,
):
    """Computes the kalman step

    Parameters
    ----------
    obs : np.ndarray, shape=(n_samples, n_time, obs_dims)
        the observations
    mu0 : np.ndarray, shape=(n_samples, state_dims)
        the mean state
    sigma0 : np.ndarray, shape=(n_samples, state_dims, state_dims)
        the sigma state
    F : np.ndarray, shape=(state_dims, state_dims)
        the transition matrix for the transition function
    Q : np.ndarray, shape=(n_samples, state_dims, state_dims)
        the transition noise for the transition function
    H : np.ndarray, shape=(obs_dim, state_dims)
        the emission matrix for the emission function
    R : np.ndarray, shape=(n_samples, obs_dim, obs_dim)
        the emission noise for the emission function
    masks : np.ndarray (Optional), default=None, shape=(n_samples, n_time, obs_dims)
        an optional mask for the observations (needed for calculating
        the log-likelihood function)
    return_predict : bool, default=False
        whether to return the transition for the states or the
        updated/corrected predictions given the observations

    Returns
    -------
    mu_t : np.ndarray, shape=(n_samples, n_time, state_dims)
        the posterior state mean given the observations if return_predict
        is False. Else the filtering predictions.
    Sigma_t : np.ndarray, shape=(n_samples, n_time, state_dims, state_dims)
        the posterior state cov given the observations if return_predict
        is False. Else the filtering predictions.
    ell : float, shape=(n_samples, n_time,)
        the likelihood value for each of the update steps
    """

    # if masks is None:
    #     masks = jnp.ones_like(obs)

    # define forward map
    fn = lambda obs, mask: filter_step_sequential(
        obs,
        mu0=mu0,
        Sigma0=Sigma0,
        F=F,
        Q=Q,
        H=H,
        R=R,
        masks=mask,
        return_predict=return_predict,
    )

    vmap_fn = jax.vmap(fn, in_axes=(0, 0))

    # handle dimensions
    state_dims = F.shape[0]
    obs_dims = H.shape[0]

    *batch_shape, timesteps, _ = obs.shape

    obs_mean_dims = (*batch_shape, timesteps, obs_dims)
    obs_cov_dims = (*batch_shape, timesteps, obs_dims, obs_dims)
    state_mean_dims = (*batch_shape, timesteps, state_dims)
    state_cov_dims = (*batch_shape, timesteps, state_dims, state_dims)

    obs = obs.reshape(-1, timesteps, obs_dims)
    if masks is not None:
        masks = masks.reshape(-1, timesteps, obs_dims)

    # forward filter
    # print("Inputs:", obs.shape, masks.shape)
    (
        mu_z_filtered,
        Sigma_z_filtered,
        mu_x_filtered,
        Sigma_x_filtered,
        log_likelihoods,
    ) = vmap_fn(obs, masks)

    # reshape to appropriate dimensions
    # print("Final:", mu_z_filtered.shape, mu_x_filtered.shape)
    mu_z_filtered = mu_z_filtered.reshape(state_mean_dims)
    Sigma_z_filtered = Sigma_z_filtered.reshape(state_cov_dims)
    mu_x_filtered = mu_x_filtered.reshape(obs_mean_dims)
    Sigma_x_filtered = Sigma_x_filtered.reshape(obs_cov_dims)
    log_likelihoods = log_likelihoods.reshape(*batch_shape, timesteps)

    return (
        mu_z_filtered,
        Sigma_z_filtered,
        mu_x_filtered,
        Sigma_x_filtered,
        log_likelihoods,
    )


def forward_filter_samples(
    obs: jnp.ndarray,
    mu0s: jnp.ndarray,
    Sigma0s: jnp.ndarray,
    F: jnp.ndarray,
    Qs: jnp.ndarray,
    H: jnp.ndarray,
    Rs: jnp.ndarray,
    masks: Optional[jnp.ndarray] = None,
    return_predict: bool = False,
):
    """Computes the kalman step

    Parameters
    ----------
    obs : np.ndarray, shape=(n_samples, n_time, obs_dims)
        the observations
    mu0 : np.ndarray, shape=(n_samples, state_dims)
        the mean state
    sigma0 : np.ndarray, shape=(n_samples, state_dims, state_dims)
        the sigma state
    F : np.ndarray, shape=(state_dims, state_dims)
        the transition matrix for the transition function
    Q : np.ndarray, shape=(n_samples, state_dims, state_dims)
        the transition noise for the transition function
    H : np.ndarray, shape=(obs_dim, state_dims)
        the emission matrix for the emission function
    R : np.ndarray, shape=(n_samples, obs_dim, obs_dim)
        the emission noise for the emission function
    masks : np.ndarray (Optional), default=None, shape=(n_samples, n_time, obs_dims)
        an optional mask for the observations (needed for calculating
        the log-likelihood function)
    return_predict : bool, default=False
        whether to return the transition for the states or the
        updated/corrected predictions given the observations

    Returns
    -------
    mu_t : np.ndarray, shape=(n_samples, n_time, state_dims)
        the posterior state mean given the observations if return_predict
        is False. Else the filtering predictions.
    Sigma_t : np.ndarray, shape=(n_samples, n_time, state_dims, state_dims)
        the posterior state cov given the observations if return_predict
        is False. Else the filtering predictions.
    ell : float, shape=(n_samples, n_time,)
        the likelihood value for each of the update steps
    """

    if masks is None:
        masks = [None] * obs.shape[0]

    # define forward map
    fn = lambda obs, mu, Sigma, Q, R, mask: filter_step_sequential(
        obs,
        mu0=mu,
        Sigma0=Sigma,
        F=F,
        Q=Q,
        H=H,
        R=R,
        masks=mask,
        return_predict=return_predict,
    )

    vmap_fn = jax.vmap(fn, in_axes=(0, 0, 0, 0, 0, 0))

    # handle dimensions
    state_dims = F.shape[0]
    obs_dims = H.shape[0]

    *batch_shape, timesteps, _ = obs.shape
    state_mean_dims = (*batch_shape, timesteps, state_dims)
    state_cov_dims = (*batch_shape, timesteps, state_dims, state_dims)

    obs = obs.reshape(-1, timesteps, obs_dims)

    # forward filter
    (
        mu_filtered,
        Sigma_filtered,
        log_likelihoods,
    ) = vmap_fn(obs, mu0s, Sigma0s, Qs, Rs, masks)

    # reshape to appropriate dimensions
    mu_filtered = mu_filtered.reshape(state_mean_dims)
    Sigma_filtered = Sigma_filtered.reshape(state_cov_dims)
    log_likelihoods = log_likelihoods.reshape(*batch_shape, timesteps)

    return mu_filtered, Sigma_filtered, log_likelihoods


def predict_step_state(mu, Sigma, F, Q):
    # predictive mean, cov (state), t|t-1
    mu_z_t = F @ mu
    Sigma_z_t = F @ Sigma @ F.T + Q

    return mu_z_t, Sigma_z_t


def predict_step_obs(mu_z_t, Sigma_z_t, H, R):
    # predictive mean, cov (obs), t
    mu_x_t = H @ mu_z_t
    Sigma_x_t = H @ Sigma_z_t @ H.T + R

    return mu_x_t, Sigma_x_t


def update_step(mu_z_t, Sigma_z_t, mu_x_t, Sigma_x_t, x_t, H):
    # innovation
    r_t = x_t - mu_x_t

    # Kalman gain
    K_t = Sigma_z_t @ H.T @ jnp.linalg.inv(Sigma_x_t)

    I = jnp.eye(Sigma_z_t.shape[-1])

    # correction
    mu_z_t = mu_z_t + K_t @ r_t
    Sigma_z_t = (I - K_t @ H) @ Sigma_z_t

    return mu_z_t, Sigma_z_t
