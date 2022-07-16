from typing import Optional, Sequence, Tuple, Union
import collections.abc
import operator
import functools
import jax
import jax.numpy as jnp
import numpy as np


from filterjax._src.losses import mvn_logpdf
from filterjax._src.ops.linalg import solve

IntLike = Union[int, np.int16, np.int32, np.int64]


def smoother_step(
    filter_mu: jnp.ndarray,
    filter_Sigma: jnp.ndarray,
    smooth_mu: jnp.ndarray,
    smooth_Sigma: jnp.ndarray,
    F: jnp.ndarray,
    Q: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    return_full: bool = False,
):
    """Computes the kalman step

    Parameters
    ----------
    filter_mean : np.ndarray, shape=(state_dims)
        the mean state
    filter_sigma : np.ndarray, shape=(state_dims, state_dims)
        the sigma state
    smooth_mean : np.ndarray, shape=(state_dims)
        the mean state
    smooth_sigma : np.ndarray, shape=(state_dims, state_dims)
        the sigma state
    F : np.ndarray, shape=(state_dims, state_dims)
        the transition matrix for the transition function
    Q : np.ndarray, shape=(state_dims, state_dims)
        the transition noise for the transition function
    H : np.ndarray, shape=(obs_dim, state_dims)
        the emission matrix for the emission function
    R : np.ndarray, shape=(obs_dim, obs_dim)
        the emission noise for the emission function
    return_full : bool, default=False
        option to return the log likelihood for the update step

    Returns
    -------
    smooth_mu : np.ndarray, shape=(state_dims)
        the posterior state mean given the observations
    smooth_Sigma : np.ndarray, shape=(state_dims, state_dims)
        the posterior state cov given the observations
    smooth_mu_obs : np.ndarray, shape=(obs_dim)
        the predicted state mean (Optional)
    smooth_Sigma_obs : np.ndarray, shape=(obs_dim, obs_dim)
        the predicted state cov (Optional)
    C : np.ndarray,
        the gain matrix
    """

    # get dims

    p_mu = F @ filter_mu
    F_fP = F @ filter_Sigma
    p_Sigma = F_fP @ F.T + Q

    C = solve(p_Sigma, F_fP).T

    smooth_mu = filter_mu + C @ (smooth_mu - p_mu)
    smooth_Sigma = filter_Sigma + C @ (smooth_Sigma - p_Sigma) @ C.T

    if return_full:

        return smooth_mu, smooth_Sigma, C
    else:
        # print("here!", H.shape, smooth_mu.shape)
        smooth_mu_obs = H @ smooth_mu
        smooth_Sigma_obs = H @ smooth_Sigma @ H.T
        # print(smooth_mu.shape, smooth_mu_obs.shape)
        return (smooth_mu, smooth_Sigma, smooth_mu_obs, smooth_Sigma_obs, C)


def smoother_step_sequential(
    filter_mus: jnp.ndarray,
    filter_Sigmas: jnp.ndarray,
    F: jnp.ndarray,
    Q: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    return_full: bool = False,
):
    """Computes the kalman step

    Parameters
    ----------
    filter_means : np.ndarray, shape=(n_time, state_dims)
        the mean state
    filter_sigmas : np.ndarray, shape=(n_time, state_dims, state_dims)
        the sigma state
    F : np.ndarray, shape=(state_dims, state_dims)
        the transition matrix for the transition function
    Q : np.ndarray, shape=(state_dims, state_dims)
        the transition noise for the transition function
    H : np.ndarray, shape=(obs_dim, state_dims)
        the emission matrix for the emission function
    R : np.ndarray, shape=(obs_dim, obs_dim)
        the emission noise for the emission function
    return_full : bool, default=False
        option to return the log likelihood for the update step

    Returns
    -------
    smooth_mus : np.ndarray, shape=(state_dims)
        the posterior state mean given the observations
    smooth_Sigma : np.ndarray, shape=(state_dims, state_dims)
        the posterior state cov given the observations
    gains : np.ndarray,
        helpful C matrix
    """

    # define ad-hoc body for kalman step
    def body(carry, inputs):
        # unroll inputs
        filter_mu, filter_Sigma = inputs
        smooth_mu, smooth_Sigma = carry

        # do Kalman Step
        smooth_mu, smooth_Sigma, smooth_obs_mu, smooth_obs_Sigma, gain = smoother_step(
            filter_mu=filter_mu,
            filter_Sigma=filter_Sigma,
            smooth_mu=smooth_mu,
            smooth_Sigma=smooth_Sigma,
            F=F,
            H=H,
            Q=Q,
            R=R,
            return_full=False,
        )
        # print(smooth_mu.shape, smooth_obs_mu.shape)
        if return_full:
            return (smooth_mu, smooth_Sigma), (smooth_mu, smooth_Sigma, gain)
        else:
            return (smooth_mu, smooth_Sigma), (smooth_obs_mu, smooth_obs_Sigma, gain)

    # loop through samples
    _, (smooth_mus, smooth_Sigmas, gains) = jax.lax.scan(
        body,
        init=(filter_mus[-1], filter_Sigmas[-1]),
        xs=(filter_mus, filter_Sigmas),
        reverse=True,
    )
    # print("Output:", smooth_mus.shape)

    return smooth_mus, smooth_Sigmas, gains


def rauch_tung_striebel_smoother(
    filter_mus: jnp.ndarray,
    filter_Sigmas: jnp.ndarray,
    F: jnp.ndarray,
    Q: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    return_full: bool = False,
):
    """Computes the kalman step

    Parameters
    ----------
    filter_mus : np.ndarray, shape=(n_samples, n_time, state_dims)
        the mean state
    filter_Sigmas : np.ndarray, shape=(n_samples, n_time, state_dims, state_dims)
        the sigma state
    F : np.ndarray, shape=(state_dims, state_dims)
        the transition matrix for the transition function
    Q : np.ndarray, shape=(state_dims, state_dims)
        the transition noise for the transition function
    H : np.ndarray, shape=(obs_dim, state_dims)
        the emission matrix for the emission function
    R : np.ndarray, shape=(obs_dim, obs_dim)
        the emission noise for the emission function
    return_full : bool, default=False
        option to return the log likelihood for the update step

    Returns
    -------
    smooth_mus : np.ndarray, shape=(state_dims)
        the posterior state mean given the observations
    smooth_Sigma : np.ndarray, shape=(state_dims, state_dims)
        the posterior state cov given the observations
    gains : np.ndarray,
        helpful C matrix
    """

    # define forward map
    fn = lambda filter_mu, filter_Sigma: smoother_step_sequential(
        filter_mus=filter_mu,
        filter_Sigmas=filter_Sigma,
        F=F,
        Q=Q,
        H=H,
        R=R,
        return_full=return_full,
    )

    vmap_fn = jax.vmap(fn, in_axes=(0, 0))

    # forward filter
    (
        mu_smoothed,
        Sigma_smoothed,
        gains,
    ) = vmap_fn(filter_mus, filter_Sigmas)

    return mu_smoothed, Sigma_smoothed, gains


def rauch_tung_striebel_smoother_samples(
    filter_mus: jnp.ndarray,
    filter_Sigmas: jnp.ndarray,
    F: jnp.ndarray,
    Qs: jnp.ndarray,
    H: jnp.ndarray,
    Rs: jnp.ndarray,
    return_full: bool = False,
):
    """Computes the kalman step

    Parameters
    ----------
    filter_mus : np.ndarray, shape=(n_samples, n_time, state_dims)
        the mean state
    filter_Sigmas : np.ndarray, shape=(n_samples, n_time, state_dims, state_dims)
        the sigma state
    F : np.ndarray, shape=(state_dims, state_dims)
        the transition matrix for the transition function
    Q : np.ndarray, shape=(state_dims, state_dims)
        the transition noise for the transition function
    H : np.ndarray, shape=(obs_dim, state_dims)
        the emission matrix for the emission function
    R : np.ndarray, shape=(obs_dim, obs_dim)
        the emission noise for the emission function
    return_full : bool, default=False
        option to return the log likelihood for the update step

    Returns
    -------
    smooth_mus : np.ndarray, shape=(state_dims)
        the posterior state mean given the observations
    smooth_Sigma : np.ndarray, shape=(state_dims, state_dims)
        the posterior state cov given the observations
    gains : np.ndarray,
        helpful C matrix
    """

    # define forward map
    fn = lambda filter_mu, filter_Sigma, Q, R: smoother_step_sequential(
        filter_mus=filter_mu,
        filter_Sigmas=filter_Sigma,
        F=F,
        Q=Q,
        H=H,
        R=R,
        return_full=return_full,
    )

    vmap_fn = jax.vmap(fn, in_axes=(0, 0, 0, 0))

    # forward filter
    (
        mu_smoothed,
        Sigma_smoothed,
        gains,
    ) = vmap_fn(filter_mus, filter_Sigmas, Qs, Rs)

    return mu_smoothed, Sigma_smoothed, gains
