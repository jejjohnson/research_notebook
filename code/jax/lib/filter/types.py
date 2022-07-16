from typing import Union, NamedTuple
import numpy as np
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

IntLike = Union[int, np.int16, np.int32, np.int64]


class KFParams(NamedTuple):
    transition_matrix: jnp.ndarray
    transition_noise: jnp.ndarray
    observation_matrix: jnp.ndarray
    observation_noise: jnp.ndarray


class KFParamsDist(NamedTuple):
    transition_matrix: jnp.ndarray
    transition_noise_dist: tfd.Distribution
    observation_matrix: jnp.ndarray
    observation_noise_dist: tfd.Distribution


class State(NamedTuple):
    mu_t: jnp.ndarray
    Sigma_t: jnp.ndarray
    t: int


class FilteredState(NamedTuple):
    mu_filtered: jnp.ndarray
    Sigma_filtered: jnp.ndarray
    log_likelihoods: jnp.ndarray
    mu_cond: jnp.ndarray
    Sigma_cond: jnp.ndarray
    ts: jnp.ndarray
