from distrax._src.utils.jittable import Jittable
from filterjax._src.ops.sample import sample_sequential, sample_n
from filterjax._src.ops.filter import forward_filter
from filterjax._src.ops.smooth import rauch_tung_striebel_smoother
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class StateSpaceModel(Jittable):
    def __init__(
        self,
        mu0,
        Sigma0,
        transition_matrix,
        transition_noise,
        observation_matrix,
        observation_noise,
    ):
        self.obs_dim = observation_matrix.shape[0]
        self.state_dim = transition_matrix.shape[0]
        self.mu0 = mu0
        self._Sigma0 = Sigma0
        self.transition_matrix = transition_matrix
        self._transition_noise = transition_noise
        self.observation_matrix = observation_matrix
        self._observation_noise = observation_noise

    @property
    def prior_noise(self):
        return NotImplementedError()

    @property
    def transition_noise(self):
        return NotImplementedError()

    @property
    def observation_noise(self):
        return NotImplementedError()

    @property
    def prior_dist(self):
        return NotImplementedError()

    def __call__(self, x, masks=None):

        return self.forward_filter(x, masks=masks)

    def sample(
        self, key, n_samples: int = 1, n_time_steps=10, sample_prior: bool = False
    ):

        return sample_n(
            key,
            self.prior_dist,
            self.transition_matrix,
            self.transition_noise,
            self.observation_matrix,
            self.observation_noise,
            n_samples=n_samples,
            n_time_steps=n_time_steps,
            sample_prior=sample_prior,
        )

    def forward_filter(self, x, masks=None):
        res = forward_filter(
            x,
            self.mu0,
            self.prior_noise,
            self.transition_matrix,
            self.transition_noise.covariance(),
            self.observation_matrix,
            self.observation_noise.covariance(),
            masks=masks,
        )

        return res

    def backward_smoothing_pass(self, filtered_means, filtered_covs, return_full=False):
        """
        Parameters
        ----------
        filtered_means : np.ndarray, shape=(batch, time, dims)
            the filtered means for the input observations
        filtered_covs : np.ndarray, shape=(batch, time, dims)
            the filtered covs for the input observations
        return_full : bool, default=False
            option to return the smoothed states

        Returns
        -------
        x_smoothed : np.ndarray, shape=(batch, time, dim)
            the mean for the smoothed observations
        smoothed_covs : np.ndarray, shape=(batch, time, dim, dim)
            the cov for the smoothed observations
        gains :
        """
        smoothed_means, smoothed_covs, gains = rauch_tung_striebel_smoother(
            filtered_means,
            filtered_covs,
            self.transition_matrix,
            self.transition_noise.covariance(),
            self.observation_matrix,
            self.observation_noise.covariance(),
            return_full=return_full,
        )

        return smoothed_means, smoothed_covs, gains

    def posterior_marginals(self, x, masks=None):
        """
        Parameters
        ----------
        x : np.ndarray, shape=(batch, time, dims)
            the input observations
        masks : np.ndarray, shape=(batch, time, dims)
            the mask for the observations

        Returns
        -------
        x_smoothed : np.ndarray, shape=(batch, time, dim)
            the mean for the smoothed observations
        smoothed_covs : np.ndarray, shape=(batch, time, dim, dim)
            the cov for the smoothed observations
        """
        # forward filter
        filter_z_means, filter_z_covs, *_ = self.forward_filter(x, masks=masks)

        # backward pass
        smoothed_means, smoothed_covs, _ = self.backward_smoothing_pass(
            filter_z_means, filter_z_covs, False
        )

        return smoothed_means, smoothed_covs

    def log_prob(self, x, masks=None):
        """
        Parameters
        ----------
        x : np.ndarray, shape=(batch, time, dims)
            the input observations for the log probability
        masks : np.ndarray, shape=(batch, time, dims)
            the mask for the observations

        Returns
        -------
        x_logprob : np.ndarray, shape=(batch)
            the log probability for all of the samples
        """
        *_, log_probs = self.forward_filter(x, masks=masks)
        # # print(log_probs.shape)
        # log_probs = jnp.nan_to_num(
        return log_probs.sum(axis=-1)

    def negative_log_likelihood(self, x, masks=None):
        """
        Parameters
        ----------
        x : np.ndarray, shape=(batch, time, dims)
            the input observations for the log probability
        masks : np.ndarray, shape=(batch, time, dims)
            the mask for the observations

        Returns
        -------
        nll : np.ndarray, shape=()
            the average negative log probability for all of the samples
            Note: usually used as a loss function
        """
        return -jnp.mean(self.log_prob(x=x, masks=masks))


class StateSpaceModelDiag(StateSpaceModel):
    @property
    def prior_noise(self):
        return jnp.diag(self._Sigma0)

    @property
    def transition_noise(self):
        return tfd.MultivariateNormalDiag(
            loc=jnp.zeros(self.state_dim), scale_diag=self._transition_noise
        )

    @property
    def observation_noise(self):
        return tfd.MultivariateNormalDiag(
            loc=jnp.zeros(self.obs_dim), scale_diag=self._observation_noise
        )

    @property
    def prior_dist(self):
        return tfd.MultivariateNormalDiag(loc=self.mu0, scale_diag=self._Sigma0)


class StateSpaceModelFull:
    @property
    def prior_noise(self):
        return self._Sigma0

    @property
    def transition_noise(self):
        return tfd.MultivariateFullCovariance(
            loc=jnp.zeros(self.state_dim), scale_diag=self._transition_noise
        )

    @property
    def observation_noise(self):
        return tfd.MultivariateFullCovariance(
            loc=jnp.zeros(self.obs_dim), scale_diag=self._observation_noise
        )

    @property
    def prior_dist(self):
        return tfd.MultivariateFullCovariance(loc=self.mu0, scale_diag=self._Sigma0)
