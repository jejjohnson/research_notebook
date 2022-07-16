import math
from einops import repeat
import torch
import torch.distributions as dist

INV2PI = (2 * math.pi) ** -1


def mask_observation_operator(operator, mask):
    """
    Parameters
    ----------
    operator : array, shape=(*batches, obs_dim, state_dim)
        the linear operator
    mask : array, shape=(*batches, obs_dim)
        the mask where the observed values are zeros

    Returns
    -------
    masked_operator : array, shape=(*batches, obs_dim, state_dim)
        the same operator with zeros where the mask is located.
    """
    # to enable broadcasting
    mask = repeat(mask, "... -> ... 1")

    operator = operator.masked_fill(mask == 1.0, 0)

    return operator


def mask_observation_noise_diag(noise, mask):
    """
    Parameters
    ----------
    noise : array, shape=(*batches, obs_dim, obs_dim)
        the linear operator
    mask : array, shape=(*batches, obs_dim)
        the mask where the observed values are zeros

    Returns
    -------
    noise : array, shape=(*batches, obs_dim, obs_dim)
        the same operator with zeros where the mask is located.
    """
    n_dims = mask.shape[0]

    mask = 1.0 - mask

    # create cov for mask
    maskv = mask.unsqueeze(0)
    mask_cov = 0.5 * (maskv + maskv.T)

    # fill zeros for all non 1.0 entries
    noise = noise.masked_fill(mask_cov != 1.0, 0.0)

    # create base identity
    identities = torch.eye(n_dims, n_dims)
    # print(identities)

    # remove 1s for masked entries
    identities = identities.masked_fill(torch.diag(mask) == 1, 0.0)
    # print("diag mask:", torch.diag(mask))
    # print(identities)

    # fill identity on non-masked regions
    noise = noise.masked_fill(identities == 1.0, 1.0)
    # print(noise)

    return noise


def stable_kalman_gain(H, P, Sigma_pred):
    # Kalman gain, a more stable implementation than naive P @ H^T @ y_sigma^{-1}
    L = torch.linalg.cholesky(Sigma_pred)
    K = torch.triangular_solve(H @ P.transpose(1, 2), L, upper=False)[0]
    K = torch.triangular_solve(K, L.transpose(1, 2))[0].transpose(1, 2)
    return K


def masked_multivariate_likelihood(x, mean, cov, mask=None):
    """Masked Likelihood for full covariance matrices

    Parameters
    ----------
    x : torch.Tensor, sha"""

    if mask is not None:

        # fill x values with zeros
        x = x.masked_fill(mask == 1.0, 0)

        # fill mean values with zeros
        mean = mean.masked_fill(mask == 1.0, 0)

        # ensure masked entries are independent
        maskv = mask.unsqueeze(0)
        cov_masked = cov.masked_fill(maskv + maskv.T > 0.0, 0)

        # ensure masked entries return log likelihood of 0
        cov = cov_masked.masked_fill(torch.diag(mask) == 1.0, INV2PI)
    # print(x)
    # print(mean)
    # print(cov)
    # cov = cov + 1e-6 * torch.eye(n=cov.shape[0], device=cov.device)
    # print(x.shape, mean.shape, cov.shape)
    return dist.MultivariateNormal(mean, cov).log_prob(x)


def transition_predict(x, P, F, Q):
    """
    Update state mean and covariance p(x_{t} | x_{t-1}) and calculate mean and
    covariance in the observation space in the case of discrete time steps

    Parameters
    ----------
    x : torch.Tensor, shape=(*batch, state_dim)
        the mean of the state
    P : torch.Tensor, shape=(*batch, state_dim, state_dim)
        the covariance of the state
    F : torch.Tensor, shape=(state_dim, state_dim)
        the transition matrix for the state
    Q : torch.Tensor, shape=(state_dim, state_dim)
        the transition noise for the state

    Returns
    -------
    x : torch.Tensor, shape=(*batch, state_dim)
        the mean of the state
    P : torch.Tensor, shape=(*batch, state_dim, state_dim)
        the covariance of the state
    """
    n_batch = x.shape[0]

    # (B, Dy) = (Dy, Dx) @ (B, Dx)
    x = torch.einsum("ij,kj->ki", F, x)

    # (B, Dy, Dy) = (Dy, Dx) @ (B, Dx. Dx) @ (Dy, Dx).T + (Dy, Dy)
    # same as H.matmul(P).matmul(H.t())
    # print(torch.einsum("ij,kjl,ml->kim", F, P, F).shape, Q.shape)
    P = torch.einsum("ij,kjl,ml->kim", F, P, F) + Q
    return x, P


def emission_predict(x, P, H, R):
    """
    Update state mean and covariance p(y_{t} | x_{t}) and calculate mean and
    covariance in the observation space in the case of discrete time steps

    Parameters
    ----------
    x : torch.Tensor, shape=(*batch, obs_dim)
        the mean of the state
    P : torch.Tensor, shape=(*batch, obs_dim, obs_dim)
        the covariance of the state
    H : torch.Tensor, shape=(state_dim, obs_dim)
        the observation matrix for the emission function
    R : torch.Tensor, shape=(state_dim, obs_dim)
        the observation noise for the emission function

    Returns
    -------
    y_mu : torch.Tensor, shape=(*batch, obs_dim)
        the mean of the state
    y_sigma : torch.Tensor, shape=(*batch, state_dim, obs_dim)
        the covariance of the state
    """
    # (B, Dy) = (Dy, Dx) @ (B, Dx)
    y_mu = torch.einsum("ij,kj->ki", H, x)

    # (B, Dy, Dy) = (Dy, Dx) @ (B, Dx, Dx) @ (Dy, Dx).T + (Dy, Dy)
    # same as H.matmul(P).matmul(H.t())
    y_sigma = torch.einsum("ij,kjl,ml->kim", H, P, H) + R

    return y_mu, y_sigma


def predict_step(x, P, F, Q, H, R):

    x, P = transition_predict(x, P, F, Q)
    y_mu, y_sigma = emission_predict(x, P, H, R)

    return x, P, y_mu, y_sigma


def update_step(obs, x, P, H, R, y_sigma, mask):
    """
    Update state x and P after the observation,
    outputs filtered state and covariance
    """
    # create masks
    if mask is not None:
        H = mask_observation_operator(H, mask)
        R = mask_observation_noise_diag(R, mask)

    # Update state mean and covariance p(x | y), Joseph Form
    # Kalman gain, a more stable implementation than naive P @ H^T @ y_sigma^{-1}
    L = torch.linalg.cholesky(
        y_sigma + 1e-6 * torch.eye(y_sigma.shape[-1], device=y_sigma.device)
    )
    K = torch.triangular_solve(H @ P.transpose(1, 2), L, upper=False)[0]
    K = torch.triangular_solve(K, L.transpose(1, 2))[0].transpose(1, 2)

    v = obs - batch_mat_predict(H, x)
    x = x + torch.einsum("bso,bo->bs", K, v)

    # P = (torch.eye(*P.shape[1:]) - K @ self.H) @ P @ (torch.eye(*P.shape[1:]) - K @ self.H).T + K @ self.trans_noise @ K.T
    identity = torch.eye(*P.shape[1:], device=P.device)
    t1 = identity - K @ H
    P = t1.matmul(P).matmul(t1.transpose(1, 2)) + K.matmul(R).matmul(K.transpose(1, 2))
    return x, P


def batch_mat_predict(operator, mu):
    return torch.einsum("ij,kj->ki", operator, mu)


def batch_mat_cov(operator, cov):
    return torch.einsum("ij,kjl,ml->kim", operator, cov, operator)
