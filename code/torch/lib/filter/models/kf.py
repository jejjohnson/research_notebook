# Take from: https://github.com/strongio/torchcast/blob/main/torchcast/kalman_filter.py

# from torch import nn, Tensor

# class GaussianStep(StateSpaceStep):
#     """
#     Used internally by `KalmanFilter` to apply the kalman-filtering algorithm. Subclasses can implement additional
#     logic such as outlier-rejection, censoring, etc.
#     """
#     use_stable_cov_update: Final[bool] = True

#     # this would ideally be a class-attribute but torch.jit.trace strips them
#     @torch.jit.ignore()
#     def get_distribution(self) -> Type[torch.distributions.Distribution]:
#         return torch.distributions.MultivariateNormal

#     def predict(self, mean: Tensor, cov: Tensor, F: Tensor, Q: Tensor) -> Tuple[Tensor, Tensor]:
#         mean = (F @ mean.unsqueeze(-1)).squeeze(-1)
#         cov = F @ cov @ F.permute(0, 2, 1) + Q
#         return mean, cov

#     def _update(self, input: Tensor, mean: Tensor, cov: Tensor, H: Tensor, R: Tensor) -> Tuple[Tensor, Tensor]:
#         K = self.kalman_gain(cov=cov, H=H, R=R)
#         measured_mean = (H @ mean.unsqueeze(-1)).squeeze(-1)
#         resid = input - measured_mean
#         new_mean = mean + (K @ resid.unsqueeze(-1)).squeeze(-1)
#         new_cov = self.covariance_update(cov=cov, K=K, H=H, R=R)
#         return new_mean, new_cov

#     def covariance_update(self, cov: Tensor, K: Tensor, H: Tensor, R: Tensor) -> Tensor:
#         I = torch.eye(cov.shape[1], dtype=cov.dtype, device=cov.device).unsqueeze(0)
#         ikh = I - K @ H
#         if self.use_stable_cov_update:
#             return ikh @ cov @ ikh.permute(0, 2, 1) + K @ R @ K.permute(0, 2, 1)
#         else:
#             return ikh @ cov

#     def kalman_gain(self, cov: Tensor, H: Tensor, R: Tensor) -> Tensor:
#         Ht = H.permute(0, 2, 1)
#         covs_measured = cov @ Ht
#         system_covariance = torch.baddbmm(R, H @ cov, Ht)
#         A = system_covariance.permute(0, 2, 1)
#         B = covs_measured.permute(0, 2, 1)
#         Kt, _ = torch.solve(B, A)
#         K = Kt.permute(0, 2, 1)
#         return K
