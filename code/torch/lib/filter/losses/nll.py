import torch
import torch.nn as nn


def nll_loss(x_hat, x):
    assert x_hat.dim() == x.dim() == 3
    assert x.size() == x_hat.size()
    return nn.BCEWithLogitsLoss(reduction="none")(x_hat, x)


def masked_likelihood(batch, pred_means, pred_sigmas, flow):
    """
    Computes masked likelihood of observed values in predicted distributions
    Keyword arguments:
    batch -- batch like constructed in datasets.utils.data_utils.collate_KF (dict)
    pred_means -- predicted means (torch.Tensor)
    pred_sigmas -- predicted sigmas (torch.Tensor)
    flow -- instance of flow (nf.Flow)
    Returns:
    loss -- mean negative log-likelihood
    """
    loss = torch.Tensor([]).cuda()
    obs = batch["z"]
    mask = batch["mask"].to(torch.bool)
    # no other implementation possible, calc now scales in number of dimensions & not in number of observations
    for d in range(1, mask.shape[1] + 1):
        mm = torch.sum(mask, dim=1) == d
        if mm.sum() > 0:
            m = mask[mm]
            means = torch.stack(torch.chunk(pred_means[mm][m], len(m), dim=0))
            sigmas = torch.stack(
                torch.chunk(pred_sigmas[mm][m], len(m), dim=0)
            ).transpose(2, 1)
            sigmas = torch.stack(torch.chunk(sigmas[m], len(m), dim=0)).transpose(2, 1)
            # special case: only one dim observed -> normal distribution
            if d == 1:
                flow.base_dist = torch.distributions.Normal(
                    means.squeeze(-1), torch.sqrt(sigmas.squeeze(-1).squeeze(-1))
                )
                transformed_obs, log_jac_diag = flow.inverse(
                    obs[mm], mask=m.to(torch.int)
                )
                res = flow.base_dist.log_prob(
                    transformed_obs[mask[mm]].squeeze(0)
                ) + log_jac_diag.sum(-1)
                res = res if len(res.shape) > 0 else res.unsqueeze(0)
            else:
                flow.base_dist = torch.distributions.MultivariateNormal(means, sigmas)
                transformed_obs, log_jac_diag = flow.inverse(
                    obs[mm], mask=m.to(torch.int)
                )
                res = flow.base_dist.log_prob(
                    torch.stack(torch.chunk(transformed_obs[m], len(m), dim=0))
                ) + log_jac_diag.sum(-1)
            loss = torch.cat((loss, res))
    return -torch.sum(loss) / mask.sum()
