import torch
import torch.nn as nn


def kl_div(mu1, logvar1, mu2=None, logvar2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if logvar2 is None:
        logvar2 = torch.zeros_like(mu1)

    return 0.5 * (
        logvar2
        - logvar1
        + (torch.exp(logvar1) + (mu1 - mu2).pow(2)) / torch.exp(logvar2)
        - 1
    )
