import torch


def discretize(F, Q, t, device):
    """
    Discretize matrices for continuous update step with matrix exponential and matrix fraction decomposition

    Keyword arguments:
    F, Q -- KF matrices for state update (torch.Tensor)
    t -- time delta to last observation (torch.Tensor)
    device -- device of model (torch.device)
    Returns:
    A, L -- discretized update matrices
    """
    if len(F.shape) == 3:
        m = F.shape[0]
        n = F.shape[1]
        M = torch.zeros(m, 2 * n, 2 * n)
        A = torch.matrix_exp(F * t.unsqueeze(-1).unsqueeze(-1).to(device))
        M[:, :n, :n] = F
        M[:, :n, n:] = Q
        M[:, n:, n:] = -F.transpose(1, 2)
        M = torch.matrix_exp(M * t.unsqueeze(-1).unsqueeze(-1)) @ torch.cat(
            [torch.zeros(n, n), torch.eye(n, n)]
        )
    else:
        n = F.shape[0]
        M = torch.zeros(2 * n, 2 * n)
        A = torch.matrix_exp(F * t.view(-1, 1, 1).to(device))
        M[:n, :n] = F
        M[:n, n:] = Q
        M[n:, n:] = -F.T
        M = torch.matrix_exp(M * t.view(-1, 1, 1)) @ torch.cat(
            [torch.zeros(n, n), torch.eye(n, n)]
        )
    C, D = M[:, :n], M[:, n:]
    L = C @ torch.inverse(D)
    return A.to(device), L.to(device)
