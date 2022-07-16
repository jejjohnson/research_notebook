import torch
import torch.nn as nn
from torchtyping import TensorType
from torch import Tensor
from typing import Tuple


scale_clamper = 1e-4


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`

    Args:
        z_dim: num of dims for the latent variable, z, (D_z)
        transition_dim: num of hidden dims for the transition function (int)
    """

    def __init__(self, z_dim: int, transition_dim: int) -> None:
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        # TODO: generic non-linear activation (e.g. RelU, tanh, softplus)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        # self.batchnorm = nn.BatchNorm1d(num_features=transition_dim)

    def forward(
        self, z_t_1: TensorType["batch", "latent_dim"]
    ) -> Tuple[TensorType["batch", "latent_dim"], TensorType["batch", "latent_dim"]]:
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`

        Args:
            z_t: input latent vector, size=(B,D_z)

        Returns:
            loc: output mean for transition function, size=(B, D_z)
            scale: output scale for the transition function, size=(B, D_z)
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        #         _gate = self.batchnorm(_gate)
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))

        # TODO: Investigate clamping
        # scale = self.softplus(self.lin_sig(self.relu(proposed_mean))).clamp(min=1e-4)

        # return loc, scale which can be fed into Normal
        return loc, scale


class ConvLSTMGatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_channels, transition_channels, kernel_size, width, height):
        super().__init__()
        padding = int((kernel_size - 1) / 2)

        # initialize the six conv transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Conv2d(
            z_channels, transition_channels, kernel_size, 1, padding, bias=True
        )
        self.lin_gate_hidden_to_z = nn.Conv2d(
            transition_channels, z_channels, kernel_size, 1, padding, bias=True
        )
        self.lin_proposed_mean_z_to_hidden = nn.Conv2d(
            z_channels, transition_channels, kernel_size, 1, padding, bias=True
        )
        self.lin_proposed_mean_hidden_to_z = nn.Conv2d(
            transition_channels, z_channels, kernel_size, 1, padding, bias=True
        )
        self.lin_sig = nn.Conv2d(
            z_channels, z_channels, kernel_size, 1, padding, bias=True
        )
        self.lin_z_to_loc = nn.Conv2d(
            z_channels, z_channels, kernel_size, 1, padding, bias=True
        )
        # modify the default initialization of lin_z_to_loc
        # # so that it's starts out as the identity function
        # self.lin_z_to_loc.weight.data = torch.ones(z_channels, z_channels, kernel_size, kernel_size)
        self.lin_z_to_loc.bias.data = torch.zeros(z_channels)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean))).clamp(
            min=scale_clamper
        )
        # return loc, scale which can be fed into Normal
        return loc, scale
