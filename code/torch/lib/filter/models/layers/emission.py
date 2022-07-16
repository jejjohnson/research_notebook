# Code adapted from: https://github.com/hmsandager/Normalizing-flow-and-deep-kalman-filter/blob/main/dkf.py

from typing import Tuple
import torch
import torch.nn as nn
from torchtyping import TensorType
import numpy as np

scale_clamper = 1e-4


class LinearEmitter(nn.Module):
    """
    Parameterizes the gaussian observation likelihood `p(x_t | z_t)`. Uses
    a non-linear transition function.

    Args:
        input_dim (int): number of dimensions for the input vector, x, (D_x,)
        z_dim (int): number of dimensions for the latent variable, z, (D_z,)
        emission_dim (int): number of dimensions for the hidden layers, (H_z,)
    """

    def __init__(self, input_dim: int, z_dim: int, emission_dim: int):
        super(LinearEmitter, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input_loc = nn.Linear(emission_dim, input_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        # TODO: Add other non-linearities (e.g. Tanh, LeakyRelU, Softplus)

    def forward(
        self, z_t: TensorType["batch", "latent_dim"]
    ) -> TensorType["batch", "input_dim"]:
        """
        Given the latent z at a particular time step t we return the vector of
        means that parameterizes the gaussian distribution `p(x_t|z_t)`

        Args:
            z_t (tensor): input latent vector, (B, D_z)

        Returns:
            mu (tensor): output mean for transition function, (B, D_x)
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        mu = self.lin_hidden_to_input_loc(h2)
        return mu


class ConvEmitter(nn.Module):
    """
    Parameterizes the normal observation likelihood `p(y_t | z_t)`
    """

    def __init__(
        self,
        width: int,
        height: int,
        input_channels: int,
        z_dim: int,
        emission_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        kernel_size = 4
        padding = int((kernel_size - 1) / 2)
        layers = 2
        stride = 2
        self.cnn_dim = min(width, height) // 2**layers
        self.cnn_shape = (emission_channels[0], self.cnn_dim, self.cnn_dim)

        # Return to original dimension using ConvCNNs
        self.lin_z_to_hidden = nn.Linear(z_dim, np.prod(self.cnn_shape))
        self.lin_hidden_to_hidden = nn.ConvTranspose2d(
            emission_channels[0],
            emission_channels[1],
            kernel_size,
            stride,
            padding,
            bias=True,
        )
        self.lin_hidden_to_conv_loc = nn.ConvTranspose2d(
            emission_channels[1],
            input_channels,
            kernel_size,
            stride,
            padding,
            bias=True,
        )
        self.lin_hidden_to_conv_scale = nn.ConvTranspose2d(
            emission_channels[1],
            input_channels,
            kernel_size,
            stride,
            padding,
            bias=True,
        )

        # Non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(
        self, z_t: TensorType["batch", "latent_dim"]
    ) -> Tuple[
        TensorType["batch", "n_channels", "height", "width"],
        TensorType["batch", "n_channels", "height", "width"],
    ]:
        """
        Given z_t, calculate mean and variance to parameterize the normal distribution `p(y_t | z_t)`

        Args:
            z_t (tensor): input latent vector, (B, D_z)

        Returns:
            mu (tensor): output mean for transition function, (B, C, H, W)
        """

        batch_size = z_t.shape[0]
        h1 = self.relu(self.lin_z_to_hidden(z_t)).view(batch_size, *self.cnn_shape)
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        loc = self.tanh(self.lin_hidden_to_conv_loc(h2))
        scale = self.softplus(self.lin_hidden_to_conv_scale(h2)).clamp(min=1e-4)
        return loc, scale


class ConvLSTMEmitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(
        self,
        input_channels,
        z_channels,
        emission_channels,
        kernel_size,
        bottleneck,
        n_layers,
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.deconv_layers = []

        self.lin_z_to_hidden = nn.Conv2d(
            z_channels, emission_channels, kernel_size, 1, padding, bias=True
        )
        if bottleneck:
            stride = 2
            kernel_size = 4
            padding = int((kernel_size - 1) / 2)
            for layer in range(n_layers - 1):
                next_emission_channels = emission_channels // 2
                self.deconv_layers.append(
                    nn.ConvTranspose2d(
                        emission_channels,
                        next_emission_channels,
                        kernel_size,
                        stride,
                        padding,
                        bias=True,
                    ).cuda()
                )
                emission_channels = next_emission_channels

            self.lin_hidden_to_input_loc = nn.ConvTranspose2d(
                emission_channels,
                input_channels,
                kernel_size,
                stride,
                padding,
                bias=True,
            )
            self.lin_hidden_to_input_scale = nn.ConvTranspose2d(
                emission_channels,
                input_channels,
                kernel_size,
                stride,
                padding,
                bias=True,
            )
        else:
            # initialize the three conv transformations used in the neural network
            for layer in range(n_layers - 1):
                self.deconv_layers.append(
                    nn.Conv2d(
                        emission_channels,
                        emission_channels,
                        kernel_size,
                        1,
                        padding,
                        bias=True,
                    ).cuda()
                )
            self.lin_hidden_to_input_loc = nn.Conv2d(
                emission_channels, input_channels, kernel_size, 1, padding, bias=True
            )
            self.lin_hidden_to_input_scale = nn.Conv2d(
                emission_channels, input_channels, kernel_size, 1, padding, bias=True
            )
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        for layer in self.deconv_layers:
            h1 = self.relu(layer(h1))
        loc = self.tanh(self.lin_hidden_to_input_loc(h1))
        scale = self.softplus(self.lin_hidden_to_input_scale(h1)).clamp(
            min=scale_clamper
        )
        return loc, scale
