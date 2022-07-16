# Code adapted from: https://github.com/hmsandager/Normalizing-flow-and-deep-kalman-filter/blob/main/dkf.py
from typing import Tuple
import torch
import torch.nn as nn
from torchtyping import TensorType

scale_clamper = 1e-4


class LinearCombiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{:t}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)

    Args:
        z_dim (int): num of dims for the latent variable, z
        rnn_dim (int): num of dims for the hidden rnn state from the input vector, x
    """

    def __init__(self, z_dim: int, rnn_dim: int) -> None:
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(
        self,
        z_t_1: TensorType["batch", "latent_dim"],
        h_rnn: TensorType["batch", "hidden_dim"],
    ) -> Tuple[TensorType["batch", "latent_dim"], TensorType["batch", "latent_dim"]]:
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{:t})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{:t})`

        Args:
            z_t_1 (tensor): latent variable input, (B, D_z)
            h_rnn (tensor): rnn latent variable input, (B, D_h)

        Returns:
            loc: output mean for transition function, size=(B, D_z)
            scale: output scale for the transition function, size=(B, D_z)
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale


class Flattener(nn.Module):
    """
    Flatten the input data
    """

    def __init__(
        self,
        width: int,
        height: int,
        input_channels: int,
        rnn_dim: int,
        flatten_channels: int,
        kernel_size: int,
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        stride = 2
        self.input_width = width // 2 ** len(flatten_channels)
        self.input_height = height // 2 ** len(flatten_channels)
        self.input_dim = flatten_channels[-1] * self.input_width * self.input_height

        # Two-layered convolution with a fully connected layer at last
        self.cnn_to_hidden = nn.Conv2d(
            input_channels, flatten_channels[0], kernel_size, stride, padding, bias=True
        )
        self.cnn_hidden_to_hidden = nn.Conv2d(
            flatten_channels[0],
            flatten_channels[1],
            kernel_size,
            stride,
            padding,
            bias=True,
        )
        self.lin_hidden_to_rnn = nn.Linear(self.input_dim, rnn_dim)

        # Non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(
        self, z_t: TensorType["batch", "n_channels", "height", "width"]
    ) -> TensorType["batch", "hidden_dims"]:
        """
        Return the flattened input to RNN
        """
        batch_size = z_t.shape[0]
        h1 = self.relu(self.cnn_to_hidden(z_t))
        h2 = self.relu(self.cnn_hidden_to_hidden(h1)).view(batch_size, -1)
        rnn_input = self.tanh(self.lin_hidden_to_rnn(h2))

        return rnn_input


class ConvLSTMCombiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_channels, rnn_channels, kernel_size):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        stride = 1

        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Conv2d(
            z_channels, rnn_channels, kernel_size, stride, padding, bias=True
        )
        self.lin_hidden_to_loc = nn.Conv2d(
            rnn_channels, z_channels, kernel_size, stride, padding, bias=True
        )
        self.lin_hidden_to_scale = nn.Conv2d(
            rnn_channels, z_channels, kernel_size, stride, padding, bias=True
        )
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined)).clamp(
            min=scale_clamper
        )
        # return loc, scale which can be fed into Normal
        return loc, scale
