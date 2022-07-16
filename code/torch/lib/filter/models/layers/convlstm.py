# Code Adopted from: https://github.com/CJHJ/convolutional-neural-markov-model/blob/master/models/convlstm.py
"""Vanilla LSTM model."""
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable

"""Vanilla LSTM model."""


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, device):
        super(ConvLSTMCell, self).__init__()

        # assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.device = device

        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whi = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whf = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whc = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Who = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(
                self.device
            )
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(
                self.device
            )
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(
                self.device
            )
        else:
            assert shape[0] == self.Wci.size()[2], "Input Height Mismatched!"
            assert shape[1] == self.Wci.size()[3], "Input Width Mismatched!"
        return (
            Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(
                self.device
            ),
            Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(
                self.device
            ),
        )


class ConvLSTM(nn.Module):
    """Vanilla ConvLSTM prediction model."""

    def __init__(
        self, input_channels, hidden_channels, kernel_size, pred_input_dim, device="cpu"
    ):
        super(ConvLSTM, self).__init__()

        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.pred_input_dim = pred_input_dim

        self.device = device

        # Define the ConvLSTM layer
        self._all_layers = []
        for i in range(self.num_layers):
            name = "cell{}".format(i)
            cell = ConvLSTMCell(
                self.input_channels[i],
                self.hidden_channels[i],
                self.kernel_size,
                device,
            ).to(device)
            setattr(self, name, cell)
            self._all_layers.append(cell)

        # Prediction layer
        self.pred_conv = nn.Conv2d(
            self.pred_input_dim * self.hidden_channels[-1],
            self.input_channels[0],
            1,
            1,
            0,
            bias=True,
        ).to(device)
        self.tanh = nn.Tanh()

    def forward(self, input_tensor, pred_length, h_0=None, c_0=None):
        """Usual (no teacher-forcing) forward pass"""
        bsize, seq_len, input_channel, height, width = input_tensor.size()
        if pred_length == 0:
            internal_state = torch.zeros(
                bsize, seq_len + 1, self.hidden_channels[-1], height, width
            ).to(self.device)
        else:
            internal_state = torch.Tensor(
                bsize, self.hidden_channels[-1] * self.pred_input_dim, height, width
            ).to(self.device)

        temp_hidden_and_cell = []
        for step in range(seq_len):
            x = input_tensor[:, step, :, :, :]
            for i in range(self.num_layers):
                # All cells are initialized in the first step
                name = "cell{}".format(i)
                if step == 0:
                    (h, c) = getattr(self, name).init_hidden(
                        batch_size=bsize,
                        hidden=self.hidden_channels[i],
                        shape=(height, width),
                    )
                    if h_0 is not None and c_0 is not None:
                        (h, c) = (h_0, c_0)

                    temp_hidden_and_cell.append((h, c))

                # Do forward
                (h, c) = temp_hidden_and_cell[i]
                x, new_c = getattr(self, name)(x, h, c)
                temp_hidden_and_cell[i] = (x, new_c)

            # Internal state
            internal_start = step * self.hidden_channels[-1]
            internal_stop = internal_start + self.hidden_channels[-1]
            if pred_length != 0:
                if internal_start < internal_state.shape[1]:
                    internal_state[:, internal_start:internal_stop, :, :] = x
                else:
                    internal_state = torch.cat(
                        (internal_state[:, self.hidden_channels[-1] :, :, :], x), 1
                    )
            else:
                internal_state[:, step + 1, :, :, :] = x

        # Prediction step
        predictions = torch.Tensor(bsize, pred_length, input_channel, height, width).to(
            self.device
        )
        for step in range(pred_length):
            predictions[:, step, :, :, :] = self.tanh(self.pred_conv(internal_state))
            x = predictions[:, step, :, :, :]
            for i in range(self.num_layers):
                name = "cell{}".format(i)

                # Do forward
                (h, c) = temp_hidden_and_cell[i]
                x, new_c = getattr(self, name)(x, h, c)
                temp_hidden_and_cell[i] = (x, new_c)

            # Update internal state
            internal_state = torch.cat(
                (internal_state[:, self.hidden_channels[-1] :, :, :], x), 1
            )

        return predictions, internal_state

    def forward_with_gt(self, input_tensor, pred_length, h_0=None, c_0=None):
        """Forward pass with teacher forcing"""
        bsize, seq_len, input_channel, height, width = input_tensor.size()
        internal_state = torch.zeros(
            bsize, self.hidden_channels[-1] * self.pred_input_dim, height, width
        ).to(self.device)

        start_pred_index = seq_len - pred_length
        predictions = torch.Tensor(bsize, pred_length, input_channel, height, width).to(
            self.device
        )

        temp_hidden_and_cell = []
        for step in tqdm(range(seq_len)):
            # Start prediction if time
            if step >= start_pred_index:
                predictions[:, step - start_pred_index, :, :, :] = self.tanh(
                    self.pred_conv(internal_state)
                )

            x = input_tensor[:, step, :, :, :]
            for i in range(self.num_layers):
                # All cells are initialized in the first step
                name = "cell{}".format(i)
                if step == 0:
                    (h, c) = getattr(self, name).init_hidden(
                        batch_size=bsize,
                        hidden=self.hidden_channels[i],
                        shape=(height, width),
                    )
                    if h_0 is not None and c_0 is not None:
                        (h, c) = (h_0, c_0)

                    temp_hidden_and_cell.append((h, c))

                # Do forward
                (h, c) = temp_hidden_and_cell[i]
                x, new_c = getattr(self, name)(x, h, c)
                temp_hidden_and_cell[i] = (x, new_c)

            # Internal state
            internal_start = step * self.hidden_channels[-1]
            internal_stop = internal_start + self.hidden_channels[-1]
            if internal_start < internal_state.shape[1]:
                internal_state[:, internal_start:internal_stop, :, :] = x
            else:
                internal_state = torch.cat(
                    (internal_state[:, self.hidden_channels[-1] :, :, :], x), 1
                )

        return predictions


if __name__ == "__main__":
    # Test
    input_channels = 1
    hidden_channels = [128, 64, 64]
    kernel_size = 5
    pred_length = 5
    input_length = 25
    width = 100
    height = 100

    input_tensor = torch.Tensor(16, input_length, input_channels, width, height)

    pred_input_dim = 5

    target = torch.Tensor(16, pred_length, input_channels, width, height)

    model = ConvLSTM(input_channels, hidden_channels, kernel_size, pred_input_dim)

    loss_fn = torch.nn.MSELoss()
    output = model(input_tensor, pred_length)

    print(output)

    # res = torch.autograd.gradcheck(
    #     loss_fn, (output, target), eps=1e-6, raise_exception=True)
    # print(res)
