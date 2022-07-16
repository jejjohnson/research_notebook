# code adapted from: https://github.com/guxd/deepHMM/blob/master/modules.py
import torch.nn as nn
import torch


class ObservationEncoder(nn.Module):
    """
    Encoder for encoding 2d data to be feed as RNN's input
    Args:
    """

    def __init__(
        self, width, height, input_channels, encoder_channels, kernel_size, n_layers
    ):
        super().__init__()
        # Parameter initialization
        padding = int((kernel_size - 1) / 2)
        stride = 2
        self.conv_layers = []

        # Initialize the three conv transformations used in the neural network
        for layer in range(n_layers):
            if layer == 0:
                temp_encoder_channels = encoder_channels
                encoder_channels = input_channels
                next_encoder_channels = temp_encoder_channels
            else:
                next_encoder_channels = encoder_channels * 2
            self.conv_layers.append(
                nn.Conv2d(
                    encoder_channels,
                    next_encoder_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=True,
                ).cuda()
            )
            encoder_channels = next_encoder_channels

        # Initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t):
        batch_size = z_t.shape[0]
        h1 = z_t
        for layer in self.conv_layers:
            h1 = self.relu(layer(h1))

        return h1


class Encoder(nn.Module):
    def __init__(
        self,
        embedder,
        input_size,
        hidden_size,
        bidir,
        n_layers,
        dropout=0.5,
        noise_radius=0.2,
    ):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.noise_radius = noise_radius
        self.n_layers = n_layers
        self.bidir = bidir
        assert type(self.bidir) == bool
        self.dropout = dropout

        self.embedding = embedder  # nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(
            input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidir
        )
        self.init_h = nn.Parameter(
            torch.randn(self.n_layers * (1 + self.bidir), 1, self.hidden_size),
            requires_grad=True,
        )  # learnable h0
        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():  # initialize the gate weights with orthogonal
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, inputs, input_lens=None, init_h=None, noise=False):
        # init_h: [n_layers*n_dir x batch_size x hid_size]
        if self.embedding is not None:
            inputs = self.embedding(
                inputs
            )  # input: [batch_sz x seq_len] -> [batch_sz x seq_len x emb_sz]

        batch_size, seq_len, emb_size = inputs.size()
        inputs = F.dropout(inputs, self.dropout, self.training)  # dropout

        if input_lens is not None:  # sort and pack sequence
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)
            inputs = pack_padded_sequence(
                inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True
            )

        if init_h is None:
            init_h = self.init_h.expand(
                -1, batch_size, -1
            ).contiguous()  # use learnable initial states, expanding along batches
        # self.rnn.flatten_parameters() # time consuming!!
        hids, h_n = self.rnn(inputs, init_h)  # hids: [b x seq x (n_dir*hid_sz)]
        # h_n: [(n_layers*n_dir) x batch_sz x hid_sz] (2=fw&bw)
        if input_lens is not None:  # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(
            self.n_layers, (1 + self.bidir), batch_size, self.hidden_size
        )  # [n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1]  # get the last layer [n_dirs x batch_sz x hid_sz]
        enc = (
            h_n.transpose(0, 1).contiguous().view(batch_size, -1)
        )  # [batch_sz x (n_dirs*hid_sz)]
        # if enc.requires_grad:
        #    enc.register_hook(self.store_grad_norm) # store grad norm
        # norms = torch.norm(enc, 2, 1) # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        # enc = torch.div(enc, norms.unsqueeze(1).expand_as(enc)+1e-5)
        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(
                means=torch.zeros(enc.size(), device=inputs.device),
                std=self.noise_radius,
            )
            enc = enc + gauss_noise

        return enc, hids


class RnnEncoder(nn.Module):
    """
    RNN encoder that outputs hidden states h_t using x_{t:T}
    Parameters
    ----------
    input_dim: int
        Dim. of inputs
    rnn_dim: int
        Dim. of RNN hidden states
    n_layer: int
        Number of layers of RNN
    drop_rate: float [0.0, 1.0]
        RNN dropout rate between layers
    bd: bool
        Use bi-directional RNN or not
    Returns
    -------
    h_rnn: tensor (b, T_max, rnn_dim * n_direction)
        RNN hidden states at every time-step
    """

    def __init__(
        self,
        input_dim,
        rnn_dim,
        n_layer=1,
        drop_rate=0.0,
        bd=False,
        nonlin="relu",
        rnn_type="rnn",
        orthogonal_init=False,
        reverse_input=True,
    ):
        super().__init__()
        self.n_direction = 1 if not bd else 2
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.n_layer = n_layer
        self.drop_rate = drop_rate
        self.bd = bd
        self.nonlin = nonlin
        self.reverse_input = reverse_input

        if not isinstance(rnn_type, str):
            raise ValueError("`rnn_type` should be type str.")
        self.rnn_type = rnn_type
        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=rnn_dim,
                nonlinearity=nonlin,
                batch_first=True,
                bidirectional=bd,
                num_layers=n_layer,
                dropout=drop_rate,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=rnn_dim,
                batch_first=True,
                bidirectional=bd,
                num_layers=n_layer,
                dropout=drop_rate,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=rnn_dim,
                batch_first=True,
                bidirectional=bd,
                num_layers=n_layer,
                dropout=drop_rate,
            )
        else:
            raise ValueError(
                "`rnn_type` must instead be ['rnn', 'gru', 'lstm'] %s" % rnn_type
            )

        if orthogonal_init:
            self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def calculate_effect_dim(self):
        return self.rnn_dim * self.n_direction

    def init_hidden(self, trainable=True):
        if self.rnn_type == "lstm":
            h0 = nn.Parameter(
                torch.zeros(self.n_layer * self.n_direction, 1, self.rnn_dim),
                requires_grad=trainable,
            )
            c0 = nn.Parameter(
                torch.zeros(self.n_layer * self.n_direction, 1, self.rnn_dim),
                requires_grad=trainable,
            )
            return h0, c0
        else:
            h0 = nn.Parameter(
                torch.zeros(self.n_layer * self.n_direction, 1, self.rnn_dim),
                requires_grad=trainable,
            )
            return h0

    def forward(self, x, seq_lengths):
        """
        x: pytorch packed object
            input packed data; this can be obtained from
            `util.get_mini_batch()`
        h0: tensor (n_layer * n_direction, b, rnn_dim)
        seq_lengths: tensor (b, )
        """
        # if self.rnn_type == 'lstm':
        #     _h_rnn, _ = self.rnn(x, (h0, c0))
        # else:
        #     _h_rnn, _ = self.rnn(x, h0)
        _h_rnn, _ = self.rnn(x)
        if self.reverse_input:
            h_rnn = pad_and_reverse(_h_rnn, seq_lengths)
        else:
            h_rnn, _ = nn.utils.rnn.pad_packed_sequence(_h_rnn, batch_first=True)
        return h_rnn
