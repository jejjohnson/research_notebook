# TODO: Add RNN Cells (e.g.  https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py)
import torch.nn as nn


def get_default_rnn_layer(
    input_dim: int,
    rnn_dim: int,
    num_layers: int,
    nonlinearity: str = "relu",
    batch_first: bool = True,
    bidirectional: bool = False,
):

    return nn.RNN(
        input_size=input_dim,
        hidden_size=rnn_dim,
        nonlinearity=nonlinearity,
        batch_first=batch_first,
        bidirectional=bidirectional,
        num_layers=num_layers,
    )
