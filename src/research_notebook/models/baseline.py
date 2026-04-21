"""Baseline model implementation."""

from __future__ import annotations


class BaselineModel:
    """A simple baseline model.

    Args:
        hidden_size: Size of the hidden layer.
        num_layers: Number of layers.
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 2) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # TODO: Initialize model weights

    def forward(self, x: object) -> object:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Model output.
        """
        # TODO: Implement forward pass
        return x

    def __repr__(self) -> str:
        return (
            f"BaselineModel(hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers})"
        )
