"""Tests for model implementations."""

from __future__ import annotations

from research_notebook.models.baseline import BaselineModel


def test_baseline_model_init() -> None:
    """Test BaselineModel initialization with default parameters."""
    model = BaselineModel()
    assert model.hidden_size == 64
    assert model.num_layers == 2


def test_baseline_model_custom_params() -> None:
    """Test BaselineModel initialization with custom parameters."""
    model = BaselineModel(hidden_size=128, num_layers=4)
    assert model.hidden_size == 128
    assert model.num_layers == 4


def test_baseline_model_repr() -> None:
    """Test BaselineModel string representation."""
    model = BaselineModel(hidden_size=32, num_layers=1)
    repr_str = repr(model)
    assert "BaselineModel" in repr_str
    assert "32" in repr_str
    assert "1" in repr_str


def test_baseline_model_forward() -> None:
    """Test BaselineModel forward pass returns input unchanged."""
    model = BaselineModel()
    x = [1, 2, 3]
    result = model.forward(x)
    assert result == x
