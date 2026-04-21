"""Tests for data loading utilities."""

from __future__ import annotations

from pathlib import Path

from research_notebook.data.loading import load_processed_data, load_raw_data


def test_load_raw_data_returns_dict(tmp_path: Path) -> None:
    """Test that load_raw_data returns a dictionary."""
    result = load_raw_data(tmp_path)
    assert isinstance(result, dict)
    assert "data" in result
    assert "labels" in result


def test_load_processed_data_returns_dict(tmp_path: Path) -> None:
    """Test that load_processed_data returns a dictionary."""
    result = load_processed_data(tmp_path)
    assert isinstance(result, dict)
    assert "train" in result
    assert "val" in result
    assert "test" in result
