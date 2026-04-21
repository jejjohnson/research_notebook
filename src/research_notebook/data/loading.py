"""Placeholder data loading functions."""

from __future__ import annotations

from pathlib import Path


def load_raw_data(data_dir: str | Path) -> dict:
    """Load raw data from the specified directory.

    Args:
        data_dir: Path to the raw data directory.

    Returns:
        Dictionary containing loaded data arrays.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {data_dir}")
    # TODO: Implement data loading logic
    return {"data": None, "labels": None}


def load_processed_data(data_dir: str | Path) -> dict:
    """Load processed data from the specified directory.

    Args:
        data_dir: Path to the processed data directory.

    Returns:
        Dictionary containing train/val/test splits.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
    # TODO: Implement processed data loading logic
    return {"train": None, "val": None, "test": None}
