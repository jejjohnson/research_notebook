"""Data preprocessing script placeholder."""

from __future__ import annotations

from pathlib import Path


def preprocess(
    raw_data_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
) -> None:
    """Preprocess raw data and save to the processed directory.

    Args:
        raw_data_dir: Path to the raw data directory.
        output_dir: Path to the output directory for processed data.
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement data preprocessing logic
    # Example steps:
    # 1. Load raw data files
    # 2. Clean and validate data
    # 3. Split into train/val/test
    # 4. Save processed splits

    print(f"Preprocessing data from {raw_data_dir} -> {output_dir}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
