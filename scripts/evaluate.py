"""Evaluation script placeholder."""

from __future__ import annotations

import json
from pathlib import Path


def evaluate(
    model_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path = "results/metrics",
) -> dict:
    """Evaluate a trained model.

    Args:
        model_path: Path to the saved model.
        data_path: Path to the evaluation data.
        output_dir: Directory to save evaluation metrics.

    Returns:
        Dictionary of evaluation metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Load model and data, run evaluation
    metrics = {
        "accuracy": 0.0,
        "loss": 0.0,
        "f1": 0.0,
    }

    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation metrics saved to {metrics_path}")
    return metrics


if __name__ == "__main__":
    evaluate(
        model_path="results/models/model.pt",
        data_path="data/processed/test",
    )
