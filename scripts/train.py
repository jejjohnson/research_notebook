"""Training entry point using Hydra for configuration management."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from research_notebook.models.baseline import BaselineModel
from research_notebook.trainers.trainer import Trainer
from research_notebook.utils.reproducibility import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object. Automatically populated from
             configs/train.yaml and any overrides passed on the command line.

    Example usage:
        # Use defaults from configs/train.yaml
        python scripts/train.py

        # Override specific parameters
        python scripts/train.py training.lr=0.001 model=transformer

        # Multirun sweep
        python scripts/train.py -m training.lr=0.001,0.01,0.1
    """
    # Set random seed for reproducibility
    seed_everything(cfg.training.seed)

    # Initialize model from config
    model = BaselineModel(
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
    )

    # Run training (data loading omitted for placeholder)
    metrics = trainer.train(train_data=None)

    # Write DVC-tracked outputs to project root (Hydra may change CWD)
    project_root = Path(get_original_cwd())

    metrics_path = project_root / "results" / "metrics" / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "train_loss": metrics["train_loss"][-1] if metrics["train_loss"] else 0.0,
        "val_loss": metrics["val_loss"][-1] if metrics["val_loss"] else None,
        "epochs": cfg.training.epochs,
        "lr": cfg.training.lr,
    }
    metrics_path.write_text(json.dumps(summary, indent=2))

    loss_curve_path = project_root / "results" / "figures" / "loss_curve.csv"
    loss_curve_path.parent.mkdir(parents=True, exist_ok=True)
    with open(loss_curve_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, train_loss in enumerate(metrics["train_loss"]):
            val_loss = metrics["val_loss"][i] if i < len(metrics["val_loss"]) else ""
            writer.writerow([i, train_loss, val_loss])

    print(f"Training complete. Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
