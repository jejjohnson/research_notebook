"""Training entry point using hydra-zen for type-safe config management.

hydra-zen benefits over pure Hydra:
- No YAML boilerplate: configs are defined as Python dataclasses
- Type-safe: configs are validated at runtime using Python type hints
- Auto-generated: configs are automatically generated from function signatures
- Composable: configs can be composed programmatically
"""

from __future__ import annotations

from hydra_zen import ZenStore, builds, launch, make_config

from research_notebook.models.baseline import BaselineModel
from research_notebook.trainers.trainer import Trainer
from research_notebook.utils.reproducibility import seed_everything


# --- Build structured configs from Python classes ---
# builds() automatically generates a dataclass config from a class/function
ModelConfig = builds(
    BaselineModel,
    hidden_size=64,
    num_layers=2,
    populate_full_signature=True,
)

TrainerConfig = builds(
    Trainer,
    model="${model}",  # interpolate model config
    lr=1e-3,
    epochs=10,
    populate_full_signature=True,
)

# --- Compose a top-level config ---
# make_config() creates a config dataclass from keyword arguments
ExperimentConfig = make_config(
    model=ModelConfig,
    trainer=TrainerConfig,
    seed=42,
)

# --- Register configs in the Zen store ---
store = ZenStore()
store(ModelConfig, name="baseline", group="model")
store(ExperimentConfig, name="experiment")
store.add_to_hydra_store()


def run_experiment(cfg: ExperimentConfig) -> None:  # type: ignore[valid-type]
    """Run the experiment with the given config.

    Args:
        cfg: The experiment configuration.
    """
    seed_everything(cfg.seed)

    # Instantiate objects from configs (hydra-zen handles this automatically)
    model = BaselineModel(
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
    )
    trainer = Trainer(
        model=model,
        lr=cfg.trainer.lr,
        epochs=cfg.trainer.epochs,
    )

    metrics = trainer.train(train_data=None)
    print(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    # launch() is hydra-zen's equivalent of @hydra.main
    # It handles CLI argument parsing and config composition
    (jobs,) = launch(
        ExperimentConfig,
        run_experiment,
        overrides=["seed=42"],
    )
